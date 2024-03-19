import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from numba import njit
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants and parameters
MIN_VALUE = -80
MAX_VALUE = 40
global_dx = 1

# PINN model
class PerocyteModel(nn.Module):
    def __init__(self):
        super(PerocyteModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        return self.layers(x).view(-1, 4)  # Reshape the output to (batch_size, 4)

# Functions from verify_params11.py for numerical stability
@njit
def safe_log(x):
    if x <= 0:
        return MIN_VALUE
    return math.log(x)

@njit
def exponential_function(x, a):
    return np.exp(a * x)

@njit
def exponential_decay_function(x, A, B):
    return A * np.exp(B * x)

# Simulate process function (similar to verify_params11.py)
@njit(parallel=False)
def simulate_process_modified_v2(g_gap_value, Ibg_init, Ikir_coef, cm, dx, K_o):
    dt = 0.001
    F = 9.6485e4
    R = 8.314e3
    loop = 600000
    Ng = 200
    Vm = np.ones(Ng) * (-33)
    g_gap = g_gap_value.item()
    eki1 = (g_gap * dt) / (dx.item()**2 * cm.item())
    eki2 = dt / cm.item()

    I_bg = np.zeros(Ng) + Ibg_init
    I_kir = np.zeros(Ng)
    distance_m = np.zeros(Ng)
    vstims = np.zeros(Ng)
    vresps = np.zeros(Ng)

    A = np.zeros((loop, Ng + 1))
    I_app = np.zeros(Ng)

    for j in range(loop):
        t = j * dt
        if 100 <= t <= 400:
            I_app[99] = 70.0
        else:
            I_app[99] = 0.0

        for kk in range(Ng):
            E_K = (R * 293 / F) * safe_log(K_o.item()/150)
            I_bg[kk] = Ibg_init * (Vm[kk] + 30)
            I_kir[kk] = Ikir_coef.item() * np.sqrt(K_o.item()) * ((Vm[kk] - E_K) / (1 + exponential_function((Vm[kk] - E_K - 25) / 7, 1)))

            new_Vm = Vm[kk]
            if kk == 0:
                new_Vm += 3 * (Vm[kk+1] - Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])
            elif kk == Ng-1:
                new_Vm += eki1 * (Vm[kk-1] - Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])
            elif kk in {98, 99, 100}:
                new_Vm += eki1 * 0.6 * (Vm[kk+1] + Vm[kk-1] - 2 * Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])
            else:
                new_Vm += eki1 * (Vm[kk+1] + Vm[kk-1] - 2 * Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])

            # Clamp new_Vm to prevent overflow/underflow
            Vm[kk] = max(min(new_Vm, MAX_VALUE), MIN_VALUE)

            distance_m[kk] = kk * dx.item()

            if kk == 99:
                vstims[kk] = Vm[kk]
            else:
                vresps[kk] = Vm[kk]

        A[j, 0] = t
        A[j, 1:] = Vm

    return A

# Physics-based loss
def physics_based_loss(model, inputs, targets):
    params = model(inputs).detach().numpy()
    g_gap_value, Ikir_coef, cm, K_o = params[:, 0], params[:, 1], params[:, 2], params[:, 3]
    dx = global_dx
    Ibg_init = 0.7 * 0.94

    A = simulate_process_modified_v2(g_gap_value, Ibg_init, Ikir_coef, cm, dx, K_o)

    cellLength = 60
    D = np.abs(A[399998, 101:135] - A[99000, 101:135]) / (np.abs(A[99000, 101:135])[0] + 1e-8)
    distance_m = cellLength * np.arange(102 - 102, 136 - 102)

    A_initial = D[0]
    B_initial = (np.log(D[1] + 1e-8) - np.log(D[0] + 1e-8)) / (distance_m[1] - distance_m[0])

    simulated_decay = exponential_decay_function(distance_m, A_initial, B_initial)
    reference_decay = 1 * np.exp(-0.003 * distance_m)

    loss = np.sum((simulated_decay - reference_decay) ** 2)
    return torch.tensor(loss, requires_grad=True)

# Data-fitting loss
def data_fitting_loss(model, inputs, targets):
    params = model(inputs)
    loss = nn.MSELoss()(params, targets.view(-1, 4))
    return loss

# Training loop
def train_pinn(model, optimizer, train_data, val_data, epochs):
    for epoch in range(epochs):
        logging.info(f"Starting epoch [{epoch+1}/{epochs}]")
        
        # Training
        model.train()
        train_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_data, 1):
            optimizer.zero_grad()
            physics_loss = physics_based_loss(model, inputs, targets)
            data_loss = data_fitting_loss(model, inputs, targets)
            loss = physics_loss + data_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logging.info(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(train_data)}], "
                             f"Train Loss: {loss.item():.4f}")
        
        train_loss /= len(train_data)
        logging.info(f"Epoch [{epoch+1}/{epochs}], Average Train Loss: {train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_data, 1):
                physics_loss = physics_based_loss(model, inputs, targets)
                data_loss = data_fitting_loss(model, inputs, targets)
                loss = physics_loss + data_loss
                val_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    logging.info(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(val_data)}], "
                                 f"Val Loss: {loss.item():.4f}")
        
        val_loss /= len(val_data)
        logging.info(f"Epoch [{epoch+1}/{epochs}], Average Val Loss: {val_loss:.4f}")
        
        logging.info(f"Epoch [{epoch+1}/{epochs}] completed")

# Generate training and validation data
def generate_data(num_samples):
    inputs = torch.rand(num_samples, 4)
    targets = torch.rand(num_samples, 4)
    return inputs, targets

# Main function
def main():
    logging.info("Starting PINN training...")

    # Generate training and validation data
    train_inputs, train_targets = generate_data(1000)
    val_inputs, val_targets = generate_data(200)
    train_data = [(train_inputs[i], train_targets[i]) for i in range(len(train_inputs))]
    val_data = [(val_inputs[i], val_targets[i]) for i in range(len(val_inputs))]

    # Create PINN model and optimizer
    model = PerocyteModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the PINN
    epochs = 100
    train_pinn(model, optimizer, train_data, val_data, epochs)

    logging.info("PINN training completed.")

    # Parameter estimation
    test_inputs = torch.rand(100, 4)
    with torch.no_grad():
        estimated_params = model(test_inputs)
    logging.info("Estimated parameters:")
    logging.info(estimated_params)

if __name__ == "__main__":
    main()