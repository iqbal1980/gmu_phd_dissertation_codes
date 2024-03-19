import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging

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

def simulate_process_modified_v2(params, Ibg_init, dx):
    g_gap_value = params[:, 0].unsqueeze(1)
    Ikir_coef = params[:, 1].unsqueeze(1)
    cm = params[:, 2].unsqueeze(1)
    K_o = params[:, 3].unsqueeze(1)
    
    dt = 0.001
    F = 9.6485e4
    R = 8.314e3
    loop = 600000
    Ng = 200
    Vm = torch.ones(Ng) * (-33)
    g_gap = g_gap_value
    eki1 = (g_gap * dt) / (dx**2 * cm)
    eki2 = dt / cm
    
    I_bg = torch.zeros(Ng) + Ibg_init
    I_kir = torch.zeros(Ng)
    distance_m = torch.zeros(Ng)
    vstims = torch.zeros(Ng)
    vresps = torch.zeros(Ng)
    
    A = torch.zeros((loop, Ng + 1))
    I_app = torch.zeros(Ng)
    
    for j in range(loop):
        t = j * dt
        if 100 <= t <= 400:
            I_app[99] = 70.0
        else:
            I_app[99] = 0.0
        
        for kk in range(Ng):
            E_K = (R * 293 / F) * torch.log(K_o / 150)
            I_bg[kk] = Ibg_init * (Vm[kk] + 30)
            I_kir[kk] = Ikir_coef * torch.sqrt(K_o) * ((Vm[kk] - E_K) / (1 + torch.exp((Vm[kk] - E_K - 25) / 7)))
            
            new_Vm = Vm[kk]
            if kk == 0:
                new_Vm = new_Vm + 3 * (Vm[kk+1] - Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])
            elif kk == Ng-1:
                new_Vm = new_Vm + eki1 * (Vm[kk-1] - Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])
            elif kk in {98, 99, 100}:
                new_Vm = new_Vm + eki1 * 0.6 * (Vm[kk+1] + Vm[kk-1] - 2 * Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])
            else:
                new_Vm = new_Vm + eki1 * (Vm[kk+1] + Vm[kk-1] - 2 * Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])
            
            # Clamp new_Vm to prevent overflow/underflow
            Vm[kk] = torch.clamp(new_Vm, MIN_VALUE, MAX_VALUE)
            
            distance_m[kk] = kk * dx
            
            if kk == 99:
                vstims[kk] = Vm[kk]
            else:
                vresps[kk] = Vm[kk]
        
        A[j, 0] = t
        A[j, 1:] = Vm
    
    return A

# Physics-based loss
def physics_based_loss(model, inputs, targets):
    params = model(inputs)
    g_gap_value, Ikir_coef, cm, K_o = torch.split(params, 1, dim=1)
    dx = global_dx
    Ibg_init = 0.7 * 0.94

    A = simulate_process_modified_v2(params, Ibg_init, dx)
    cellLength = 60
    D = torch.abs(A[399998, 101:135] - A[99000, 101:135]) / torch.abs(A[99000, 101:135])[0]
    distance_m = cellLength * torch.arange(102 - 102, 136 - 102).float()

    simulated_decay = D[0] * torch.exp(torch.log(D[1] / D[0]) / (distance_m[1] - distance_m[0]) * distance_m)
    reference_decay = torch.ones_like(distance_m) * torch.exp(-0.003 * distance_m)

    loss = torch.sum((simulated_decay - reference_decay) ** 2)
    return loss

# Data-fitting loss
def data_fitting_loss(model, inputs, targets):
    params = model(inputs)
    loss = nn.MSELoss()(params, targets)
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