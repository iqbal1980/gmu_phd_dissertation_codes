import numpy as np
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Real
import time
from numba import njit
import random
import torch
import torch.nn as nn
import torch.optim as optim
from skopt.space import Real

#Random number not needed really!
random_number = 1

def HK_deltas_vstim_vresponse_graph_modified_v2(ggap=1.0, Ibg_init=0.0, Ikir_coef=0.94, dt=0.001, 
                                                cm=9.4, a=0.01, dx=0.06, F=9.6485e4, R=8.314e3, K_o=5):
    max_val = 0.51
    min_val = 0.5
    images = []

    for counter in np.arange(min_val, max_val, 0.01):
        ggapval = counter * ggap
        print(f"ggapval={ggapval}")
        A = simulate_process_modified_v2(ggapval, Ibg_init, Ikir_coef, dt, cm, a, dx, F, R, K_o)
        x = A[:, 0]
        y = A[:, 98:135]

        # Plot
        #plt.figure()
        #plt.plot(x, y, linewidth=3)
        #plt.title(f"G gap = {ggapval}")
        #images.append(f"Image{ggapval}.png")
        #plt.savefig(f"Image{ggapval}.png")
        #print("saved 1")
        
        plot_data2_modified(A)
        
        #print("saved 2")

    return images


@njit(parallel=False)
def simulate_process_modified_v2(g_gap_value, Ibg_init, Ikir_coef, dt, cm, a, dx, F, R, K_o):
    loop = 600000
    Ng = 200
    Vm = np.ones(Ng) * (-33)
    g_gap = g_gap_value
    eki1 = (g_gap * dt) / (dx**2 * cm)
    eki2 = dt / cm

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
            I_app[99] = 50.0
        else:
            I_app[99] = 0.0

        for kk in range(Ng):
            E_K = (R * 293 / F) * np.log(K_o/150)

            I_bg[kk] = Ibg_init * (Vm[kk] + 30)
            I_kir[kk] = Ikir_coef * np.sqrt(K_o) * ((Vm[kk] - E_K) / (1 + np.exp((Vm[kk] - E_K - 25) / 7)))

            if kk == 0:
                Vm[kk] += random_number * eki1 * (Vm[kk+1] - Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])
            elif kk == Ng-1:
                Vm[kk] += random_number * eki1 * (Vm[kk-1] - Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])
            elif kk == 98:
                Vm[kk] += random_number * eki1 * 0.6 * (Vm[kk+1] + Vm[kk-1] - 2 * Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])
            elif kk == 99:
                Vm[kk] += random_number * eki1 * 0.6 * (Vm[kk+1] + Vm[kk-1] - 2 * Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])
            elif kk == 100:
                Vm[kk] += random_number * eki1 * 0.6 * (Vm[kk+1] + Vm[kk-1] - 2 * Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])
            else:
                Vm[kk] += random_number * eki1 * (Vm[kk+1] + Vm[kk-1] - 2 * Vm[kk]) - eki2 * (I_bg[kk] + I_kir[kk] + I_app[kk])

            distance_m[kk] = kk * dx

            if kk == 99:
                vstims[kk] = Vm[kk]
            else:
                vresps[kk] = Vm[kk]

        A[j, 0] = t
        A[j, 1:] = Vm

    return A

@njit(parallel=False)
def plot_data2_modified(A):  
    dx = 0.06
    D = np.abs(A[-2, 98:135] - A[int(0.1 * len(A)), 98:135]) / np.abs(A[int(0.1 * len(A)), 98:135])[0]

    distance_m = dx * np.arange(99, 136)
    #plt.figure()
    #plt.plot(distance_m, D, '.', markersize=8)
    c = np.polyfit(distance_m, D, 1)
    y_est = np.polyval(c, distance_m)
    #plt.plot(distance_m, y_est, 'r--', linewidth=2)
    #plt.savefig(f"Image2.png")

#Bayesian Optimization Code
def objective(params):
    ggap, Ibg_init, Ikir_coef, dt, cm, a, dx, F, R, K_o = params
    
    #Run the simulation with the provided parameters
    A = simulate_process_modified_v2(ggap, Ibg_init, Ikir_coef, dt, cm, a, dx, F, R, K_o)
    
    dx = 0.06
    D = np.abs(A[-2, 98:135] - A[int(0.1 * len(A)), 98:135]) / np.abs(A[int(0.1 * len(A)), 98:135])[0]
    distance_m = dx * np.arange(99, 136)
    
    #Compute the polynomial coefficients from the model's result
    coefficients = np.polyfit(distance_m, D, 1)
    
    #Compute the loss as the squared difference between the model's coefficients and the target coefficients
    loss = (coefficients[0] - 2)**2 + (coefficients[1] - 3.2)**2
    
    return loss

#Define the Parameter Space, TODO: need to discuss the value ranges here!
space = [
    Real(0.5, 1.5, name="ggap"),
    Real(0, 1, name="Ibg_init"),
    Real(0.8, 1.0, name="Ikir_coef"),
    Real(0.0005, 0.005, name="dt"),
    Real(8, 11, name="cm"),
    Real(0, 0.05, name="a"),
    Real(0.05, 0.07, name="dx"),
    Real(9.5e4, 1e5, name="F"),
    Real(8.2e3, 8.4e3, name="R"),
    Real(4, 6, name="K_o")
]

# Generate training data
N_SAMPLES = 5000
X_train = np.zeros((N_SAMPLES, len(space)))
y_train = np.zeros(N_SAMPLES)

print("Generating training data...")
for i in range(N_SAMPLES):
    print("Sample->"+str(i))
    params = [float(s.rvs()[0]) for s in space]

    X_train[i, :] = params
    y_train[i] = objective(params)  # Assign the objective function output to y_train
    
print("Training data generation completed!")

# Neural Network Architecture
class RegressionNN(nn.Module):
    def __init__(self):
        super(RegressionNN, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(len(space), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.fc(x)

# Convert numpy arrays to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(-1)  # Add an extra dimension to match network output

# Hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 2000

# DataLoader
dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model, Loss, Optimizer
model = RegressionNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Weights Initialization
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)

model.apply(weights_init)





# Early stopping parameters
patience = 20
best_loss = float('inf')
counter = 0

# Training loop
for epoch in range(EPOCHS):
    for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(X_batch)
        
        # Loss
        loss = criterion(predictions, y_batch)
        
        # Backward pass
        loss.backward()
        
        # Parameter update
        optimizer.step()
    
    # Validation loss (using training set as a proxy in this example)
    with torch.no_grad():
        val_predictions = model(X_train_tensor)
        val_loss = criterion(val_predictions, y_train_tensor)
    
    print(f"Epoch {epoch + 1}/{EPOCHS}, Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}")
    
    # Early stopping logic
    if val_loss < best_loss:
        best_loss = val_loss
        counter = 0
    else:
        counter += 1
        print(f"EarlyStopping counter: {counter} out of {patience}")
        if counter >= patience:
            print("Early stopping.")
            break

print("Training complete!")





# Predict losses for training data
with torch.no_grad():  # No need to calculate gradients
    predicted_losses = model(X_train_tensor)

# Convert to numpy for easier handling
predicted_losses_np = predicted_losses.numpy().flatten()

# Find the index of the minimum loss
min_loss_idx = np.argmin(predicted_losses_np)

# Find the corresponding parameters
optimal_params = X_train[min_loss_idx, :]

print(f"Optimal parameters are: {optimal_params}")
print(f"Minimum loss is: {predicted_losses_np[min_loss_idx]}")
