import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm

# Constants
MIN_VALUE = -80
MAX_VALUE = 40
MIN_VM = -80  # Minimum physiologically reasonable value for Vm
MAX_VM = 40  # Maximum physiologically reasonable value for Vm

# Safe logarithm function
def safe_log(x):
    return torch.where(x <= 0, MIN_VALUE, torch.log(x))

# Exponential function
def exponential_function(x, a):
    return torch.exp(a * x)

# Physics Informed Neural Network (PINN)
class PINN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PINN, self).__init__()
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x

def ode_system(t, y, g_gap, Ibg_init, Ikir_coef, cm, dx, K_o, I_app_val):
    #print("Start ODE system")
    # Constants within the function
    F = 9.6485e4
    R = 8.314e3
    Ng = len(y)
    Vm = torch.from_numpy(y)  # Convert y to a PyTorch tensor

    # Calculating currents
    I_bg = Ibg_init * (Vm + 30)
    E_K = (R * 293 / F) * safe_log(K_o / 150)
    I_kir = Ikir_coef * torch.sqrt(K_o) * ((Vm - E_K) / (1 + exponential_function((Vm - E_K - 25) / 7, 1)))

    # Application current adjustment
    I_app = torch.zeros_like(Vm)
    I_app[99] = I_app_val if 100 <= t <= 400 else 0.0

    dVm_dt = torch.zeros_like(Vm)
    for kk in range(Ng):
        if kk == 0:
            dVm_dt[kk] = (g_gap / (dx**2 * cm)) * (Vm[kk+1] - Vm[kk]) - (1 / cm) * (I_bg[kk] + I_kir[kk] + I_app[kk])
        elif kk == Ng-1:
            dVm_dt[kk] = (g_gap / (dx**2 * cm)) * (Vm[kk-1] - Vm[kk]) - (1 / cm) * (I_bg[kk] + I_kir[kk] + I_app[kk])
        else:
            dVm_dt[kk] = (g_gap / (dx**2 * cm)) * (Vm[kk+1] + Vm[kk-1] - 2 * Vm[kk]) - (1 / cm) * (I_bg[kk] + I_kir[kk] + I_app[kk])

    return dVm_dt.numpy()  # Convert the result back to a NumPy array

def ode_residual(t, Vm, g_gap, Ibg_init, Ikir_coef, cm, dx, K_o, I_app_val):
    dVm_dt = ode_system(t, Vm.detach().numpy(), g_gap, Ibg_init, Ikir_coef, cm, dx, K_o, I_app_val)
    return torch.from_numpy(dVm_dt)

def run_simulation(ggap, Ibg_init, Ikir_coef, cm, dx, K_o, I_app_val, t_span, Ng, time_in_ms, cell_index):
    print(f"Running simulation for sample {i+1}/{len(x_train)}")
    y0 = torch.ones(Ng) * (-33)  # Initial condition

    # Solving the ODE with specified method
    sol = solve_ivp(lambda t, y: ode_system(t, y, ggap, Ibg_init, Ikir_coef, cm, dx, K_o, I_app_val), t_span, y0.numpy(), method='Radau')

    # Extract the voltage for the specified time and cell.
    time_idx = np.argmin(np.abs(sol.t - time_in_ms))  # Find index of the closest time point to time_in_ms
    voltage = sol.y[cell_index, time_idx]  # Extract voltage

    return voltage

def train_pinn(pinn, x_train, y_train, x_collocation, num_epochs, learning_rate, alpha):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(pinn.parameters(), lr=learning_rate)

    print("Starting training...")
    for epoch in tqdm(range(num_epochs), desc="Training"):
        optimizer.zero_grad()
        
        # Data loss
        data_outputs = pinn(x_train)
        data_loss = criterion(data_outputs, y_train.unsqueeze(1))
        
        # Physics-based loss
        t_collocation = x_collocation[:, 0]  # Time points for collocation
        g_gap, Ibg_init, Ikir_coef, cm, dx, K_o, I_app_val = x_collocation[:, 1:].T
        Vm_collocation = pinn(x_collocation)
        residual = ode_residual(t_collocation, Vm_collocation, g_gap, Ibg_init, Ikir_coef, cm, dx, K_o, I_app_val)
        physics_loss = torch.mean(residual ** 2)
        
        # Total loss
        loss = data_loss + alpha * physics_loss
        loss.backward()
        optimizer.step()

    return pinn

# Generate training data
Ibg_init_val = 0.7 * 0.94
t_span = (0, 600)
Ng = 200
time_in_ms = 300  # Example time in ms
cell_index = 100  # Example cell index

x_train = torch.rand(1000, 7)  # Random input parameters
y_train = torch.zeros(1000)

print("Generating training data...")
for i in range(len(x_train)):
    ggap, Ibg_init, Ikir_coef, cm, dx, K_o, I_app_val = x_train[i]
    y_train[i] = run_simulation(ggap, Ibg_init, Ikir_coef, cm, dx, K_o, I_app_val, t_span, Ng, time_in_ms, cell_index)

# Generate collocation points
num_collocation_points = 1000
t_collocation = torch.rand(num_collocation_points, 1) * (t_span[1] - t_span[0]) + t_span[0]
x_collocation = torch.rand(num_collocation_points, 7)
x_collocation = torch.cat((t_collocation, x_collocation), dim=1)

# Create and train the PINN
input_size = 8  # Include time as an input
hidden_size = 64
output_size = 1
num_epochs = 1000
learning_rate = 0.001
alpha = 1.0  # Hyperparameter for physics-based loss

pinn = PINN(input_size, hidden_size, output_size)
pinn = train_pinn(pinn, torch.cat((x_train, torch.ones(len(x_train), 1) * time_in_ms), dim=1), y_train, x_collocation, num_epochs, learning_rate, alpha)

# Test the trained PINN
test_input = torch.tensor([[time_in_ms, 20.008702984240095, Ibg_init_val, 0.9, 11.0, 1, 4.502039575569403, -70]])
predicted_voltage = pinn(test_input)
print(f"Predicted voltage at time {time_in_ms}ms for cell index {cell_index}: {predicted_voltage.item()}")