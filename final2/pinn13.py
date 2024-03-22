import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numba import njit
from numba.extending import overload
from scipy.optimize import curve_fit
from numba import generated_jit
from numba import types
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)


MIN_VALUE = -80
MAX_VALUE = 40


@njit
def safe_log(x):
    if x <= 0:
        return np.log(MIN_VALUE)
    return np.log(float(x))
    
@njit
def exponential_function(x, a):
    return np.exp(a * x)

@njit
def exponential_decay_function(x, A, B):
    return A * np.exp(B * x)

MIN_VM = -80 # Minimum physiologically reasonable value for Vm
MAX_VM = 40 # Maximum physiologically reasonable value for Vm






@njit(parallel=False)
def ode_system(t, y, g_gap, Ibg_init, Ikir_coef, cm, dx, K_o, I_app_val, stimulation_time_start, stimulation_time_end, activated_cell_index):
    F = 9.6485e4
    R = 8.314e3
    Ng = len(y)
    Vm = y
    I_bg = Ibg_init * (Vm + 30)
    E_K = (R * 293 / F) * safe_log(K_o / 150)
    I_kir = Ikir_coef * np.sqrt(K_o) * ((Vm - E_K) / (1 + exponential_function((Vm - E_K - 25) / 7, 1)))
    I_app = np.zeros_like(Vm)
    I_app[activated_cell_index] = I_app_val if stimulation_time_start <= t <= stimulation_time_end else 0.0
    dVm_dt = np.zeros_like(Vm)
    for kk in range(Ng):
        if kk == 0:
            dVm_dt[kk] = (g_gap / (dx**2 * cm)) * (Vm[kk+1] - Vm[kk]) - (1 / cm) * (I_bg[kk] + I_kir[kk] + I_app[kk])
        elif kk == Ng-1:
            dVm_dt[kk] = (g_gap / (dx**2 * cm)) * (Vm[kk-1] - Vm[kk]) - (1 / cm) * (I_bg[kk] + I_kir[kk] + I_app[kk])
        else:
            dVm_dt[kk] = (g_gap / (dx**2 * cm)) * (Vm[kk+1] + Vm[kk-1] - 2 * Vm[kk]) - (1 / cm) * (I_bg[kk] + I_kir[kk] + I_app[kk])
    return dVm_dt








def HK_deltas_vstim_vresponse_graph_modified_v2(ggap, Ibg_init, Ikir_coef, cm, dx, K_o, I_app_val, y0_init, t_span, Ng, stimulation_time_start, stimulation_time_end, what_cell_is_activated):
    y0 = np.ones(Ng) * y0_init # Initial condition
    #sol = solve_ivp(ode_system, t_span, y0, args=(ggap, Ibg_init, Ikir_coef, cm, dx, K_o, I_app_val, stimulation_time_start, stimulation_time_end, activated_cell_index), method='Radau')#, max_step=0.01)
    #sol = solve_ivp(ode_system, t_span, y0, args=(ggap, Ibg_init, Ikir_coef, cm, dx, K_o, I_app_val, stimulation_time_start, stimulation_time_end, activated_cell_index), method='RK23')#, max_step=0.01)
    sol = solve_ivp(ode_system, t_span, y0, args=(ggap, Ibg_init, Ikir_coef, cm, dx, K_o, I_app_val, stimulation_time_start, stimulation_time_end, activated_cell_index), method='RK45')#, max_step=0.01)
    return sol







    
    
def generate_random_params(param_bounds):
    ggap = np.random.choice(param_bounds[0])
    Ikir_coef = np.random.choice(param_bounds[1])
    cm = np.random.choice(param_bounds[2])
    K_o = np.random.choice(param_bounds[3])
    I_app_val = np.random.choice(param_bounds[4])
    time_in_s = np.random.choice(param_bounds[5])
    y0_init = np.random.choice(param_bounds[6])
    Ng = np.random.choice(param_bounds[7])
    activated_cell_index = int(np.random.choice(param_bounds[8]))  # Convert to integer

    print(y0_init)
    print(time_in_s)
    stimulation_time_start = random.uniform(0, int(0.75*time_in_s))
    stimulation_time_end = random.uniform(stimulation_time_start + 85, min(stimulation_time_start + 90 + random.uniform(0, 300), time_in_s))
    
    print("stimulation_time_start="+str(int(stimulation_time_start)))
    print("stimulation_time_end="+str(int(stimulation_time_end)))
    print("activated_cell_index="+str(activated_cell_index))

    t_span = (0, time_in_s)
    Ng = 200

    return ggap, Ikir_coef, cm, K_o, I_app_val, y0_init, t_span, Ng, stimulation_time_start, stimulation_time_end, activated_cell_index

# Example usage
param_bounds = [
    np.linspace(0.1, 35, 10),  # ggap (10 values) 0
    np.linspace(0.90, 0.96, 5),  # Ikir_coef (5 values) 1
    np.linspace(8, 11, 4),  # cm (4 values) 2
    np.linspace(1, 8, 8),  # K_o (8 values) 3
    np.linspace(-70, 70, 35),  # I_app_val (15 values) 4
    np.linspace(600, 600, 1),  # time_in_s (100 values) 5
    np.linspace(-33, -33, 1),  # y0_init (20 values) 6
    np.linspace(200, 200, 1),  # ng number of cells (200) 7
    np.arange(85, 115)  # activated_cell_index (all values from 10 to 195) 8  
]

Ibg_init_val = 0.7 * 0.94
dx = 1
 



def V_function(t, cell_id, g_gap, Ibg_init, Ikir_coef, cm, dx, K_o, I_app_val, stimulation_time_start, stimulation_time_end, activated_cell_index):
    Ng = 200  # Number of cells
    y0_init = -33  # Initial condition for voltage
    t_span = (0, t)  # Time span for the simulation

    sol = HK_deltas_vstim_vresponse_graph_modified_v2(g_gap, Ibg_init, Ikir_coef, cm, dx, K_o, I_app_val, y0_init, t_span, Ng, stimulation_time_start, stimulation_time_end, activated_cell_index)

    # Extract the voltage for the specified cell at the specified time
    voltage = sol.y[cell_id, np.argmin(np.abs(sol.t - t))]

    return voltage

# Example usage
ggap, Ikir_coef, cm, K_o, I_app_val, y0_init, t_span, Ng, stimulation_time_start, stimulation_time_end, activated_cell_index = generate_random_params(param_bounds)

# Get the voltage for cell 100 at time 300
cell_id = 100
time = 300
voltage = V_function(time, cell_id, ggap, Ibg_init_val, Ikir_coef, cm, dx, K_o, I_app_val, stimulation_time_start, stimulation_time_end, activated_cell_index)
print(f"Voltage for cell {cell_id} at time {time}: {voltage}")











#########################################################################################################################

# Define the physics loss function
def physics_loss(x, y):
    t = x[:, 0:1]
    cell_id = x[:, 1:2]
    g_gap = x[:, 2:3]
    Ibg_init = x[:, 3:4]
    Ikir_coef = x[:, 4:5]
    cm = x[:, 5:6]
    dx = x[:, 6:7]
    K_o = x[:, 7:8]
    I_app_val = x[:, 8:9]
    stimulation_time_start = x[:, 9:10]
    stimulation_time_end = x[:, 10:11]
    activated_cell_index = x[:, 11:12]

    # Compute the derivatives of the predicted voltage with respect to time and space
    dt = t[:, 1:] - t[:, :-1]
    dy_dt = (y[:, 1:] - y[:, :-1]) / dt
    dy_dx = (y[:, 2:] - 2 * y[:, 1:-1] + y[:, :-2]) / (dx ** 2)

    # Compute the residual of the PDE
    residual = dy_dt - (g_gap / (dx ** 2 * cm)) * dy_dx + (1 / cm) * (Ibg_init * (y[:, 1:-1] + 30) + Ikir_coef * torch.sqrt(K_o) * ((y[:, 1:-1] - ((8.314e3 * 293) / 9.6485e4) * torch.log(K_o / 150)) / (1 + torch.exp(((y[:, 1:-1] - ((8.314e3 * 293) / 9.6485e4) * torch.log(K_o / 150) - 25) / 7)))) + I_app_val * ((stimulation_time_start <= t[:, 1:-1]) & (t[:, 1:-1] <= stimulation_time_end)) * (cell_id[:, 1:-1] == activated_cell_index))

    return torch.mean(residual ** 2)




# Define the PINN architecture
class PINN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PINN, self).__init__()
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x

# Generate random input data
num_samples = 1000
t = np.random.rand(num_samples, 1) * 600
cell_id = np.random.randint(0, 200, size=(num_samples, 1))
g_gap = np.random.uniform(0.1, 35, size=(num_samples, 1))
Ibg_init = np.full((num_samples, 1), 0.7 * 0.94)
Ikir_coef = np.random.uniform(0.90, 0.96, size=(num_samples, 1))
cm = np.random.uniform(8, 11, size=(num_samples, 1))
dx = np.full((num_samples, 1), 1.0)
K_o = np.random.uniform(1, 8, size=(num_samples, 1))
I_app_val = np.random.uniform(-70, 70, size=(num_samples, 1))
stimulation_time_start = np.random.randint(0, 450, size=(num_samples, 1))
stimulation_time_end = stimulation_time_start + np.random.randint(85, 390, size=(num_samples, 1))
activated_cell_index = np.random.randint(85, 115, size=(num_samples, 1))



# Compute the target voltage values using the V_function
target_voltages = np.array([V_function(t[i], cell_id[i], g_gap[i], Ibg_init[i], Ikir_coef[i], cm[i], dx[i], K_o[i], I_app_val[i], stimulation_time_start[i], stimulation_time_end[i], activated_cell_index[i]) for i in range(num_samples)])

# Prepare input data and target voltages
input_data = np.concatenate((t, cell_id, g_gap, Ibg_init, Ikir_coef, cm, dx, K_o, I_app_val, stimulation_time_start, stimulation_time_end, activated_cell_index), axis=1)
target_voltages = target_voltages.reshape(-1, 1)

# Convert input data and target voltages to PyTorch tensors
input_tensor = torch.tensor(input_data, dtype=torch.float32)
target_tensor = torch.tensor(target_voltages, dtype=torch.float32)




# Create an instance of the PINN
input_dim = 12
hidden_dim = 64
output_dim = 1

pinn = PINN(input_dim, hidden_dim, output_dim)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(pinn.parameters(), lr=0.001)

# Training loop
num_epochs = 1000
batch_size = 32

for epoch in range(num_epochs):
    # Shuffle and batch the data
    indices = torch.randperm(num_samples)
    for i in range(0, num_samples, batch_size):
        batch_indices = indices[i:i+batch_size]
        batch_input = input_tensor[batch_indices]
        batch_target = target_tensor[batch_indices]
        
        # Forward pass
        batch_output = pinn(batch_input)
        
        # Compute the data loss
        data_loss = criterion(batch_output, batch_target)
        
        # Compute the physics loss
        physics_loss_value = physics_loss(batch_input, batch_output)
        
        # Combine the losses
        loss = data_loss + physics_loss_value
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Print the loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        
#########################################################################################################################