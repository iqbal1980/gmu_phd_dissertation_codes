import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numba import njit, vectorize
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pickle
import os

# Ensure reproducibility
np.random.seed(42)
torch.manual_seed(42)

@vectorize
def safe_log(x):
    return np.log(x) if x > 0 else np.log(1e-20)

@njit
def exponential_function(x, a):
    return np.exp(a * x)

@njit
def ode_system(t, y, g_gap, Ibg_init, Ikir_coef, cm, dx, K_o, I_app_val, stimulation_time_start, stimulation_time_end, activated_cell_index):
    F = 9.6485e4
    R = 8.314e3
    Ng = len(y)
    Vm = y
    I_bg = Ibg_init * (Vm + 30)
    E_K = (R * 293 / F) * safe_log(np.maximum(K_o / 150, 1e-20))  # Clamp K_o / 150 to avoid log(0)
    I_kir = Ikir_coef * np.sqrt(np.maximum(K_o, 1e-20)) * (Vm - E_K) / (1 + np.exp((Vm - E_K - 25) / 7))
    I_app = np.zeros_like(Vm)
    I_app[int(activated_cell_index)] = float(I_app_val if stimulation_time_start <= t <= stimulation_time_end else 0.0)

    dVm_dt = np.zeros_like(Vm)
    for kk in range(Ng):
        if kk == 0 or kk == Ng-1:
            continue  # Skip boundary conditions for simplicity
        dVm_dt[kk] = (g_gap / (dx**2 * cm)) * (Vm[kk+1] + Vm[kk-1] - 2 * Vm[kk]) - (1 / cm) * (I_bg[kk] + I_kir[kk] + I_app[kk])
    return dVm_dt

def HK_deltas_vstim_vresponse_graph_modified_v2(ggap, Ibg_init, Ikir_coef, cm, dx, K_o, I_app_val, y0_init, t_span, Ng, stimulation_time_start, stimulation_time_end, activated_cell_index):
    y0 = np.ones(Ng) * y0_init
    sol = solve_ivp(ode_system, t_span, y0, args=(ggap, Ibg_init, Ikir_coef, cm, dx, K_o, I_app_val, stimulation_time_start, stimulation_time_end, activated_cell_index), method='RK45')
    return sol

def generate_random_params():
    ggap = np.random.choice(np.linspace(0.1, 35, 10))
    Ikir_coef = np.random.choice(np.linspace(0.90, 0.96, 5))
    cm = np.random.choice(np.linspace(8, 11, 4))
    K_o = np.random.choice(np.linspace(1, 8, 8))
    I_app_val = np.random.choice(np.linspace(-70, 70, 35))
    # Assuming a fixed simulation time for simplicity
    time_in_s = 600
    y0_init = -33
    Ng = 200
    activated_cell_index = np.random.randint(85, 115)  # Use np.random.randint for integer values

    # Generate stimulation times within valid ranges
    stimulation_time_start = random.uniform(0, 0.75 * time_in_s)
    stimulation_time_end = random.uniform(stimulation_time_start + 85, min(stimulation_time_start + 90 + random.uniform(0, 300), time_in_s))

    # Set the value of Ibg_init (you can adjust this as needed)
    Ibg_init = 0.7 * 0.94  # Example value

    # Set the value of dx (you can adjust this as needed)
    dx = 1  # Example value

    params = {
        "ggap": ggap,
        "Ikir_coef": Ikir_coef,
        "cm": cm,
        "K_o": K_o,
        "I_app_val": I_app_val,
        "y0_init": y0_init,
        "Ng": Ng,
        "activated_cell_index": activated_cell_index,
        "stimulation_time_start": stimulation_time_start,
        "stimulation_time_end": stimulation_time_end,
        "time_in_s": time_in_s,
        "Ibg_init": Ibg_init,
        "dx": dx
    }

    print("Generated parameters: ggap, Ikir_coef, cm, K_o, I_app_val, time_in_s, y0_init, Ng, activated_cell_index")
    print(f"stimulation_time_start: {stimulation_time_start}, stimulation_time_end: {stimulation_time_end}")

    return params

sample_counter = 0

def V_function(time, cell_id, params):
    global sample_counter
    sample_counter = sample_counter + 1
    # Prepare the time span for the simulation based on the input
    t_span = (0, time)

    # Call the simulation function with parameters unpacked from the `params` dictionary
    sol = HK_deltas_vstim_vresponse_graph_modified_v2(
        params["ggap"], params["Ibg_init"], params["Ikir_coef"], params["cm"], params["dx"],
        params["K_o"], params["I_app_val"], params["y0_init"], t_span,
        int(params["Ng"]), params["stimulation_time_start"],
        params["stimulation_time_end"], int(params["activated_cell_index"])
    )

    # Find the closest time point in the solution to the requested time and extract the voltage
    voltage = sol.y[cell_id, np.argmin(np.abs(sol.t - time))]

    print(f"Sample: {sample_counter}, V_function called with time: {time}, cell_id: {cell_id}, voltage: {voltage}")

    return voltage

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

    # Assuming dy_dt and dy_dx calculations are placeholders;
    # actual implementation depends on your model's specific needs
    dy_dt = torch.zeros_like(y)
    dy_dx = torch.zeros_like(y)

    # Compute the residual of the PDE
    F = 9.6485e4
    R = 8.314e3
    E_K = (R * 293 / F) * torch.log(torch.clamp(K_o / 150, min=1e-20))  # Clamp to avoid log(0)

    I_kir = Ikir_coef * torch.sqrt(torch.clamp(K_o, min=1e-20)) * ((y - E_K) / (1 + torch.exp(((y - E_K - 25) / 7))))
    I_app = I_app_val * ((stimulation_time_start <= t) & (t <= stimulation_time_end)) * (cell_id == activated_cell_index)

    residual = dy_dt - (g_gap / (dx ** 2 * cm)) * dy_dx + (1 / cm) * (Ibg_init * (y + 30) + I_kir + I_app)

    return torch.mean(residual ** 2)
    
    
num_samples = 10000    

# Check if the pickled data file exists
if os.path.exists("data.pkl"):
    with open("data.pkl", "rb") as f:
        data_dict = pickle.load(f)
    input_data = data_dict["input_data"]
    target_voltages = data_dict["target_voltages"]
    params = data_dict["params"]
    print("Input and target data loaded from 'data.pkl'")
else:
    params = generate_random_params()

    # Generate input data for training
    print(f"Generating {num_samples} samples for training...")
    sampled_times = np.random.rand(num_samples, 1) * params["time_in_s"]
    sampled_cell_ids = np.random.randint(0, params["Ng"], size=(num_samples, 1))
    sampled_activated_cell_index = np.full((num_samples, 1), params["activated_cell_index"])

    # Compute the target voltage values using the corrected approach
    target_voltages = np.array([V_function(sampled_times[i][0], sampled_cell_ids[i][0], params) for i in range(num_samples)])
    print("Target voltages computed.")

    input_data = np.hstack((sampled_times, sampled_cell_ids, np.full((num_samples, 1), params["ggap"]), np.full((num_samples, 1), 0.7 * 0.94), np.full((num_samples, 1), params["Ikir_coef"]), np.full((num_samples, 1), params["cm"]), np.full((num_samples, 1), 1), np.full((num_samples, 1), params["K_o"]), np.full((num_samples, 1), params["I_app_val"]), np.full((num_samples, 1), params["stimulation_time_start"]), np.full((num_samples, 1), params["stimulation_time_end"]), sampled_activated_cell_index))

    # Save the input and target data
    data_dict = {
        "input_data": input_data,
        "target_voltages": target_voltages,
        "params": params
    }

    with open("data.pkl", "wb") as f:
        pickle.dump(data_dict, f)

    print("Input and target data saved to 'data.pkl'")

target_voltages = target_voltages.reshape(-1, 1)

print("Target voltages computed.")

# Convert input data and target voltages to PyTorch tensors
input_tensor = torch.tensor(input_data, dtype=torch.float32)
target_tensor = torch.tensor(target_voltages, dtype=torch.float32)

# Define the PINN architecture
class PINN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(PINN, self).__init__()
        self.hidden_layer1 = nn.Linear(input_dim, hidden_dim1)
        self.hidden_layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.output_layer = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        x = torch.relu(self.hidden_layer1(x))
        x = torch.relu(self.hidden_layer2(x))
        x = self.output_layer(x)
        return x

# Initialize the PINN
input_dim = 12
hidden_dim1 = 64
hidden_dim2 = 128
output_dim = 1
pinn = PINN(input_dim, hidden_dim1, hidden_dim2, output_dim)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(pinn.parameters(), lr=0.001)

# Training loop
num_epochs = 5000
batch_size = 32

for epoch in range(num_epochs):
    indices = torch.randperm(num_samples)
    epoch_loss = 0.0
    for i in range(0, num_samples, batch_size):
        batch_indices = indices[i:i + batch_size]
        batch_input = input_tensor[batch_indices]
        batch_target = target_tensor[batch_indices]
        
        batch_output = pinn(batch_input)
        data_loss = criterion(batch_output, batch_target)
        physics_loss_value = physics_loss(batch_input, batch_output)
        
        if torch.isnan(data_loss) or torch.isnan(physics_loss_value):
            print(f"NaN detected in epoch {epoch+1}, batch {i//batch_size+1}")
            print(f"data_loss: {data_loss}, physics_loss_value: {physics_loss_value}")
            break  # Exit the training loop
        
        loss = data_loss + physics_loss_value
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    average_loss = epoch_loss / (num_samples / batch_size)
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {average_loss:.4f}")