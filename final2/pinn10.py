import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm

dx = 1.0

param_bounds = [
    np.linspace(0.1, 35, 10),  # ggap (10 values)
    np.linspace(0.90, 0.96, 5),  # Ikir_coef (5 values)
    np.linspace(8, 11, 4),  # cm (4 values)
    np.linspace(1, 8, 8),  # K_o (8 values)
    np.linspace(-70, 70, 15),  # I_app_val (15 values)
    np.linspace(1, 600000, 100),  # time_in_ms (100 values)
    np.arange(1, 195)  # cell_index (all values from 1 to 194)
]




def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

def run_inference(pinn, test_data):
    pinn.eval()
    with torch.no_grad():
        predictions = pinn(test_data)
    return predictions

def compare_with_simulation(pinn, test_data, t_span, Ng):
    predictions = run_inference(pinn, test_data)
    
    for i, sample in enumerate(test_data):
        t, Vm, g_gap, Ikir_coef, cm, dx, K_o, I_app_val = sample
        time_in_ms = t.item()
        cell_index = int(Vm.item())
        
        simulated_voltage = run_simulation(g_gap, Ibg_init_val, Ikir_coef, cm, dx, K_o, I_app_val, t_span, Ng, time_in_ms, cell_index)
        predicted_voltage = predictions[i].item()
        
        print(f"Sample {i+1}:")
        print(f"Simulated voltage: {simulated_voltage}")
        print(f"Predicted voltage: {predicted_voltage}")
        print("-" * 20)




###########################################
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

# Custom tokenization function
def tokenize(t, Vm, g_gap, Ikir_coef, cm, dx, K_o, I_app_val):
    token = torch.tensor([t, Vm, g_gap, Ikir_coef, cm, dx, K_o, I_app_val])
    return token

# Positional encoding
def get_positional_encoding(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.input_layer = nn.Linear(input_size, d_model)
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src = self.input_layer(src)
        output = self.transformer_encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.output_layer(output)
        return output

# Physics Informed Neural Network (PINN)
class PINN(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, dropout=0.1):
        super(PINN, self).__init__()
        self.transformer_encoder = TransformerEncoder(input_size, d_model, nhead, num_layers, dropout)

    def forward(self, x):
        return self.transformer_encoder(x)
###########################################

















 

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

    # Convert K_o to a PyTorch tensor
    K_o = torch.tensor(K_o)

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


# Generate training data
num_train_samples = 10000  # Increase the number of training samples
x_train = torch.zeros(num_train_samples, 7)
y_train = torch.zeros(num_train_samples)

print("Generating training data...")
for i in range(num_train_samples):
    ggap = np.random.choice(param_bounds[0])
    Ikir_coef = np.random.choice(param_bounds[1])
    cm = np.random.choice(param_bounds[2])
    K_o = np.random.choice(param_bounds[3])
    I_app_val = np.random.choice(param_bounds[4])
    time_in_ms = np.random.choice(param_bounds[5])
    cell_index = np.random.choice(param_bounds[6])
    
    x_train[i] = torch.tensor([ggap, Ibg_init_val, Ikir_coef, cm, dx, K_o, I_app_val])
    y_train[i] = run_simulation(ggap, Ibg_init_val, Ikir_coef, cm, dx, K_o, I_app_val, t_span, Ng, time_in_ms, cell_index)


# Generate collocation points
num_collocation_points = 10000  # Increase the number of collocation points
t_collocation = torch.rand(num_collocation_points, 1) * (t_span[1] - t_span[0]) + t_span[0]
x_collocation = torch.zeros(num_collocation_points, 7)

for i in range(num_collocation_points):
    ggap = np.random.choice(param_bounds[0])
    Ikir_coef = np.random.choice(param_bounds[1])
    cm = np.random.choice(param_bounds[2])
    K_o = np.random.choice(param_bounds[3])
    I_app_val = np.random.choice(param_bounds[4])
    
    x_collocation[i] = torch.tensor([ggap, Ibg_init_val, Ikir_coef, cm, dx, K_o, I_app_val])

x_collocation = torch.cat((t_collocation, x_collocation), dim=1)



# Create and train the PINN
input_size = 8  # Include time as an input
hidden_size = 64
output_size = 1
num_epochs = 1000
learning_rate = 0.001
alpha = 1.0  # Hyperparameter for physics-based loss

pinn = PINN(input_size, hidden_size, output_size)
# Train the PINN model
pinn = train_pinn(pinn, torch.cat((x_train, torch.ones(len(x_train), 1) * time_in_ms), dim=1), y_train, x_collocation, num_epochs, learning_rate, alpha)


# Test the trained PINN
test_input = torch.tensor([[time_in_ms, 20.008702984240095, Ibg_init_val, 0.9, 11.0, 1, 4.502039575569403, -70]])
predicted_voltage = pinn(test_input)
print(f"Predicted voltage at time {time_in_ms}ms for cell index {cell_index}: {predicted_voltage.item()}")





# Save the trained model
save_model(pinn, 'trained_pinn.pth')

# Load the saved model
pinn = PINN(input_size, d_model, nhead, num_layers, dropout)
pinn = load_model(pinn, 'trained_pinn.pth')

# Create test data
num_test_samples = 10
test_data = torch.zeros(num_test_samples, 8)

for i in range(num_test_samples):
    t = np.random.choice(param_bounds[5])
    cell_index = np.random.choice(param_bounds[6])
    ggap = np.random.choice(param_bounds[0])
    Ikir_coef = np.random.choice(param_bounds[1])
    cm = np.random.choice(param_bounds[2])
    K_o = np.random.choice(param_bounds[3])
    I_app_val = np.random.choice(param_bounds[4])
    
    test_data[i] = torch.tensor([t, cell_index, ggap, Ikir_coef, cm, dx, K_o, I_app_val])

# Run inference and compare with simulation
compare_with_simulation(pinn, test_data, t_span, Ng)