import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from numba import njit
from scipy.optimize import curve_fit
import csv
import time

# Constants for safe_log and safe_exponential
MIN_VALUE = -80  # Minimum physiologically reasonable value for Vm
MAX_VALUE = 40   # Maximum physiologically reasonable value for Vm

# Functions from verify_params11.py for numerical stability
@njit
def safe_log(x):
    if x <= 0:
        return MIN_VALUE
    return np.log(x)

@njit
def exponential_function(x, a):
    return np.exp(a * x) 

@njit
def exponential_decay_function(x, A, B):
    return A * np.exp(B * x)

# Simulate process function (similar to verify_params11.py)
@njit(parallel=False)
def simulate_process_modified_v2(g_gap_value, Ibg_init, Ikir_coef, cm, dx, K_o, I_app):
    dt = 0.001
    F = 9.6485e4
    R = 8.314e3
    loop = 600000
    Ng = 200
    Vm = np.ones(Ng) * (-33)
    g_gap = g_gap_value
    eki1 = (g_gap * dt) / (dx**2 * cm)
    eki2 = dt / cm

    I_bg = np.zeros(Ng) + Ibg_init
    I_kir = np.zeros(Ng)

    for j in range(loop):
        t = j * dt
        if 100 <= t <= 400:
            I_app[99] = I_app[99]
        else:
            I_app[99] = 0.0

        for kk in range(Ng):
            E_K = (R * 293 / F) * safe_log(K_o/150)
            I_bg[kk] = Ibg_init * (Vm[kk] + 30)
            I_kir[kk] = Ikir_coef * np.sqrt(K_o) * ((Vm[kk] - E_K) / (1 + exponential_function((Vm[kk] - E_K - 25) / 7, 1)))

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

    return Vm

# Generate training data
def generate_training_data(num_samples, output_file):
    start_time = time.time()
    
    param_ranges = [
        (0.1, 35),    # ggap
        (0.1, 10.96), # Ikir_coef
        (1, 40),      # cm
        (1, 45),       # K_o
        (-150, 150)     # I_app
    ]

    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ggap', 'Ikir_coef', 'cm', 'K_o', 'I_app'] + [f'Vm_{i}' for i in range(200)])

        for i in range(num_samples):
            params = [np.random.uniform(low, high) for low, high in param_ranges]
            ggap, Ikir_coef, cm, K_o, I_app = params
            
            dx = 1
            Ibg_init = 0.7 * 0.94
            
            I_app_array = np.zeros(200)
            I_app_array[99] = I_app
            
            Vm = simulate_process_modified_v2(ggap, Ibg_init, Ikir_coef, cm, dx, K_o, I_app_array)
            
            writer.writerow(params + Vm.tolist())
            
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1} samples...")
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Generating {num_samples} samples took {execution_time:.2f} seconds.")

# Create a custom dataset
class SimulationDataset(Dataset):
    def __init__(self, params, Vm):
        self.params = params
        self.Vm = Vm

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        params = self.params[idx]
        Vm = self.Vm[idx]
        return params, Vm

# Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, output_dim, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        # Input embedding
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        
        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*4, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Output layer
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Input embedding
        x = self.embedding(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Output layer
        x = self.fc_out(x)
        
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# Training loop
def train(model, dataloader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        for params, Vm in dataloader:
            optimizer.zero_grad()
            output = model(params)
            loss = criterion(output, Vm)
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Main function
def main():
    # Generate and save training data
    num_samples = 150000
    output_file = 'training_data.csv'
    generate_training_data(num_samples, output_file)

    # Load the saved training data
    params = []
    Vm = []
    with open(output_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            params.append(row[:5])
            Vm.append(row[5:])
    
    params = np.array(params, dtype=float)
    Vm = np.array(Vm, dtype=float)

    # Create a dataset and dataloader
    dataset = SimulationDataset(params, Vm)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Define the Transformer model
    input_dim = params.shape[1]
    hidden_dim = 256
    num_layers = 4
    num_heads = 8
    output_dim = Vm.shape[1]
    dropout = 0.1
    model = TransformerModel(input_dim, hidden_dim, num_layers, num_heads, output_dim, dropout)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training
    num_epochs = 100
    train(model, dataloader, criterion, optimizer, num_epochs)

if __name__ == "__main__":
    main()