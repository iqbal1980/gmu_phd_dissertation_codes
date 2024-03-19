import csv
import matplotlib.pyplot as plt

file_path = 'training_data.csv'  # Update this to your file path

plt.figure(figsize=(10, 6))

# Initialize the counter for the number of lines (data rows) processed
line_count = 0

with open(file_path, 'r') as file:
    csv_reader = csv.reader(file)
    header = next(csv_reader)  # Skip the header

    vm_indices = [i for i, col in enumerate(header) if col.startswith('Vm_')]
    
    for row in csv_reader:
        if line_count >= 200:  # Process only the first 3 data lines
            break
        line_count += 1
        
        y = [float(row[i]) for i in vm_indices[:10]]  # Adjust according to your x-axis length

        I_app = row[header.index('I_app')]
        cellid = row[header.index('cellid')]
        
        plt.plot(range(len(y)), y, label=f'I_app: {I_app}, Cell ID: {cellid}')

plt.xlabel('Time step')
plt.ylabel('Vm')
plt.title('Vm over Time for First 3 Cells')
plt.legend()
plt.tight_layout()
plt.show()
