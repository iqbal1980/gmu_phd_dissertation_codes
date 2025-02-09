import pandas as pd  # Ensure pandas is imported
import matplotlib.pyplot as plt

# Load the Excel file (update the path if necessary)
file_path = "channels.xlsx"  # Ensure the file is in the same directory or provide the correct path
df = pd.read_excel(file_path, sheet_name="channels", engine="openpyxl")

# Check if the correct column name exists
print("Column names in dataset:", df.columns)  # Debugging step

# Ensure column names match exactly
expected_column = "mRNA average counts/cell"
if expected_column not in df.columns:
    raise ValueError(f"Column '{expected_column}' not found. Check column names.")

# Convert RNA expression values to numeric, handling any errors
df[expected_column] = pd.to_numeric(df[expected_column], errors='coerce')

# Sort data for better visualization
df_sorted = df.sort_values(by=expected_column, ascending=False)

# Plot RNA expression for each ion channel
plt.figure(figsize=(12, 6))
plt.bar(df_sorted["Channel protein"], df_sorted[expected_column], color="skyblue")
plt.xlabel("Ion Channel Protein")
plt.ylabel("mRNA Average Counts/Cell")
plt.title("mRNA Expression Levels for Ion Channels")
plt.xticks(rotation=90)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Show the plot
plt.show()
