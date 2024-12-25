import pandas as pd

# Load the CSV file
csv_file = 'data/test.csv'  # Replace with the path to your CSV file
df = pd.read_csv(csv_file)

# Modify the `path` column to add "real_vs_fake/" to each entry
df['path'] = df['path'].apply(lambda x: f"data/{x}")

# Save the modified DataFrame back to a CSV file
output_file = 'data/test.csv'  # Replace with the desired output file name
df.to_csv(output_file, index=False)

print(f"Modified CSV saved as {output_file}")
