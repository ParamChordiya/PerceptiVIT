import pandas as pd

csv_file = 'data/test.csv'  
df = pd.read_csv(csv_file)

df['path'] = df['path'].apply(lambda x: f"data/{x}")

output_file = 'data/test.csv' 
df.to_csv(output_file, index=False)

print(f"Modified CSV saved as {output_file}")
