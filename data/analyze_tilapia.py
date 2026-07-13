import pandas as pd
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(base_dir, "data", "raw", "tilapia_iot.csv")
df = pd.read_csv(csv_path)

print("Tilapia CSV Columns:")
print(df.columns)
print("\nShape:", df.shape)

sr_col = 'Survival Rate (%)'
if sr_col in df.columns:
    df[sr_col] = pd.to_numeric(df[sr_col], errors='coerce')
    print("\nSurvival Rate (%) Summary:")
    print(df[sr_col].describe())
    
    # Check distribution of survival rate
    print("\nValue counts of Survival Rate (%):")
    print(df[sr_col].value_counts().sort_index().head(20))
    print("...")
    print(df[sr_col].value_counts().sort_index().tail(20))
    
    # Let's count how many rows have survival rate < 100%
    less_than_100 = df[df[sr_col] < 100]
    print(f"\nNumber of rows where Survival Rate < 100%: {len(less_than_100)} ({len(less_than_100)/len(df)*100:.2f}%)")
    
    # Check health status vs survival rate
    if 'Health Status' in df.columns:
        print("\nHealth Status vs Mean Survival Rate:")
        print(df.groupby('Health Status')[sr_col].mean())
        print("\nHealth Status counts:")
        print(df['Health Status'].value_counts())
else:
    print(f"Column '{sr_col}' not found.")
