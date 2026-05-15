import pandas as pd
import sys
sys.path.append('.')

print("Testing imports...")
try:
    df = pd.read_csv('data/comments.csv')
    print(f"Data loaded: {len(df)} rows")
    print(df['sentiment'].value_counts())
    print("Success!")
except Exception as e:
    print(f"Error: {e}")