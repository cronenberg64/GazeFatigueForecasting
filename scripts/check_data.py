import pandas as pd
import glob
import numpy as np
import os

def check_data():
    files = glob.glob(os.path.join("data", "S_*001_*.csv"))
    print(f"Found {len(files)} files for Subject 001.")
    
    for sample_file in files[:5]:
        print(f"\n--- Checking file: {os.path.basename(sample_file)} ---")
        df = pd.read_csv(sample_file)
        
        lab_counts = df['lab'].value_counts(dropna=False).to_dict()
        print(f"Lab distribution: {lab_counts}")

if __name__ == "__main__":
    check_data()
