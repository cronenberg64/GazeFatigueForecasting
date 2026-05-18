import os
import glob
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

def process_file_events(filepath, threshold=50.0):
    basename = os.path.basename(filepath)
    # Parse filename: S_{round}{subject}_S{session}_{task}.csv
    match = re.search(r'S_(\d)(\d{3})_S(\d)_(\w+)\.csv$', basename)
    if not match:
        return None
    
    round_num = int(match.group(1))
    subject = match.group(2)
    session = int(match.group(3))
    task = match.group(4)
    
    if task == 'FXS':
        task = 'FIX'
        
    df = pd.read_csv(filepath)
    if len(df) < 2:
        return None
        
    # Validity masking: set x, y coordinates to NaN when val != 0
    if 'val' in df.columns:
        invalid_mask = df['val'] != 0
        df.loc[invalid_mask, ['x', 'y']] = np.nan
        
    # Calculate continuous velocity (deg/sec)
    df['dx'] = df['x'].diff()
    df['dy'] = df['y'].diff()
    df['dt'] = df['n'].diff() / 1000.0 # Convert ms to seconds
    
    valid_dt = df['dt'] > 0
    df['velocity'] = np.nan
    df.loc[valid_dt, 'velocity'] = np.sqrt(df.loc[valid_dt, 'dx']**2 + df.loc[valid_dt, 'dy']**2) / df.loc[valid_dt, 'dt']
    
    # Drop NaNs in continuous velocity to compute differences correctly
    vel_series = df[['n', 'velocity']].dropna()
    if len(vel_series) < 2:
        return None
        
    # Compute continuous velocity differences
    vel_series['v_diff'] = vel_series['velocity'].diff()
    
    # Filter by threshold
    events_df = vel_series[vel_series['v_diff'].abs() > threshold].copy()
    if events_df.empty:
        return None
        
    events_df['subject'] = subject
    events_df['round'] = round_num
    events_df['task'] = task
    events_df['t_ms'] = events_df['n'].astype(int)
    events_df['polarity'] = np.where(events_df['v_diff'] > 0, 1, -1)
    events_df['magnitude'] = events_df['v_diff'].abs()
    
    # Keep only required columns
    return events_df[['subject', 'round', 'task', 't_ms', 'polarity', 'magnitude']]

def main():
    files = glob.glob(os.path.join("data", "*.csv"))
    files = [f for f in files if "processed_saccades" not in f]
    print(f"Extracting event streams from {len(files)} files...")
    
    # Run in parallel using all available cores
    results = Parallel(n_jobs=-1)(
        delayed(process_file_events)(f, threshold=50.0) 
        for f in tqdm(files, desc="Extracting events")
    )
    
    # Filter out None values
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        print("No events extracted!")
        return
        
    # Combine into a single dataframe
    print("Combining event dataframes...")
    final_df = pd.concat(valid_results, ignore_index=True)
    
    # Save as Parquet
    parquet_path = os.path.join("data", "events.parquet")
    print(f"Saving {len(final_df)} events to {parquet_path}...")
    final_df.to_parquet(parquet_path, engine='pyarrow', index=False)
    print("Parquet saved successfully!")
    
    # Step 10 Sanity Check Plot: pick Subject 001, Round 1, Session 1, Task 'RAN'
    sanity_file = glob.glob(os.path.join("data", "S_1001_S1_RAN.csv"))
    if sanity_file:
        filepath = sanity_file[0]
        print(f"Generating sanity check plot from {filepath}...")
        
        # Re-load full continuous data
        df = pd.read_csv(filepath)
        if 'val' in df.columns:
            df.loc[df['val'] != 0, ['x', 'y']] = np.nan
        df['dx'] = df['x'].diff()
        df['dy'] = df['y'].diff()
        df['dt'] = df['n'].diff() / 1000.0
        df['velocity'] = np.nan
        valid_dt = df['dt'] > 0
        df.loc[valid_dt, 'velocity'] = np.sqrt(df.loc[valid_dt, 'dx']**2 + df.loc[valid_dt, 'dy']**2) / df.loc[valid_dt, 'dt']
        
        # Extract events for this specific file
        file_events = process_file_events(filepath, threshold=50.0)
        
        plt.figure(figsize=(12, 6))
        plt.plot(df['n'] / 1000.0, df['velocity'], color='#1f77b4', alpha=0.6, label='Raw 1000 Hz Velocity')
        
        if file_events is not None and not file_events.empty:
            pos_events = file_events[file_events['polarity'] == 1]
            neg_events = file_events[file_events['polarity'] == -1]
            
            plt.scatter(pos_events['t_ms'] / 1000.0, pos_events['magnitude'], 
                        color='#2ca02c', marker='^', s=40, label='Positive Events (+1)', alpha=0.8)
            plt.scatter(neg_events['t_ms'] / 1000.0, neg_events['magnitude'], 
                        color='#d62728', marker='v', s=40, label='Negative Events (-1)', alpha=0.8)
            
        plt.title("Step 10: Neuromorphic Event Stream Sanity Check (Subject 001, RAN Task)")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Velocity / Change Magnitude (deg/sec)")
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # Enforce limits for visual clarity
        plt.xlim(0, 10)  # Zoom in on first 10 seconds to see the structure of events
        plt.ylim(0, 1000)
        
        os.makedirs("plots", exist_ok=True)
        plot_path = os.path.join("plots", "event_sanity_check.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Sanity check plot saved to {plot_path}")

if __name__ == "__main__":
    main()
