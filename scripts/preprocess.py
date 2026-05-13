import os
import glob
import re
import pandas as pd
import numpy as np
from tqdm import tqdm

def process_file(filepath):
    basename = os.path.basename(filepath)
    # Parse filename: S_{round}{subject}_S{session}_{task}.csv
    match = re.search(r'S_(\d)(\d{3})_S(\d)_(\w+)\.csv$', basename)
    if not match:
        return []
    
    round_num = int(match.group(1))
    subject = match.group(2)
    session = int(match.group(3))
    task = match.group(4)
    
    df = pd.read_csv(filepath)
    
    if 'lab' not in df.columns:
        return []
        
    # Calculate instantaneous velocity
    df['dx'] = df['x'].diff()
    df['dy'] = df['y'].diff()
    df['dt'] = df['n'].diff()
    
    # Avoid division by zero
    valid_dt = df['dt'] > 0
    df.loc[valid_dt, 'velocity'] = np.sqrt(df.loc[valid_dt, 'dx']**2 + df.loc[valid_dt, 'dy']**2) / (df.loc[valid_dt, 'dt'] / 1000.0)
    
    # Identify contiguous blocks of saccades (lab == 2)
    is_saccade = (df['lab'] == 2)
    if not is_saccade.any():
        return []
        
    # Create block IDs
    block_id = (is_saccade != is_saccade.shift()).cumsum()
    
    saccades = []
    
    for _, group in df[is_saccade].groupby(block_id):
        # We also need to check validity. GazeBase uses val=0 for valid.
        # If the saccade has invalid samples, we might want to skip it, or just ignore invalid samples.
        # Let's count how many valid samples it has.
        if 'val' in group.columns:
            if (group['val'] != 0).all():
                continue # completely invalid
                
        # Duration is roughly the number of samples since it's 1000Hz,
        # but let's use the timestamp difference just to be precise.
        t_start = group['n'].iloc[0]
        t_end = group['n'].iloc[-1]
        
        # In ms, duration is (t_end - t_start) + 1 if perfect 1ms sampling
        duration = len(group)
        
        if 6 <= duration <= 100:
            peak_vel = group['velocity'].max()
            if pd.isna(peak_vel) or np.isinf(peak_vel):
                continue
                
            saccades.append({
                'round': round_num,
                'subject': subject,
                'session': session,
                'task': task,
                'onset_time': t_start,
                'duration': duration,
                'peak_velocity': peak_vel
            })
            
    return saccades

def main():
    files = glob.glob(os.path.join("data", "*.csv"))
    print(f"Found {len(files)} CSV files to process.")
    
    all_saccades = []
    
    for f in tqdm(files, desc="Processing files"):
        file_saccades = process_file(f)
        all_saccades.extend(file_saccades)
        
    out_df = pd.DataFrame(all_saccades)
    print(f"\nProcessed {len(out_df)} valid saccades.")
    
    out_path = os.path.join("data", "processed_saccades.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Saved processed data to {out_path}")

if __name__ == "__main__":
    main()
