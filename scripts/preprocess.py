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
    
    # Map FXS to FIX
    if task == 'FXS':
        task = 'FIX'
        
    df = pd.read_csv(filepath)
    
    # Filter validity: set x, y coordinates to NaN when val != 0
    if 'val' in df.columns:
        invalid_mask = df['val'] != 0
        df.loc[invalid_mask, ['x', 'y']] = np.nan
        
    # Calculate differences
    df['dx'] = df['x'].diff()
    df['dy'] = df['y'].diff()
    df['dt'] = df['n'].diff() / 1000.0 # Convert ms to seconds
    
    # Avoid division by zero
    valid_dt = df['dt'] > 0
    df['velocity'] = np.nan
    df.loc[valid_dt, 'velocity'] = np.sqrt(df.loc[valid_dt, 'dx']**2 + df.loc[valid_dt, 'dy']**2) / df.loc[valid_dt, 'dt']
    
    # Check if pre-labeled saccades exist
    has_labels = 'lab' in df.columns and not df['lab'].isna().all()
    
    saccades = []
    MAX_VEL = 1000.0 # Physical limit for human eye saccade peak velocity
    
    if has_labels:
        # Pre-labeled: lab == 2 is saccade
        is_saccade = (df['lab'] == 2)
        if not is_saccade.any():
            return []
            
        block_id = (is_saccade != is_saccade.shift()).cumsum()
        
        for _, group in df[is_saccade].groupby(block_id):
            duration = len(group)
            if 6 <= duration <= 100:
                peak_vel = group['velocity'].max()
                if pd.isna(peak_vel) or np.isinf(peak_vel) or peak_vel > MAX_VEL:
                    continue
                saccades.append({
                    'round': round_num,
                    'subject': subject,
                    'session': session,
                    'task': task,
                    'onset_time': group['n'].iloc[0],
                    'duration': duration,
                    'peak_velocity': peak_vel
                })
    else:
        # Velocity-based detector
        is_above_thresh = (df['velocity'] >= 30) & (df['velocity'] <= MAX_VEL)
        if not is_above_thresh.any():
            return []
            
        block_id = (is_above_thresh != is_above_thresh.shift()).cumsum()
        
        for _, group in df[is_above_thresh].groupby(block_id):
            duration = len(group)
            if 6 <= duration <= 100:
                peak_vel = group['velocity'].max()
                if pd.isna(peak_vel) or np.isinf(peak_vel) or peak_vel > MAX_VEL:
                    continue
                saccades.append({
                    'round': round_num,
                    'subject': subject,
                    'session': session,
                    'task': task,
                    'onset_time': group['n'].iloc[0],
                    'duration': duration,
                    'peak_velocity': peak_vel
                })
                
    return saccades

def main():
    files = glob.glob(os.path.join("data", "*.csv"))
    # Exclude processed_saccades.csv itself if present
    files = [f for f in files if "processed_saccades" not in f]
    print(f"Found {len(files)} CSV files to process.")
    
    all_saccades = []
    
    for f in tqdm(files, desc="Processing files"):
        file_saccades = process_file(f)
        all_saccades.extend(file_saccades)
        
    out_df = pd.DataFrame(all_saccades)
    print(f"\nProcessed {len(out_df)} valid saccades across all subjects and tasks.")
    
    out_path = os.path.join("data", "processed_saccades.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Saved processed data to {out_path}")

if __name__ == "__main__":
    main()
