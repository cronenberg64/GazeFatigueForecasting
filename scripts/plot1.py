import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    # Load processed saccades
    df = pd.read_csv('data/processed_saccades.csv')
    
    # Select ONE representative subject, round, and session
    # Let's pick subject 001, round 1, session 1
    # We need to compare TEX vs VD1
    subj_id = df['subject'].iloc[0] # usually 1 or '001'
    df_subj = df[(df['subject'] == subj_id) & (df['round'] == 1) & (df['session'] == 1)]
    
    df_tex = df_subj[df_subj['task'] == 'TEX'].copy()
    df_vd1 = df_subj[df_subj['task'] == 'VD1'].copy()
    
    # If VD1 is empty, try VD2
    if df_vd1.empty:
        df_vd1 = df_subj[df_subj['task'] == 'VD2'].copy()
        
    # Sort by onset time
    df_tex = df_tex.sort_values('onset_time').reset_index(drop=True)
    df_vd1 = df_vd1.sort_values('onset_time').reset_index(drop=True)
    
    # Convert onset time from ms to seconds
    df_tex['time_sec'] = df_tex['onset_time'] / 1000.0
    df_vd1['time_sec'] = df_vd1['onset_time'] / 1000.0
    
    # Compute rolling mean (window=30)
    window_size = 30
    df_tex['rolling_vel'] = df_tex['peak_velocity'].rolling(window=window_size, min_periods=1, center=True).mean()
    df_vd1['rolling_vel'] = df_vd1['peak_velocity'].rolling(window=window_size, min_periods=1, center=True).mean()
    
    # Plotting
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    # Left panel: TEX
    axes[0].scatter(df_tex['time_sec'], df_tex['peak_velocity'], alpha=0.3, color='steelblue', s=15, label='Saccade')
    axes[0].plot(df_tex['time_sec'], df_tex['rolling_vel'], color='red', linewidth=2, label=f'Rolling Mean (n={window_size})')
    axes[0].set_title('TEX (Reading, Low Complexity)', fontsize=14)
    axes[0].set_xlabel('Time within session (seconds)', fontsize=12)
    axes[0].set_ylabel('Saccadic peak velocity (deg/sec)', fontsize=12)
    axes[0].tick_params(axis='both', which='major', labelsize=10)
    axes[0].legend()
    
    # Right panel: VD1
    axes[1].scatter(df_vd1['time_sec'], df_vd1['peak_velocity'], alpha=0.3, color='darkorange', s=15, label='Saccade')
    axes[1].plot(df_vd1['time_sec'], df_vd1['rolling_vel'], color='red', linewidth=2, label=f'Rolling Mean (n={window_size})')
    task_name = df_vd1['task'].iloc[0] if not df_vd1.empty else "VD"
    axes[1].set_title(f'{task_name} (Video Viewing, High Complexity)', fontsize=14)
    axes[1].set_xlabel('Time within session (seconds)', fontsize=12)
    axes[1].set_ylabel('Saccadic peak velocity (deg/sec)', fontsize=12)
    axes[1].tick_params(axis='both', which='major', labelsize=10)
    axes[1].legend()
    
    # Shared Y-axis limits for better comparison
    y_max = max(df_tex['peak_velocity'].max(), df_vd1['peak_velocity'].max())
    axes[0].set_ylim(0, y_max * 1.05)
    axes[1].set_ylim(0, y_max * 1.05)
    
    plt.suptitle("Saccadic peak velocity over session: low vs high visual complexity", fontsize=16, y=1.05)
    plt.tight_layout()
    
    os.makedirs('plots', exist_ok=True)
    out_path = os.path.join('plots', '01_input_feature_timeseries.png')
    plt.savefig(out_path, bbox_inches='tight')
    print(f"Saved plot to {out_path}")
    
if __name__ == "__main__":
    main()
