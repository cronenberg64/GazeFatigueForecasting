import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    # Load processed saccades
    df = pd.read_csv('data/processed_saccades.csv')
    
    # Filter to Subject 001, Round 1, Session 1
    # We use a robust filter that handles both string ('001') and integer (1) types
    df_subj = df[
        (df['subject'].astype(str).isin(['001', '1', '1.0'])) & 
        (df['round'] == 1) & 
        (df['session'] == 1)
    ]
    
    print(f"Loaded {len(df_subj)} saccades for Subject 001, Round 1, Session 1")
    print(f"Tasks available in this session: {df_subj['task'].unique()}")
    
    df_tex = df_subj[df_subj['task'] == 'TEX'].copy()
    df_vd1 = df_subj[df_subj['task'] == 'VD1'].copy()
    
    print(f"TEX saccades: {len(df_tex)}")
    print(f"VD1 saccades: {len(df_vd1)}")
    
    # Sort by onset time
    df_tex = df_tex.sort_values('onset_time').reset_index(drop=True)
    df_vd1 = df_vd1.sort_values('onset_time').reset_index(drop=True)
    
    # Convert onset time from ms to seconds
    df_tex['time_sec'] = df_tex['onset_time'] / 1000.0
    df_vd1['time_sec'] = df_vd1['onset_time'] / 1000.0
    
    # Compute rolling mean (window=30)
    window_size = 30
    if len(df_tex) >= window_size:
        df_tex['rolling_vel'] = df_tex['peak_velocity'].rolling(window=window_size, min_periods=1, center=True).mean()
    else:
        df_tex['rolling_vel'] = df_tex['peak_velocity'].expanding().mean()
        
    if len(df_vd1) >= window_size:
        df_vd1['rolling_vel'] = df_vd1['peak_velocity'].rolling(window=window_size, min_periods=1, center=True).mean()
    else:
        df_vd1['rolling_vel'] = df_vd1['peak_velocity'].expanding().mean()
        
    # Plotting
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    # Left panel: TEX
    if not df_tex.empty:
        axes[0].scatter(df_tex['time_sec'], df_tex['peak_velocity'], alpha=0.4, color='steelblue', s=15, label='Saccades')
        axes[0].plot(df_tex['time_sec'], df_tex['rolling_vel'], color='firebrick', linewidth=2.5, label=f'Rolling Mean (n={window_size})')
        axes[0].legend(frameon=True, facecolor='white', edgecolor='none')
    axes[0].set_title('TEX (Reading, Low Complexity)', fontsize=14, fontweight='semibold')
    axes[0].set_xlabel('Time within session (seconds)', fontsize=12)
    axes[0].set_ylabel('Saccadic peak velocity (deg/sec)', fontsize=12)
    axes[0].tick_params(axis='both', which='major', labelsize=10)
    
    # Right panel: VD1
    if not df_vd1.empty:
        axes[1].scatter(df_vd1['time_sec'], df_vd1['peak_velocity'], alpha=0.4, color='darkorange', s=15, label='Saccades')
        axes[1].plot(df_vd1['time_sec'], df_vd1['rolling_vel'], color='firebrick', linewidth=2.5, label=f'Rolling Mean (n={window_size})')
        axes[1].legend(frameon=True, facecolor='white', edgecolor='none')
    axes[1].set_title('VD1 (Video Viewing, High Complexity)', fontsize=14, fontweight='semibold')
    axes[1].set_xlabel('Time within session (seconds)', fontsize=12)
    axes[1].set_ylabel('Saccadic peak velocity (deg/sec)', fontsize=12)
    axes[1].tick_params(axis='both', which='major', labelsize=10)
    
    # Shared Y-axis limits for better comparison
    y_max = max(
        df_tex['peak_velocity'].max() if not df_tex.empty else 600, 
        df_vd1['peak_velocity'].max() if not df_vd1.empty else 600
    )
    axes[0].set_ylim(0, y_max * 1.05)
    axes[1].set_ylim(0, y_max * 1.05)
    
    plt.suptitle("Saccadic peak velocity over session: low vs high visual complexity", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    os.makedirs('plots', exist_ok=True)
    out_path = os.path.join('plots', '01_input_feature_timeseries.png')
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    print(f"Saved plot to {out_path}")
    
if __name__ == "__main__":
    main()
