import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def main():
    # Load processed saccades
    df = pd.read_csv('data/processed_saccades.csv')
    df['time_sec'] = df['onset_time'] / 1000.0
    
    # Define visual complexity mapping
    complexity_map = {
        'FIX': 'Low',
        'TEX': 'Low',
        'VD1': 'High',
        'VD2': 'High',
        'RAN': 'High'
    }
    
    sns.set_theme(style="whitegrid")
    
    # ----------------------------------------------------
    # PLOT 2: MAIN FATIGUE PROXIES (1x3 Panel)
    # ----------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    
    # --- Subplot A: Within-session proxy (Time-on-Task) ---
    # Bin time into 5-second intervals within session
    df['time_bin'] = (df['time_sec'] // 5) * 5
    
    # Group by task and time bin, average across all subjects & sessions
    mean_over_time = df.groupby(['task', 'time_bin'])['peak_velocity'].mean().reset_index()
    # Sort tasks for consistent coloring
    mean_over_time = mean_over_time.sort_values('task')
    
    sns.lineplot(
        data=mean_over_time, 
        x='time_bin', 
        y='peak_velocity', 
        hue='task', 
        marker='o', 
        linewidth=1.5,
        ax=axes[0]
    )
    axes[0].set_title('A. Within-Session Proxy (Time-on-Task)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Time within session (seconds)', fontsize=11)
    axes[0].set_ylabel('Mean Saccadic Peak Velocity (deg/sec)', fontsize=11)
    axes[0].legend(title='Task Type', frameon=True, facecolor='white')
    
    # --- Subplot B: Between-session proxy (Session Order) ---
    # Average velocity per subject, session, and task
    session_means = df.groupby(['subject', 'task', 'session'])['peak_velocity'].mean().reset_index()
    
    sns.barplot(
        data=session_means, 
        x='task', 
        y='peak_velocity', 
        hue='session', 
        palette='Set2',
        errorbar='se', 
        capsize=0.08,
        ax=axes[1]
    )
    axes[1].set_title('B. Between-Session Proxy (Session Order)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Task Type', fontsize=11)
    axes[1].set_ylabel('Mean Peak Velocity (deg/sec)', fontsize=11)
    axes[1].legend(title='Session', frameon=True, facecolor='white')
    
    # --- Subplot C: Complexity modulator of decline ---
    # Filter to tasks in visual complexity classification
    df_complexity = df[df['task'].isin(complexity_map.keys())].copy()
    df_complexity['complexity'] = df_complexity['task'].map(complexity_map)
    
    # Calculate within-session decline (early - late) per subject, round, session, task
    declines = []
    grouped = df_complexity.groupby(['subject', 'round', 'session', 'task'])
    for name, group in grouped:
        if len(group) < 10:
            continue
        group = group.sort_values('onset_time')
        n_saccades = len(group)
        early_idx = int(n_saccades * 0.25)
        late_idx = int(n_saccades * 0.75)
        
        early_mean = group.iloc[:early_idx]['peak_velocity'].mean()
        late_mean = group.iloc[late_idx:]['peak_velocity'].mean()
        
        decline = early_mean - late_mean
        
        declines.append({
            'subject': name[0],
            'task': name[3],
            'complexity': complexity_map[name[3]],
            'decline': decline
        })
        
    df_decline = pd.DataFrame(declines)
    
    # Plot decline grouped by visual complexity
    sns.barplot(
        data=df_decline, 
        x='complexity', 
        y='decline', 
        order=['Low', 'High'],
        palette=['royalblue', 'crimson'],
        errorbar='se', 
        capsize=0.1,
        ax=axes[2]
    )
    
    # Overlay individual subject paired comparisons (averaged decline per subject per complexity)
    subj_decline = df_decline.groupby(['subject', 'complexity'])['decline'].mean().reset_index()
    pivot_decline = subj_decline.pivot(index='subject', columns='complexity', values='decline').dropna()
    
    for i, row in pivot_decline.iterrows():
        axes[2].plot(['Low', 'High'], [row['Low'], row['High']], marker='o', color='gray', alpha=0.4, linestyle='--')
        
    axes[2].set_title('C. Visual Complexity Moderation', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Visual Complexity', fontsize=11)
    axes[2].set_ylabel('Mean Decline (Early - Late) [deg/sec]', fontsize=11)
    
    plt.suptitle("Fatigue proxies in GazeBase (no KSS labels available — using time-on-task and session order)", fontsize=15, fontweight='bold', y=1.03)
    plt.tight_layout()
    
    os.makedirs('plots', exist_ok=True)
    out_path1 = os.path.join('plots', '02_fatigue_proxies.png')
    plt.savefig(out_path1, bbox_inches='tight', dpi=300)
    print(f"Saved main plot to {out_path1}")
    
    # ----------------------------------------------------
    # PLOT 2B: FACETED SESSION ORDER BY COMPLEXITY (1x2 Panel)
    # ----------------------------------------------------
    df_comp_sess = df[df['task'].isin(complexity_map.keys())].copy()
    df_comp_sess['complexity'] = df_comp_sess['task'].map(complexity_map)
    
    # Group by subject, complexity, task, and session to get mean peak velocity
    task_session_means = df_comp_sess.groupby(['subject', 'complexity', 'task', 'session'])['peak_velocity'].mean().reset_index()
    
    fig, axes2 = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True, dpi=300)
    
    # Left Panel: Low Complexity tasks (FIX, TEX)
    df_low = task_session_means[task_session_means['complexity'] == 'Low']
    sns.barplot(
        data=df_low, 
        x='task', 
        y='peak_velocity', 
        hue='session', 
        palette='Set2', 
        errorbar='se', 
        capsize=0.08,
        ax=axes2[0]
    )
    axes2[0].set_title('Low Visual Complexity (FIX, TEX)', fontsize=13, fontweight='semibold')
    axes2[0].set_xlabel('Task Type', fontsize=12)
    axes2[0].set_ylabel('Mean Peak Velocity (deg/sec)', fontsize=12)
    axes2[0].legend(title='Session', frameon=True, facecolor='white')
    
    # Right Panel: High Complexity tasks (VD1, VD2, RAN)
    df_high = task_session_means[task_session_means['complexity'] == 'High']
    sns.barplot(
        data=df_high, 
        x='task', 
        y='peak_velocity', 
        hue='session', 
        palette='Set2', 
        errorbar='se', 
        capsize=0.08,
        ax=axes2[1]
    )
    axes2[1].set_title('High Visual Complexity (VD1, VD2, RAN)', fontsize=13, fontweight='semibold')
    axes2[1].set_xlabel('Task Type', fontsize=12)
    axes2[1].set_ylabel('', fontsize=12) # Shared y-axis label
    axes2[1].legend(title='Session', frameon=True, facecolor='white')
    
    plt.suptitle("Session-Order Fatigue Effect (Session 1 vs Session 2) Faceted by Visual Complexity", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    out_path2 = os.path.join('plots', '02b_session_order_by_complexity.png')
    plt.savefig(out_path2, bbox_inches='tight', dpi=300)
    print(f"Saved faceted plot to {out_path2}")

if __name__ == "__main__":
    main()
