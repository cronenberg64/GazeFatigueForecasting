import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def main():
    # Load processed saccades
    df = pd.read_csv('data/processed_saccades.csv')
    df['time_sec'] = df['onset_time'] / 1000.0
    
    # Define complexity
    complexity_map = {'FIX': 'Low', 'TEX': 'Low', 'VD1': 'High', 'VD2': 'High', 'RAN': 'High'}
    
    # Filter to only tasks we care about for complexity analysis (optional, but requested to exclude some)
    # We will exclude BLG and HSS for complexity analysis, but keep all for Subplot A and B.
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    sns.set_theme(style="whitegrid")
    
    # --- Subplot A: Within-session proxy ---
    # Bin time into 5-second intervals
    df['time_bin'] = (df['time_sec'] // 5) * 5
    
    # Group by task and time_bin
    mean_over_time = df.groupby(['task', 'time_bin'])['peak_velocity'].mean().reset_index()
    
    sns.lineplot(data=mean_over_time, x='time_bin', y='peak_velocity', hue='task', ax=axes[0], marker='o')
    axes[0].set_title('A. Within-Session Proxy (Time-on-Task)')
    axes[0].set_xlabel('Time within session (seconds)')
    axes[0].set_ylabel('Mean Peak Velocity (deg/sec)')
    
    # --- Subplot B: Between-session proxy ---
    # Mean velocity per subject, task, and session
    # We'll limit to round 1 if there are multiple rounds, to avoid over-averaging,
    # or just average across all rounds where session 1 & 2 exist.
    session_means = df.groupby(['subject', 'task', 'session'])['peak_velocity'].mean().reset_index()
    
    sns.barplot(data=session_means, x='task', y='peak_velocity', hue='session', errorbar='se', ax=axes[1])
    axes[1].set_title('B. Between-Session Proxy')
    axes[1].set_xlabel('Task Type')
    axes[1].set_ylabel('Mean Peak Velocity (deg/sec)')
    
    # --- Subplot C: Task-complexity effect ---
    # Calculate within-session decline per subject, session, task
    df_complexity = df[df['task'].isin(complexity_map.keys())].copy()
    df_complexity['complexity'] = df_complexity['task'].map(complexity_map)
    
    # To compute early vs late, we'll split each (subject, round, session, task) session into first 25% and last 25%
    # by saccade index as per step 6 description (Wait, Step 5 says "bar chart of mean saccadic peak velocity grouped by low-complexity vs high-complexity tasks"
    # Actually, Step 5 just says "bar chart of mean saccadic peak velocity grouped by low-complexity vs high-complexity tasks."
    # Let's just plot the mean peak velocity overall for Low vs High complexity, or maybe the decline.
    # The prompt says: "Subplot C — TASK-COMPLEXITY effect: bar chart of mean saccadic peak velocity grouped by low-complexity vs high-complexity tasks. Include per-subject paired comparisons if possible. Expectation: complexity should modulate the magnitude of the within-session decline."
    # Oh! To show it modulates the *magnitude of the decline*, I should plot the DECLINE grouped by complexity.
    
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
    
    sns.barplot(data=df_decline, x='complexity', y='decline', ax=axes[2], order=['Low', 'High'], errorbar='se', capsize=0.1)
    
    # Overlay subject pairs if possible (average decline per subject per complexity)
    subj_decline = df_decline.groupby(['subject', 'complexity'])['decline'].mean().reset_index()
    # Pivot to match subjects
    pivot_decline = subj_decline.pivot(index='subject', columns='complexity', values='decline').dropna()
    
    for i, row in pivot_decline.iterrows():
        axes[2].plot(['Low', 'High'], [row['Low'], row['High']], marker='o', color='gray', alpha=0.5, linestyle='--')
        
    axes[2].set_title('C. Complexity Effect on Velocity Decline')
    axes[2].set_xlabel('Visual Complexity')
    axes[2].set_ylabel('Velocity Decline (Early - Late) [deg/sec]')
    
    plt.suptitle("Fatigue proxies in GazeBase (no KSS labels available — using time-on-task and session order)", fontsize=16, y=1.05)
    plt.tight_layout()
    
    os.makedirs('plots', exist_ok=True)
    out_path = os.path.join('plots', '02_fatigue_proxies.png')
    plt.savefig(out_path, bbox_inches='tight')
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
