import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
import os

def cohen_d(x, y):
    # Paired Cohen's d
    diff = x - y
    return np.mean(diff) / np.std(diff, ddof=1)

def main():
    # Load processed saccades
    df = pd.read_csv('data/processed_saccades.csv')
    
    # Complexity mapping
    complexity_map = {
        'FIX': 'Low',
        'TEX': 'Low',
        'VD1': 'High',
        'VD2': 'High',
        'RAN': 'High'
    }
    
    # Filter to tasks we care about
    df_comp = df[df['task'].isin(complexity_map.keys())].copy()
    df_comp['complexity'] = df_comp['task'].map(complexity_map)
    
    results = []
    
    # Group by subject, round, session, and task to split into early vs late
    grouped = df_comp.groupby(['subject', 'round', 'session', 'task'])
    for name, group in grouped:
        if len(group) < 10:
            continue
            
        group = group.sort_values('onset_time')
        n_saccades = len(group)
        early_idx = int(n_saccades * 0.25)
        late_idx = int(n_saccades * 0.75)
        
        early_vel = group.iloc[:early_idx]['peak_velocity'].mean()
        late_vel = group.iloc[late_idx:]['peak_velocity'].mean()
        
        results.append({
            'subject': name[0],
            'round': name[1],
            'session': name[2],
            'task': name[3],
            'complexity': complexity_map[name[3]],
            'early_vel': early_vel,
            'late_vel': late_vel
        })
        
    df_res = pd.DataFrame(results)
    
    # Set plotting theme
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9, 8), dpi=300)
    
    colors = {'Low': 'royalblue', 'High': 'crimson'}
    
    for comp in ['Low', 'High']:
        comp_df = df_res[df_res['complexity'] == comp].dropna(subset=['early_vel', 'late_vel'])
        x = comp_df['early_vel']
        y = comp_df['late_vel']
        
        if len(x) < 2:
            continue
            
        # Run paired t-test
        t_stat, p_val = stats.ttest_rel(x, y)
        d = cohen_d(x, y)
        
        mean_early = x.mean()
        mean_late = y.mean()
        mean_diff = mean_early - mean_late
        
        label = (f"{comp} Complexity (n={len(x)})\n"
                 f"  Early: {mean_early:.1f}, Late: {mean_late:.1f} deg/sec\n"
                 f"  Mean Δ: {mean_diff:.2f} deg/sec\n"
                 f"  p = {p_val:.5f}, Cohen's d = {d:.2f}")
                 
        plt.scatter(x, y, alpha=0.6, color=colors[comp], s=35, label=label, edgecolor='none')
        
    # Identity line (y = x)
    all_vals = pd.concat([df_res['early_vel'], df_res['late_vel']]).dropna()
    min_val = all_vals.min()
    max_val = all_vals.max()
    padding = (max_val - min_val) * 0.05
    
    plt.plot(
        [min_val - padding, max_val + padding], 
        [min_val - padding, max_val + padding], 
        color='black', 
        linestyle='--', 
        alpha=0.7, 
        linewidth=1.5,
        label='Identity Line (y=x)'
    )
    
    plt.title("Within-Session Velocity Decline by Visual Complexity\n(Early vs Late Saccades)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Early-Window Mean Peak Velocity (deg/sec)", fontsize=12)
    plt.ylabel("Late-Window Mean Peak Velocity (deg/sec)", fontsize=12)
    plt.xlim(min_val - padding, max_val + padding)
    plt.ylim(min_val - padding, max_val + padding)
    
    # Legend outside plot for clarity
    plt.legend(loc='upper left', bbox_to_anchor=(1.03, 1), title="Statistics", title_fontsize=11, frameon=True, facecolor='white')
    
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    out_path = os.path.join('plots', '03_early_late_velocity.png')
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
