import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
import os

def cohen_d(x, y):
    # paired Cohen's d
    diff = x - y
    return np.mean(diff) / np.std(diff, ddof=1)

def main():
    df = pd.read_csv('data/processed_saccades.csv')
    
    complexity_map = {'FIX': 'Low', 'TEX': 'Low', 'VD1': 'High', 'VD2': 'High', 'RAN': 'High'}
    df_comp = df[df['task'].isin(complexity_map.keys())].copy()
    df_comp['complexity'] = df_comp['task'].map(complexity_map)
    
    results = []
    
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
            'task': name[3],
            'complexity': complexity_map[name[3]],
            'early_vel': early_vel,
            'late_vel': late_vel
        })
        
    df_res = pd.DataFrame(results)
    
    # Calculate stats
    stats_text = []
    colors = {'Low': 'blue', 'High': 'red'}
    
    plt.figure(figsize=(8, 8), dpi=300)
    
    for comp in ['Low', 'High']:
        comp_df = df_res[df_res['complexity'] == comp].dropna(subset=['early_vel', 'late_vel'])
        x = comp_df['early_vel']
        y = comp_df['late_vel']
        
        if len(x) < 2:
            continue
            
        t_stat, p_val = stats.ttest_rel(x, y)
        d = cohen_d(x, y)
        
        mean_early = x.mean()
        mean_late = y.mean()
        
        label = (f"{comp} Comp (n={len(x)})\n"
                 f"Early: {mean_early:.1f}, Late: {mean_late:.1f}\n"
                 f"p = {p_val:.3e}, d = {d:.2f}")
                 
        plt.scatter(x, y, alpha=0.5, color=colors[comp], label=label)
        
    # Identity line
    min_val = min(df_res['early_vel'].min(), df_res['late_vel'].min())
    max_val = max(df_res['early_vel'].max(), df_res['late_vel'].max())
    
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, label='Identity (y=x)')
    
    plt.title("Within-session velocity decline by visual complexity\n(early vs late saccades)", fontsize=14)
    plt.xlabel("Early-window Mean Velocity (deg/sec)")
    plt.ylabel("Late-window Mean Velocity (deg/sec)")
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title="Statistics")
    
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    out_path = os.path.join('plots', '03_early_late_velocity.png')
    plt.savefig(out_path, bbox_inches='tight')
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
