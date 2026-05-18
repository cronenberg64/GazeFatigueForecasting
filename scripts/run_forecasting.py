import os
import pandas as pd
import numpy as np
import scipy.stats as stats
from tqdm import tqdm
from joblib import Parallel, delayed
import json
import warnings
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import pmdarima as pm

# Suppress warnings from models
warnings.filterwarnings('ignore')

def run_adf_test(series):
    if len(series) < 20:
        return 1.0
    try:
        res = adfuller(series, autolag='AIC')
        return float(res[1])  # p-value
    except:
        return 1.0

def fit_within_session_models(subject, round_num, task, series_data, subject_mean_val):
    if len(series_data) < 50:
        return None
        
    # Temporal split: 80% train, 20% test
    n_train = int(len(series_data) * 0.8)
    train = series_data[:n_train]
    test = series_data[n_train:]
    n_test = len(test)
    
    # Run ADF test on training data
    adf_p = run_adf_test(train)
    
    # Baselines
    rw_forecast = np.full(n_test, train[-1])
    mean_forecast = np.full(n_test, np.mean(train))
    sub_mean_forecast = np.full(n_test, subject_mean_val)
    
    # ARIMA fitting
    arima_forecast = rw_forecast.copy()
    arima_order = (1, 0, 0)
    try:
        model = pm.auto_arima(
            train, 
            start_p=1, start_q=1,
            max_p=5, max_q=5,
            max_d=1,
            stepwise=True,
            approximation=False,
            seasonal=False,
            error_action='ignore',
            suppress_warnings=True
        )
        arima_forecast = model.predict(n_periods=n_test)
        arima_order = model.order
    except Exception as e:
        pass
        
    # Exponential Smoothing
    es_forecast = rw_forecast.copy()
    try:
        es_model = SimpleExpSmoothing(train, initialization_method='estimated').fit()
        es_forecast = es_model.forecast(n_test)
    except:
        pass
        
    # Compute MAEs
    mae_rw = float(np.mean(np.abs(test - rw_forecast)))
    mae_mean = float(np.mean(np.abs(test - mean_forecast)))
    mae_sub_mean = float(np.mean(np.abs(test - sub_mean_forecast)))
    mae_arima = float(np.mean(np.abs(test - arima_forecast)))
    mae_es = float(np.mean(np.abs(test - es_forecast)))
    
    # Compute RMSEs
    rmse_rw = float(np.sqrt(np.mean((test - rw_forecast)**2)))
    rmse_mean = float(np.sqrt(np.mean((test - mean_forecast)**2)))
    rmse_sub_mean = float(np.sqrt(np.mean((test - sub_mean_forecast)**2)))
    rmse_arima = float(np.sqrt(np.mean((test - arima_forecast)**2)))
    rmse_es = float(np.sqrt(np.mean((test - es_forecast)**2)))
    
    return {
        'subject': subject,
        'round': round_num,
        'task': task,
        'n_train': n_train,
        'n_test': n_test,
        'adf_p': adf_p,
        'arima_order': arima_order,
        'metrics': {
            'MAE': {
                'RW': mae_rw,
                'Mean': mae_mean,
                'SubMean': mae_sub_mean,
                'ARIMA': mae_arima,
                'ES': mae_es
            },
            'RMSE': {
                'RW': rmse_rw,
                'Mean': rmse_mean,
                'SubMean': rmse_sub_mean,
                'ARIMA': rmse_arima,
                'ES': rmse_es
            }
        }
    }

def fit_cross_session_models(subject, task, series_data, subject_mean_val):
    # series_data is a sequence of round mean velocities, e.g. length up to 9
    if len(series_data) < 3:
        return None
        
    # Split: Train on first N-1 rounds, forecast last round N
    train = series_data[:-1]
    test_val = series_data[-1]
    
    # Baselines
    rw_forecast = train[-1]
    mean_forecast = np.mean(train)
    sub_mean_forecast = subject_mean_val
    
    # Exponential Smoothing
    es_forecast = rw_forecast
    try:
        es_model = SimpleExpSmoothing(train, initialization_method='estimated').fit()
        es_forecast = es_model.forecast(1)[0]
    except:
        pass
        
    # Compute MAEs (which is absolute error for a single forecast point)
    mae_rw = float(np.abs(test_val - rw_forecast))
    mae_mean = float(np.abs(test_val - mean_forecast))
    mae_sub_mean = float(np.abs(test_val - sub_mean_forecast))
    mae_es = float(np.abs(test_val - es_forecast))
    
    return {
        'subject': subject,
        'task': task,
        'n_train': len(train),
        'metrics': {
            'MAE': {
                'RW': mae_rw,
                'Mean': mae_mean,
                'SubMean': mae_sub_mean,
                'ES': mae_es
            },
            'RMSE': {
                'RW': mae_rw,  # Since it's a single point, RMSE = MAE
                'Mean': mae_mean,
                'SubMean': mae_sub_mean,
                'ES': mae_es
            }
        }
    }

def main():
    saccades_path = os.path.join("data", "processed_saccades.csv")
    if not os.path.exists(saccades_path):
        print(f"Error: {saccades_path} does not exist. Run preprocess.py first.")
        return
        
    print("Loading processed saccades...")
    df = pd.read_csv(saccades_path)
    
    # Group within-session series by (subject, round, task), sort by (session, onset_time)
    print("Grouping within-session sequences...")
    df_sorted = df.sort_values(by=['subject', 'round', 'session', 'onset_time'])
    grouped = df_sorted.groupby(['subject', 'round', 'task'])
    
    within_series_dict = {}
    subject_train_velocities = {}  # To compute subject-mean baseline from training portion
    
    # Collect series and define train portions for subject-mean baseline
    for (subj, rnd, tsk), group in tqdm(grouped, desc="Extracting within-session series"):
        vel_seq = group['peak_velocity'].values
        if len(vel_seq) >= 50:
            within_series_dict[(subj, rnd, tsk)] = vel_seq
            
            # Keep training portion for subject mean computation
            n_train = int(len(vel_seq) * 0.8)
            train_vels = vel_seq[:n_train]
            if subj not in subject_train_velocities:
                subject_train_velocities[subj] = []
            subject_train_velocities[subj].extend(train_vels)
            
    # Calculate Subject Means
    subject_means = {}
    for subj, vels in subject_train_velocities.items():
        subject_means[subj] = float(np.mean(vels))
        
    print(f"Constructed {len(within_series_dict)} within-session series across {len(subject_means)} subjects.")
    
    # Parallelize within-session modeling
    print("Fitting within-session forecasting models in parallel (ARIMA, ES, Baselines)...")
    within_tasks = [
        (subj, rnd, tsk, vel_seq, subject_means[subj])
        for (subj, rnd, tsk), vel_seq in within_series_dict.items()
    ]
    
    # Use all CPU cores
    within_results = Parallel(n_jobs=-1)(
        delayed(fit_within_session_models)(*t)
        for t in tqdm(within_tasks, desc="Within-session ARIMA/ES")
    )
    within_results = [r for r in within_results if r is not None]
    
    # ----------------------------------------------------
    # Cross-Session Series Computation
    # ----------------------------------------------------
    print("Constructing cross-session sequences (mean velocity per round)...")
    # For each (subject, round, task), compute mean peak velocity of all extracted saccades
    round_means = df.groupby(['subject', 'round', 'task'])['peak_velocity'].mean().reset_index()
    round_means = round_means.sort_values(by=['subject', 'task', 'round'])
    
    cross_series_dict = {}
    for (subj, tsk), group in round_means.groupby(['subject', 'task']):
        series = group['peak_velocity'].values
        if len(series) >= 3:
            cross_series_dict[(subj, tsk)] = series
            
    # Parallelize cross-session modeling (Exponential Smoothing, Naive, SubjectMean)
    print("Fitting cross-session forecasting models in parallel...")
    cross_tasks = [
        (subj, tsk, series, subject_means[subj])
        for (subj, tsk), series in cross_series_dict.items()
        if subj in subject_means
    ]
    
    cross_results = Parallel(n_jobs=-1)(
        delayed(fit_cross_session_models)(*t)
        for t in tqdm(cross_tasks, desc="Cross-session ES")
    )
    cross_results = [r for r in cross_results if r is not None]
    
    # ----------------------------------------------------
    # Save Results
    # ----------------------------------------------------
    results_path = os.path.join("data", "forecasting_results.json")
    print(f"Saving forecasting results to {results_path}...")
    
    with open(results_path, 'w') as f:
        json.dump({
            'within_session': within_results,
            'cross_session': cross_results,
            'subject_means': subject_means
        }, f, indent=4)
        
    print("Forecasting baseline computations completed successfully!")

if __name__ == "__main__":
    main()
