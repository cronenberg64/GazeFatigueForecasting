# GazeBase Data Exploration for Fatigue Forecasting

This project explores the [GazeBase](https://figshare.com/articles/dataset/GazeBase_data_Repository/12912257) dataset (Griffith et al. 2021) to investigate if oculomotor fatigue can be forecasted from gaze time series. 

Since GazeBase does not contain Karolinska Sleepiness Scale (KSS) ratings or any subjective fatigue questionnaires, this exploration uses **time-on-task** and **session-order** as empirically supported proxies for oculomotor fatigue. A moderating variable of visual task complexity is also considered.

## Environment Setup
- Python 3.11+
- Install dependencies: `pip install pandas numpy matplotlib seaborn scipy requests tqdm`

## Directory Structure
- `data/`: Local cache for downloaded GazeBase dataset CSV files.
- `scripts/`: Python scripts for data downloading, preprocessing, and plotting.
- `plots/`: Generated plots (PNG format).

## Phases
1. **Data Acquisition:** Download a subset of the 50 GB dataset (5-10 subjects across multiple rounds) using HTTP Range streaming.
2. **Preprocessing:** Verify dataset constraints (1000 Hz sampling, `val` / `lab` columns) and compute saccadic peak velocities.
3. **Exploration & Plotting:** Generate plots representing the input feature time series, fatigue proxies, and early-vs-late session velocity contrasts.
4. **Summary Report:** Produce a written report detailing the statistical findings of the exploration.