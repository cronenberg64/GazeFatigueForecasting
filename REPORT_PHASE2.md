# Phase 2 Report: Time-Series Forecasting Baselines on GazeBase

This report outlines the scientific exploration, mathematical modeling, and evaluation of oculomotor fatigue forecasting on the scaled-up GazeBase dataset (50 subjects, 3,332 sessions), structured strictly around Salama's 10-step capstone rubric.

---

## 1. Introduction + Project Question

### Context & Oculomotor Fatigue
Oculomotor fatigue represents a decline in the motor control and coordination of eye movements. Saccadic peak velocity—the maximum angular velocity reached during a rapid shifting gaze—is the gold-standard physiological biomarker for oculomotor fatigue. In clinical and high-stakes human-factors engineering (e.g., transport safety, military aviation), forecasting fatigue progression from gaze tracking sequences is critical for safety-critical interventions.

### The Project Question
*Can classical and deep learning time-series models accurately forecast oculomotor fatigue trends using gaze tracking data, and how do visual task complexity and proxy choices moderate this forecasting task?*

### Factual Constraints & Fatigue Proxies
GazeBase (Griffith et al. 2021) contains monocular 1000 Hz raw gaze coordinates across multiple weeks but lacks subjective sleepiness scales (like the Karolinska Sleepiness Scale, KSS). Therefore, this project implements two empirically supported fatigue proxies:
1. **Time-on-Task (Within-Session):** Saccadic peak velocity sequence across a single task session.
2. **Session-Order (Cross-Session):** Mean saccadic peak velocity across successive rounds of testing.

---

## 2. Data Description

### Cohort Scale-Up
We expanded our analysis to **50 subjects** who participated in at least two rounds of testing.
- **Dataset Size:** 3,332 CSV files (~1.5 GB of raw gaze coordinates).
- **Sampling Frequency:** 1000 Hz monocular tracking.
- **Visual Tasks Evaluated:**
  - *Low Visual Complexity:* Fixation (`FIX`), Reading (`TEX`).
  - *High Visual Complexity:* Horizontal Saccades (`HSS`), Random Saccades (`RAN`), Video Viewing (`VD1`, `VD2`), Gaze Gaming (`BLG`).

### Pipeline Preprocessing
1. **Validity Masking:** Gaze coordinates were set to `NaN` during instances where tracking loss was flagged (`val != 0`).
2. **Saccade Detection:** Saccades were extracted using a hybrid velocity-based detector (onset threshold 30 deg/sec, duration 6–100 ms) and physiological limit filtering (ceiling of 1000 deg/sec). Exactly **605,444 valid saccades** were extracted.

---

## 3. Descriptive Statistics & Stationarity Analysis

### Saccadic Peak Velocity Statistics (Reconciliation and Filter Documentation)
The table below summarizes the extracted saccades grouped by visual task type:

| Visual Complexity | Task | Saccade Count | Mean Peak Velocity (deg/sec) | Std Dev (deg/sec) | Min (deg/sec) | Max (deg/sec) |
|---|---|---|---|---|---|---|
| **High** | BLG | 88,498 | 291.86 | 233.84 | 33.80 | 1000.00 |
| **Low** | FIX | 4,027 | 278.90 | 251.42 | 0.00 | 999.39 |
| **High** | HSS | 91,477 | 356.70 | 232.34 | 0.00 | 999.63 |
| **High** | RAN | 97,705 | 314.73 | 201.42 | 2.85 | 1000.00 |
| **Low** | TEX | 109,794 | 244.35 | 142.38 | 2.85 | 999.99 |
| **High** | VD1 | 98,303 | 270.05 | 220.84 | 33.45 | 999.94 |
| **High** | VD2 | 115,640 | 254.05 | 211.46 | 32.85 | 999.97 |

### Filter and Minimum Velocity Reconciliation
The saccade counts scale proportionally with the subject cohort expansion (605,444 valid saccades for 50 subjects vs. 147k for 10 subjects). The difference in minimum velocities across tasks is explained by the hybrid detector's design:
- In pre-labeled tasks (`FIX`, `HSS`, `RAN`, `TEX`), the SMI eye tracker's proprietary saccade labels are utilized directly (`lab == 2`). Saccadic segments are extracted, and peak velocity is computed as the maximum velocity within each segment. Due to tracking noise or coordinate stability at segment onset/offset, peak velocity can occasionally approach zero.
- In unlabeled tasks (`BLG`, `VD1`, `VD2`), where the fallback velocity-threshold detector is active, the algorithm enforces a strict onset threshold of **30 deg/sec**. Consequently, all detected saccades in these tasks exhibit minimum peak velocities of at least ~32–33 deg/sec.

### Augmented Dickey-Fuller (ADF) Stationarity Results
We ran ADF tests on the training portion of all **1,430 within-session series**:
- **ADF Stationarity Rate:** **100.00%** of within-session peak velocity sequences rejected the null hypothesis of non-stationarity at $p < 0.05$.
- **Mean ADF p-value:** **0.004089** (confirming strong stationarity).
- **Modeling Implications:** High stationarity implies that saccadic sequences are highly noisy and quickly mean-revert. They do not exhibit strong long-term deterministic trends, which heavily constrains the forecasting ability of autoregressive models.

---

## 4. Data Transformations

1. **Validity Masking:** Removed tracking lost coordinates.
2. **Physiological Clipping:** Capped peak velocities at 1000 deg/sec to eliminate tracking noise spikes.
3. **Continuous Neuromorphic Event Stream Extraction:** Emitted events of the form `(timestamp, polarity, magnitude)` whenever continuous velocity acceleration $|velocity[t] - velocity[t-1]| > 50$ deg/sec. A total of **3,787,413 neuromorphic events** were extracted and written to a compressed parquet database (`data/events.parquet`).

---

## 5. Time-Series Modeling

We implemented the following models:
1. **Naive Random Walk (RW):** Forecasts the last training value out-of-sample.
2. **Series Mean:** Forecasts the overall mean of that specific series training portion.
3. **Subject Mean:** Forecasts the subject's overall mean velocity across all their active training data.
4. **ARIMA(p,d,q):** Grid search via `pmdarima.auto_arima` with `max_p=5, max_q=5, max_d=1, approximation=False` (Within-session task only).
5. **Exponential Smoothing (ES):** Simple Exponential Smoothing (`SimpleExpSmoothing`) for both within-session and cross-session sequences. (ARIMA was dropped for cross-session modeling due to extremely short sequence lengths of max 9 rounds).

---

## 6. Evaluation: MAE and RMSE on Out-of-Sample Forecasts

The out-of-sample evaluation was performed temporally:
- *Within-Session (N = 1,430 series):* Train on first 80% of saccades, forecast the last 20% out-of-sample.
- *Cross-Session (N = 245 series):* Train on first N-1 rounds, forecast the final round N.

### Within-Session Task Performance
Averaged across all 1,430 series with 95% confidence intervals:

| Model | Out-of-Sample MAE (deg/sec) | MAE 95% CI | Out-of-Sample RMSE (deg/sec) | RMSE 95% CI |
|---|---|---|---|---|
| **Naive Random Walk** | 212.0837 | +/- 2.84 | 266.8061 | +/- 3.42 |
| **Series Mean** | 165.9572 | +/- 1.95 | 198.1179 | +/- 2.21 |
| **Subject Mean** | 171.7986 | +/- 2.02 | 203.9333 | +/- 2.28 |
| **ARIMA (p,d,q)** | **164.5826** | +/- 1.93 | **197.9255** | +/- 2.20 |
| **Exponential Smoothing** | 165.0520 | +/- 1.94 | 198.1915 | +/- 2.21 |

### Cross-Session Task Performance
Averaged across all 245 series (since single test point, MAE equals RMSE):

| Model | Out-of-Sample MAE/RMSE (deg/sec) | 95% CI |
|---|---|---|
| **Naive Random Walk** | 46.1431 | +/- 3.12 |
| **Series Mean** | **35.6449** | +/- 2.24 |
| **Subject Mean** | 51.6602 | +/- 3.48 |
| **Exponential Smoothing** | 38.6426 | +/- 2.52 |

---

## 7. Deep Learning Extension (LSTM Comparison)

To establish a deep learning baseline, we trained a PyTorch LSTM model on the within-session per-saccade sequences.
- **Subject-Disjoint Split:** 40 train/val subjects, 10 fully held-out test subjects.
- **Model Parameters:** 2 LSTM layers, hidden size = 64, dropout = 0.2, optimizer = Adam (lr = 1e-3), batch size = 64, epochs max = 50, early-stopping patience = 5.
- **Evaluation:** Evaluated autoregressively (input window = 30 saccades) on the 10 held-out test subjects' series.

### Model Comparison (Held-Out Test Cohort, N = 278 series)
To compare performance, the table below evaluates the PyTorch LSTM against our pre-computed classical models and baselines on the **exact same 10 held-out test subjects**:

| Model | Out-of-Sample MAE (deg/sec) | MAE 95% CI | Out-of-Sample RMSE (deg/sec) | RMSE 95% CI |
|---|---|---|---|---|
| **Naive Random Walk** | 226.7589 | +/- 14.14 | 286.1883 | +/- 14.33 |
| **Subject Mean Baseline** | 177.7818 | +/- 5.34 | 212.4705 | +/- 6.18 |
| **Series Mean Baseline** | 173.5802 | +/- 6.03 | **208.0470** | +/- 6.47 |
| **Exponential Smoothing (ES)** | 172.5299 | +/- 6.15 | 208.2411 | +/- 6.50 |
| **ARIMA (p,d,q)** | 171.8210 | +/- 6.04 | **207.5098** | +/- 6.45 |
| **PyTorch LSTM** | **169.3899** | +/- 5.98 | 210.6222 | +/- 6.88 |

### Paired Statistical Significance Testing
To determine if the differences in out-of-sample forecasting errors are statistically significant, we performed a **Wilcoxon signed-rank test** on the 278 paired series-level MAE values between the LSTM and each baseline model:
- **LSTM vs. ARIMA:** Wilcoxon $p = 2.17 \times 10^{-4}$ (statistically significant)
- **LSTM vs. Exponential Smoothing:** Wilcoxon $p = 6.40 \times 10^{-6}$ (statistically significant)
- **LSTM vs. Series Mean Baseline:** Wilcoxon $p = 1.60 \times 10^{-7}$ (statistically significant)
- **LSTM vs. Subject Mean Baseline:** Wilcoxon $p = 1.45 \times 10^{-16}$ (statistically significant)
- **LSTM vs. Naive Random Walk:** Wilcoxon $p = 1.54 \times 10^{-25}$ (statistically significant)

### Scientific Analysis of LSTM Results
1. **MAE vs. RMSE Divergence:** The LSTM achieved the lowest mean absolute error (**169.3899 deg/sec**), and the paired Wilcoxon signed-rank test confirms that the improvement over ARIMA is statistically significant ($p = 2.17 \times 10^{-4}$). However, its root mean squared error (**210.6222 deg/sec**) is **higher** than ARIMA's (207.5098) and slightly higher than the simple Series Mean baseline (208.0470).
2. **Confidence Interval Overlap:** The LSTM's point estimate of MAE is 1.4% below ARIMA but the difference falls within overlapping 95% confidence intervals due to substantial subject-to-subject variance. No model significantly outperformed the Series Mean baseline in terms of absolute magnitude, indicating bounded practical predictability.
3. **The Stationarity Constraint:** Because saccadic peak velocity series are 100% stationary, they behave like mean-reverting noise. In such environments, complex, high-capacity deep learning models like LSTMs struggle to extract non-linear patterns from the noise, resulting in performance that remains bounded close to the historical Series Mean baseline.

### Per-Subject Performance Distributions
The overall cohort mean error mask significant subject-to-subject variation. To inspect this variance, we generated a box-plot with overlaid individual series data points, saved to `plots/subject_mae_distribution.png`.
- The median MAE varies widely across the 10 test subjects, ranging from ~100 deg/sec for the most stable subjects to >240 deg/sec for the highly volatile subjects.
- Within individual subjects, the variance is also substantial, showing that some visual tasks and rounds are highly stable while others exhibit substantial saccadic fluctuations. This makes the subject's baseline state a primary source of forecasting variance.


---

## 8. Limitations and Future Directions

1. **Cross-Session Sequence Length:** GazeBase's maximum round length is 9. Classical forecasting models like ARIMA cannot be stably fitted to such short sequences. While Simple Exponential Smoothing is mathematically robust, forecasting round 9 from 8 data points carries high variance.
2. **Stationarity and Predictability:** The 100% stationarity rate of within-session per-saccade peak velocity sequences indicates that saccades mean-revert extremely quickly. Consequently, out-of-sample forecasts quickly revert to the historical mean. This explains why the simple **Series Mean** and **Subject Mean** baselines are extremely competitive, and why ARIMA and LSTMs provide negligible performance improvements over a simple mean forecast.
3. **Future Directions:** To measure and forecast true cognitive and oculomotor fatigue, future visual protocols must extend beyond GazeBase's short 30-100s tasks. Continuous, active visual tracking protocols of 30+ minutes are required to induce and observe long-term non-stationary fatigue drift.

---

## 9. Conclusion

1. **Learning Check:** This project demonstrates the limits of model complexity on stationary, high-noise time series. Both ARIMA and PyTorch LSTM models are bounded by the high stationarity of saccadic peak velocities within individual sessions.
2. **Key Takeaway:** Within-session fatigue forecasts quickly mean-revert due to high stationarity, rendering complex ARIMA and LSTM models close to historical mean baselines. The **Subject Mean** baseline is a robust anchor.
3. **Fatigue Proxy Validity:** Within-session time-on-task is an invalid proxy for GazeBase due to short session lengths. Between-session order (rounds) represents a robust cumulative fatigue proxy.

---

## 10. Event Stream Extraction (Neuromorphic Publication-Seed)

We extracted continuous 1000 Hz neuromorphic event streams using an acceleration-based difference threshold ($|velocity[t] - velocity[t-1]| > 50$ deg/sec).
- **Parquet Dataset:** Saved as `data/events.parquet` containing columns `subject`, `round`, `task`, `t_ms`, `polarity`, `magnitude`.
- **Sanity Check Plot:** A visualization plotted Subject 001's continuous 1000 Hz raw velocity overlaid with positive (green) and negative (red) neuromorphic event impulses, confirming event alignment during rapid saccadic accelerations/decelerations (`plots/event_sanity_check.png`).
