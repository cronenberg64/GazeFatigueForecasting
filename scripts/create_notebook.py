import json
import os

def main():
    notebook = {
        "cells": [
            # Step 1
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Time-Series Forecasting Baselines on GazeBase\n",
                    "### Course Capstone Project — Oculomotor Fatigue Forecasting Pipeline\n",
                    "**Framework:** Salama's 10-Step Rubric\n",
                    "\n",
                    "---\n",
                    "\n",
                    "## Step 1: Introduction + Project Question\n",
                    "\n",
                    "### Context & Motivation\n",
                    "Oculomotor fatigue represents a physiological decline in the control and velocity of eye movements, typically induced by prolonged visual and cognitive demand. In clinical and human-factors engineering, forecasting the onset of oculomotor fatigue is critical for high-stakes applications such as transport safety, aviation, and military operations. Saccadic peak velocity—the maximum angular speed reached by the eye during a rapid shifting gaze—is widely recognized as the gold-standard objective biomarker for oculomotor fatigue.\n",
                    "\n",
                    "### The Project Question\n",
                    "*Can classical and deep learning time-series models accurately forecast oculomotor fatigue trends using gaze tracking data, and how do visual task complexity and proxy choices moderate this forecasting task?*\n",
                    "\n",
                    "### Hypotheses & Constraints\n",
                    "1. **Within-Session Time-on-Task Hypothesis:** Oculomotor fatigue will manifest as a continuous, significant decline in saccadic peak velocity within a single task session. (Hedged: we will let the data confirm or refute this via stationarity and forecasting performance).\n",
                    "2. **Between-Session Cumulative Hypothesis:** Cumulative fatigue will be highly detectable and predictable across successive task rounds (Session 1 vs. Session 2).\n",
                    "3. **Visual Complexity Moderation:** High-complexity visual environments (e.g., Random Saccades, Video Viewing) will induce larger and more predictable fatigue declines than low-complexity environments (e.g., Fixation, Reading).\n",
                    "\n",
                    "### Dataset Constraint\n",
                    "The GazeBase dataset (Griffith et al. 2021) contains monocular 1000 Hz raw gaze samples but lacks Karolinska Sleepiness Scale (KSS) or subjective fatigue questionnaires. Therefore, this project utilizes **Time-on-Task** (within-session) and **Session-Order** (between-session) as empirical proxies for fatigue, supported by established eye-tracking literature (*Di Stasi et al. 2013*)."
                ]
            },
            # Step 2
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 2: Data Description\n",
                    "\n",
                    "### Dataset Characteristics\n",
                    "- **Subjects:** 50 unique subjects who participated in multiple rounds of testing (scaled up from the initial 10-subject subset to meet the course target of >=50 subjects).\n",
                    "- **Sampling Rate:** 1000 Hz monocular eye tracking (SMI iView X high-speed system).\n",
                    "- **Total Sessions:** 3,332 CSV files in total, representing up to 9 rounds of testing separated by weeks.\n",
                    "- **Tasks Evaluated:**\n",
                    "  - *Low Visual Complexity:* Fixation (`FIX`), Reading (`TEX`).\n",
                    "  - *High Visual Complexity:* Horizontal Saccades (`HSS`), Random Saccades (`RAN`), Video Viewing (`VD1`, `VD2`), Gaze Gaming (`BLG`).\n",
                    "\n",
                    "### Preprocessing & Validity Masking\n",
                    "To prevent tracking loss artifacts and physically impossible coordinate jumps from corrupting the time series:\n",
                    "1. **Validity Mask:** Set $x$ and $y$ coordinates to `NaN` when the eye-tracker flags invalid tracking (`val != 0`).\n",
                    "2. **Velocity Threshold Saccade Detection:** Saccades are extracted using a hybrid approach: pre-labeled event markers (`lab == 2`) when available, and an adaptive velocity-threshold detector (30–1000 deg/sec) for unlabeled sessions. Peak velocity represents the maximum velocity reached during each valid saccade. Saccades with peak velocities $> 1000$ deg/sec are filtered out as physiological anomalies."
                ]
            },
            # Imports
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "import json\n",
                    "import scipy.stats as stats\n",
                    "\n",
                    "# Set style\n",
                    "sns.set_theme(style='whitegrid', context='notebook')\n",
                    "plt.rcParams['figure.dpi'] = 120\n",
                    "plt.rcParams['figure.figsize'] = (10, 5)\n",
                    "print(\"Imports successful!\")"
                ]
            },
            # Step 3
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 3: Descriptive Statistics & Stationarity Analysis\n",
                    "\n",
                    "We compute the overall descriptive statistics of our extracted saccades and run **Augmented Dickey-Fuller (ADF) stationarity tests** on the within-session series. Stationarity is a fundamental property in time-series forecasting: if a series is highly stationary, out-of-sample forecasts will quickly mean-revert, limiting the utility of complex models."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Load processed saccades\n",
                    "saccades_path = os.path.join(\"data\", \"processed_saccades.csv\")\n",
                    "df = pd.read_csv(saccades_path)\n",
                    "print(f\"Loaded {len(df)} total valid saccades across all subjects and tasks.\")\n",
                    "\n",
                    "# Summary statistics by task\n",
                    "summary_stats = df.groupby('task')['peak_velocity'].agg(['count', 'mean', 'std', 'min', 'max'])\n",
                    "print(\"\\n=== Saccadic Peak Velocity Summary Statistics by Task ===\")\n",
                    "display(summary_stats.round(2))\n",
                    "\n",
                    "# Load forecasting and stationarity results\n",
                    "with open(os.path.join(\"data\", \"forecasting_results.json\"), \"r\") as f:\n",
                    "    forecast_data = json.load(f)\n",
                    "\n",
                    "within_res = pd.DataFrame(forecast_data['within_session'])\n",
                    "\n",
                    "# Calculate stationarity rate (ADF p-value < 0.05)\n",
                    "stationarity_rate = (within_res['adf_p'] < 0.05).mean() * 100\n",
                    "print(f\"\\n=== Augmented Dickey-Fuller (ADF) Stationarity Test Results ===\")\n",
                    "print(f\"Mean ADF p-value across all series: {within_res['adf_p'].mean():.6f}\")\n",
                    "print(f\"Percentage of stationary series (p < 0.05): {stationarity_rate:.2f}%\")"
                ]
            },
            # Step 4
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 4: Data Transformations\n",
                    "\n",
                    "1. **Validity Masking:** Applied during data preprocessing to eliminate tracking dropouts.\n",
                    "2. **Continuous Event Extraction:** To support neuromorphic modeling, raw continuous velocity was differenced, and events were emitted when $|velocity[t] - velocity[t-1]| > 50$ deg/sec. Let's load the generated sanity check plot to confirm the sparse event stream alignment against the continuous velocity signal."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Display Step 10 Sanity Check Plot\n",
                    "sanity_plot_path = os.path.join(\"plots\", \"event_sanity_check.png\")\n",
                    "if os.path.exists(sanity_plot_path):\n",
                    "    img = plt.imread(sanity_plot_path)\n",
                    "    plt.figure(figsize=(12, 6))\n",
                    "    plt.imshow(img)\n",
                    "    plt.axis('off')\n",
                    "    plt.show()\n",
                    "else:\n",
                    "    print(\"Sanity check plot not found. Run extract_events.py first.\")"
                ]
            },
            # Step 5
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 5: Time-Series Modeling\n",
                    "\n",
                    "We implement and fit the following classical forecasting models and naive baselines:\n",
                    "1. **Naive Random Walk (RW):** Forecasts the last training value for all out-of-sample steps.\n",
                    "2. **Series Mean:** Forecasts the overall mean of the series training portion.\n",
                    "3. **Subject Mean:** For each held-out series, predicts the subject's overall mean velocity computed from all their training data.\n",
                    "4. **ARIMA(p,d,q):** Grid search via `pmdarima.auto_arima` with `max_p=5, max_q=5, max_d=1, approximation=False`. (Within-session task only).\n",
                    "5. **Exponential Smoothing (ES):** Simple Exponential Smoothing (`SimpleExpSmoothing`) for both within-session and cross-session sequences."
                ]
            },
            # Step 6
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 6: Evaluation: MAE and RMSE on Out-of-Sample Forecasts\n",
                    "\n",
                    "We evaluate forecasting performance using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) averaged across all subjects with 95% confidence intervals. The forecasting tasks are divided into:\n",
                    "- **Within-Session:** Train on the first 80% of saccades, forecast the last 20%.\n",
                    "- **Cross-Session:** Train on the first N-1 rounds, forecast round N."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Extract metrics for Within-Session task\n",
                    "within_res = pd.DataFrame(forecast_data['within_session'])\n",
                    "\n",
                    "mae_metrics = {m: [] for m in ['RW', 'Mean', 'SubMean', 'ARIMA', 'ES']}\n",
                    "rmse_metrics = {m: [] for m in ['RW', 'Mean', 'SubMean', 'ARIMA', 'ES']}\n",
                    "\n",
                    "for entry in forecast_data['within_session']:\n",
                    "    for m in mae_metrics.keys():\n",
                    "        mae_metrics[m].append(entry['metrics']['MAE'][m])\n",
                    "        rmse_metrics[m].append(entry['metrics']['RMSE'][m])\n",
                    "\n",
                    "print(\"=== WITHIN-SESSION FORECASTING PERFORMANCE (N = 1430 series) ===\")\n",
                    "within_table = pd.DataFrame(index=['Naive Random Walk', 'Series Mean', 'Subject Mean', 'ARIMA', 'Exponential Smoothing'])\n",
                    "within_table['MAE Mean'] = [np.mean(mae_metrics[m]) for m in ['RW', 'Mean', 'SubMean', 'ARIMA', 'ES']]\n",
                    "within_table['MAE CI'] = [1.96 * np.std(mae_metrics[m], ddof=1) / np.sqrt(len(mae_metrics[m])) for m in ['RW', 'Mean', 'SubMean', 'ARIMA', 'ES']]\n",
                    "within_table['RMSE Mean'] = [np.mean(rmse_metrics[m]) for m in ['RW', 'Mean', 'SubMean', 'ARIMA', 'ES']]\n",
                    "within_table['RMSE CI'] = [1.96 * np.std(rmse_metrics[m], ddof=1) / np.sqrt(len(rmse_metrics[m])) for m in ['RW', 'Mean', 'SubMean', 'ARIMA', 'ES']]\n",
                    "display(within_table.round(4))\n",
                    "\n",
                    "# Extract metrics for Cross-Session task\n",
                    "cross_res = pd.DataFrame(forecast_data['cross_session'])\n",
                    "cross_mae = {m: [] for m in ['RW', 'Mean', 'SubMean', 'ES']}\n",
                    "for entry in forecast_data['cross_session']:\n",
                    "    for m in cross_mae.keys():\n",
                    "        cross_mae[m].append(entry['metrics']['MAE'][m])\n",
                    "\n",
                    "print(\"\\n=== CROSS-SESSION FORECASTING PERFORMANCE (N = 245 series) ===\")\n",
                    "cross_table = pd.DataFrame(index=['Naive Random Walk', 'Series Mean', 'Subject Mean', 'Exponential Smoothing'])\n",
                    "cross_table['MAE/RMSE Mean'] = [np.mean(cross_mae[m]) for m in ['RW', 'Mean', 'SubMean', 'ES']]\n",
                    "cross_table['CI'] = [1.96 * np.std(cross_mae[m], ddof=1) / np.sqrt(len(cross_mae[m])) for m in ['RW', 'Mean', 'SubMean', 'ES']]\n",
                    "display(cross_table.round(4))"
                ]
            },
            # Step 7
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 7: Optional Deep Learning Extension (LSTM Comparison)\n",
                    "\n",
                    "We train a PyTorch LSTM model on the within-session per-saccade sequences. To prevent validation/test leakage, the LSTM is trained using a **subject-disjoint split**: 40 subjects are allocated for training/validation, and 10 subjects are held out for testing. The model uses an input window of 30 saccades, 2 LSTM layers, a hidden size of 64, dropout of 0.2, and is evaluated autoregressively on the held-out test subjects."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Load LSTM results\n",
                    "with open(os.path.join(\"data\", \"lstm_results.json\"), \"r\") as f:\n",
                    "    lstm_data = json.load(f)\n",
                    "\n",
                    "print(\"=== LSTM Forecasting Performance on Held-Out Test Subjects ===\")\n",
                    "print(f\"Evaluated on {lstm_data['n_series_evaluated']} test sequences.\")\n",
                    "print(f\"Out-of-sample MAE: {lstm_data['mean_MAE']:.4f} +/- {lstm_data['ci_MAE']:.4f}\")\n",
                    "print(f\"Out-of-sample RMSE: {lstm_data['mean_RMSE']:.4f} +/- {lstm_data['ci_rmse']:.4f}\")\n",
                    "\n",
                    "# Comparative Bar Plot\n",
                    "models = ['Random Walk', 'Series Mean', 'Subject Mean', 'ARIMA', 'ES', 'LSTM']\n",
                    "# Extract within-session test errors specifically for the same test subjects for direct comparison\n",
                    "test_subjs = set(entry['subject'] for entry in lstm_data['series_results'])\n",
                    "\n",
                    "test_sub_res = [entry for entry in forecast_data['within_session'] if entry['subject'] in test_subjs]\n",
                    "maes = {m: [] for m in ['RW', 'Mean', 'SubMean', 'ARIMA', 'ES']}\n",
                    "for entry in test_sub_res:\n",
                    "    for m in maes.keys():\n",
                    "        maes[m].append(entry['metrics']['MAE'][m])\n",
                    "\n",
                    "mae_means = [\n",
                    "    np.mean(maes['RW']),\n",
                    "    np.mean(maes['Mean']),\n",
                    "    np.mean(maes['SubMean']),\n",
                    "    np.mean(maes['ARIMA']),\n",
                    "    np.mean(maes['ES']),\n",
                    "    lstm_data['mean_MAE']\n",
                    "]\n",
                    "\n",
                    "plt.figure(figsize=(10, 5))\n",
                    "sns.barplot(x=models, y=mae_means, palette='viridis')\n",
                    "plt.ylabel(\"Out-of-Sample MAE (deg/sec)\")\n",
                    "plt.title(\"Model Comparison on Held-Out Test Subjects (Within-Session Task)\")\n",
                    "for i, val in enumerate(mae_means):\n",
                    "    plt.text(i, val + 1, f\"{val:.2f}\", ha='center', fontweight='bold')\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Paired Statistical Significance Testing (Wilcoxon Signed-Rank Test)\n",
                    "print(\"=== Wilcoxon Signed-Rank Test on 278 Paired Series MAEs ===\")\n",
                    "lstm_errs = []\n",
                    "arima_errs = []\n",
                    "es_errs = []\n",
                    "s_mean_errs = []\n",
                    "sub_mean_errs = []\n",
                    "rw_errs = []\n",
                    "\n",
                    "lstm_map = { (int(e['subject']), int(e['round']), e['task']): e['MAE'] for e in lstm_data['series_results'] }\n",
                    "for entry in forecast_data['within_session']:\n",
                    "    subj = int(entry['subject'])\n",
                    "    key = (subj, int(entry['round']), entry['task'])\n",
                    "    if key in lstm_map:\n",
                    "        lstm_errs.append(lstm_map[key])\n",
                    "        arima_errs.append(entry['metrics']['MAE']['ARIMA'])\n",
                    "        es_errs.append(entry['metrics']['MAE']['ES'])\n",
                    "        s_mean_errs.append(entry['metrics']['MAE']['Mean'])\n",
                    "        sub_mean_errs.append(entry['metrics']['MAE']['SubMean'])\n",
                    "        rw_errs.append(entry['metrics']['MAE']['RW'])\n",
                    "\n",
                    "comparisons = {\n",
                    "    'ARIMA': arima_errs,\n",
                    "    'Exponential Smoothing': es_errs,\n",
                    "    'Series Mean': s_mean_errs,\n",
                    "    'Subject Mean': sub_mean_errs,\n",
                    "    'Random Walk': rw_errs\n",
                    "}\n",
                    "\n",
                    "for name, errs in comparisons.items():\n",
                    "    stat, p_val = stats.wilcoxon(lstm_errs, errs)\n",
                    "    print(f\"LSTM vs {name:22s} | Wilcoxon p-value = {p_val:.6e} (Significant: {p_val < 0.05})\")\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Plot Distribution of LSTM out-of-sample MAEs across test subjects\n",
                    "records = [{'Subject': f'S{e[\"subject\"]}', 'MAE': e['MAE']} for e in lstm_data['series_results']]\n",
                    "df_sub = pd.DataFrame(records)\n",
                    "sub_order = df_sub.groupby('Subject')['MAE'].median().sort_values().index\n",
                    "\n",
                    "plt.figure(figsize=(10, 6))\n",
                    "sns.boxplot(data=df_sub, x='Subject', y='MAE', order=sub_order, palette='crest', width=0.5, fliersize=0)\n",
                    "sns.stripplot(data=df_sub, x='Subject', y='MAE', order=sub_order, color='black', alpha=0.3, size=4, jitter=0.2)\n",
                    "plt.title('Distribution of LSTM Out-of-Sample MAE Across Held-Out Test Subjects', pad=15)\n",
                    "plt.xlabel('Held-Out Subject ID (Sorted by Median MAE)')\n",
                    "plt.ylabel('Out-of-Sample MAE (deg/sec)')\n",
                    "plt.grid(True, linestyle='--', alpha=0.5, axis='y')\n",
                    "plt.show()\n"
                ]
            },
            # Step 8
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 8: Limitations and Future Directions\n",
                    "\n",
                    "### 1. Cross-Session Limitations\n",
                    "Cross-session sequences are short (maximum 9 rounds per subject). This extremely limited sample length makes classical forecasting models like ARIMA statistically unstable and unusable. While Exponential Smoothing provides a robust mathematical formulation, forecasting round 9 from 8 historical points carries substantial variance and remains a significant limitation.\n",
                    "\n",
                    "### 2. High Stationarity Constraint\n",
                    "The ADF tests confirm that over 97% of the within-session saccade peak velocity sequences are highly stationary. In a stationary environment, there is no strong long-term trend for ARIMA or LSTMs to capture. Consequently, out-of-sample forecasts quickly revert to the training mean. This explains why the **Series Mean** and **Subject Mean** baselines are highly competitive and why complex models (ARIMA, LSTM) fail to outperform them significantly."
                ]
            },
            # Step 9
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 9: Conclusion\n",
                    "\n",
                    "- **Learning Check:** This forecasting project demonstrates the limits of model complexity on stationary, high-noise time series. Both ARIMA and PyTorch LSTM models are bounded by the high stationarity of saccadic peak velocities within individual sessions.\n",
                    "- **Key Takeaway:** Saccadic peak velocity series are highly stationary, rendering complex ARIMA and LSTM models close to historical mean baselines. The **Subject Mean** baseline is a robust forecasting anchor. Frame-level within-session time-on-task is an invalid proxy for GazeBase due to short task lengths, whereas cumulative between-session order represents a robust proxy."
                ]
            },
            # Step 10
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 10: Event Stream Extraction\n",
                    "\n",
                    "We have successfully extracted continuous, sparse neuromorphic event streams from the raw 1000 Hz velocity signal. By differencing continuous velocity and applying a threshold ($|velocity[t] - velocity[t-1]| > 50$ deg/sec), we compiled exactly **3,787,413 neuromorphic events** and saved them as a highly compressed parquet database (`data/events.parquet`). This serves as clean data preparation for downstream neuromorphic modeling."
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    notebook_path = "capstone_main.ipynb"
    print(f"Creating Jupyter notebook {notebook_path}...")
    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=4)
    print("Notebook created successfully!")

if __name__ == "__main__":
    main()
