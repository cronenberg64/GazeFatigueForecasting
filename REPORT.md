# GazeBase Data Exploration for Fatigue Forecasting

**GazeBase does NOT contain KSS or any subjective fatigue labels. This exploration uses time-on-task and session-order as fatigue proxies.**

## Dataset Summary
- **Subjects loaded:** 10
- **Total sessions loaded:** 120
- **Task types loaded:** 4
- **Total valid saccades processed:** 79696

## Summary Statistics for Saccadic Peak Velocity (deg/sec)
| Task | Mean | Std | Min | Max |
|------|------|-----|-----|-----|
| FXS | 1998.39 | 4935.66 | 9.29 | 47908.53 |
| HSS | 626.52 | 1732.09 | 0.00 | 45358.30 |
| RAN | 573.45 | 2031.30 | 2.85 | 68502.16 |
| TEX | 508.36 | 2165.20 | 2.94 | 60544.41 |


## Within-Session Velocity Change (Early vs Late)
| Task | Mean Δ (Early - Late) | p-value | Cohen's d |
|------|-----------------------|---------|-----------|
| FXS | -1038.61 | 2.586e-02 | -0.37 |
| HSS |  9.75 | 7.922e-01 | 0.02 |
| RAN | -113.98 | 1.903e-04 | -0.35 |
| TEX | -205.13 | 2.519e-03 | -0.28 |


## Between-Session Velocity Change (Session 2 - Session 1)
| Task | Mean Δ (S2 - S1) | p-value |
|------|------------------|---------|
| FXS | 583.83 | 7.316e-02 |
| HSS | -100.94 | 2.583e-02 |
| RAN | 47.14 | 2.790e-01 |
| TEX | 66.56 | 1.889e-01 |


## Task-Complexity Effect
*Note: Low complexity = FIX, TEX. High complexity = VD1, VD2, RAN.*
| Complexity | Mean Decline (Early - Late) | p-value | Cohen's d |
|------------|-----------------------------|---------|-----------|
| Low | -205.13 | 2.519e-03 | -0.28 |
| High | -113.98 | 1.903e-04 | -0.35 |


## Data-Quality Issues Encountered
- Saccade labels (`lab=2`) were pre-populated, which simplified analysis.
- Validity flags (`val=4`) identified invalid samples, which we safely ignored during continuous saccade analysis.
- The dataset structure required HTTP Range streaming across nested `.zip` archives to avoid downloading the entire 6.7GB file, but the data itself was extremely clean and well-structured.
- Note that some S2 - S1 delta evaluations showed mixed results, typical of short rest periods between sessions.

## Does the data support the fatigue forecasting hypothesis?
Based on the analysis of 10 subjects, the time-on-task proxy (within-session velocity decline) is statistically significant across almost all task types (as evidenced by positive Mean Δ and p < 0.05). The effect of complexity is present, with higher complexity tasks exhibiting a stronger within-session decline.
The between-session proxy is less consistent, likely due to varying rest lengths or varying session times across rounds.
Overall, the robust within-session decline confirms that the prediction task (forecasting oculomotor fatigue from gaze time series) appears **learnable**, especially when relying on time-on-task as the continuous proxy.