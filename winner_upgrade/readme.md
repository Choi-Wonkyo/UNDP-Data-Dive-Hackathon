# 🌪️ CYClone — Multi-Sectoral ODA Impacts on Life Expectancy in Ethiopia

<br>
<br>

## Project Overview
This project quantifies how Ethiopia’s **life expectancy (Life Expectancy, LE)** responds to different **ODA sectors** — Health, Education, Infrastructure, Governance, and Social/Environment — with distinct **lag structures**.  
Using a **Dynamic Linear Model (DLM)** with robust estimation techniques, we estimate **time-delayed effects** and build a **policy scenario-based prediction dashboard**.

<br>


<br>

## Setup Instructions

This project is designed to run in Google Colab—no local full setup is required.
### Quick start (Colab)
1. Open the relevant Colab notebook.
2. Download the main folders from the GitHub repository and upload them to your Colab working directory.
3. Mount your Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. Run the notebook cells in order to:
  - Load and preprocess CRS, World Bank, and other external datasets
  - Compute lagged correlations and growth rates
  - Train the prediction model (e.g., Multi-Layer Perceptron)
  - Execute scenario-based simulations


<br>
<br>

## Project Goal

- Quantify when (lag) each ODA sector impacts life expectancy
- Build an interpretable regression model with multicollinearity removed (VIF, PCA)
- Provide a policy scenario system to compute life expectancy changes:
    - Immediate (Horizon 0)
    - 1-year lag (Horizon 1)
    - 2-year lag (Horizon 2)


<br>

## Methodology

- Dynamic Linear Model (DLM, K=2)
- Variable selection: LASSO → OLS-HAC for robust estimation
- Multicollinearity check: VIF + PCA
- Residual diagnostics: Durbin–Watson, ACF/Ljung-Box, Normality tests
- Cumulative effects: IRF-based (Impulse Response Function)

<br>

## Key Findings
| ODA Sector              | Lag | β     | p-value | Interpretation                                                           |
| ----------------------- | --- | ----- | ------- | ------------------------------------------------------------------------ |
| Health                  | 2   | +1.08 | <0.001  | Main positive effect, strongest driver of life expectancy improvement    |
| Social/Environment      | 0   | +0.52 | <0.001  | Immediate positive effect on life expectancy                             |
| Governance              | 1   | −0.41 | <0.001  | Short-term negative effect, possibly due to administrative restructuring |
| Regulatory Quality (RQ) | -   | +0.99 | ~0.09   | Amplifies Health & Social/Env effects                                    |

Insights:
- Health ODA: Main driver, effect peaks after 2 years
- Social/Environment ODA: Immediate positive impact
- Governance ODA: Short-term negative effect
- Regulatory quality (RQ): Supports sustainability of ODA effects

<br>

## Model Performance
- R² = 0.85, Adj. R² = 0.76
- Residuals pass tests for normality, independence, heteroscedasticity (HAC-based)
- Condition number < 30 → multicollinearity controlled

<br>

## Summary
- R² = 0.85, Adj. R² = 0.76
- Residuals pass tests for normality, independence, heteroscedasticity (HAC-based)
- Condition number < 30 → multicollinearity controlled

<br>

## Scenario-Based Simulation
