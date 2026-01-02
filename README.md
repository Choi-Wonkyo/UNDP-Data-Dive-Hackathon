Live Simulation Web Link: https://undp-odaproj.vercel.app/

# 🌪️ CYClone - Analyzing the Past, Designing the Future: A Multi-target ODA Forecasting System

-------------------------------
<br>

## 📋 Project Overview
This project develops a simulation-based forecasting tool to support UNDP's decision-making in Official Development Assistance (ODA). 
Using a Multi-Layer Perceptron (MLP) model, the system quantitatively estimates how sector-specific ODA investments lead to measurable development outcomes over time. 
It enables scenario-based comparisons, facilitates the assessment of sustainability and effectiveness, and provides evidence-based recommendations for optimal aid allocation.

<br>
<br>

## 📁 Repository Structure
```text
undp-odaproj/
├── data_collection/    # Scripts for collecting ODA flow data and WB indicators
├── data_analysis/      # Trend analysis, clustering, and lagged correlation
├── modeling/           # MLP (Final model) and benchmarking models
├── dashboard/          # Streamlit-based UI source code
├── data/               # Processed datasets and data-specific README
└── README.md           # Main documentation'''

<br>
<br>

## 🚀 Setup Instructions

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

## 🛠️ Code documentation
The project is organized into four main modules:

- **data_collection**: Scripts for collecting and preprocessing raw ODA flow data and development indicators. Includes both original and cleaned CSV files.
  
- **data_analysis**:
  - crs_data analysis: Includes trend analysis by purpose and country, concentration, clustering (country × purpose), and assessment of sectoral importance.
  - Integrated Data Analysis: Causal inference between ODA and outcome indicators, indicator growth rate calculation, and preprocessing (missing value handling and normalization).
  - Lagged Correlation Analysis: Identifies time-lagged relationships between ODA inputs and development outcomes, and determines the optimal lag for each target variable.
 
- **modeling**:
  - Final Model: Multi-target regression using a Multi-Layer Perceptron (MLP) to predict the impact of ODA on development indicators.
  - Candidate Models: XGBoost, LightGBM, and CatBoost for benchmarking and comparison.

- **dashboard**:
  - A Streamlit-based user interface for setting country- and sector-level ODA allocation values.
  - Enables real-time visualizaion of predicted indicator changes and simulation results.

<br>
<br>

## 🔌 API Documentation

This project does not use any external APIs.  
All processes run locally within Python scripts and the Streamlit dashboard.  
The analysis modules prepare processed datasets and a trained MLP model,  
and the dashboard loads these files to generate predictions and visualizations  
based on user-selected parameters.

<br>
<br>

## 🧪 Analysis Methodology
This project aims to quantitatively analyze the relationships between country- and sector-specific ODA flows and key development indicators, and to build foundational dataset for deep learning-based predictive modeling. The following analytical procedures were conducted:

1. **Time Series Flow Analysis**  
   - Annual ODA recipient data was organized by country, and time series trends were analyzed to understand cumulative beneficiary patterns.

2. **Sector-wise ODA Flow Analysis**
   - Linear regression was used to visualize annual trends across major ODA purpose groups. Changes were compared based on year-over-year growth rates.

3. **Clustering-based Country Grouping** 
   - KMeans clustering was applied to group countries with similar ODA purpose distributions. Beneficiary characteristics and development indicator patterns were compared across clusters.

4. **Inequality and Dispersion Analysis**
   - The degree of concentration in ODA allocation by purpose and country was assessed using variance and Gini coefficients.

5. **Lagged Correlation Analysis**
   - Time-lagged correlations between ODA inputs and changes in development indicators were analyzed to explore potential causal relationship.
   
<br>
<br>

## 📊 Datasets Choice Justification

- **CRS ODA Data (`crs_data.csv`)**
  - Source: UNDP - Seoul Policy Centre
  - Period: 2014–2023
  - Variables: Year, RecipientName, SectorName, PurposeName, USD_Disbursement, USD_Disbursement_Defl, RegionName, IncomegroupName
  - Purpose: To identify ODA disbursement amounts by country and sector, and to determine major beneficiary countries

- **Development Indicator Data**
  - Source: World Bank Open Data
  - Period: 2014–2023
  - Variables: 33 development indicators, including those related to education, health, environment, and social welfare
  - Purpose: To quantify development outcomes in recipient countries and enable lag-based causal analysis

- **Data Processing Methods**
  - ODA amounts are standardized in USD
  - Development indicators were interpolated along each country's time series (NaNs allowed at both ends)
  - Country names were matched using ISO3 codes

  **For detailed data structure and descriptions, refer to [`data/README.md`](data/README.md)**

<br>
<br>

## 📈 Key Findings
- **CRS Data Analysis**
  - Time Series Analysis of ODA by Country: Sudden increases in annual totals reflect exogenous shocks (e.g., disasters, conflicts) and serve as important event signals for policy and modeling.
  - Sector-wise ODA Flow Analysis: Changes in sectoral shares over time provide key context for understanding lagged relationships with development indicators.
  - Clustering by Country-sector ODA Patterns: Groups of countries with similar ODA support patterns are identified to facilitate tailored ODA strategy formulation. <br>
    <img src="https://github.com/user-attachments/assets/98c1c60c-abed-4020-8fbe-a4201c2780da" width="400"/>
  - Beneficiary Group Analysis: Distinct differences in ODA purposes exist based on region and income level.

- **Integrated Data Analysis**
  - Evidence of Lagged ODA Effects: High correlations observed within lag periods of 1 to 3 years suggest that ODA can have short-term impacts on development indicators, indicating the need for strategies aiming for effects within 5 years.
  - Predictive Performance by Sector: Using an MLP-based time series prediction model, the quantitative influence of sector-specific ODA on particular development indicators is demonstrated. <br>
    <img src="https://github.com/user-attachments/assets/9d5ce7eb-f9e6-412d-938b-e0b3a1b9e825" width="900"/>
  - Feature Importance Analysis: SHAP analysis using XGBoost reveals lag-based correlations between sectoral support and development indicators.
  - ODA Impact Simulator Development: A policy decision-support tool that simulates changes in development indicators based on user inputs for country and sector allocations.

- **Utilization of UNDP Outputs**
  - Serves as a policy simulation tool to evaluate the impact of proactive aid strategies.
  - Provides empirical evidence to guide country- and sector-specific ODA allocation planning.

<br>
<br>

## ⚙️ Technical Decisions
- **Model Selection**
  - Tree-based models (XGBoost, LightGBM) excel at interpreting feature importance.
  - However, to capture time-lagged causality and continuous prediction in time series, **MLP (Multi-Layer Perceptron)** was chosen.

- **Missing Data Handling**
  - Interpolation performed based on time series within the same country, allowing missing values at both ends.
  - Data entries where `RecipientName` could not be reliably identified as a country were removed.

- **Feature Importance Interpretation**
  - SHAP-based analysis was used to identify influential variables and their lagged effects.

- **Visualization and Interface**
  - **Why Streamlit?**
    - Enables real-time visualization of predictions by country and sector based on user inputs.
    - Intuitive dashboard UI facilitates demonstration and simulation.


<br>

## 🚀 Future Possibilities
**Limitations**:
- Significant missing data in country-specific features limits analysis and prediction accuracy.
- The model's performance is suboptimal, resulting in constraints on prediction accuracy.
- Since the model is trained on historical data, its generalization performance for unseen future points, such as beyond 2025, may be limited.

**Improvement Directions**:
- Improve data quality through re-collection and sample adjustment by country.
- Advance the MLP model with hyperparameter tuning, SHAP-based feature selection, and alternative models (tree-based, ensemble) for low-performing indicators
- Advanced uncertainty quantification: complete confidence interval estimation through MC Dropout to support risk-aware policy decision-making
- Expand into a future-oriented simulation pipeline:
  - Sequential multi-year performance forecasting
  - Optimization modules to maximize composite SDG outcomes

**Long-term Potential**:
- With continuous updates of ODA and development indicator data, the system can evolve into a highly reliable policy simulator.
- By integrating predictive modeling, uncertainty estimation, and optimization, it can be expanded into a core decision-support tool for UNDP and partner organizations.
