# 🌪️ CYClone - Analyzing the Past, Designing the Future: A Multi-target ODA Forecasting System

-------------------------------
<br>

## Project Overview
This project develops a simulation-based forecasting tool to support UNDP's decision-making in Official Development Assistance (ODA). 
Using a Multi-Layer Perceptron (MLP) model, the system quantitatively estimates how sector-specific ODA investments lead to measurable development outcomes over time. 
It enables scenario-based comparisons, facilitates the assessment of sustainability and effectiveness, and provides evidence-based recommendations for optimal aid allocation.

<br>
<br>

## Setup Instructions(설치 및 실행 방법)    !!!! <span style="color:red"> Streamlit 완성되면 수정해야 됨 </span>

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

### 시뮬레이션
(채워질 것임)



본 프로젝트는 Google Colab 기반 분석 환경에서 수행되었습니다.
### Quick start (Colab)
1. 해당 Colab 노트북을 연다.
2. GitHub 저장소의 메인 폴더들을 다운로드하여 Colab 작업 폴더에 업로드한다.
3. Google Drive를 마운트한다.
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. 노트북 셀을 순서대로 실행하여 다음 작업을 수행한다:
  - CRS 데이터, World Bank 데이터, 기타 외부 데이터 불러오기 및 전처리
  - 시차 상관관계 및 변화율 계산
  - 예측 모델(Multi-Layer Perceptron 등) 학습
  - 시나리오 기반 시뮬레이션 실행
<br>
<br>

## Code documentation
The project is organized into four main modules:

- **data_collection**: Scripts for collecting and preprocessing raw ODA flow data and development indicators. Includes both original and cleaned CSV files.
- 
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

## API 문서 (API Documentation)    !!!! Streamlit 내부 함수와 입출력 구조 정도만 간단히 설명

**streamlit
내부에서 API 서버는 사용하지 않지만, Streamlit 내에서 사용자 입력에 따른 모델 예측 결과를 반환하는 구조로 되어 있음.
주요 함수 및 상호작용은 dashboard/utils.py와 modeling/predict.py에 정리되어 있음.

<br>
<br>

## Analysis Methodology
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

## Datasets Choice Justification

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

## Key Findings
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

## Technical Decisions(기술적 결정 사항)
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
- **모델 선택**
  - 트리 기반 모델(XGBoost, LightGBM)은 변수 중요도 해석에 강점
  - 하지만 시계열적 시차 인과성 및 연속 예측을 반영하기 위해 **MLP(다층 퍼셉트론)** 선택

- **결측치 처리**
  - 동일 국가 내 시계열 기준으로 보간 (양쪽 결측 허용)
  - RecipientName에서 국가명이라 보기 어려운 데이터는 제거 처리

- **변수 중요도 해석**
  - SHAP 기반 변수 중요도 분석으로 영향력 있는 변수 및 시차 확인

- **시각화 및 인터페이스**
  - **Streamlit 사용 이유**:
    - 사용자의 입력에 따라 국가·분야별 예측값 실시간 확인 가능
    - 대시보드 형태의 직관적 UI로 시연 및 시뮬레이션이 용이

<br>

## Future Possibilities(미래 가능성)
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



한계점: 
- 국가별 특성 데이터의 결측치가 많아 분석 및 예측에 제한이 있음
- 모델 성능이 완벽하지 않아 예측 정확도에 한계가 존재함
- 본 모델은 과거 데이터에 의해 학습되었기 때문에, 2025년 이후와 같은 관측되지 않은 시점에 대해서는 예측의 일반화 성능이 낮을 수 있음


개선 방향: 
- 데이터 재수집 및 국가별 표본 조정으로 데이터 품질 개선
- MLP 모델의 고도화: 하이퍼파라미터 튜닝, SHAP 기반 변수 선택, 저성과 지표에 대한 대체 모델(트리 기반, 앙상블) 통합 검토
- 불확실성 정량화 고도화: MC Dropout을 통한 신뢰 구간 산출을 완성하여, 리스크를 고려한 정책 의사결정 지원
- 미래 지향적 시뮬레이션 파이프라인 확장
  - 연속적 연도별 성과 예측
  - 복합 SDG 성과 극대화를 위한 최적화 모듈 개발

장기적 잠재력:
- ODA 및 개발 지표 데이터의 지속적 업데이트를 통해, 신뢰성 높은 정책 시뮬레이터로 발전 가능  
- 예측 모델링, 불확실성 추정, 최적화 기능을 결합함으로써 **UNDP 및 협력 기관의 의사결정을 지원하는 핵심 도구**로 확장될 수 있음
