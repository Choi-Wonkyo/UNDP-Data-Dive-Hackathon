# 🌪️ CYClone - 제목 미정


-------------------------------
<br>

## Project Overview(프로젝트 개요)
This project develops a simulation tool to support UNDP’s decision-making in Official Development Assistance (ODA).
Using a Multi-Layer Perceptron (MLP) model, we quantitatively predict how sectoral ODA investments translate into outcomes with time lags and compare results across different scenarios.
This enables the assessment of ODA’s sustainability and efficiency, while providing evidence-based recommendations for optimal allocation strategies.

본 프로젝트는 UNDP의 국제개발협력(ODA)의 의사결정을 지원하기 위해, 수혜국의 보건·교육·경제·환경 등 핵심 지표를 시계열 분석·예측하는 시뮬레이션 도구입니다.
다층 퍼셉트론(MLP) 기반 모델을 활용해, 분야별 ODA 투입이 시차를 두고 성과로 이어지는 과정을 정량적으로 예측하고 시나리오별 성과를 비교할 수 있습니다.
이를 통해 ODA의 지속가능성과 효율성을 평가하고, 근거 기반의 최적 배분 전략을 제안할 수 있습니다.
<br>
<br>

## Setup Instructions(설치 및 실행 방법)    !!!! <span style="color:red"> Streamlit 완성되면 수정해야 됨 </span>
본 프로젝트는 Google Colab 기반 분석 환경에서 수행되었습니다.
모델 학습 및 시뮬레이션 실행을 재현하려면 아래 과정을 따르세요.
- Google Drive 연동  <- 지워도 될 듯;;
```python
drive.mount('/content/drive')
```
- 분석 및 시뮬레이션 실행
```python
git clone https://github.com/your-id/cyclone-oda-impact.git
cd cyclone-oda-impact
pip install -r requirements.txt
streamlit run dashboard/main.py
```
<br>
<br>

## Code documentation(코드 문서화)
프로젝트는 다음의 네 모듈로 구성되어 있습니다:
- **data_collection** :  ODA 흐름 및 성과 지표 관련 원천 데이터를 수집하고 정제하는 스크립트와 원본/전처리된 CSV 파일을 포함
- **data_analysis** : 
  - crs_data 분석 : 목적별·국가별 ODA 추이, 집중도, 클러스터링(국가×목적), 분야별 중요도 등
  - Integrated Data Analysis : ODA와의 인과 관계 분석, 성과 지표의 변화율 계산, 결측/정규화 처리
  - ODA와 성과 간 시차 분석 : ODA 투입과 성과(지표 변화율) 간의 lagged correlation 분석 및 각 타겟별 최적 lag 도출.
- **modeling** : 
  - 최종 선택 모델 : MLP 기반 다중 타겟 회귀 모델로 ODA가 개발 성과에 미치는 영향을 예측.
  - 후보/보조 모델 : XGBoost, LightGBM, CatBoost 등 비교용 모델들
- **dashboard** : Streamlit 기반 인터페이스로, 사용자가 국가 및 목적별 입력값을 설정하면 예측 결과(개발 지표 변화, 시뮬레이션)를 실시간으로 확인할 수 있는 대시보드 구성 요소
<br>
<br>

## API 문서 (API Documentation)    !!!! <span style="color:red"> Streamlit 완성되면 수정해야 됨 </span>
내부에서 API 서버는 사용하지 않지만, Streamlit 내에서 사용자 입력에 따른 모델 예측 결과를 반환하는 구조로 되어 있음.
주요 함수 및 상호작용은 dashboard/utils.py와 modeling/predict.py에 정리되어 있음.

<br>
<br>

## 분석 방법론 (Analysis Methodology)
본 프로젝트는 국가별·분야별 ODA 흐름과 주요 성과 지표 간의 관계를 정량적으로 분석하고, 딥러닝 기반 예측 모델링을 위한 기반 데이터를 구축하는 데 목적을 두었습니다. 이를 위해 다음과 같은 분석 절차를 수행하였습니다: <br>
1. 시계열 기반 흐름 분석
- 국가별 ODA 수혜 내역을 연도별로 정리하고 시계열 추세를 분석하여 국가별 누적 수혜 경향을 파악

2. 분야(Purpose)별 ODA 흐름 분석
- 주요 목적군에 대해 연도별 지원 추이를 선형 회귀 분석(Linear Regression)을 통해 시각화하였고, 변화율을 기반으로 증감 패턴을 비교

3. 클러스터링 기반 국가 그룹화
- KMeans를 활용하여 ODA 목적 구조가 유사한 국가를 군집화하고, 그룹별 수혜 특성과 성과 지표 패턴을 비교 분석

4. 불균형 및 분산도 분석
- 목적 및 국가 간 ODA 분배의 편중 정도를 분산(Variance)과 Gini 계수 등을 통해 분석

5. 시차 기반 상관관계 분석 (Lagged Correlation)
- ODA 투입과 성과 지표 변화 간의 시간 지연 효과를 고려한 시차 기반 상관 분석을 통해 인과 가능성 탐색
   
<br>
<br>

## 데이터셋 선택 근거 (Datasets Choice Justification)
- **CRS ODA 데이터 (`crs_data.csv`)**
  - 출처: UNDP - 서울정책센터
  - 기간: 2014–2023
  - 변수: Year, RecipientName, SectorName, PurposeName, USD_Disbursement, USD_Disbursement_Defl, RegionName, IncomegroupName
  - 목적: 국가별, 분야별 ODA 지원금액 파악 및 주요 수혜국 선정

- **개발 지표 데이터**
  - 출처: World Bank Open Data, 
  - 기간: 2014–2023
  - 변수: 교육/보건/환경/사회복지 등 33개 개발 지표
  - 목적: 수혜국의 개발 성과를 계량적으로 측정하여 시차 기반 인과분석 수행

- **데이터 처리 방법**
  - ODA 금액은 USD 기준으로 정리
  - 개발 지표는 국가별 시계열 기준으로 보간 (양쪽 NaN 허용)
  - ISO3 코드 기준으로 국가 간 일치


  ** **자세한 데이터 구성 설명은 [`data/README.md`](data/README.md) 참고**

<br>

## 핵심 결과 (Key Findings)  -> 분석 과정이나 인사이트 많이 적고 싶은데 중요한 것만 쓰랬음
- **crs_data 분석**
  - 국가별 ODA 시계열 분석: 연도별 총액의 급증 포인트는 외생 충격(재난·분쟁 등)을 반영하며, 정책·모델링 시 중요한 이벤트 신호가 됨
  - 분야별 ODA 흐름 분석: 연도별 분야 비중 변화는 성과 지표와의 시차(Lag) 관계를 이해하는 핵심 컨텍스트
  - 국가 × 목적 클러스터링: 국유사한 지원 패턴을 가진 국가 그룹을 도출해 맞춤형 ODA 전략 수립에 활용 가능 <br>
    <img src="https://github.com/user-attachments/assets/98c1c60c-abed-4020-8fbe-a4201c2780da" width="400"/>
  - 수혜국 그룹 분석: 지역별·소득 수준별로 ODA 목적의 뚜렷한 차이가 존재
 
- **Integrated Data 분석**
  - ODA의 시차 효과 존재: Lag 1~3년 구간에서 상관계수가 높게 나타나, ODA가 단기적으로 개발 지표에 영향을 줄 수 있음을 시사함. 효과가 5년 이내에 나타나도록 전략 수립 필요.
  - 분야별 예측 가능성 검증: MLP 기반 시계열 예측 모델을 통해 분야별 ODA가 특정 개발 지표에 미치는 영향을 정량화할 수 있음.
    <img src="https://github.com/user-attachments/assets/9d5ce7eb-f9e6-412d-938b-e0b3a1b9e825" width="900"/>
  - 중요도 분석 결과: XGBoost + SHAP 분석을 통해, 분야별 지원 내역과 개발 지표 간 시차 기반 상관관계 확인
  - ODA Impact Simulator 개발: 사용자 입력(국가·분야 비중)에 따라 개발 지표 변화를 시뮬레이션할 수 있는 정책 의사결정 지원 도구
    
- UNDP 활용 방안:
  - 사전 정책 효과를 검토할 수 있는 시뮬레이션 도구로 활용 가능
  - 특정 국가 및 분야에 대한 ODA 배분 전략 수립 시 참고 자료로 활용 가능
<br>
<br>

## 기술적 결정 사항 (Technical Decisions)
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
<br>

## 미래 가능성 (Future Possibilities)
한계점: 
- 국가별 특성 데이터의 결측치가 많아 분석 및 예측에 제한이 있음
- 모델 성능이 완벽하지 않아 예측 정확도에 한계가 존재함
- 현재 모델은 학습 데이터 기간 내에서만 신뢰할 수 있으며, 2025년 이후 등 범위를 벗어난 예측에서는 일반화 신뢰도가 낮음

개선 방향:
- 데이터 재수집 및 국가별 표본 조정으로 데이터 품질 개선
- 모델 성능 고도화를 위한 추가 연구 및 튜닝
- 미래 예측을 위한 모델 재설계 및 입력 데이터 포맷 개선
