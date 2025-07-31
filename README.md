# 🌪️ CYClone - ODA Sustatinability Impact Prototype

An interactive data-driven prototype to evaluate the sustainability and policy impact of Korea's ODA (Official Development Assistance), developed for the UNDP ODA Data Hackthon 2025.

-------------------------------
<br>

## 프로젝트 개요 (Project Overview)
UN에서 제공하는 ODA(공적개발원조)가 수혜국에 미친 지속가능한 개발 영향력을 분석하고, UNDP와 정책 입안자들이 보다 효과적인 ODA 전략을 수립할 수 있도록 지원하는 도구입니다.
수혜국의 사회·경제·환경 지표에 대한 시계열 분석 및 예측을 통해, 특정 목적의 ODA가 시간차를 두고 어떤 성과로 이어지는지를 가시적으로 제시합니다.

이거 넣어야되나?(원교언니 지피티) -> 본 프로젝트는 UNDP(유엔개발계획)의 목적에 부합하는 국제개발협력(ODA) 데이터 분석을 통해 수원국의 개발 성과(보건·교육·경제 등) 를 정량적으로 예측하고, ODA의 지속가능성과 효율성을 평가하기 위한 다중 회귀 기반 MLP 모델을 구축한 것입니다.
<br>

## 설치 및 실행 방법 (Setup Instructions)
```python
git clone https://github.com/your-id/cyclone-oda-impact.git
cd cyclone-oda-impact
pip install -r requirements.txt
streamlit run dashboard/main.py
```
<br>

## 코드 문서화 (Code documentation)
프로젝트는 다음의 네 모듈로 구성되어 있습니다:
- **data_collection** : 주요 ODA 흐름과 성과 지표 관련 데이터 수집 및 전처리 스크립트
- **data_analysis** : 
  - crs_data 분석 : 목적별 ODA 추이 분석, 수혜국 분류 등 분석 코드
  - world bank data 분석 : 성과지표 상관분석 등 분석 코드
- **modeling** : 시계열 예측 모델 구축 (MLT 기반, XGBoost, LightGBM), 평가 지표 및 시차 반영 로직 구현
- **dashboard** : Streamlit 기반 대시보드 UI 및 사용자 입력 → 예측 결과 반환 인터페이스
<br>

## API 문서 (API Documentation)    !!!! <span style="color:red"> Streamlit 완성되면 수정해야 됨 </span>
내부에서 API 서버는 사용하지 않지만, Streamlit 내에서 사용자 입력에 따른 모델 예측 결과를 반환하는 구조로 되어 있음.
주요 함수 및 상호작용은 dashboard/utils.py와 modeling/predict.py에 정리되어 있음.
<br>

## 분석 방법론 (Analysis Methodology)
본 프로젝트는 국가별·분야별 ODA 흐름과 주요 성과 지표 간의 관계를 정량적으로 분석하고, 딥러닝 기반 예측 모델링을 위한 기반 데이터를 구축하는 데 목적을 두었습니다. 이를 위해 다음과 같은 분석 절차를 수행하였습니다: <br>
1. 시계열 기반 흐름 분석
- 국가별 ODA 수혜 내역을 연도별로 정리하고 시계열 추세를 분석하여 국가별 누적 수혜 경향을 파악하였습니다.

2. 분야(Purpose)별 ODA 흐름 분석
- 주요 목적군에 대해 연도별 지원 추이를 선형 회귀 분석(Linear Regression)을 통해 시각화하였고, 변화율을 기반으로 증감 패턴을 비교하였습니다.

3. 클러스터링 기반 국가 그룹화
- KMeans를 활용하여 ODA 목적 구조가 유사한 국가를 군집화하고, 그룹별 수혜 특성과 성과 지표 패턴을 비교 분석하였습니다.

4. 불균형 및 분산도 분석
- 목적 및 국가 간 ODA 분배의 편중 정도를 분산(Variance)과 Gini 계수 등을 통해 분석하였습니다.

5. 시차 기반 상관관계 분석 (Lagged Correlation)
- ODA 투입과 성과 지표 변화 간의 시간 지연 효과를 고려한 시차 기반 상관 분석을 통해 인과 가능성 탐색
   
<br>

## 데이터셋 선택 근거 (Datasets Choice Justification)
- **CRS ODA 데이터 (`crs_data.csv`)**
  - 출처: UNDP - 서울정책센터
  - 기간: 2014–2023
  - 변수: Year, RecipientName, SectorName, PurposeName, USD_Disbursement, USD_Disbursement_Defl, RegionName, IncomegroupName
  - 목적: 국가별, 분야별 ODA 지원금액 파악 및 주요 수혜국 선정

- **World Bank 개발 지표 데이터**
  - 출처: World Bank Open Data
  - 기간: 2014–2023
  - 변수: 교육/보건/환경/사회복지 등 33개 개발 지표
  - 목적: 수혜국의 개발 성과를 계량적으로 측정하여 시차 기반 인과분석 수행

- **데이터 처리 방법**
  - ODA 금액은 USD 기준으로 정리
  - 개발 지표는 국가별 시계열 기준으로 보간 (양쪽 NaN 허용)
  - ISO3 코드 기준으로 국가 간 일치
<br>

## 핵심 결과 (Key Findings)
- 중요한 인사이트 요약 및 개발 협력에서의 의의
  - 국가 × 목적 클러스터링: 국가별 목적 패턴을 기반으로 유사 국가 그룹을 도출할 수 있으며, 이는 맞춤형 ODA 전략 수립에 활용 가능함.
  - ODA의 시차 효과 존재: Lag 1~3년 구간에서 상관계수가 높게 나타나, ODA가 단기적으로 개발 지표에 영향을 줄 수 있음을 시사함. 효과가 5년 이내에 나타나도록 전략 수립 필요.
  - 지원 분야별 예측 가능성: MLP 기반 시계열 예측 모델을 통해 분야별 ODA가 특정 개발 지표에 미치는 영향을 정량화할 수 있음.
  - 중요도 분석 결과: XGBoost 및 SHAP 분석을 통해, 분야별 지원 내역과 개발 지표 간 시차 기반 상관관계가 존재함을 확인함.
  - ODA Impact Simulator 개발: 사용자 입력(국가, 목적별 비중)에 따라 개발 지표 변화를 예측할 수 있는 정책 시뮬레이션 도구를 개발함.
- UNDP 활용 방안:
  - 사전 정책 효과를 검토할 수 있는 시뮬레이션 도구로 활용 가능
  - 특정 국가 및 분야에 대한 ODA 배분 전략 수립 시 참고 자료로 활용 가능
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

## 미래 가능성 (Future Possibilities)
한계점: 
- 국가별 특성 데이터의 결측치가 많아 분석 및 예측에 제한이 있음
- 모델 성능이 완벽하지 않아 예측 정확도에 한계가 존재함
- 현재 모델은 학습 데이터 기간 내에서만 신뢰할 수 있으며, 2025년 이후 등 범위를 벗어난 예측에서는 일반화 신뢰도가 낮음

개선 방향:
- 데이터 재수집 및 국가별 표본 조정으로 데이터 품질 개선
- 모델 성능 고도화를 위한 추가 연구 및 튜닝
- 미래 예측을 위한 모델 재설계 및 입력 데이터 포맷 개선
