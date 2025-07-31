# 🌪️ CYClone - ODA Sustatinability Impact Prototype

An interactive data-driven prototype to evaluate the sustainability and policy impact of Korea's ODA (Official Development Assistance), developed for the UNDP ODA Data Hackthon 2025.

## 프로젝트 개요 (Project Overview)
UN에서 제공하는 ODA(공적개발원조)가 수혜국에 미친 지속가능한 개발 영향력을 분석하고, UNDP와 정책 입안자들이 보다 효과적인 ODA 전략을 수립할 수 있도록 지원하는 도구입니다.
수혜국의 사회·경제·환경 지표에 대한 시계열 분석 및 예측을 통해, 특정 목적의 ODA가 시간차를 두고 어떤 성과로 이어지는지를 가시적으로 제시합니다.
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
전체 프로젝트는 data collection, analysis, modeling, dashboard 네 부분으로 구성되어있습니다.
- data_collection: 주요 ODA 흐름과 성과지표를 포함한 데이터 수집 및 전처리 스크립트
- data_analysis: 
  - crs_data 분석 -> 목적별 ODA 추이 분석, 수혜국 분류 등 분석 코드
  - world bank data 분석 -> 성과지표 상관분석 등 분석 코드
- modeling: MLT 등등등 기반 시계열 예측 모델 및 평가 지표 구현
- dashboard: Streamlit 기반 인터페이스
<br>

## API 문서 (API Documentation)
내부에서 API 서버는 사용하지 않지만, Streamlit 내에서 사용자 입력에 따른 모델 예측 결과를 반환하는 구조로 되어 있음.
주요 함수 및 상호작용은 dashboard/utils.py와 modeling/predict.py에 정리되어 있음.
<br>

## 분석 방법론 (Analysis Methodology)
본 프로젝트는 ODA(공적개발원조)의 국가별·분야별 흐름과 그에 따른 성과 지표 변화를 정량적으로 분석하고, 이를 기반으로 딥러닝 기반 성과 예측 모델링이 가능한 구조를 설계하는 데 초점을 두었습니다. <br>
1. 국가별 ODA 시계열 분석
2. 국가별 누적 수혜 및 군집 분석
3. 목적(Purpose)별 연도 추세 분석: 선형 회귀 (Linear Regression)
4. ODA 목적의 특성에 따른 클러스터링 분석: KMeans
5. 분산도 및 불균형 분석
6. 시차 기반 상관관계 분석 (Lagged Correlation Analysis)
7. 변화율 기반 영향 비교 (Rate-of-Change Comparison)
   
<br>

## 데이터셋 선택 근거 (Datasets Choice Justification)
  - crs_data
  - world bank data
  - 처리 방법 설명
<br>

## 핵심 결과 (Key Findings)
  - 중요한 인사이트 요약 및 개발 협력에서의 의의
  - UNDP에서 이 도구를 어떻게 활용할 수 있는지 설명
<br>

## 기술적 결정 사항 (Technical Decisions)
- 트리 기반 부스팅 모델 : XGBoost, LightGBM / 신경망 모델 : MLP(다층 퍼셉트론) 중 (이유 어쩌고)를 고려하여  MLP 선택

- 결측치는 국가별 시계열 기준 보간 처리

- SHAP 기반 중요도 분석으로 변수 영향 해석

- Streamlit 사용 이유: 시연 시 직관적 시각화 및 사용자 입력 기반 예측 기능


<br>

## 미래 가능성 (Future Possibilities)
한계점:

개선 방향:
