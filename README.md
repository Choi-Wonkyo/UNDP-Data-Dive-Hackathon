# 🌪️ CYClone - ODA Sustatinability Impact Prototype

An interactive data-driven prototype to evaluate the sustainability and policy impact of Korea's ODA (Official Development Assistance), developed for the UNDP ODA Data Hackthon 2025.

## 프로젝트 개요 (Project Overview)
UN에서 제공하는 ODA(공적개발원조)가 수혜국에 미친 지속가능한 개발 영향력을 분석하고, UNDP와 정책 입안자들이 보다 효과적인 ODA 전략을 수립할 수 있도록 지원하는 도구입니다.
수혜국의 사회·경제·환경 지표에 대한 시계열 분석 및 예측을 통해, 특정 목적의 ODA가 시간차를 두고 어떤 성과로 이어지는지를 가시적으로 제시합니다.

## 설치 및 실행 방법 (Setup Instructions)
```python
git clone https://github.com/your-id/cyclone-oda-impact.git
cd cyclone-oda-impact
pip install -r requirements.txt
streamlit run dashboard/main.py
```

## 코드 문서화 (Code documentation)
전체 프로젝트는 data collection, analysis, modeling, dashboard 네 부분으로 구성되어있습니다.
- data_collection: 주요 ODA 흐름과 성과지표를 포함한 데이터 수집 및 전처리 스크립트
- data_analysis: 목적별 ODA 추이 분석, 수혜국 분류, 성과지표 상관분석 등 분석 코드

- modeling: LSTM/GRU 기반 시계열 예측 모델 및 평가 지표 구현

- dashboard: Streamlit 기반 사용자 인터페이스
## API 문서 (API Documentation)

## 분석 방법론 (Analysis Methodology)
  - 사용한 분석 기법과 그 적합성 설명

## 데이터셋 선택 근거 (Datasets Choice Justification)
  - crs_data
  - world bank data
  - 처리 방법 설명

## 핵심 결과 (Key Findings)
  - 중요한 인사이트 요약 및 개발 협력에서의 의의
  - UNDP에서 이 도구를 어떻게 활용할 수 있는지 설명

## 기술적 결정 사항 (Technical Decisions)

## 미래 가능성 (Future Possibilities)
한계점:

개선 방향:
