** 코드 한 파일로 합치기엔 너무 길어짐 -> 합치는 게 나을거 같으면 말 ㄱㄱ
** 삭제 예정

## crs_data 분석




## world bank 데이터 분석


[분석 3] 목적별 연도별 ODA 흐름 분석
1. 전체 ODA 규모 (목적별 누적합)
2. 목적별 연도별 증감 추세 분석 (증가/감소 추세 판단) → LinearRegression
<img src="https://github.com/user-attachments/assets/f200ec06-37f9-4b91-ab83-de2b630584e5" width="400"/>
<img src="https://github.com/user-attachments/assets/dc1a4bd8-70e4-430d-92c3-b476b08bfb77" width="400" /><br>
4. 특정 국가들은 어떤 목적에 집중적으로 지원 받는지?



[분석 4] 수혜국 그룹 분석
1. 지역별 수혜국 수
2. 소득 수준별 수혜국 수
3. 수혜국 그룹별 주요 목적 분포
  <img src="https://github.com/user-attachments/assets/24d7577d-7ea5-481f-baa3-34e2ba047c1a" width="500"/>
    

## 성과 지표 분석
<1차 분석>
- 1단계: 성과지표 후보 수집 → 총 16개
- 2단계: 각 국가별로 연도별 변화율 계산 → 성과의 변화
  - 형태: 국가(Country) × 연도(Year) 단위의 행 + 각 성과지표별 변화율(%) 열
  - (지표_t+2 - 지표_t) / 지표_t × 100   → t에 저장(주의: t+2에 저장 x)
- 3단계: 시차(Lag) 기반 상관분석 → ODA 투입량 vs 성과 변화율
  - 각 지표에 대해 상관계수 / p-value 계산
- 4단계: 타겟 후보 확정
  - 지표별로 -> 의미 있는 상관 있음, 결측률 낮음, 충분한 국가 수 존재
  - 분야별로 그룹핑 (보건/교육/경제)
 
<2차 분석>
