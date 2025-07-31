## **< crs_data 데이터>**
| csv 파일명 | 추출한 열 | 설명 |
| --- | --- | --- |
| **1. 국가별 ODA 규모 시계열 분석용** | `Year`, `RecipientName`, `USD_Disbursement` | 연도별 국가별 전체 ODA 흐름 확인 |
| **2. 분야별 ODA 흐름 분석용** | `Year`, `RecipientName`, `SectorName`, `USD_Disbursement` | 교육, 보건, 환경 등 분야별 흐름 |
| **3. 목적별 세부 ODA 분석용** | `Year`, `RecipientName`, `PurposeName`, `USD_Disbursement` | "초등교육", "기초보건" 등 세부항목 분석 |
| **4. 수혜국 그룹 분석용** | `RecipientName`, `RegionName`, `IncomegroupName` | 지역, 소득 수준별 분포 파악 |
| **5. 시계열 예측용 Input 구성용** | `Year`, `RecipientName`, `SectorName`, `USD_Disbursement`, `USD_Disbursement_Defl` | 모델 Input용 |
| **6. 외부 성과지표 병합용 기준표** | `Year`, `RecipientName` | World Bank 지표와 병합 키 |

## **< world bank 데이터 >** <br>

output 해당 데이터  <br>
| 분야             | 지표 데이터                                                                |
|------------------|--------------------------------------------------------------------------|
| **경제** | - 1인당 GDP (GDP per capita) |
| **보건** | - 영아 사망률 (Under-1 Mortality Rate)<br>- 기대 수명 (Life Expectancy)<br>- 신생아 사망률 (Neonatal Mortality Rate) |
| **교육** | - 초등학교 순취학률 (Net Primary Enrollment Rate)<br>- 초등학교 이수율 (Primary Completion Rate) |
| **빈곤 및 사회복지** | - 빈곤율 (Poverty Rate)<br>- 1인당 GNI<br>- 의료 접근성 (Access to Healthcare) |
| **환경** | - 1인당 CO2 배출량 (CO2 Emissions)<br>- 대기오염 지수 (Air Pollution Index)<br>- 재생 에너지 사용률 (Renewable Energy Usage Rate)  |
| **생산** | - 서비스업 부가가치 (% of GDP)<br>- 제조업 부가가치 (% of GDP)<br>- 농작물 생산지수 <br>- 가축 생산지수  |

input 해당 데이터 <br>
외생 충격 데이터
|데이터          | 설명                      |
|------------------|-------------------------------|
| new_cases_per_million | 백만 명당 신규 확진자 수 |
| new_deaths_per_million | 백만 명당 코로나 신규 사망자  |
| stringency_index | 코로나 대응 강도 지수 |
| natural_disaster_count | 해당 연도에 발생한 자연재해 횟수 |
| log_battle_deaths | 전쟁 및 분쟁으로 인한 사망자 수 |

일반 요인 변수

| 분야 | 데이터 |
| --- | --- |
| 경제 | - 연간 물가상승률 |
| 보건 | - 인구 1,000명당 간호사 및 조산사 수 |
|  | - GDP 대비 보건 지출 비율 |
| 교육 | - GDP 대비 공교육 지출 비율 |
|  | - 정부예산 중 교육비 비중 |
|  | - 중등학교 총등록률 |
| 빈곤 및 사회복지 | - 정부효율성 (WGI)  |
|  | - 실업률 |
|  | - 정치 안정성 |
|  | - 소득 불평등 |
| 환경/기후 | - 도시화율   |
|  | - 인구밀도 |
|  | - 환경적 지속가능성 |
|  | - 산림면적 비율 |
| 생산 | - 농업 부가가치 비중 |
|  | - 산업 부가가치 비중 |
|  | - 식량 생산 지수 |
