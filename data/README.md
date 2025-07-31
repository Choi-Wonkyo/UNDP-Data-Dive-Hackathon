< undp 제공 데이터>
| 분석 목적 | 필요한 열 | 설명 |
| --- | --- | --- |
| **1. 국가별 ODA 규모 시계열 분석** | `Year`, `RecipientName`, `USD_Disbursement` | 연도별 국가별 전체 ODA 흐름 확인 |
| **2. 분야별 ODA 흐름 분석** | `Year`, `RecipientName`, `SectorName`, `USD_Disbursement` | 교육, 보건, 환경 등 분야별 흐름 |
| **3. 목적별 세부 ODA 분석** | `Year`, `RecipientName`, `PurposeName`, `USD_Disbursement` | "초등교육", "기초보건" 등 세부항목 분석 |
| **4. 수혜국 그룹 분석** | `RecipientName`, `RegionName`, `IncomegroupName` | 지역, 소득 수준별 분포 파악 |
| **5. 시계열 예측용 Input 구성** | `Year`, `RecipientName`, `SectorName`, `USD_Disbursement`, `USD_Disbursement_Defl` | 모델 Input용 |
| **6. 외부 성과지표 병합용 기준** | `Year`, `RecipientName` | World Bank 지표와 병합 키 |

< world bank data > <br>
| 분야             | 지표 데이터                                                                |
|------------------|--------------------------------------------------------------------------|
| **경제** | - 1인당 GDP (GDP per capita) |
| **보건** | - 영아 사망률 (Under-1 Mortality Rate)<br>- 기대 수명 (Life Expectancy)<br>- 신생아 사망률 (Neonatal Mortality Rate) |
| **교육** | - 초등학교 순취학률 (Net Primary Enrollment Rate)<br>- 초등학교 이수율 (Primary Completion Rate) |
| **빈곤 및 사회복지** | - 빈곤율 (Poverty Rate)<br>- 1인당 GNI<br>- 의료 접근성 (Access to Healthcare) |
| **환경** | - 1인당 CO2 배출량 (CO2 Emissions)<br>- 대기오염 지수 (Air Pollution Index)<br>- 재생 에너지 사용률 (Renewable Energy Usage Rate) (2014~2022) |
| **생산** | - 서비스업 부가가치 (% of GDP)<br>- 제조업 부가가치 (% of GDP)<br>- 농작물 생산지수 (2014~2022)<br>- 가축 생산지수 (2014~2022) |

