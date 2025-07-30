!pip install pandas matplotlib seaborn requests wbdata statsmodels

!pip install pycountry




import wbdata
import pandas as pd
import datetime
from pandas_datareader import wb
import pycountry

# 국가 리스트
country_names = [
    'Tunisia', 'Afghanistan', 'Mozambique', 'Pakistan', 'Peru', 'Philippines',
    'Senegal', 'South Africa', 'Zimbabwe', 'Sudan', 'Tajikistan', 'Cameroon',
    'Georgia', 'Viet Nam', "China (People's Republic of)", 'India', 'Kenya',
    'Ethiopia', 'Niger', 'Mali', 'Ghana', 'Democratic Republic of the Congo',
    'Chile', 'Colombia', 'Costa Rica', 'Ecuador', 'West Bank and Gaza Strip',
    'Guatemala', 'Morocco', 'Algeria', 'Libya', 'Indonesia', 'Malaysia',
    'Mexico', 'Namibia', 'Nigeria', 'Rwanda', 'Türkiye', 'Uruguay',
    'Venezuela', 'El Salvador', 'Tanzania', 'Iran', 'Guinea-Bissau',
    'Timor-Leste', 'Bolivia', 'Egypt', 'Serbia', 'Thailand', 'Ukraine',
    'Uzbekistan', 'Iraq', 'Kosovo', 'Mongolia', 'Moldova', 'Montenegro',
    "Côte d'Ivoire", 'Belarus', 'Haiti', 'Dominican Republic', 'Argentina',
    'Brazil', 'Cabo Verde', 'Kazakhstan', 'Albania', 'North Macedonia',
    'Mauritius', 'Uganda', 'Syrian Arab Republic', 'Kyrgyzstan', 'Lebanon',
    'Paraguay', 'Panama', 'Bangladesh', 'Burkina Faso', 'Jordan', 'Cambodia',
    'Cuba', 'Liberia', 'Nicaragua', 'Sri Lanka', 'Equatorial Guinea', 'Angola',
    'South Sudan', 'Armenia', 'Bosnia and Herzegovina', 'Sierra Leone', 'Togo',
    'Turkmenistan', 'Yemen', 'Malawi', 'Gambia', 'Burundi', 'Bhutan',
    'Somalia', 'Central African Republic', 'Mauritania', 'Botswana', 'Myanmar',
    'Congo', 'Madagascar', 'Guinea', 'Djibouti', 'Eswatini', 'Jamaica',
    'Azerbaijan', 'Nepal', 'Benin', 'Comoros', 'Chad', 'Eritrea', 'Gabon',
    'Honduras', "Democratic People's Republic of Korea",
    "Lao People's Democratic Republic", 'Seychelles', 'Belize', 'Lesotho',
    'Fiji', 'Vanuatu', 'Saint Lucia', 'Suriname', 'Tonga', 'Marshall Islands',
    'Papua New Guinea', 'Guyana', 'Saint Vincent and the Grenadines',
    'Dominica', 'Solomon Islands', 'Grenada', 'Maldives', 'Palau', 'Tuvalu',
    'Antigua and Barbuda', 'Samoa', 'Kiribati', 'Nauru'
]

# 날짜 설정
data_date = (datetime.datetime(2014, 1, 1), datetime.datetime(2023, 12, 31))

# 1. 나라 이름 → ISO3 코드 매핑
def get_country_code(name):
    try:
        return pycountry.countries.lookup(name).alpha_3
    except:
        return None

# 2. 국가 이름 → ISO3 코드 변환
mapped = {name: get_country_code(name) for name in country_names}
iso3_list = list(filter(None, mapped.values()))  # None 제외

# 3. 지표 목록
indicators = {
    'poverty_percent': 'SI.POV.NAHC',       # 빈곤율
    'gni_per_capita': 'NY.GNP.PCAP.CD',     # 중위소득 proxy
    'Hospital beds': 'SH.MED.BEDS.ZS',          # 인구 1,000명당 병상 수
    'Physicians': 'SH.MED.PHYS.ZS'    # 인구 1,000명당 의사 수
}

# 4. World Bank API 호출 (2010~2022)
data1 = wb.download(
    indicator=list(indicators.values()),
    country=iso3_list,
    start=2014,
    end=2023
)

# 5. 정리
data1 = data1.reset_index()
data1 = data1.rename(columns={v: k for k, v in indicators.items()})
data1 = data1[['country', 'year'] + list(indicators.keys())]
data1 = data1.sort_values(['country', 'year']).reset_index(drop=True)

# 결과 확인
print(data1.head())

# CSV 저장
data1.to_csv('/content/drive/MyDrive/딥러닝프로젝트/UNDP Data Dive 해커톤/단위별 데이터/외부데이터_빈곤 및 사회복지 지표1.csv', index=False)


#  환경/기후 데이터

# World Bank 지표 전체 불러오기
all_indicators = wb.get_indicators()

# 'co2' 포함하는 지표만 필터링 (소문자 포함)
co2_indicators = all_indicators[all_indicators['name'].str.lower().str.contains('co2')]

# 컬럼 몇 개만 보여주기
print(co2_indicators[['id', 'name', 'sourceOrganization']].head(30))

import pandas_datareader.wb as wb

indicators_to_test = ['EN.ATM.CO2E.KT', 'EN.ATM.CO2E.PC', 'EG.FEC.RNEW.ZS', 'EN.ATM.PM25.MC.M3']

for ind in indicators_to_test:
    try:
        # 예시 국가 'KOR' 로 1년치만 조회 시도
        data = wb.download(indicator=ind, country='KOR', start=2020, end=2020)
        if data.empty:
            print(f"지표 {ind} 데이터가 없습니다.")
        else:
            print(f"지표 {ind} 정상 조회됨, 데이터 수: {len(data)}")
    except Exception as e:
        print(f"지표 {ind} 조회 실패: {e}")

# 환경/기후 지표 코드
env_indicators = {
    'renewable_energy_pct': 'EG.FEC.RNEW.ZS',      # 재생에너지 사용 비율
    'pm25_concentration': 'EN.ATM.PM25.MC.M3'      # PM2.5 농도
}

# 데이터 다운로드
env_data = wb.download(
    indicator=list(env_indicators.values()),
    country=iso3_list,
    start=2014,
    end=2023
)

# 데이터 정리
env_data = env_data.reset_index()
env_data = env_data.rename(columns={v: k for k, v in env_indicators.items()})
env_data = env_data[['country', 'year'] + list(env_indicators.keys())]
env_data = env_data.sort_values(['country', 'year']).reset_index(drop=True)


print(env_data)

# CO2 데이터 불러오기
co2_df = pd.read_csv(
    '/content/drive/MyDrive/딥러닝프로젝트/UNDP Data Dive 해커톤/단위별 데이터/외부 데이터/P_Data_Extract_From_Environment_Social_and_Governance_(ESG)_Data/312d6359-c398-42ec-bb4b-1c7976fe4153_Data.csv'
)

print(co2_df.columns)  # 먼저 실제 열 이름 확인!

print(co2_df)

# 필요한 열만 추출
value_vars = [col for col in co2_df.columns if '[YR' in col]
co2_long = co2_df.melt(
    id_vars=['Country Name'],
    value_vars=value_vars,
    var_name='year',
    value_name='co2_per_capita'
)

# 연도만 숫자 형태로 추출 (예: "2014 [YR2014]" → 2014)
co2_long['year'] = co2_long['year'].str.extract(r'(\d{4})').astype(int)

# 컬럼명 정리
co2_long = co2_long.rename(columns={'Country Name': 'country'})

# co2_per_capita 타입 정리
co2_long['co2_per_capita'] = pd.to_numeric(co2_long['co2_per_capita'], errors='coerce')

# env_data['year']가 문자열이면 int로 변환
env_data['year'] = env_data['year'].astype(int)

merged_data = pd.merge(env_data, co2_long, on=['country', 'year'], how='left')

print(merged_data.head())
print(merged_data.columns)

# CSV 저장
merged_data.to_csv('/content/drive/MyDrive/딥러닝프로젝트/UNDP Data Dive 해커톤/단위별 데이터/외부데이터_환경기후_지표.csv', index=False)

# 기타 개발 지표 코드
dev_indicators = {
    'female_employment_rate': 'SL.TLF.CACT.FE.ZS',    # 여성 경제활동 참가율
    'internet_user_pct': 'IT.NET.USER.ZS',            # 인터넷 사용자 비율
    'ict_exports_pct': 'BX.GSR.CCIS.ZS'                # ICT 수출 비중
}

# 데이터 다운로드
dev_data = wb.download(
    indicator=list(dev_indicators.values()),
    country=iso3_list,
    start=2014,
    end=2023
)

# 데이터 정리
dev_data = dev_data.reset_index()
dev_data = dev_data.rename(columns={v: k for k, v in dev_indicators.items()})
dev_data = dev_data[['country', 'year'] + list(dev_indicators.keys())]
dev_data = dev_data.sort_values(['country', 'year']).reset_index(drop=True)

# CSV 저장
dev_data.to_csv('/content/drive/MyDrive/딥러닝프로젝트/UNDP Data Dive 해커톤/단위별 데이터/외부데이터_기타 개발 지표_지표.csv', index=False)



# 생산 데이터

# 생산 관련 지표 코드 (WBGAPI 기준)
production_indicators = {
    'services_value_pct_gdp': 'NV.SRV.TOTL.ZS',        # 서비스업 부가가치 (% of GDP)
    'manufacturing_pct_gdp': 'NV.IND.MANF.ZS',         # 제조업 부가가치 (% of GDP)
    'crop_prod_index': 'AG.PRD.CROP.XD',               # 농작물 생산지수 (base 2004-2006 = 100)
    'livestock_prod_index': 'AG.PRD.LVSK.XD'           # 가축 생산지수 (base 2004-2006 = 100)
}

# ⬇데이터 다운로드
prod_data = wb.download(
    indicator=list(production_indicators.values()),
    country=iso3_list,  # 이건 네가 미리 정의한 국가코드 리스트
    start=2014,
    end=2023
)

# 데이터 정리
prod_data = prod_data.reset_index()
prod_data = prod_data.rename(columns={v: k for k, v in production_indicators.items()})
prod_data = prod_data[['country', 'year'] + list(production_indicators.keys())]
prod_data = prod_data.sort_values(['country', 'year']).reset_index(drop=True)

# CSV 저장
prod_data.to_csv('/content/drive/MyDrive/딥러닝프로젝트/UNDP Data Dive 해커톤/단위별 데이터/외부데이터_생산지표.csv', index=False)

print(prod_data)

prod_data.groupby('year').apply(lambda x: x.isnull().mean())


# 경제 관련 지표
econ_indicators = {
    'gdp_per_capita': 'NY.GDP.PCAP.KD'  # 1인당 GDP (constant 2015 USD)
}

# 데이터 다운로드
econ_data = wb.download(
    indicator=list(econ_indicators.values()),
    country=iso3_list,
    start=2014,
    end=2023
)

# 정리
econ_data = econ_data.reset_index()
econ_data = econ_data.rename(columns={v: k for k, v in econ_indicators.items()})
econ_data = econ_data[['country', 'year'] + list(econ_indicators.keys())]
econ_data = econ_data.sort_values(['country', 'year']).reset_index(drop=True)

# 저장
econ_data.to_csv('/content/drive/MyDrive/딥러닝프로젝트/UNDP Data Dive 해커톤/단위별 데이터/외부데이터_경제지표.csv', index=False)

# 보건 관련 지표
health_indicators = {
    'infant_mortality': 'SH.DYN.MORT',           # 영아 사망률 (Under-1)
    'neonatal_mortality': 'SH.DYN.NMRT',         # 신생아 사망률
    'life_expectancy': 'SP.DYN.LE00.IN'          # 기대 수명
}

# 다운로드
health_data = wb.download(
    indicator=list(health_indicators.values()),
    country=iso3_list,
    start=2014,
    end=2023
)

# 정리
health_data = health_data.reset_index()
health_data = health_data.rename(columns={v: k for k, v in health_indicators.items()})
health_data = health_data[['country', 'year'] + list(health_indicators.keys())]
health_data = health_data.sort_values(['country', 'year']).reset_index(drop=True)

# 저장
health_data.to_csv('/content/drive/MyDrive/딥러닝프로젝트/UNDP Data Dive 해커톤/단위별 데이터/외부데이터_보건지표.csv', index=False)

# 교육 관련 지표
edu_indicators = {
    'primary_enrollment_rate': 'SE.PRM.ENRR',      # 초등학교 순취학률
    'primary_completion_rate': 'SE.PRM.CMPT.ZS'    # 초등학교 이수율
}

# 다운로드
edu_data = wb.download(
    indicator=list(edu_indicators.values()),
    country=iso3_list,
    start=2014,
    end=2023
)

# 정리
edu_data = edu_data.reset_index()
edu_data = edu_data.rename(columns={v: k for k, v in edu_indicators.items()})
edu_data = edu_data[['country', 'year'] + list(edu_indicators.keys())]
edu_data = edu_data.sort_values(['country', 'year']).reset_index(drop=True)

# 저장
edu_data.to_csv('/content/drive/MyDrive/딥러닝프로젝트/UNDP Data Dive 해커톤/단위별 데이터/외부데이터_교육지표.csv', index=False)
