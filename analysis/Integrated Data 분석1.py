# Google Drive 마운트 및 데이터 불러오기
from google.colab import drive
import pandas as pd
import wbgapi as wb
import pycountry
import matplotlib.pyplot as plt
import pickle

drive.mount('/content/drive')

# ODA 데이터 불러오기 및 국가 관련 필터링
df = pd.read_csv('/content/drive/MyDrive/UNDP Data Dive 해커톤/단위별 데이터/분야별 연도별 ODA 흐름.csv')

# 수치형 코드와 특정 명칭 제외 (지역 및 불분명 데이터)
exclude_codes = ['40000', '41000', '42000', '43000', '44000', '45000', '46000', '47000']
df_country = df[~df['RecipientName'].isin(exclude_codes)]
non_country_names = ['Unspecified', 'Bilateral, unspecified', 'Developing countries, unspecified', 'Region, unspecified']
df_country_only = df_country[~df_country['RecipientName'].isin(non_country_names)]

# 국가명 리스트 추출 및 불필요한 키워드 포함 국가 제외
country_list = df_country_only['RecipientName'].unique().tolist()
exclude_keywords = [
    'regional', 'Region', 'Unspecified', 'States Ex-', 'Micronesia, regional',
    'Africa, ', 'America, ', 'Asia, ', 'Europe, ', 'Middle East', 'Central America',
    'Southern Africa', 'Western Africa', 'Melanesia', 'Polynesia', 'Micronesia',
    '10003'
]
cleaned_countries = [c for c in country_list if not any(keyword in c for keyword in exclude_keywords)]

# pycountry를 이용한 국가명 → ISO3 코드 매핑
def get_country_code(name):
    try:
        return pycountry.countries.lookup(name).alpha_3
    except:
        return None

mapped = {name: get_country_code(name) for name in cleaned_countries}
unmatched = [k for k, v in mapped.items() if v is None]

# 수동 매핑 보완
manual_corrections = {
    "China (People's Republic of)": "CHN",
    "Democratic Republic of the Congo": "COD",
    "West Bank and Gaza Strip": "PSE",
    "Kosovo": "XKX"
}
final_codes = [v for v in mapped.values() if v is not None] + list(manual_corrections.values())
final_codes = sorted(set(final_codes))  # 중복 제거

# 결과 출력
print(f"변환 성공 국가 수: {len(final_codes)}")
if unmatched:
    print("변환 실패 국가:", unmatched)

# final_codes 저장
with open("final_codes.pkl", "wb") as f:
    pickle.dump(final_codes, f)
with open('./country_name.txt', 'w') as f:
    f.write(','.join(final_codes))

# World Bank API로 GDP 데이터 불러오기
indicators = ['NY.GDP.MKTP.KD', 'NY.GDP.PCAP.KD']
df_gdp = wb.data.DataFrame(
    indicators,
    economy=final_codes,
    time=range(2014, 2024),
    columns='time',
    index=['economy', 'series'],
    skipBlanks=True
)

# 결측값 있는 국가 제외 (모든 연도 모두 값 있어야 함)
year_cols = [col for col in df_gdp.columns if col.startswith('YR')]
valid_rows = ~df_gdp[year_cols].isnull().any(axis=1)
df_gdp_clean = df_gdp[valid_rows]

# GDP 총합 평균 기준 Top 10 국가 선정
df_gdp_clean['mean_gdp'] = df_gdp_clean.loc[:, year_cols].mean(axis=1)
top10_gdp_codes = df_gdp_clean.loc[df_gdp_clean.index.get_level_values('series') == 'NY.GDP.MKTP.KD']\
    .sort_values('mean_gdp', ascending=False).head(10).index.get_level_values('economy').tolist()

# GDP 총합과 1인당 GDP 시계열 그래프 시각화
indicator_names = {
    'NY.GDP.MKTP.KD': 'GDP (constant 2015 US$)',
    'NY.GDP.PCAP.KD': 'GDP per capita (constant 2015 US$)'
}
for indicator in indicators:
    plt.figure(figsize=(12,6))
    for code in top10_gdp_codes:
        try:
            row = df_gdp.loc[(code, indicator)]
            plt.plot(year_cols, row[year_cols].values, label=code)
        except KeyError:
            continue
    plt.title(f"Top 10 Countries by {indicator_names[indicator]}")
    plt.xlabel("Year")
    plt.ylabel(indicator_names[indicator])
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 경제 지표 추가 수집 및 매핑
econ_indicators = {
    'NY.GDP.MKTP.KD.ZG': 'GDP_growth_rate',
    'SL.UEM.TOTL.ZS': 'Unemployment_rate',
    'NE.GDI.TOTL.ZS': 'Capital_formation',
    'FP.CPI.TOTL.ZG': 'Inflation_rate'
}
df_econ = wb.data.DataFrame(
    list(econ_indicators.keys()),
    economy=final_codes,
    time=range(2014, 2024),
    columns='time',
    index=['economy', 'series'],
    skipBlanks=True
)
df_econ = df_econ.rename(index=econ_indicators, level=1)

# 결측치 확인 및 시각화 (Top 10 국가)
for indicator_key, indicator_label in econ_indicators.items():
    df_sub = df_econ.loc[(slice(None), indicator_label), year_cols].dropna()
    df_sub['avg'] = df_sub.mean(axis=1)
    top10 = df_sub.sort_values('avg', ascending=False).head(10)

    plt.figure(figsize=(12,6))
    for code in top10.index.get_level_values(0):
        row = df_econ.loc[(code, indicator_label)]
        plt.plot(year_cols, row[year_cols].values, label=code)
    plt.title(f"{indicator_label} (Top 10 Countries)")
    plt.xlabel("Year")
    plt.ylabel(indicator_label)
    plt.xticks(rotation=45)
    plt.legend(title="Country Code", bbox_to_anchor=(1.05,1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 보건 지표 수집 및 시각화
health_indicators = {
    'Infant_Mortality': 'SH.DYN.MORT',
    'Life_Expectancy': 'SP.DYN.LE00.IN',
    'Maternal_Mortality': 'SH.STA.MMRT',
    'Neonatal_Mortality': 'SH.DYN.NMRT'
}
df_health = wb.data.DataFrame(
    list(health_indicators.values()),
    economy=final_codes,
    time=range(2014, 2024),
    columns='time',
    index=['economy', 'series'],
    skipBlanks=True
)
df_health = df_health.rename(index={v:k for k,v in health_indicators.items()}, level=1)

year_cols = [c for c in df_health.columns if c.startswith('YR')]
for indicator_key, indicator_label in health_indicators.items():
    df_sub = df_health.loc[(slice(None), indicator_key), year_cols].copy()
    df_sub['avg'] = df_sub.mean(axis=1)
    ascending = True if indicator_key == 'Life_Expectancy' else False
    top10 = df_sub.sort_values('avg', ascending=ascending).head(10)

    plt.figure(figsize=(12,6))
    for code in top10.index.get_level_values('economy'):
        row = df_health.loc[(code, indicator_key)]
        plt.plot(year_cols, row[year_cols].values, label=code)
    plt.title(f"{indicator_key} (Top 10 Countries)")
    plt.xlabel("Year")
    plt.ylabel(indicator_key)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 교육 지표 수집 및 시각화
education_indicators = {
    'SE.ADT.1524.LT.ZS': 'Youth Literacy Rate (15-24) (%)',
    'SE.PRM.ENRR': 'Primary School Enrollment Rate (%)',
    'SE.PRM.CMPT.ZS': 'Primary School Completion Rate (%)',
    'SE.XPD.TOTL.GD.ZS': 'Education Expenditure (% of GDP)'
}

df_edu = wb.data.DataFrame(
    list(education_indicators.keys()),
    economy=final_codes,
    time=range(2014, 2024),
    columns='time',
    index=['economy', 'series'],
    skipBlanks=True
)
year_cols = [f'YR{y}' for y in range(2014, 2024)]

for code, name in education_indicators.items():
    df_sub = df_edu.loc[(slice(None), code), year_cols].dropna()
    df_sub['avg'] = df_sub.mean(axis=1)
    top10 = df_sub.sort_values('avg', ascending=False).head(10)

    plt.figure(figsize=(12,6))
    for c in top10.index.get_level_values(0):
        row = df_edu.loc[(c, code)]
        plt.plot(year_cols, row[year_cols].values, label=c)
    plt.title(f"{name} (Top 10 Countries)")
    plt.xlabel("Year")
    plt.ylabel(name)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
