pip install wbgapi

import pandas as pd
import wbgapi as wb
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt
import joblib

# 최종 데이터프레임 만들기

df_oda_cat = pd.read_csv('/content/drive/MyDrive/딥러닝프로젝트/UNDP Data Dive 해커톤/2차 데이터/에티오피아_섹터별 수여액_대분류.csv')
df = df_oda_cat.rename(columns={'TIME_PERIOD':'year'}).copy()

# 6개 축으로 통합
df['oda_health'] = (
    df['Health'] +
    df['Population policies/Programmes & reproductive health']
)

df['oda_edu'] = df['Education']

df['oda_infra'] = (
    df['Energy'] +
    df['Transport and storage'] +
    df['Communications'] +
    df['Water supply & sanitation']
)

df['oda_econ'] = (
    df['Agriculture, forestry, fishing'] +
    df['Industry, mining, construction'] +
    df['Business and other services'] +
    df['Trade policies and regulations'] +
    df['Tourism']
)

df['oda_gov'] = (
    df['Government and civil society'] +
    df['Banking and financial services']
)

df['oda_social_env'] = (
    df['Other social infrastructure and services'] +
    df['General environment protection'] +
    df['Other multisector']
)

oda_grp_cols = ['oda_health','oda_edu','oda_infra','oda_econ','oda_gov','oda_social_env']

df_wgi = pd.read_csv('/content/drive/MyDrive/딥러닝프로젝트/UNDP Data Dive 해커톤/2차 데이터/WGI 6개 지표.csv')

# 1) 지표 정의 (여러 개여도 OK)
indicators = {
    "gdp_pc": "NY.GDP.PCAP.CD",   # gdp per capita

}

years = list(range(2002, 2024))
codes = list(indicators.values())

# 2) 데이터 다운로드
wb_df = wb.data.DataFrame(codes, "ETH", time=years, labels=True).reset_index()

# 3) 단일 지표 처리 보강:
value_cols = [c for c in wb_df.columns if c.startswith("YR")]

if {"series", "Series"}.issubset(set(wb_df.columns)):
    # (A) 지표 여러 개일 때 일반 경로
    long_df = wb_df.melt(id_vars=["series", "Series"], value_vars=value_cols,
                         var_name="Year", value_name="value")
    long_df["Year"] = long_df["Year"].str.replace("YR", "").astype(int)
    long_df = long_df.rename(columns={"series": "indicator", "Series": "IndicatorName"})
else:
    # (B) 지표 1개일 때 보강 경로
    code = codes[0]
    name = list(indicators.keys())[0]
    tmp = wb_df[value_cols].T.reset_index()
    tmp.columns = ["Year", "value"]
    tmp["Year"] = tmp["Year"].str.replace("YR", "").astype(int)
    tmp["indicator"] = code
    tmp["IndicatorName"] = name
    long_df = tmp[["indicator", "IndicatorName", "Year", "value"]]

# 4) Year × 지표 pivot
pivot = long_df.pivot(index="Year", columns="IndicatorName", values="value").reset_index()
pivot = pivot.rename(columns={'Year': 'year'})
education_df = pd.read_csv('/content/drive/MyDrive/딥러닝프로젝트/UNDP Data Dive 해커톤/2차 데이터/education.csv')
# education_df: ['year', 'Education expenditure (% of GDP)', ...] 라고 가정
df_edu = education_df[['year', 'Education expenditure (% of GDP)']].copy()
df_edu = df_edu.rename(columns={'Education expenditure (% of GDP)': 'edu_expenditure'})

# 필요시 수치형 변환
df_edu['edu_expenditure'] = pd.to_numeric(df_edu['edu_expenditure'], errors='coerce')

# 2002–2023 범위로 정렬
df_macro = (
    pivot.merge(df_edu, on='year', how='left')
    .query('2002 <= year <= 2023')
    .reset_index(drop=True)
)

# gdp_pc: 명목 USD 기준 사용
df_macro.loc[df_macro['year'] == 2023, 'gdp_pc'] = 1272.0
df_macro

df_y = pd.read_csv('/content/drive/MyDrive/딥러닝프로젝트/UNDP Data Dive 해커톤/2차 데이터/health.csv')

# 1) 컬럼 정리 및 연도 범위 필터
def clean_year(df, col='year'):
    df = df.copy()
    df[col] = df[col].astype(int)
    return df

# ODA 분야별
df = clean_year(df, 'year')
df = df.query('2002 <= year <= 2023').reset_index(drop=True)

# WGI
df_wgi = df_wgi.drop(columns=[c for c in df_wgi.columns if 'Unnamed' in c], errors='ignore')
df_wgi = clean_year(df_wgi, 'year')
df_wgi = df_wgi.query('2002 <= year <= 2023').reset_index(drop=True)

# 매크로
df_macro = clean_year(df_macro, 'year')
df_macro = df_macro.query('2002 <= year <= 2023').reset_index(drop=True)

# Y
df_y = df_y.rename(columns={
    'Life Expectancy':'life_expectancy',
    'Under-5 Mortality Rate':'under5_mortality',
    '신생아 사망률':'infant_mortality'
})
df_y = clean_year(df_y, 'year')
df_y = df_y.query('2002 <= year <= 2023').reset_index(drop=True)

# 2) 결측 처리 유틸(연도별 단조 시계열 가정)
def interp_inner(series):
    return series.astype(float).interpolate(method='linear', limit_direction='both')

# Education expenditure: 내부 보간(2006~2023), 2002~2005는 NA 유지
if 'edu_expenditure' in df_macro.columns:
    df_macro['edu_expenditure'] = pd.to_numeric(df_macro['edu_expenditure'], errors='coerce')
    # 2006~2023 범위에서만 보간
    mask_edu = df_macro['year'].between(2006, 2023)
    df_macro.loc[mask_edu, 'edu_expenditure'] = (
        df_macro.loc[mask_edu, 'edu_expenditure'].interpolate('linear', limit_direction='both')
    )
    # 2002~2005는 그대로 NA 두기

# 3) 내부 결측 보정(WGI, ODA 분야별, Y)
for col in ['oda_health','oda_edu','oda_infra','oda_econ','oda_gov','oda_social_env']:
    if col in df.columns:
        df[col] = interp_inner(df[col])

for col in ['cc','ge','pv','rl','rq','va']:
    if col in df_wgi.columns:
        df_wgi[col] = interp_inner(df_wgi[col])

# 4) 병합: 2002–2023 내부 조인
dfs = [df[['year','oda_health','oda_edu','oda_infra','oda_econ','oda_gov','oda_social_env']],
       df_wgi[['year','cc','ge','pv','rl','rq','va']],
       df_macro[['year','gdp_pc','edu_expenditure']],
       df_y[['year','life_expectancy','under5_mortality','infant_mortality']]]

df_master = dfs[0]
for t in dfs[1:]:
    df_master = df_master.merge(t, on='year', how='inner')

# 최종 정렬
df_master = df_master.sort_values('year').reset_index(drop=True)
