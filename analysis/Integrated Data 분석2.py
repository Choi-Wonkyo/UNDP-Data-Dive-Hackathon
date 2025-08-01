import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# 1. 데이터 로드
df4 = pd.read_csv("/content/drive/MyDrive/딥러닝프로젝트/UNDP Data Dive 해커톤/단위별 데이터/외부데이터_빈곤 및 사회복지 지표1.csv")
df5 = pd.read_csv("/content/drive/MyDrive/딥러닝프로젝트/UNDP Data Dive 해커톤/단위별 데이터/외부데이터_환경기후_지표.csv")
oda_df = pd.read_csv("/content/drive/MyDrive/딥러닝프로젝트/UNDP Data Dive 해커톤/단위별 데이터/전체 분석 대비.csv")

# 2. 빈곤 관련 데이터 전처리
poverty_df = df4[['year', 'country', 'poverty_percent']]
poverty_df = poverty_df[(poverty_df['year'] >= 2014) & (poverty_df['year'] <= 2021)]

# 3. 빈곤율 데이터 유효한 국가 필터링 (50% 이상 데이터 보유)
valid_countries = (
    poverty_df
    .groupby('country')['poverty_percent']
    .apply(lambda x: x.notna().mean())
    .reset_index(name='valid_ratio')
)
valid_countries = valid_countries[valid_countries['valid_ratio'] >= 0.5]['country']
poverty_df_filtered = poverty_df[poverty_df['country'].isin(valid_countries)]

# 4. 빈곤 관련 ODA 필터링 (키워드 포함)
poverty_keywords = ['poverty', 'basic needs', 'social protection']
poverty_oda_df = oda_df[oda_df['PurposeName'].str.contains('|'.join(poverty_keywords), case=False, na=False)]

poverty_oda_grouped = (
    poverty_oda_df
    .groupby(['RecipientName', 'Year'], as_index=False)
    .agg(poverty_oda_usd=('USD_Disbursement', 'sum'))
    .rename(columns={'RecipientName': 'country'})
)
# 시차 고려: ODA 금액 1년 앞당기기
poverty_oda_grouped['Year'] += 1

# 병합
poverty_df_filtered = poverty_df_filtered.rename(columns={'year': 'Year'})
merged_df = pd.merge(poverty_df_filtered, poverty_oda_grouped, on=['country', 'Year'], how='inner')

# 5. 국가별 빈곤 관련 ODA와 빈곤율 상관분석 (최소 3개 연도 이상)
corr_list = []
for country, group in merged_df.groupby('country'):
    if len(group) >= 3:
        corr, pval = pearsonr(group['poverty_oda_usd'], group['poverty_percent'])
        corr_list.append({'country': country, 'corr': corr, 'pval': pval, 'n_obs': len(group)})

corr_df = pd.DataFrame(corr_list)

# 6. 상관계수 상위/하위 국가 출력
print(corr_df.sort_values('corr', ascending=False).head(10))
print(corr_df.sort_values('corr', ascending=True).head(10))

# 7. 예시: 특정 국가(우루과이) 빈곤율과 빈곤 관련 ODA 시계열 시각화
uru_df = merged_df[merged_df['country'] == 'Uruguay'].sort_values('Year')
fig, ax1 = plt.subplots(figsize=(10,6))
ax1.plot(uru_df['Year'], uru_df['poverty_percent'], marker='o', color='tab:blue')
ax1.set_xlabel('Year')
ax1.set_ylabel('Poverty Rate (%)', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax2 = ax1.twinx()
ax2.plot(uru_df['Year'], uru_df['poverty_oda_usd'], marker='o', color='tab:orange')
ax2.set_ylabel('Poverty-related ODA (USD)', color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')
plt.title('Uruguay: Poverty Rate and Poverty-related ODA Over Time')
plt.show()

# 8. 빈곤율 변화량과 해당 기간 빈곤 관련 ODA 총액 집계 및 시각화
start_poverty = (
    poverty_df_filtered.sort_values(['country', 'Year'])
    .groupby('country').first().reset_index()[['country', 'Year', 'poverty_percent']]
    .rename(columns={'Year': 'start_year', 'poverty_percent': 'start_poverty_percent'})
)
end_poverty = (
    poverty_df_filtered.sort_values(['country', 'Year'])
    .groupby('country').last().reset_index()[['country', 'Year', 'poverty_percent']]
    .rename(columns={'Year': 'end_year', 'poverty_percent': 'end_poverty_percent'})
)
poverty_change = pd.merge(start_poverty, end_poverty, on='country')
poverty_change['poverty_diff'] = poverty_change['start_poverty_percent'] - poverty_change['end_poverty_percent']

# 해당 기간 빈곤 관련 ODA 합산
oda_poverty = oda_df[oda_df['PurposeName'].str.lower().str.contains('|'.join(poverty_keywords), na=False)]
oda_by_country = []
for _, row in poverty_change.iterrows():
    total_oda = oda_poverty[
        (oda_poverty['RecipientName'] == row['country']) &
        (oda_poverty['Year'] >= row['start_year']) &
        (oda_poverty['Year'] <= row['end_year'])
    ]['USD_Disbursement'].sum()
    oda_by_country.append({'country': row['country'], 'poverty_related_oda': total_oda})

oda_df_sum = pd.DataFrame(oda_by_country)
poverty_merged = pd.merge(poverty_change, oda_df_sum, on='country')

sns.scatterplot(data=poverty_merged, x='poverty_diff', y='poverty_related_oda')
plt.xlabel('빈곤율 감소 (%)')
plt.ylabel('빈곤 관련 ODA 총액')
plt.title('빈곤율 감소 vs 빈곤 관련 ODA')
plt.show()

print(poverty_merged[['poverty_diff', 'poverty_related_oda']].corr())

# 9. 중위소득(GNI) 관련 데이터 및 ODA 경제 분야 필터링 후 시차 상관분석

gni_df = df4[['year', 'country', 'gni_per_capita']]
valid_countries = (
    gni_df.groupby('country')['gni_per_capita']
    .apply(lambda x: x.notna().mean())
    .reset_index(name='valid_ratio')
)
valid_countries = valid_countries[valid_countries['valid_ratio'] >= 0.5]['country']
gni_df_filtered = gni_df[gni_df['country'].isin(valid_countries)]

economic_keywords = ['Transport', 'Energy', 'Communications', 'Banking', 'Business', 'Industry', 'Trade', 'Agriculture', 'Forestry', 'Fishing']
economic_oda = oda_df[oda_df['PurposeName'].str.contains('|'.join(economic_keywords), case=False, na=False)]

target_income_groups = ['LICs', 'LMICs', 'UMICs', 'HICs', 'LDCs', 'Part I', 'Part II']
filtered_oda = economic_oda[economic_oda['IncomegroupName'].isin(target_income_groups)]

economic_oda_sum = (
    filtered_oda.groupby(['RecipientName', 'Year', 'IncomegroupName'])['USD_Disbursement']
    .sum().reset_index()
    .rename(columns={'RecipientName': 'country', 'Year': 'year', 'USD_Disbursement': 'economic_oda'})
)

merged = pd.merge(gni_df_filtered, economic_oda_sum, on=['country', 'year'], how='inner').sort_values(['country', 'year'])
merged['gni_change'] = merged.groupby('country')['gni_per_capita'].diff()
merged['oda_change'] = merged.groupby('country')['economic_oda'].diff()

sns.scatterplot(data=merged, x='oda_change', y='gni_change')
plt.title("ODA 변화 vs GNI 변화")
plt.show()

print(f"ODA 변화량과 GNI 변화량 간의 상관계수: {merged[['gni_change', 'oda_change']].corr().iloc[0,1]:.3f}")

# Lag별 상관계수 계산 (1~5년)
results = []
lags = range(1, 6)
groups = merged['IncomegroupName'].unique()

for lag in lags:
    for group in groups:
        df_group = merged[merged['IncomegroupName'] == group].copy()
        df_group['oda_lag'] = df_group.groupby('country')['oda_change'].shift(lag)
        valid_df = df_group.dropna(subset=['oda_lag', 'gni_change'])
        if not valid_df.empty:
            corr = valid_df['oda_lag'].corr(valid_df['gni_change'])
            results.append({'Lag': lag, 'Group': group, 'Correlation': corr})

results_df = pd.DataFrame(results)

plt.figure(figsize=(10, 6))
ax = sns.lineplot(data=results_df, x='Lag', y='Correlation', hue='Group', marker='o')
ax.set_xticks(sorted(results_df['Lag'].unique()))
plt.title('Lag vs Correlation between ODA Change and GNI Change by Income Group')
plt.xlabel('Lag (years)')
plt.ylabel('Correlation Coefficient')
plt.axhline(0, color='gray', linestyle='--')
plt.legend(title='Income Group')
plt.tight_layout()
plt.show()

# 10. 의료접근성 지표와 의료 관련 ODA 상관분석

hos = df4[['country', 'year', 'Hospital beds', 'Physicians']].copy()
hos.columns = hos.columns.str.strip()
hos_grouped = hos.groupby(['country', 'year'])[['Hospital beds', 'Physicians']].mean().reset_index()

oda_health = oda_df[oda_df['PurposeName'].str.contains('health|medical|hospital', case=False, na=False)].copy()
oda_health_grouped = oda_health.groupby(['RecipientName', 'Year', 'IncomegroupName'])['USD_Disbursement'].sum().reset_index()
oda_health_grouped.rename(columns={'RecipientName': 'country', 'Year': 'year', 'USD_Disbursement': 'ODA_Health'}, inplace=True)

health_merged = pd.merge(hos_grouped, oda_health_grouped, on=['country', 'year'], how='inner')

results = []
income_groups = ['LICs', 'LMICs', 'UMICs', 'HICs', 'LDCs', 'Part I', 'Part II']

for group in income_groups:
    group_countries = oda_df[oda_df['IncomegroupName'] == group]['RecipientName'].unique()
    group_data = health_merged[health_merged['country'].isin(group_countries)]
    for lag in range(6):
        temp = group_data.copy()
        temp['year'] += lag
        merged_lag = pd.merge(group_data[['country', 'year', 'Hospital beds', 'Physicians']],
                              temp[['country', 'year', 'ODA_Health']], on=['country', 'year'], how='inner')
        if len(merged_lag) < 3:
            continue
        corr_beds = merged_lag['Hospital beds'].corr(merged_lag['ODA_Health'])
        corr_phys = merged_lag['Physicians'].corr(merged_lag['ODA_Health'])
        results.append({'Group': group, 'Lag': lag, 'Indicator': 'Hospital beds', 'Correlation': corr_beds})
        results.append({'Group': group, 'Lag': lag, 'Indicator': 'Physicians', 'Correlation': corr_phys})

results_df = pd.DataFrame(results)
plt.figure(figsize=(12, 7))
sns.lineplot(data=results_df, x='Lag', y='Correlation', hue='Group', style='Indicator', markers=True, dashes=False)
plt.title('Health ODA vs Medical Access by Income Group and Lag')
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel('Lag (years)')
plt.ylabel('Correlation Coefficient')
plt.legend(title='Income Group & Indicator', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 11. Health ODA 상위 수혜국별 lag별 상관분석 및 시각화

top_oda_countries = oda_health_grouped.groupby('country')['ODA_Health'].sum().nlargest(10).index.tolist()
top_oda_data = health_merged[health_merged['country'].isin(top_oda_countries)]

results_top = []
for country in top_oda_countries:
    country_data = top_oda_data[top_oda_data['country'] == country]
    for lag in range(6):
        temp = country_data.copy()
        temp['year'] += lag
        merged_lag = pd.merge(country_data[['year', 'Hospital beds', 'Physicians']],
                              temp[['year', 'ODA_Health']], on='year', how='inner')
        if len(merged_lag) < 3:
            continue
        corr_beds = merged_lag['Hospital beds'].corr(merged_lag['ODA_Health'])
        corr_phys = merged_lag['Physicians'].corr(merged_lag['ODA_Health'])
        results_top.append({'Country': country, 'Lag': lag, 'Indicator': 'Hospital beds', 'Correlation': corr_beds})
        results_top.append({'Country': country, 'Lag': lag, 'Indicator': 'Physicians', 'Correlation': corr_phys})

results_top_df = pd.DataFrame(results_top)
plt.figure(figsize=(13, 7))
sns.lineplot(data=results_top_df, x='Lag', y='Correlation', hue='Country', style='Indicator', markers=True)
plt.title('Top 10 Health ODA Recipient Countries: Correlation by Lag')
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel('Lag (years)')
plt.ylabel('Correlation Coefficient')
plt.legend(title='Country & Indicator', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 1. 데이터 로드
df5 = pd.read_csv("/content/drive/MyDrive/딥러닝프로젝트/UNDP Data Dive 해커톤/단위별 데이터/외부데이터_환경기후_지표.csv")
oda_df = pd.read_csv("/content/drive/MyDrive/딥러닝프로젝트/UNDP Data Dive 해커톤/단위별 데이터/전체 분석 대비.csv")

# 2. 결측치 확인 및 연도 필터링
print(df5.isnull().sum())
print(df5.groupby('year').apply(lambda x: x.isnull().mean()))

env_df = df5[df5['year'] <= 2021]

# 3. 환경 관련 키워드 기반 ODA 필터링
env_keywords = [
    'environment', 'climate', 'biodiversity', 'renewable',
    'energy efficiency', 'pollution', 'sustainable', 'clean energy',
    'air quality', 'waste', 'natural resource', 'water'
]

df_env_oda = oda_df[oda_df['PurposeName'].str.lower().str.contains('|'.join(env_keywords), na=False)]

# 국가-연도별 환경 ODA 총액
env_oda_grouped = (
    df_env_oda.groupby(['RecipientName', 'Year'])['USD_Disbursement']
    .sum().reset_index()
    .rename(columns={'USD_Disbursement': 'env_oda_usd'})
)

# 4. 국가명 통일
env_replace_map = {
    'Congo, Rep.': 'Congo',
    "Cote d'Ivoire": "Côte d'Ivoire",
    'Egypt, Arab Rep.': 'Egypt',
    'Gambia, The': 'Gambia',
    'Iran, Islamic Rep.': 'Iran',
    "Korea, Dem. People's Rep.": "Democratic People's Republic of Korea",
    'Kyrgyz Republic': 'Kyrgyzstan',
    'Lao PDR': "Lao People's Democratic Republic",
    'St. Lucia': 'Saint Lucia',
    'St. Vincent and the Grenadines': 'Saint Vincent and the Grenadines',
    'Turkey': 'Turkmenistan',   # 주의: 실제 확인 필요
    'Venezuela, RB': 'Venezuela',
    'Yemen, Rep.': 'Yemen'
}
env_df['country'] = env_df['country'].replace(env_replace_map)

# 5. 병합
env_df_renamed = env_df.rename(columns={'country': 'RecipientName', 'year': 'Year'})
merged_df = pd.merge(env_oda_grouped, env_df_renamed, on=['RecipientName', 'Year'], how='inner')

# 6. 기본 상관관계 확인
print(merged_df[['env_oda_usd', 'renewable_energy_pct', 'pm25_concentration', 'co2_per_capita']].corr())

# 7. 예시 시각화: ODA vs 재생에너지 비율
sns.lmplot(
    data=merged_df, x='env_oda_usd', y='renewable_energy_pct',
    aspect=1.5, scatter_kws={"alpha": 0.3}
)
plt.title("ODA vs Renewable Energy %")
plt.show()

# 8. 시차 변수 생성 (1~3년)
for lag in [1, 2, 3]:
    merged_df[f'oda_lag{lag}'] = merged_df.groupby('RecipientName')['env_oda_usd'].shift(lag)

# 상관 확인
print(merged_df[['renewable_energy_pct', 'pm25_concentration', 'co2_per_capita',
                 'oda_lag1', 'oda_lag2', 'oda_lag3']].corr())

# 9. 환경 ODA Top 10 국가 변화율 분석
top10_countries = merged_df.groupby('RecipientName')['env_oda_usd'].sum().nlargest(10).index
top10_df = merged_df[merged_df['RecipientName'].isin(top10_countries)].copy()
top10_df.sort_values(['RecipientName', 'Year'], inplace=True)

# 변화율(%) 계산
for col in ['renewable_energy_pct', 'pm25_concentration', 'co2_per_capita', 'env_oda_usd']:
    top10_df[f'{col}_pct_change'] = top10_df.groupby('RecipientName')[col].pct_change() * 100

# 변화율 시각화
fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
metrics = [
    'renewable_energy_pct_pct_change', 'pm25_concentration_pct_change',
    'co2_per_capita_pct_change', 'env_oda_usd_pct_change'
]
titles = ['재생에너지 비중 변화율(%)', 'PM2.5 농도 변화율(%)', 'CO2 배출량 변화율(%)', '환경 ODA 금액 변화율(%)']

for ax, metric, title in zip(axes, metrics, titles):
    sns.lineplot(data=top10_df, x='Year', y=metric, hue='RecipientName', ax=ax)
    ax.set_title(title)
    ax.set_ylabel('변화율 (%)')
    ax.legend(loc='best')

plt.xlabel('Year')
plt.tight_layout()
plt.show()

# 10. 환경 ODA 상위 5개국 상세 추이 분석
top5_countries = merged_df.groupby('RecipientName')['env_oda_usd'].sum().nlargest(5).index
top5_df = merged_df[merged_df['RecipientName'].isin(top5_countries)].copy()

# 스케일 조정(시각화 편의)
top5_df['env_oda_usd_scaled'] = top5_df['env_oda_usd'] / 100  # 또는 /1e6

# 상위 4개국만 시각화 (2x2 subplot)
metrics = ['env_oda_usd_scaled', 'renewable_energy_pct', 'pm25_concentration', 'co2_per_capita']
metric_names = ['환경 ODA(축소)', '재생에너지 비중(%)', 'PM2.5 농도', '1인당 CO₂ 배출량']
colors = ['green', 'orange', 'red', 'blue']

top4_countries = top5_df['RecipientName'].unique()[:4]
fig, axs = plt.subplots(2, 2, figsize=(18, 12))

for idx, country in enumerate(top4_countries):
    ax = axs[idx // 2, idx % 2]
    country_df = top5_df[top5_df['RecipientName'] == country].sort_values('Year')

    for col, name, color in zip(metrics, metric_names, colors):
        ax.plot(country_df['Year'], country_df[col], label=name, marker='o', color=color)

    ax.set_title(f'{country} - 환경 지표 추이')
    ax.set_xlabel('연도')
    ax.set_ylabel('값')
    ax.grid(True)
    ax.legend(fontsize=9)

plt.tight_layout()
plt.suptitle('상위 4개국 환경 지표 및 ODA 추이 (ODA 1/100 단위)', fontsize=16, y=1.02)
plt.subplots_adjust(top=0.92)
plt.show()
