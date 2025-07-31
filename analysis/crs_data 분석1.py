import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('/content/drive/MyDrive/딥러닝프로젝트/UNDP Data Dive 해커톤/단위별 데이터/국가별 연도별 ODA 흐름.csv')


# 1. 전반적 흐름 파악 - 연도별 총 ODA 지급액 시계열
yearly_total = df_country_only.groupby('Year')['USD_Disbursement'].sum().reset_index()

plt.figure(figsize=(10, 6))
sns.lineplot(data=yearly_total, x='Year', y='USD_Disbursement', marker='o', linewidth=2)
plt.title('Total USD Disbursement by Year', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Total USD Disbursement', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. 최근 몇 년간 상위 10개 수원국 ODA 수령액 변화 추이
top10_recipients = (
    df_country_only.groupby('RecipientName')['USD_Disbursement']
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .index
)

top10_df = df_country_only[df_country_only['RecipientName'].isin(top10_recipients)]
grouped_top10 = top10_df.groupby(['Year', 'RecipientName'])['USD_Disbursement'].sum().reset_index()

plt.figure(figsize=(12, 7))
sns.lineplot(data=grouped_top10, x='Year', y='USD_Disbursement', hue='RecipientName', marker='o')
plt.title('Top 10 Recipient Countries - ODA Trend', fontsize=14)
plt.xlabel('Year')
plt.ylabel('USD Disbursement')
plt.legend(title='Recipient')
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. 전체 ODA 중 상위 10개 국가가 차지하는 비중 분석 (연도별)
year_country = df_country_only.groupby(['Year', 'RecipientName'])['USD_Disbursement'].sum().reset_index()

def get_top10_fraction(year_df):
    year_df_sorted = year_df.sort_values('USD_Disbursement', ascending=False).reset_index(drop=True)
    year_df_sorted['Rank'] = year_df_sorted.index + 1
    total = year_df_sorted['USD_Disbursement'].sum()
    top10_total = year_df_sorted.loc[year_df_sorted['Rank'] <= 10, 'USD_Disbursement'].sum()
    return pd.Series({'Total': total, 'Top10': top10_total})

top10_share = year_country.groupby('Year').apply(get_top10_fraction).reset_index()

plt.figure(figsize=(10, 6))
plt.bar(top10_share['Year'], top10_share['Top10'], label='Top 10 Recipients')
plt.bar(top10_share['Year'], top10_share['Total'] - top10_share['Top10'],
        bottom=top10_share['Top10'], label='Others', color='gray')
plt.title('Share of Top 10 Recipients in Total ODA by Year', fontsize=14)
plt.xlabel('Year')
plt.ylabel('USD Disbursement')
plt.legend()
plt.tight_layout()
plt.show()

# 4. 지역/국가별 분석 - 국가별 총 ODA 수령액 Top 20
top20_recipients = (
    df_country_only.groupby('RecipientName')['USD_Disbursement']
    .sum()
    .sort_values(ascending=False)
    .head(20)
    .reset_index()
)

plt.figure(figsize=(12, 8))
sns.barplot(data=top20_recipients, x='USD_Disbursement', y='RecipientName', palette='viridis')
plt.title('Top 20 Recipient Countries by Total ODA Received', fontsize=16)
plt.xlabel('Total USD Disbursement')
plt.ylabel('Recipient Country')
plt.tight_layout()
plt.show()

# 5. 특정 국가 5개 ODA 흐름 시계열 시각화
target_countries = ['India', 'Ukraine', 'Syrian Arab Republic', 'Egypt', 'Indonesia']
df_selected = df_country_only[df_country_only['RecipientName'].isin(target_countries)]

country_trend = df_selected.groupby(['Year', 'RecipientName'])['USD_Disbursement'].sum().reset_index()

plt.figure(figsize=(10, 6))
sns.lineplot(data=country_trend, x='Year', y='USD_Disbursement', hue='RecipientName', marker='o')
plt.title('ODA Flow Over Time for Selected Countries', fontsize=14)
plt.xlabel('Year')
plt.ylabel('USD Disbursement')
plt.legend(title='Recipient Country')
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. 분산도 및 불균형 분석 - 국가별 연도별 누적 ODA 수령액의 지니계수 계산

def gini(array):
    array = np.array(array, dtype=np.float64)
    if np.amin(array) < 0:
        array -= np.amin(array)
    array += 1e-9  # 0 나누기 방지
    array = np.sort(array)
    n = len(array)
    cumulative = np.cumsum(array)
    return (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n

years = sorted(df_country_only['Year'].unique())
gini_values = []

for year in years:
    df_year = df_country_only[df_country_only['Year'] == year]
    by_country = df_year.groupby('RecipientName')['USD_Disbursement'].sum()
    gini_values.append(gini(by_country.values))

plt.figure(figsize=(10,6))
plt.plot(years, gini_values, marker='o', linestyle='-', color='purple')
plt.title('Yearly Gini Coefficient of ODA Distribution by Recipient Country', fontsize=14)
plt.xlabel('Year')
plt.ylabel('Gini Coefficient')
plt.grid(True)
plt.tight_layout()
plt.show()

# 7. 국가별 총 ODA 수령액 분포 히스토그램
total_by_country = df_country_only.groupby('RecipientName')['USD_Disbursement'].sum()

plt.figure(figsize=(10,6))
plt.hist(total_by_country, bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Total ODA Received by Country', fontsize=14)
plt.xlabel('Total USD Disbursement')
plt.ylabel('Number of Countries')
plt.grid(True)
plt.tight_layout()
plt.show()

# 8. 국가별 연도별 수령액 기반 군집화 (KMeans)

# 피벗 테이블 생성 (RecipientName x Year)
pivot_df = df_country_only.pivot_table(
    index='RecipientName',
    columns='Year',
    values='USD_Disbursement',
    aggfunc='sum',
    fill_value=0
)

# 표준화
scaler = StandardScaler()
pivot_scaled = scaler.fit_transform(pivot_df)

# KMeans 클러스터링 (클러스터 수 k=4)
k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(pivot_scaled)

# 클러스터 결과 데이터프레임에 추가
pivot_df['Cluster'] = clusters

# 군집별 평균 시계열 시각화
plt.figure(figsize=(12, 8))
for cluster_num in range(k):
    cluster_data = pivot_df[pivot_df['Cluster'] == cluster_num].drop('Cluster', axis=1)
    mean_series = cluster_data.mean()
    plt.plot(mean_series.index, mean_series.values, marker='o', label=f'Cluster {cluster_num}')

plt.title('Average ODA Disbursement Time Series by Cluster', fontsize=14)
plt.xlabel('Year')
plt.ylabel('Average USD Disbursement')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 클러스터별 국가 리스트 출력
for cluster_num in range(k):
    countries_in_cluster = pivot_df[pivot_df['Cluster'] == cluster_num].index.tolist()
    print(f"Cluster {cluster_num}:")
    print(countries_in_cluster)
    print()


# --------------------------------------
# 분야별 ODA 흐름 분석
# --------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 국가 및 비국가 수혜자 필터링
df = pd.read_csv('/content/drive/MyDrive/딥러닝프로젝트/UNDP Data Dive 해커톤/단위별 데이터/분야별 연도별 ODA 흐름.csv')

# 국가코드 기반 제외
exclude_codes = ['40000', '41000', '42000', '43000', '44000', '45000', '46000', '47000']
df_country = df[~df['RecipientName'].isin(exclude_codes)]
df_country = df_country[~df_country['RecipientName'].astype(str).isin(exclude_codes)]

# 비국가명도 제외
non_country = ['Unspecified', 'Bilateral, unspecified', 'Developing countries, unspecified', 'Region, unspecified']
df_country_only = df_country[~df_country['RecipientName'].isin(non_country)]

# 1) 연도별 전체 ODA 금액 변화 (절대값)
oda_yearly = df_country_only.groupby('Year')['USD_Disbursement'].sum().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(data=oda_yearly, x='Year', y='USD_Disbursement', marker='o')
plt.title('Total ODA Disbursement by Year', fontsize=16)
plt.xlabel('Year')
plt.ylabel('Total ODA (USD)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# 2) 분야별 연도별 ODA 금액 변화 (절대값)

sector_trend = df_country_only.groupby(['Year', 'SectorName'])['USD_Disbursement'].sum().reset_index()

# Pivot 테이블 생성 (Year x SectorName)
sector_pivot = sector_trend.pivot(index='Year', columns='SectorName', values='USD_Disbursement').fillna(0)

plt.figure(figsize=(14, 8))
sector_pivot.plot(kind='line', marker='o', figsize=(14, 8))
plt.title('ODA Disbursement by Sector (Yearly)', fontsize=16)
plt.xlabel('Year')
plt.ylabel('USD Disbursement')
plt.grid(True)
plt.legend(title='Sector', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 3) 분야별 ODA 비중 변화 (Stacked Area Chart, % 기준)

sector_percent = sector_pivot.div(sector_pivot.sum(axis=1), axis=0) * 100

plt.figure(figsize=(14, 8))
sector_percent.plot.area(colormap='tab20')
plt.title('Sectoral Share of ODA by Year (%)', fontsize=16)
plt.xlabel('Year')
plt.ylabel('Share of Total ODA (%)')
plt.legend(title='Sector', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 4) 상위 10개 분야 선정 및 연도별 추이 분석

top_sectors = (
    df_country_only.groupby('SectorName')['USD_Disbursement']
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .index
)

df_top = df_country_only[df_country_only['SectorName'].isin(top_sectors)]

sector_trend_top = (
    df_top.groupby(['Year', 'SectorName'])['USD_Disbursement']
    .sum()
    .reset_index()
)

pivot_abs = sector_trend_top.pivot(index='Year', columns='SectorName', values='USD_Disbursement').fillna(0)

plt.figure(figsize=(14, 8))
pivot_abs.plot(kind='line', marker='o', figsize=(14, 8))
plt.title('Top 10 Sectors - ODA Disbursement Over Time', fontsize=16)
plt.xlabel('Year')
plt.ylabel('USD Disbursement')
plt.grid(True)
plt.legend(title='Sector', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 상위 10개 분야의 비중 변화 (Stacked Area Chart, % 기준)

pivot_pct = pivot_abs.div(pivot_abs.sum(axis=1), axis=0) * 100

plt.figure(figsize=(14, 8))
pivot_pct.plot.area(colormap='tab20')
plt.title('Top 10 Sectors - Share of Total ODA by Year (%)', fontsize=16)
plt.xlabel('Year')
plt.ylabel('Share of ODA (%)')
plt.legend(title='Sector', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# --------------------------------------
# 분야별 ODA 비중 분석 및 추가 인사이트
# --------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 필수 전처리 (중복 시 제거)
df = df_country_only[['Year', 'SectorName', 'USD_Disbursement']].dropna()

# 1) 연도별 분야별 총 ODA 금액 계산
sector_year = df.groupby(['Year', 'SectorName'])['USD_Disbursement'].sum().reset_index()

# 2) 연도별 총 ODA 금액 계산
year_total = df.groupby('Year')['USD_Disbursement'].sum().reset_index()
year_total.columns = ['Year', 'Total_Disbursement']

# 3) 분야별 연도별 비중 계산 (%)
merged = pd.merge(sector_year, year_total, on='Year')
merged['Percentage'] = (merged['USD_Disbursement'] / merged['Total_Disbursement']) * 100

# 4) 피벗 테이블 생성 (Year x SectorName)
pivot_pct = merged.pivot(index='Year', columns='SectorName', values='Percentage').fillna(0)

# 5) 상위 5개 분야 선정 (전체 누적 기준)
top_sectors = df.groupby('SectorName')['USD_Disbursement'].sum().sort_values(ascending=False).head(5).index

# 6) 상위 5개 분야 연도별 비중 시계열 플롯
plt.figure(figsize=(14, 8))
for sector in top_sectors:
    plt.plot(pivot_pct.index, pivot_pct[sector], label=sector)
plt.title('Annual ODA Sectoral Concentration (Top 5)', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Share of Total ODA (%)', fontsize=12)
plt.legend(title='Sector', loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# 7) 연도별 분야 비중 순위 히트맵 (Top 5 분야)
ranks = pivot_pct[top_sectors].rank(axis=1, ascending=False)

plt.figure(figsize=(10, 6))
sns.heatmap(ranks, annot=True, cmap='coolwarm_r', cbar_kws={'label': 'Rank'})
plt.title("Sector Rank per Year (1 = Highest Share)")
plt.ylabel("Year")
plt.yticks(rotation=0)
plt.xlabel("Sector")
plt.tight_layout()
plt.show()

# 8) 상위 5개 분야의 연도별 ODA 비중 합계 (%)

year_total_full = df_country_only.groupby('Year')['USD_Disbursement'].sum()
df_top5 = df_country_only[df_country_only['SectorName'].isin(top_sectors)]
year_top5_total = df_top5.groupby('Year')['USD_Disbursement'].sum()
top5_share = (year_top5_total / year_total_full * 100).dropna()

plt.figure(figsize=(10, 5))
top5_share.plot(marker='o', color='darkblue')
plt.title('Share of Top 5 Sectors in Total ODA by Year')
plt.xlabel('Year')
plt.ylabel('Top 5 Share (%)')
plt.grid(True)
plt.tight_layout()
plt.show()

# 9) 분야별 평균 비중 계산 및 시각화

avg_share_by_sector = pivot_pct.mean().sort_values(ascending=False)

print("📌 분야별 연도 평균 비중 (%):")
print(avg_share_by_sector)

plt.figure(figsize=(12, 10))
bars = plt.barh(y=avg_share_by_sector.index, width=avg_share_by_sector.values, color='skyblue', edgecolor='black')
plt.xlabel("Average Share of Total ODA (%)")
plt.title("ODA Sectoral Importance (2014–2023)")
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.5)

for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.2, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', va='center')

plt.tight_layout()
plt.show()

# 10) PCA + KMeans 클러스터링으로 분야별 연도별 패턴 군집화

pivot_sector_year = df_country_only.groupby(['SectorName', 'Year'])['USD_Disbursement'].sum().reset_index()
pivot_sector_year = pivot_sector_year.pivot(index='SectorName', columns='Year', values='USD_Disbursement').fillna(0)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(pivot_sector_year)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(pca_result)

pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
pca_df['SectorName'] = pivot_sector_year.index
pca_df['Cluster'] = labels

plt.figure(figsize=(12, 8))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster', style='Cluster', palette='Set2', s=120, alpha=0.8)

for label in pca_df['Cluster'].unique():
    cluster_group = pca_df[pca_df['Cluster'] == label]
    center_x = cluster_group['PC1'].mean()
    center_y = cluster_group['PC2'].mean()
    representative = cluster_group.loc[((cluster_group['PC1'] - center_x)**2 + (cluster_group['PC2'] - center_y)**2).idxmin()]
    plt.text(representative['PC1'] + 0.2, representative['PC2'], f"[C{label}] {representative['SectorName']}", fontsize=10, fontweight='bold')

plt.title('PCA + KMeans of ODA Sectors (Cluster Labeled)', fontsize=14)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.tight_layout()
plt.show()

for cluster_id in sorted(pca_df['Cluster'].unique()):
    sectors_in_cluster = pca_df[pca_df['Cluster'] == cluster_id]['SectorName'].tolist()
    print(f"🔹 Cluster {cluster_id} ({len(sectors_in_cluster)} sectors):")
    for sector in sectors_in_cluster:
        print(f"   - {sector}")
    print()

# 11) 군집별 연도별 평균 ODA 패턴 시각화

year_columns = sorted(df['Year'].unique())
pivot_with_cluster = pivot_sector_year.merge(pca_df[['SectorName', 'Cluster']], left_index=True, right_on='SectorName')
cluster_trends = pivot_with_cluster.groupby('Cluster')[year_columns].mean().T

plt.figure(figsize=(12, 6))
for cluster_id in cluster_trends.columns:
    plt.plot(cluster_trends.index, cluster_trends[cluster_id], label=f'Cluster {cluster_id}')
plt.title('Average ODA Pattern by Cluster (Yearly)', fontsize=14)
plt.xlabel('Year')
plt.ylabel('Average USD Disbursement')
plt.legend(title='Cluster')
plt.grid(True)
plt.tight_layout()
plt.show()

# 12) 연도별 ODA 수혜국 수 변화 시각화

oda_yearly_count = df_country_only.groupby('Year')['RecipientName'].nunique()

plt.figure(figsize=(10, 5))
oda_yearly_count.plot(marker='o', color='teal')
plt.title('Changes in the Number of ODA Beneficiaries by Year')
plt.xlabel('Year')
plt.ylabel('Number of Beneficiary Countries')
plt.grid(True)
plt.tight_layout()
plt.show()

# 13) 상위 5개 분야 연평균 성장률 계산

top5_sectors = top_sectors.tolist()  # 이미 top 5로 정의됨

sector_year_df = df_country_only[df_country_only['SectorName'].isin(top5_sectors)] \
    .groupby(['SectorName', 'Year'])['USD_Disbursement'].sum().unstack()

sector_growth_rate = sector_year_df.pct_change(axis=1, fill_method=None)
sector_growth_avg = sector_growth_rate.mean(axis=1, skipna=True)

print("상위 5개 분야의 연평균 성장률:")
print(sector_growth_avg.sort_values(ascending=False))

# 14) 연도별 주요 수원국 TOP 10 순위 변화 히트맵

pivot_df = df_country_only.groupby(['Year', 'RecipientName'])['USD_Disbursement'].sum().reset_index()
pivot_df['rank'] = pivot_df.groupby('Year')['USD_Disbursement'].rank(ascending=False)

top_ranks = pivot_df[pivot_df['rank'] <= 10].pivot(index='Year', columns='RecipientName', values='rank')

plt.figure(figsize=(12, 8))
sns.heatmap(top_ranks, annot=True, cmap='YlGnBu', cbar=False)
plt.yticks(rotation=0)
plt.title('Top 10 ODA Beneficiaries Ranking Changes by Year', fontsize=14)
plt.xlabel('Country')
plt.ylabel('Year')
plt.tight_layout()
plt.show()
