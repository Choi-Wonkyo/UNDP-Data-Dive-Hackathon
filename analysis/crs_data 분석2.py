import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import networkx as nx
import plotly.express as px
import squarify

# --- 1. 목적별 누적 ODA 규모 상위 10개 추출 및 한글명 매핑 ---
purpose_total = df.groupby('PurposeName')['USD_Disbursement'].sum().sort_values(ascending=False)

purpose_translate = {
    'Sectors not specified': '구체적 부문 미지정',
    'Material relief assistance and services': '구호 물자 및 서비스 지원',
    'Refugees/asylum seekers in donor countries (non-sector allocable)': '공여국 내 난민/망명자 지원 (부문 분류 불가)',
    'General budget support-related aid': '일반 예산 지원 관련 원조',
    'Administrative costs (non-sector allocable)': '행정 비용 (부문 분류 불가)',
    'Road transport': '도로 운송',
    'Formal sector financial intermediaries': '공식 금융 중개기관 지원',
    'Public sector policy and administrative management': '공공 정책 및 행정 관리',
    'STD control including HIV/AIDS': '성병/HIV/AIDS 통제',
}

top9 = list(purpose_translate.keys())
df_top9 = df[df['PurposeName'].isin(top9)].copy()
df_top9['PurposeName_KR'] = pd.Categorical(df_top9['PurposeName'].map(purpose_translate), categories=[purpose_translate[p] for p in top9], ordered=True)

# --- 2. 연도별 주요 목적 ODA 흐름 시각화 (한글) ---
yearly_trend = df_top9.groupby(['Year', 'PurposeName_KR'])['USD_Disbursement'].sum().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(data=yearly_trend, x='Year', y='USD_Disbursement', hue='PurposeName_KR', marker='o')
plt.title("연도별 주요 목적 ODA 흐름 (Top 9)")
plt.ylabel("USD Disbursement")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- 3. 연도별 증감 추세 분석 (Top 10 목적군, 최근 10년 기준) ---
max_year = df['Year'].max()
min_year = max_year - 9
df_recent = df[(df['Year'] >= min_year) & (df['Year'] <= max_year)].copy()

purpose_total_recent = df_recent.groupby('PurposeName')['USD_Disbursement'].sum().sort_values(ascending=False)
top10_recent = list(purpose_total_recent.head(10).index)

yearly_sum = df_recent[df_recent['PurposeName'].isin(top10_recent)] \
                .groupby(['PurposeName', 'Year'])['USD_Disbursement'].sum().reset_index()

def calc_trend(df_group):
    X = df_group['Year'].values.reshape(-1, 1)
    y = df_group['USD_Disbursement'].values
    model = LinearRegression()
    model.fit(X, y)
    return model.coef_[0], model.intercept_

trends = []
for purpose in top10_recent:
    sub_df = yearly_sum[yearly_sum['PurposeName'] == purpose]
    slope, intercept = calc_trend(sub_df)
    total_disbursement = sub_df['USD_Disbursement'].sum()
    trends.append({'PurposeName': purpose, 'Slope': slope, 'Intercept': intercept, 'TotalDisbursement': total_disbursement})

trend_df = pd.DataFrame(trends).sort_values(by='Slope', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=trend_df, x='Slope', y='PurposeName', palette='Blues_r')
plt.title("누적합 상위 목적군의 연도별 증감 추세 (기울기 기준)")
plt.xlabel("기울기 (Slope)")
plt.ylabel("목적")
plt.tight_layout()
plt.show()

# --- 4. 최근 10년 전체 목적군 증감 추세 및 누적액 분석 ---
results_recent = []
for purpose in df_recent['PurposeName'].unique():
    data = df_recent[df_recent['PurposeName'] == purpose].groupby('Year')['USD_Disbursement'].sum().reset_index()
    if len(data) < 4:
        continue
    slope, intercept = calc_trend(data)
    results_recent.append({'PurposeName': purpose, 'Slope': slope, 'Intercept': intercept})

trend_df_recent = pd.DataFrame(results_recent)
total_disbursement = df_recent.groupby('PurposeName')['USD_Disbursement'].sum().reset_index().rename(columns={'USD_Disbursement': 'TotalDisbursement'})
trend_df_recent = pd.merge(trend_df_recent, total_disbursement, on='PurposeName', how='left')

top20_total = trend_df_recent.sort_values(by='TotalDisbursement', ascending=False).head(20)
top10_slope_up = trend_df_recent[trend_df_recent['Slope'] > 0].sort_values(by='Slope', ascending=False).head(10)
top10_slope_down = trend_df_recent[trend_df_recent['Slope'] < 0].sort_values(by='Slope').head(10)

combined_df = pd.concat([top20_total, top10_slope_up, top10_slope_down]).drop_duplicates().reset_index(drop=True)

# 누적액 대비 증감 추세 산점도
plt.figure(figsize=(12, 8))
sns.scatterplot(data=combined_df,
                x='TotalDisbursement',
                y='Slope',
                size=combined_df['Slope'].abs(),
                hue='PurposeName',
                legend='brief',
                sizes=(50, 300),
                alpha=0.7)
plt.title("최근 10년 목적별 누적액 대비 증감 추세 (기울기)")
plt.xlabel("누적 ODA 금액 (USD)")
plt.ylabel("증감 추세 기울기 (Slope)")
plt.axhline(0, color='gray', linestyle='--')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 사분면 플롯
median_slope = combined_df['Slope'].median()
median_total = combined_df['TotalDisbursement'].median()

plt.figure(figsize=(12, 8))
sns.scatterplot(data=combined_df, x='TotalDisbursement', y='Slope', hue='Slope', palette='coolwarm', s=100)
plt.axvline(median_total, color='gray', linestyle='--')
plt.axhline(median_slope, color='gray', linestyle='--')
plt.xlabel('누적 ODA 금액 (Total Disbursement)')
plt.ylabel('연도별 증감 추세 (Slope)')
plt.title('최근 10년 목적별 누적액과 증감 추세 사분면 플롯')

for _, row in combined_df.iterrows():
    plt.text(row['TotalDisbursement'], row['Slope'], row['PurposeName'], fontsize=8, alpha=0.7)

plt.tight_layout()
plt.show()

# 누적액과 증감 추세 히트맵 (0~1 스케일링)
heatmap_data = combined_df.set_index('PurposeName')[['TotalDisbursement', 'Slope']]
scaler = MinMaxScaler()
heatmap_scaled = pd.DataFrame(scaler.fit_transform(heatmap_data), columns=heatmap_data.columns, index=heatmap_data.index)

plt.figure(figsize=(10, len(heatmap_scaled) * 0.4))
sns.heatmap(heatmap_scaled, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
plt.title("최근 10년 목적별 누적액 및 증감 추세 히트맵 (0~1 스케일링)")
plt.tight_layout()
plt.show()

# --- 5. 국가별 주요 목적 클러스터링 ---
df_filtered = df[df['PurposeName'] != 'Sectors not specified']

pivot_df = df_filtered.pivot_table(
    index='RecipientName',
    columns='PurposeName',
    values='USD_Disbursement',
    aggfunc='sum',
    fill_value=0
)

scaled_data = StandardScaler().fit_transform(pivot_df)
kmeans = KMeans(n_clusters=5, random_state=42)
pivot_df['Cluster'] = kmeans.fit_predict(scaled_data)
pivot_df_reset = pivot_df.reset_index()

df_with_cluster = pd.merge(df_filtered, pivot_df_reset[['RecipientName', 'Cluster']], on='RecipientName', how='left')

cluster_purpose = df_with_cluster.groupby(['Cluster', 'PurposeName'])['USD_Disbursement'].sum().reset_index()

top_purposes = cluster_purpose.groupby('Cluster').apply(
    lambda x: x.sort_values(by='USD_Disbursement', ascending=False).head(3)
).reset_index(drop=True)

plt.figure(figsize=(12, 6))
sns.barplot(data=top_purposes, x='Cluster', y='USD_Disbursement', hue='PurposeName')
plt.title("클러스터별 상위 목적 Top 3 (Sectors not specified 제외)")
plt.ylabel("USD Disbursement")
plt.tight_layout()
plt.show()

# 클러스터별 국가 수 출력
cluster_counts = pivot_df['Cluster'].value_counts().sort_index()
print("클러스터별 국가 수:")
print(cluster_counts)

# 클러스터별 샘플 국가 출력
for cluster in sorted(pivot_df['Cluster'].unique()):
    group = pivot_df[pivot_df['Cluster'] == cluster]
    sample = group.sample(n=min(5, len(group)), random_state=cluster)
    print(f"\n[Cluster {cluster}] (총 {len(group)}개 국가)")
    print(sample.index.tolist())

# --- 6. 클러스터별 주요 목적 시각화 (Treemap) ---
for cluster in sorted(pivot_df['Cluster'].unique()):
    countries_in_cluster = pivot_df[pivot_df['Cluster'] == cluster].index
    cluster_df = df_filtered[df_filtered['RecipientName'].isin(countries_in_cluster)]
    top_purposes = cluster_df.groupby('PurposeName')['USD_Disbursement'].sum().sort_values(ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    squarify.plot(sizes=top_purposes.values, label=top_purposes.index, alpha=0.8)
    plt.title(f"Cluster {cluster} – 주요 목적별 비중 (Top 10, 'Sectors not specified' 제외)")
    plt.axis('off')
    plt.show()

# --- 7. 클러스터별 상위 목적 및 국가 구조 시각화 (Sunburst) ---
df_clustered = df_filtered[df_filtered['RecipientName'].isin(pivot_df.index)].copy()
df_clustered['Cluster'] = df_clustered['RecipientName'].map(pivot_df['Cluster'])

sunburst_data = (
    df_clustered.groupby(['Cluster', 'PurposeName', 'RecipientName'])['USD_Disbursement']
    .sum()
    .reset_index()
)

top_n = 10
filtered_data = pd.DataFrame()
for cluster in sunburst_data['Cluster'].unique():
    subset = sunburst_data[sunburst_data['Cluster'] == cluster]
    top_subset = subset.sort_values(by='USD_Disbursement', ascending=False).head(top_n)
    filtered_data = pd.concat([filtered_data, top_subset])

fig = px.sunburst(filtered_data,
                  path=['Cluster', 'PurposeName', 'RecipientName'],
                  values='USD_Disbursement',
                  color='Cluster',
                  title=f"클러스터별 상위 {top_n}개 목적 및 수혜국 구조 ('Sectors not specified' 제외)")
fig.show()

# --- 8. 클러스터별 주요 목적 TOP 3 및 국가 목록 출력 ---
for cluster in sorted(pivot_df['Cluster'].unique()):
    countries = pivot_df[pivot_df['Cluster'] == cluster].index
    df_c = df_filtered[df_filtered['RecipientName'].isin(countries)]

    top_purposes = df_c.groupby('PurposeName')['USD_Disbursement'].sum().sort_values(ascending=False).head(3)
    print(f"\n🔹 Cluster {cluster} — 국가 수: {len(countries)}개")
    print("Top 3 목적:")
    for purpose, amount in top_purposes.items():
        print(f"  - {purpose}: ${amount:,.0f}")
    print("샘플 국가:")
    print(", ".join(countries[:10]) + (" ..." if len(countries) > 10 else ""))

# --- 9. 국가-목적 네트워크 시각화 (NetworkX) ---
G = nx.Graph()
df_clustered = df[df['RecipientName'].isin(pivot_df.index)].copy()
df_clustered['Cluster'] = df_clustered['RecipientName'].map(pivot_df['Cluster'])

top_purposes = df_clustered['PurposeName'].value_counts().head(20).index
df_sub = df_clustered[df_clustered['PurposeName'].isin(top_purposes)]

for _, row in df_sub.iterrows():
    country = row['RecipientName']
    purpose = row['PurposeName']
    G.add_node(country, type='country', cluster=row['Cluster'])
    G.add_node(purpose, type='purpose')
    G.add_edge(country, purpose, weight=row['USD_Disbursement'])

node_colors = ['C' + str(data['cluster']) if data.get('type') == 'country' else 'lightgray' for node, data in G.nodes(data=True)]

plt.figure(figsize=(14, 10))
pos = nx.spring_layout(G, k=0.35)
nx.draw(G, pos, with_labels=True, node_color=node_colors, font_size=7, edge_color='lightblue')
plt.title("국가-목적 네트워크 (클러스터 색상)")
plt.show()
