import pandas as pd
from scipy.stats import pearsonr

# ----------------------------
# 1. 데이터 로드
# ----------------------------
oda = pd.read_csv("/content/drive/MyDrive/딥러닝프로젝트/UNDP Data Dive 해커톤/input_pivot_2차.csv")
성과 = pd.read_csv("/content/drive/MyDrive/딥러닝프로젝트/UNDP Data Dive 해커톤/혜원 - output/지표별_변화율.csv")

# ----------------------------
# 2. 국가명 전처리 및 매핑
# ----------------------------
oda_countries = set(oda['Country'].str.lower().unique())
성과_countries = set(성과['country'].str.lower().unique())

# crs_data와 world data 국가명 맵핑
name_mapping = {
    'kyrgyz republic': 'kyrgyzstan',
    'cote d ivoire': "côte d'ivoire",
    "cote d'ivoire": "côte d'ivoire",
    'egypt, arab rep.': 'egypt',
    'iran, islamic rep.': 'iran',
    'st. vincent and the grenadines': 'saint vincent and the grenadines',
    "korea, dem. people's rep.": "democratic people's republic of korea",
    'congo, rep.': 'congo',
    'gambia, the': 'gambia',
    'lao pdr': "lao people's democratic republic",
    'st. lucia': 'saint lucia',
    'turkiye': 'türkiye',
    'venezuela, rb': 'venezuela',
    'yemen, rep.': 'yemen'
}

performance_df = 성과.copy()
performance_df['country'] = performance_df['country'].str.lower().str.strip()
performance_df['country'] = performance_df['country'].replace(name_mapping)

oda['Country'] = oda['Country'].str.lower().str.strip()
oda = oda.rename(columns={'Year': 'year', 'Country': 'country'})
성과 = performance_df.rename(columns={'year': 'year', 'country': 'country'})

# ----------------------------
# 3. ODA → 성과 지표 매핑
# ----------------------------
oda_to_perf = {
    '교육_초등학교 순취학률': '교육_초등학교 순취학률_증가율',
    '교육_초등학교 이수율': '교육_초등학교 이수율_증가율',
    '보건_영아 사망률': '보건_영아 사망률_감소율',
    '보건_신생아 사망률': '보건_신생아 사망률_감소율',
    '보건_기대 수명': '보건_기대 수명_증가율',
    '생산_서비스업 부가가치': '생산_서비스업 부가가치_증가율',
    '생산_제조업 부가가치': '생산_제조업 부가가치_증가율',
    '생산_농작물 생산지수': '생산_농작물 생산지수_증가율',
    '생산_가축 생산지수': '생산_가축 생산지수_증가율',
    '경제_1인당 GDP': '경제_1인당 GDP_증가율',
    '빈곤 및 사회복지_빈곤율': '빈곤 및 사회복지_빈곤율_감소율',
    '빈곤 및 사회복지_1인당 GNI': '빈곤 및 사회복지_1인당 GNI_증가율',
    '빈곤 및 사회복지_의료 접근성': '빈곤 및 사회복지_의료 접근성_증가율',
    '환경/기후_재생 에너지 사용률': '환경/기후_재생 에너지 사용률_증가율',
    '환경/기후_대기오염 지수': '환경/기후_대기오염 지수_감소율',
    '환경/기후_CO2 배출량': '환경/기후_1인당 co2 배출량_감소율'
}

# 특정 코드 제거 (예: 10003)
oda = oda[oda['country'] != '10003']

# ----------------------------
# 4. Lag 상관분석
# ----------------------------
results = []

for oda_col in oda.columns[3:]:  # 연도/국가 컬럼 제외
    perf_col = oda_to_perf.get(oda_col)
    if perf_col not in 성과.columns:
        continue

    for lag in [1, 2, 3]:  # 1~3년 시차
        oda_shifted = oda[['year', 'country', oda_col]].copy()
        oda_shifted['year'] = oda_shifted['year'] + lag  # 시차 적용

        perf_df = 성과[['year', 'country', perf_col]].copy()
        merged = pd.merge(oda_shifted, perf_df, on=['year', 'country'], how='inner')

        xy = merged[[oda_col, perf_col]].dropna()
        if len(xy) > 5:  # 최소 샘플 수
            r, p = pearsonr(xy[oda_col], xy[perf_col])
            results.append({
                'lag': lag,
                'oda_variable': oda_col,
                'target_variable': perf_col,
                'correlation': r,
                'p_value': p,
                'n_samples': len(xy)
            })

lag_correlation_df = pd.DataFrame(results)
lag_correlation_df = lag_correlation_df.sort_values(by=['oda_variable', 'lag']).reset_index(drop=True)
lag_correlation_df.to_csv('/content/drive/MyDrive/딥러닝프로젝트/UNDP Data Dive 해커톤/혜원 - output/lag_correlation_summary_2차(전체).csv', index=False)

# ----------------------------
# 5. 유의미 상관관계 및 best lag 추출
# ----------------------------
filtered_corr = lag_correlation_df[
    (lag_correlation_df['p_value'] < 0.2) & 
    (lag_correlation_df['correlation'].abs() > 0.05)
]

best_lags = (
    lag_correlation_df.loc[
        lag_correlation_df.groupby('target_variable')['correlation'].apply(lambda x: x.abs().idxmax())
    ]
    .reset_index(drop=True)
    .rename(columns={'lag': 'best_lag'})[
        ['target_variable', 'best_lag', 'correlation', 'p_value', 'n_samples']
    ]
)

best_lags.to_csv(
    "/content/drive/MyDrive/딥러닝프로젝트/UNDP Data Dive 해커톤/혜원 - output/best_lag_target.csv", 
    index=False
)
