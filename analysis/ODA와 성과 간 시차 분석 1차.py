import pandas as pd
import numpy as np
from functools import reduce
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# 1. 데이터 로드 및 리네이밍
# ----------------------------

def load_and_rename():
    # 파일 경로는 필요에 따라 변수화해도 됨
    경제 = pd.read_csv("외부데이터_경제지표.csv")
    보건 = pd.read_csv("외부데이터_보건지표.csv")
    교육 = pd.read_csv("외부데이터_교육지표.csv")
    빈곤 = pd.read_csv("외부데이터_빈곤_사회복지_지표.csv")
    생산 = pd.read_csv("외부데이터_생산지표.csv")
    환경 = pd.read_csv("외부데이터_환경기후_지표.csv")

    # 컬럼 이름 일관화 / 해석을 쉽게 하기 위한 접두사 부여
    교육 = 교육.rename(columns={
        'primary_enrollment_rate': '교육_초등학교 순취학률',
        'primary_completion_rate': '교육_초등학교 이수율'
    })

    보건 = 보건.rename(columns={
        'infant_mortality': '보건_영아 사망률',
        'life_expectancy': '보건_기대 수명',
        'neonatal_mortality': '보건_신생아 사망률'
    })

    경제 = 경제.rename(columns={
        'gdp_per_capita': '경제_1인당 GDP'
    })

    빈곤 = 빈곤.rename(columns={
        'poverty_percent': '빈곤 및 사회복지_빈곤율',
        'gni_per_capita': '빈곤 및 사회복지_1인당 GNI',
        'Hospital beds': '빈곤 및 사회복지_의료 접근성'
    })

    생산 = 생산.rename(columns={
        'services_value_pct_gdp': '생산_서비스업 부가가치',
        'manufacturing_pct_gdp': '생산_제조업 부가가치',
        'crop_prod_index': '생산_농작물 생산지수',
        'livestock_prod_index': '생산_가축 생산지수'
    })

    환경 = 환경.rename(columns={
        'co2_per_capita': '환경/기후_1인당 CO2 배출량',
        'renewable_energy_pct': '환경/기후_재생에너지 비중',
        'pm25_concentration': '환경/기후_대기오염 지수'
    })

    return {
        '교육': 교육,
        '보건': 보건,
        '경제': 경제,
        '빈곤': 빈곤,
        '생산': 생산,
        '환경': 환경
    }

# ----------------------------
# 2. 정렬 및 병합 공통 함수
# ----------------------------

def standardize(df):
    return df.sort_values(by=['year', 'country']).reset_index(drop=True)

def merge_indicators(indicator_dfs):
    # 외부 지표들을 year, country 기준 outer 조인
    dfs = [standardize(df) for df in indicator_dfs]
    merged = reduce(lambda left, right: pd.merge(left, right, on=['year', 'country'], how='outer'), dfs)
    # 기본 정리: year, country 먼저
    cols = ['year', 'country'] + [c for c in merged.columns if c not in ['year', 'country']]
    merged = merged[cols]
    return merged

# ----------------------------
# 3. 지표 변화율 계산
# ----------------------------

def compute_change_rates(df, lag=2, negative_indicators=None):
    """
    각 국가별로 lag년 후 변화율 ((t+lag - t)/t *100)을 계산.
    negative_indicators: 감소할수록 좋은 항목 리스트 (부호 반전)
    """
    df = df.sort_values(['country', 'year']).reset_index(drop=True)
    value_cols = [c for c in df.columns if c not in ['year', 'country']]

    change_df = df.copy()
    # 퍼센트 변화: (future - current)/current * 100
    change_df[value_cols] = (
        df.groupby('country')[value_cols]
          .apply(lambda g: (g.shift(-lag) - g) / g * 100)
          .reset_index(drop=True)
    )

    if negative_indicators:
        for col in negative_indicators:
            if col in change_df.columns:
                change_df[col] = -change_df[col]

    # 이름에 접미사 붙이기
    renamed = {}
    for col in value_cols:
        suffix = '_감소율' if (negative_indicators and col in negative_indicators) else '_증가율'
        renamed[col] = f"{col}{suffix}"
    change_df = change_df.rename(columns=renamed)

    # year/country 유지
    change_df[['year', 'country']] = df[['year', 'country']]
    return change_df

# ----------------------------
# 4. 국가명 정규화 (ODA <-> 성과 매칭용)
# ----------------------------

def normalize_country_names(df, column='country', mapping=None):
    df = df.copy()
    df[column] = df[column].str.lower().str.strip()
    if mapping:
        df[column] = df[column].replace(mapping)
    return df

# ----------------------------
# 5. Lagged correlation analysis
# ----------------------------

def lagged_correlation(oda_df, performance_df, field_targets, lags=(1,2,3), min_samples=6):
    """
    oda_df: ODA input data with columns ['year','country', <field columns>]
    performance_df: performance metrics (change rates) with ['year','country', <target columns>]
    field_targets: dict mapping oda field name -> list of corresponding performance target columns
    """
    # Normalize names
    oda_df = oda_df.rename(columns={'Year':'year', 'Country':'country'})
    oda_df = normalize_country_names(oda_df, 'country')
    performance_df = performance_df.rename(columns={'year':'year', 'country':'country'})
    performance_df = normalize_country_names(performance_df, 'country')

    results = []
    for lag in lags:
        for oda_field, target_list in field_targets.items():
            if oda_field not in oda_df.columns:
                continue
            oda_lagged = oda_df[['year', 'country', oda_field]].copy()
            for target_col in target_list:
                target_lagged = performance_df[['year', 'country', target_col]].copy()
                target_lagged['year'] = target_lagged['year'] - lag  # shift performance backward

                merged = pd.merge(oda_lagged, target_lagged, on=['year', 'country'], how='inner')
                merged = merged.dropna(subset=[oda_field, target_col])
                if len(merged) >= min_samples:
                    corr, pval = pearsonr(merged[oda_field], merged[target_col])
                    results.append({
                        'lag': lag,
                        'oda_variable': oda_field,
                        'target_variable': target_col,
                        'correlation': corr,
                        'p_value': pval,
                        'n_samples': len(merged)
                    })
    return pd.DataFrame(results)

# ----------------------------
# 6. 실행 흐름
# ----------------------------

def main():
    # 1. Load & rename source indicator tables
    dfs = load_and_rename()
    merged_indicators = merge_indicators([
        dfs['교육'], dfs['보건'], dfs['생산'], dfs['경제'], dfs['빈곤'], dfs['환경']
    ])

    # 2. Save base merged performance indicators
    merged_indicators.to_csv("성과지표_원본.csv", index=False)
    print("✅ 성과지표_원본.csv 생성 완료")

    # 3. Compute change rates (lag=2, 예: 2년 뒤 변화), 부정적 지표는 부호 반전
    neg = [
        '보건_영아 사망률', '보건_신생아 사망률',
        '빈곤 및 사회복지_빈곤율', '환경/기후_대기오염 지수',
        '환경/기후_1인당 CO2 배출량'
    ]
    change_df = compute_change_rates(merged_indicators, lag=2, negative_indicators=neg)
    # 4. 제한된 기간(예: 2014~2021)만 남기기
    change_df = change_df[~change_df['year'].isin([2022, 2023])]
    change_df.to_csv("지표별_변화율.csv", index=False)

    # 5. 결측률 요약
    indicator_cols = [c for c in change_df.columns if c not in ['year', 'country']]
    missing_rate = change_df[indicator_cols].isnull().mean() * 100
    missing_summary = pd.DataFrame({
        '지표명': missing_rate.index,
        '결측률(%)': missing_rate.values
    }).sort_values('결측률(%)', ascending=False)
    missing_summary.to_csv("지표별_변화율_결측률요약.csv", index=False)

    # 6. Lagged correlation: ODA input과 performance change 간 대응
    oda = pd.read_csv("input_pivot.csv")  # ODA fields like '교육','보건',...
    # Field to target mapping (예시는 접두사 기준으로 자동 구성)
    field_targets = {
        '교육': [col for col in change_df.columns if col.startswith('교육')],
        '보건': [col for col in change_df.columns if col.startswith('보건')],
        '생산': [col for col in change_df.columns if col.startswith('생산')],
        '경제': [col for col in change_df.columns if col.startswith('경제')],
        '빈곤 및 사회복지': [col for col in change_df.columns if col.startswith('빈곤 및 사회복지')],
        '환경/기후': [col for col in change_df.columns if col.startswith('환경/기후')],
    }

    lag_corr_df = lagged_correlation(oda_df=oda, performance_df=change_df, field_targets=field_targets, lags=(1,2,3))
    lag_corr_df.to_csv("lag_correlation_summary.csv", index=False)

    # 7. 필터링 -> 유의미한 상관관계만 추출
    filtered = lag_corr_df[(lag_corr_df['p_value'] < 0.1) & (lag_corr_df['correlation'].abs() > 0.15)]
    filtered.to_csv("lag_correlation_significant.csv", index=False)
