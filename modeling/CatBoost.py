import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 파일 경로
input_path = '/content/drive/MyDrive/딥러닝프로젝트/UNDP Data Dive 해커톤/모델/input_final.csv'
output_path = '/content/drive/MyDrive/딥러닝프로젝트/UNDP Data Dive 해커톤/모델/y.csv'

# 데이터 로드 및 병합
input_df = pd.read_csv(input_path)
output_df = pd.read_csv(output_path)
output_df.rename(columns={'country': 'Country', 'year': 'Year'}, inplace=True)
merged_df = pd.merge(input_df, output_df, on=['Country', 'Year'], how='inner')

# 컬럼 목록 정의
oda_cols = ['경제_1인당 GDP_ODA', '보건_영아 사망률_ODA', '보건_신생아 사망률_ODA', '보건_기대 수명_ODA',
            '교육_초등학교 순취학률_ODA', '교육_초등학교 이수율_ODA', '빈곤 및 사회복지_빈곤율_ODA',
            '빈곤 및 사회복지_1인당 GNI_ODA', '빈곤 및 사회복지_의료 접근성_ODA', '환경/기후_재생 에너지 사용률_ODA',
            '환경/기후_대기오염 지수_ODA', '환경/기후_CO2 배출량_ODA', '생산_서비스업 부가가치_ODA',
            '생산_제조업 부가가치_ODA', '생산_농작물 생산지수_ODA', '생산_가축 생산지수_ODA']

shock_cols = ['new_cases_per_million', 'new_deaths_per_million', 'stringency_index',
              'natural_disaster_count', 'log_battle_deaths']

context_cols = ['연간 물가상승률 (CPI 기준, %)', '인구 1,000명당 간호사 및 조산사 수', 'GDP 대비 보건 지출 비율 (%)',
                'GDP 대비 공교육 지출 비율 (%)', '정부예산 중 교육비 비중', '중등학교 총등록률', '정부효율성', '실업률',
                '소득 불평등', '정치 안정성', '도시화율', '환경적 지속가능성', '인구밀도', '산림면적 비율',
                '농업 부가가치 비중', '산업 부가가치 비중', '작물 생산 지수', '식량 생산 지수']

input_features = oda_cols + shock_cols + context_cols
output_targets = output_df.columns.drop(['Country', 'Year'])

x = merged_df[['Country', 'Year'] + input_features].copy()
y = merged_df[['Country', 'Year'] + list(output_targets)].copy()

results = []

# 결측 평균 대체 함수
def impute_mean(df):
    cols = df.columns.difference(['Year'])
    df_imputed = df.copy()
    df_imputed[cols] = df_imputed[cols].fillna(df_imputed[cols].mean())
    return df_imputed

for target in output_targets:
    x_train = x[(x['Year'] >= 2014) & (x['Year'] <= 2021)].drop(columns=['Country'])
    y_train = y[(y['Year'] >= 2014) & (y['Year'] <= 2021)][target]

    x_test1 = x[(x['Year'] >= 2018) & (x['Year'] <= 2019)].drop(columns=['Country'])
    y_test1 = y[(y['Year'] >= 2018) & (y['Year'] <= 2019)][target]

    x_test2 = x[(x['Year'] >= 2020) & (x['Year'] <= 2021)].drop(columns=['Country'])
    y_test2 = y[(y['Year'] >= 2020) & (y['Year'] <= 2021)][target]

    x_train_imputed = impute_mean(x_train).drop(columns=['Year'])
    x_test1_imputed = impute_mean(x_test1).drop(columns=['Year'])
    x_test2_imputed = impute_mean(x_test2).drop(columns=['Year'])

    y_train_imputed = y_train.fillna(y_train.mean())
    y_test1_imputed = y_test1.fillna(y_test1.mean())
    y_test2_imputed = y_test2.fillna(y_test2.mean())

    model = CatBoostRegressor(
        iterations=1000,
        depth=8,
        learning_rate=0.05,
        l2_leaf_reg=3,
        loss_function='RMSE',
        verbose=0,
        random_seed=42
    )
    model.fit(x_train_imputed, y_train_imputed)

    y_pred1 = model.predict(x_test1_imputed)
    y_pred2 = model.predict(x_test2_imputed)

    rmse1 = mean_squared_error(y_test1_imputed, y_pred1, squared=False)
    r2_1 = r2_score(y_test1_imputed, y_pred1)
    rmse2 = mean_squared_error(y_test2_imputed, y_pred2, squared=False)
    r2_2 = r2_score(y_test2_imputed, y_pred2)

    print(f"[Target: {target}] | 18~19 RMSE: {rmse1:.4f}, R²: {r2_1:.4f} | 20~21 RMSE: {rmse2:.4f}, R²: {r2_2:.4f}")

    results.append({
        "target": target,
        "test1_rmse": rmse1,
        "test1_r2": r2_1,
        "test2_rmse": rmse2,
        "test2_r2": r2_2,
    })

# 전체 성능 요약
summary_df = pd.DataFrame(results).sort_values(by="test1_r2", ascending=False)
print("\n전체 성능 요약:")
print(summary_df.to_string(index=False))

# 마지막 모델 기준 피처 중요도 시각화
feature_names = x_train_imputed.columns
importances = model.get_feature_importance()
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

print("\nFeature Importance (상위 20):")
print(importance_df.head(20))
