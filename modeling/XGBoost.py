# 💡 XGBoost 기반 다변량 회귀 모델 파이프라인

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor


# ----------------------------
# 1. 데이터 전처리 및 정제
# ----------------------------

# 연도 형식 통일
x['Year'] = x['Year'].astype(int)
y['year'] = y['year'].astype(int)
y.rename(columns={'country': 'Country', 'year': 'Year'}, inplace=True)

# 결측값 처리 - 버전1: 0으로 채움
y_filled = y.copy()
y_filled.iloc[:, 2:] = y_filled.iloc[:, 2:].replace([np.inf, -np.inf], 0).fillna(0)

# 공통 키 (Year, Country) 기준 정렬 및 필터링
x_keys = set(zip(x['Year'], x['Country']))
y_keys = set(zip(y_filled['Year'], y_filled['Country']))
common_keys = x_keys & y_keys

x_filtered = x[x[['Year', 'Country']].apply(tuple, axis=1).isin(common_keys)].copy()
y_filtered = y_filled[y_filled[['Year', 'Country']].apply(tuple, axis=1).isin(common_keys)].copy()

X_filtered = x_filtered.reset_index(drop=True)
y_filtered = y_filtered.reset_index(drop=True)

# Label Encoding
X_filtered['Country'] = LabelEncoder().fit_transform(X_filtered['Country'])

# ----------------------------
# 2. 학습/테스트 데이터 분할
# ----------------------------

x_train = X_filtered[(X_filtered['Year'] >= 2014) & (X_filtered['Year'] <= 2021)]
y_train = y_filtered[(y_filtered['Year'] >= 2014) & (y_filtered['Year'] <= 2021)]

x_test1 = X_filtered[(X_filtered['Year'] >= 2018) & (X_filtered['Year'] <= 2019)]
y_test1 = y_filtered[(y_filtered['Year'] >= 2018) & (y_filtered['Year'] <= 2019)]

x_test2 = X_filtered[(X_filtered['Year'] >= 2020) & (X_filtered['Year'] <= 2021)]
y_test2 = y_filtered[(y_filtered['Year'] >= 2020) & (y_filtered['Year'] <= 2021)]

X_train = x_train.drop(columns=['Unnamed: 0'])
X_test1 = x_test1.drop(columns=['Unnamed: 0'])
X_test2 = x_test2.drop(columns=['Unnamed: 0'])

# 타겟 분리
y_train_targets = y_train.drop(columns=['Year', 'Country'])
y_test1_targets = y_test1.drop(columns=['Year', 'Country'])
y_test2_targets = y_test2.drop(columns=['Year', 'Country'])

# ----------------------------
# 3. 모델 학습 및 예측
# ----------------------------

model = MultiOutputRegressor(XGBRegressor(random_state=42, n_jobs=-1))
model.fit(X_train, y_train_targets)

pred_test1 = model.predict(X_test1)
pred_test2 = model.predict(X_test2)

# ----------------------------
# 4. 평가 함수 정의
# ----------------------------

def evaluate_each_target(y_true, y_pred, label, target_columns):
    print(f"\n[{label}]")
    for i, col in enumerate(target_columns):
        true_vals = y_true[:, i] if isinstance(y_true, np.ndarray) else y_true[col].values
        pred_vals = y_pred[:, i] if isinstance(y_pred, np.ndarray) else y_pred[col].values
        mse = mean_squared_error(true_vals, pred_vals)
        mae = mean_absolute_error(true_vals, pred_vals)
        print(f" - {col}: MSE = {mse:.4f}, MAE = {mae:.4f}")

# 평가 실행
y_columns = y_train_targets.columns.tolist()
evaluate_each_target(y_test1_targets, pred_test1, "Test 1 (정상기)", y_columns)
evaluate_each_target(y_test2_targets, pred_test2, "Test 2 (충격기)", y_columns)


# ----------------------------
# 5. 시각화
# ----------------------------

def plot_predictions(y_true, y_pred, years, label, target_columns):
    num_targets = len(target_columns)
    fig, axes = plt.subplots(nrows=num_targets, figsize=(10, 4 * num_targets), sharex=True)
    if num_targets == 1:
        axes = [axes]
    for i, col in enumerate(target_columns):
        true_vals = y_true[:, i] if isinstance(y_true, np.ndarray) else y_true[col].values
        pred_vals = y_pred[:, i] if isinstance(y_pred, np.ndarray) else y_pred[col].values
        axes[i].plot(years, true_vals, label='실제값', marker='o')
        axes[i].plot(years, pred_vals, label='예측값', marker='x')
        axes[i].set_title(f'{label} - {col}')
        axes[i].legend()
        axes[i].grid(True)
    plt.xlabel('연도')
    plt.tight_layout()
    plt.show()

years_test1 = y_test1['Year'].values
years_test2 = y_test2['Year'].values

plot_predictions(y_test1_targets, pred_test1, years_test1, "정상기", y_columns)
plot_predictions(y_test2_targets, pred_test2, years_test2, "충격기", y_columns)


# ----------------------------
# 6. SHAP 분석
# ----------------------------

def get_shap_values_for_target(estimator, X_data, feature_names, target_name):
    explainer = shap.Explainer(estimator)
    shap_values = explainer(X_data)
    shap_df = pd.DataFrame(shap_values.values, columns=feature_names)
    shap_mean = shap_df.abs().mean().sort_values(ascending=False)
    result_df = shap_mean.reset_index()
    result_df.columns = ['Feature', 'Mean(|SHAP|)']
    result_df['Target'] = target_name
    return result_df

feature_names = X_train.columns.tolist()
shap_result_list = []

for i, target in enumerate(y_train_targets.columns):
    est = model.estimators_[i]
    shap_df = get_shap_values_for_target(est, X_train, feature_names, target)
    print(f"\n🔍 {target}")
    print(shap_df.head(10))
    shap_result_list.append(shap_df.head(10))

shap_summary_all = pd.concat(shap_result_list, ignore_index=True)

# 시각화
for i, est in enumerate(model.estimators_):
    explainer = shap.Explainer(est)
    shap_values = explainer(X_train)
    shap.summary_plot(shap_values, X_train, feature_names=feature_names, show=True)
    shap.summary_plot(shap_values, X_train, feature_names=feature_names, plot_type="bar", max_display=10, show=True)


# ----------------------------
# 버전2: 시계열 보간
# ----------------------------

y_interpolated = (
    y.sort_values(['Country', 'Year'])
     .groupby('Country')
     .apply(lambda g: g.interpolate(method='linear', limit_direction='both'))
     .reset_index(drop=True)
)

# 교집합 추출 및 재정렬
x_keys = set(zip(x['Year'], x['Country']))
y_keys = set(zip(y_interpolated['Year'], y_interpolated['Country']))
common_keys = x_keys & y_keys

x_filt = x[x[['Year', 'Country']].apply(tuple, axis=1).isin(common_keys)].copy()
y_filt = y_interpolated[y_interpolated[['Year', 'Country']].apply(tuple, axis=1).isin(common_keys)].copy()

X_interp = x_filt.reset_index(drop=True)
y_interp = y_filt.reset_index(drop=True)
X_interp['Country'] = LabelEncoder().fit_transform(X_interp['Country'])

# 분할
X_tr2 = X_interp[(X_interp['Year'] >= 2014) & (X_interp['Year'] <= 2021)].drop(columns=['Unnamed: 0'])
y_tr2 = y_interp[(y_interp['Year'] >= 2014) & (y_interp['Year'] <= 2021)]

X_ts1 = X_interp[(X_interp['Year'] >= 2018) & (X_interp['Year'] <= 2019)].drop(columns=['Unnamed: 0'])
y_ts1 = y_interp[(y_interp['Year'] >= 2018) & (y_interp['Year'] <= 2019)]

X_ts2 = X_interp[(X_interp['Year'] >= 2020) & (X_interp['Year'] <= 2021)].drop(columns=['Unnamed: 0'])
y_ts2 = y_interp[(y_interp['Year'] >= 2020) & (y_interp['Year'] <= 2021)]

# 모델2
model2 = MultiOutputRegressor(XGBRegressor(random_state=42, n_jobs=-1))
model2.fit(X_tr2, y_tr2.drop(columns=['Year', 'Country']))

pred_ts1 = model2.predict(X_ts1)
pred_ts2 = model2.predict(X_ts2)

y_cols2 = y_tr2.drop(columns=['Year', 'Country']).columns.tolist()
evaluate_each_target(y_ts1.drop(columns=['Year', 'Country']), pred_ts1, "Test 1 (보간 y)", y_cols2)
evaluate_each_target(y_ts2.drop(columns=['Year', 'Country']), pred_ts2, "Test 2 (보간 y)", y_cols2)

# SHAP 버전2
feature_names2 = X_tr2.columns.tolist()
shap_result_list2 = []

for i, target in enumerate(y_cols2):
    est = model2.estimators_[i]
    shap_df = get_shap_values_for_target(est, X_tr2, feature_names2, target)
    print(f"\n🔍 {target}")
    print(shap_df.head(10))
    shap_result_list2.append(shap_df.head(10))

for i, est in enumerate(model2.estimators_):
    explainer = shap.Explainer(est)
    shap_values = explainer(X_tr2)
    shap.summary_plot(shap_values, X_tr2, feature_names=feature_names2, show=True)
    shap.summary_plot(shap_values, X_tr2, feature_names=feature_names2, plot_type="bar", max_display=10, show=True)
