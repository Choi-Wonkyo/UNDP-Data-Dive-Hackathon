import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_squared_error, r2_score

# 파일 불러오기
input_df = pd.read_csv('/content/drive/MyDrive/.../input_final.csv')
output_df = pd.read_csv('/content/drive/MyDrive/.../y.csv')

# 컬럼명 정리
def clean_colnames(df):
    df.columns = (
        df.columns.str.strip()
                  .str.replace(r"[^\w]", "_", regex=True)
                  .str.replace(r"_+", "_", regex=True)
                  .str.rstrip('_')
                  .str.lower()
    )
    return df

input_df = clean_colnames(input_df)
output_df = clean_colnames(output_df)

# 피처 및 타겟 컬럼 지정
oda_cols = [...]; shock_cols = [...]; context_cols = [...]
features = oda_cols + shock_cols + context_cols
targets = output_df.columns.drop(['country', 'year'])

# 연도별 분할
def filter_years(df, years):
    return df[df['year'].isin(years)].reset_index(drop=True)

X_train = filter_years(input_df, range(2014, 2018))
X_test1 = filter_years(input_df, [2018, 2019])
X_test2 = filter_years(input_df, [2020, 2021])

y_train = filter_years(output_df, range(2014, 2018))
y_test1 = filter_years(output_df, [2018, 2019])
y_test2 = filter_years(output_df, [2020, 2021])

# 병합
def merge_xy(x, y):
    return pd.merge(x, y, on=['country', 'year'], how='inner')

train = merge_xy(X_train, y_train)
test1 = merge_xy(X_test1, y_test1)
test2 = merge_xy(X_test2, y_test2)

X_train_final = train[features].fillna(train[features].median())
X_test1_final = test1[features].fillna(train[features].median())
X_test2_final = test2[features].fillna(train[features].median())

y_train_final = train[targets].fillna(train[targets].median())
y_test1_final = test1[targets].fillna(train[targets].median())
y_test2_final = test2[targets].fillna(train[targets].median())

models = {}
results = {}

for target in targets:
    print(f'\n[Target: {target}]')
    
    # 결측 제거
    mask_tr = ~y_train_final[target].isna()
    X_tr = X_train_final[mask_tr]
    y_tr = y_train_final.loc[mask_tr, target]

    model = LGBMRegressor(
        objective='regression',
        boosting_type='gbdt',
        random_state=42,
        n_estimators=1000,
        verbose=-1
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_tr, y_tr)],
        eval_metric='rmse',
        callbacks=[early_stopping(50), log_evaluation(100)]
    )

    models[target] = model

    res = {}

    for label, X_te, y_te in [('test1', X_test1_final, y_test1_final), ('test2', X_test2_final, y_test2_final)]:
        mask_te = ~y_te[target].isna()
        if mask_te.sum() == 0:
            print(f"{label}: 평가 데이터 없음")
            res[f"{label}_rmse"] = None
            res[f"{label}_r2"] = None
        else:
            pred = model.predict(X_te[mask_te])
            true = y_te.loc[mask_te, target]
            rmse = mean_squared_error(true, pred) ** 0.5
            r2 = r2_score(true, pred)
            print(f"{label.upper()} RMSE: {rmse:.4f}, R²: {r2:.4f}")
            res[f"{label}_rmse"] = rmse
            res[f"{label}_r2"] = r2

    results[target] = res
