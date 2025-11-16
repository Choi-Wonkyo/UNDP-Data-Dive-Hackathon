## VIF (다중공산성) 체크

selected_vars = ['dln_oda_health_lag1', 'dln_oda_health_lag2', 'dln_oda_edu_lag0', 'dln_oda_infra_lag0', 'dln_oda_gov_lag1', 'dln_oda_social_env_lag0', 'ge', 'rq', 'va']
X = df[selected_vars]
vif = pd.DataFrame({'Variable': X.columns,
                    'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]})

df = pd.read_csv("/content/drive/MyDrive/딥러닝프로젝트/UNDP Data Dive 해커톤/2차 데이터/df_master_sec_lag2.csv")

# 1️⃣ WGI 변수 6개 선택
wgi_vars = ['ge', 'rq', 'va']

# 2️⃣ 표준화
X_std = StandardScaler().fit_transform(df[wgi_vars])

# 3️⃣ PCA (1개 주성분)
pca = PCA(n_components=1)
df['Gov_PC1'] = pca.fit_transform(X_std)

# 4️⃣ 설명된 분산 비율 확인
explained_var = pca.explained_variance_ratio_[0]
print(f"Governance_PC1 explains {explained_var*100:.2f}% of variance")

# 5️⃣ 새 변수 포함하여 DML / LASSO 재수행
new_df = df.drop(columns=wgi_vars)

selected_vars = ['dln_oda_health_lag1', 'dln_oda_health_lag2', 'dln_oda_edu_lag0', 'dln_oda_infra_lag0', 'dln_oda_gov_lag1', 'dln_oda_social_env_lag0', 'rq']
X = df[selected_vars]
vif = pd.DataFrame({'Variable': X.columns,
                    'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]})

vif


## OLS-HAC 재추정**

import statsmodels.api as sm

# X, Y 설정
y = df['d_lifeexp']  # 종속변수
X = df[['dln_oda_health_lag1', 'dln_oda_health_lag2',
        'dln_oda_edu_lag0', 'dln_oda_infra_lag0',
        'dln_oda_gov_lag1', 'dln_oda_social_env_lag0',
        'rq']]

X = sm.add_constant(X)

# OLS-HAC 추정
model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags':2})

print(model.summary())

# 검증 단계

**Durbin-Watson, ACF/Ljung-Box**

# 잔차
resid = model.resid

# ACF plot
sm.graphics.tsa.plot_acf(resid, lags=10)
plt.title("Residual ACF (Life Expectancy Model)")
plt.show()

# Ljung-Box test
lb = acorr_ljungbox(resid, lags=[1,2,3], return_df=True)
print(lb)

**Q-Q Plot (정규성 확인)**

sm.qqplot(resid, line='45')
plt.title("Q-Q Plot of Residuals")
plt.show()

resid.hist(bins=6)
plt.title("Histogram of Residuals")

**Observed vs Fitted (Life Expectancy)**

plt.plot(y, label='Observed')
plt.plot(model.fittedvalues, label='Fitted', linestyle='--')
plt.legend()
plt.title("Observed vs Fitted (Life Expectancy)")
plt.show()
