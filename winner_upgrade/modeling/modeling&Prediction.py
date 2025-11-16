## 표준화 계수 Bar Plot

# 종속변수와 설명변수 이름
y_col = 'd_lifeexp'
x_cols = [
    'dln_oda_health_lag1',
    'dln_oda_health_lag2',
    'dln_oda_edu_lag0',
    'dln_oda_infra_lag0',
    'dln_oda_gov_lag1',
    'dln_oda_social_env_lag0',
    'rq'  # 거버넌스는 바차트에는 포함, IRF에는 제외(정책 시차 구조가 아님)
]

# --- 표준화 계수 Bar Plot ---
# beta_std = beta * (std(X) / std(y))
params = model.params.drop('const')  # 상수 제외
std_X = df[x_cols].std()
std_y = df[y_col].std()
beta_std = params.reindex(x_cols) * (std_X / std_y)

# 보기 좋게 |표준화계수| 기준 내림차순 정렬
beta_std_sorted = beta_std.reindex(beta_std.abs().sort_values(ascending=False).index)

plt.figure(figsize=(8,5))
beta_std_sorted.plot(kind='bar')
plt.axhline(0, linestyle='--')
plt.title('Standardized Coefficients (Life Expectancy Model)')
plt.ylabel('Standardized β')
plt.tight_layout()
plt.show()


## IRF(누적효과) With 95% CI

# === 0) 준비: 계수와 공분산행렬(상수항 제외) ===
beta = model.params.drop('const', errors='ignore')
V = model.cov_params().loc[beta.index, beta.index]  # HAC cov

# 편의상 필요한 계수명을 변수로
HN1 = 'dln_oda_health_lag1'
HN2 = 'dln_oda_health_lag2'
SE0 = 'dln_oda_social_env_lag0'
IF0 = 'dln_oda_infra_lag0'
ED0 = 'dln_oda_edu_lag0'
GV1 = 'dln_oda_gov_lag1'

# 해당 계수가 없으면 0으로 처리할 수 있도록 인덱스 매핑
names = list(beta.index)
name_to_pos = {n:i for i,n in enumerate(names)}

def L_vec(terms):
    """terms: [(name, weight), ...]  → beta 순서에 맞는 L 벡터 반환"""
    L = np.zeros(len(beta))
    for n, w in terms:
        if n in name_to_pos:
            L[name_to_pos[n]] += w
        # 없으면 자동으로 0 (무시)
    return L

def irf_point_se(terms_list):
    """
    terms_list: 특정 horizon에서의 누적효과를 이루는 [(name, weight), ...]
    return: (estimate, se)
    """
    L = L_vec(terms_list)
    est = float(L @ beta.values)
    var = float(L @ V.values @ L)
    se = np.sqrt(max(var, 0))
    return est, se

# === 1) Horizon별 IRF 추정치 및 95% CI 계산 ===
horizons = [0,1,2]
result = { 'HEALTH':[], 'SOCIAL_ENV':[], 'INFRA':[], 'EDU':[], 'GOV':[] }
ci_low = {k:[] for k in result}
ci_high= {k:[] for k in result}

z = 1.96  # 95% CI

for h in horizons:
    # HEALTH: [0, β1, β1+β2]
    if h == 0:
        terms = []
    elif h == 1:
        terms = [(HN1, 1)]
    else:  # h==2
        terms = [(HN1, 1), (HN2, 1)]
    est, se = irf_point_se(terms)
    for d in (result, ci_low, ci_high): pass
    result['HEALTH'].append(est)
    ci_low['HEALTH'].append(est - z*se)
    ci_high['HEALTH'].append(est + z*se)

    # SOCIAL_ENV: [β0, β0, β0]
    est, se = irf_point_se([(SE0, 1)])
    result['SOCIAL_ENV'].append(est); ci_low['SOCIAL_ENV'].append(est - z*se); ci_high['SOCIAL_ENV'].append(est + z*se)

    # INFRA: [β0, β0, β0]
    est, se = irf_point_se([(IF0, 1)])
    result['INFRA'].append(est); ci_low['INFRA'].append(est - z*se); ci_high['INFRA'].append(est + z*se)

    # EDU: [β0, β0, β0]
    est, se = irf_point_se([(ED0, 1)])
    result['EDU'].append(est); ci_low['EDU'].append(est - z*se); ci_high['EDU'].append(est + z*se)

    # GOV: [0, β1, β1]
    if h == 0:
        terms = []
    else:
        terms = [(GV1, 1)]
    est, se = irf_point_se(terms)
    result['GOV'].append(est); ci_low['GOV'].append(est - z*se); ci_high['GOV'].append(est + z*se)

# === 2) 플로팅: 라인 + 95% CI 리본 ===
plt.figure(figsize=(9,5))
for k, color in zip(['HEALTH','SOCIAL_ENV','INFRA','EDU','GOV'],
                    ['C0','C1','C2','C3','C4']):
    y  = result[k]
    lo = ci_low[k]
    hi = ci_high[k]
    plt.plot(horizons, y, marker='o', label=k, color=color)
    plt.fill_between(horizons, lo, hi, alpha=0.20, step='mid', color=color)

plt.axhline(0, linestyle='--', linewidth=1)
plt.xlabel('Horizon (years)')
plt.ylabel('Cumulative Effect (Σβ)')
plt.title('Cumulative Effects by Sector with 95% CI (HAC)')
plt.legend(ncol=3)
plt.tight_layout()
plt.show()


sig = {}
for k in ['HEALTH','SOCIAL_ENV','INFRA','EDU','GOV']:
    sig[k] = []
    for lo, hi in zip(ci_low[k], ci_high[k]):
        sig[k].append( (lo>0) or (hi<0) )  # 95% CI가 0을 완전히 벗어나면 True

plt.figure(figsize=(9,5))
colors = {'HEALTH':'C0','SOCIAL_ENV':'C1','INFRA':'C2','EDU':'C3','GOV':'C4'}
for k in colors:
    y  = result[k]; lo = ci_low[k]; hi = ci_high[k]
    plt.plot(horizons, y, marker='o', label=k, color=colors[k])
    plt.fill_between(horizons, lo, hi, alpha=0.20, step='mid', color=colors[k])
    # 별표 표시
    for h, yv, ok in zip(horizons, y, sig[k]):
        if ok:
            plt.text(h, yv, '★', ha='center', va='bottom', fontsize=11, color=colors[k])

plt.axhline(0, linestyle='--', linewidth=1)
plt.xlabel('Horizon (years)'); plt.ylabel('Cumulative Effect (Σβ)')
plt.title('Cumulative Effects with 95% CI (★ = statistically significant)')
plt.legend(ncol=3); plt.tight_layout(); plt.show()


**정책 시나리오형 예측 모델**

# --------------------------------------------------------------------
# 0) 준비: model(OLS-HAC 결과)이 메모리에 있다고 가정
# --------------------------------------------------------------------
beta_all = model.params.drop('const', errors='ignore')
V_all    = model.cov_params().loc[beta_all.index, beta_all.index]

# 파라미터 명 단축
HN1 = 'dln_oda_health_lag1'
HN2 = 'dln_oda_health_lag2'
ED0 = 'dln_oda_edu_lag0'
IF0 = 'dln_oda_infra_lag0'
GV1 = 'dln_oda_gov_lag1'
SE0 = 'dln_oda_social_env_lag0'
RQ  = 'rq'

# 이름 ↔ 인덱스
name2pos = {n:i for i,n in enumerate(beta_all.index)}

def L_vec(terms):
    """terms: [(param_name, weight), ...] → beta 순서에 맞는 L 벡터"""
    L = np.zeros(len(beta_all))
    for n, w in terms:
        if n in name2pos:
            L[name2pos[n]] += w
    return L

def linear_est_se(terms):
    """선형결합 추정치/표준오차 (HAC 공분산)"""
    L   = L_vec(terms)
    est = float(L @ beta_all.values)
    var = float(L @ V_all.values @ L)
    se  = np.sqrt(max(var, 0))
    return est, se

def policy_scenario_predict(
    oda_rates=None,
    rq_delta=0.0,
    rq_persistent=True,
    return_cumulative=True,
    ci=0.95,
):
    """
    oda_rates: dict with keys in {'HEALTH','SOCIAL_ENV','INFRA','EDU','GOV'}
               values = 증가율(예: 0.10 = +10%)
    rq_delta:  RQ의 절대 변화량 (예: 표준화 z-score라면 +0.2 등)
    rq_persistent: True면 H0,H1,H2 모두에 RQ 변화 반영(지속개입 가정). False면 H0에만.
    return_cumulative: 누적효과(Σ)까지 반환할지 여부
    ci: 신뢰구간 (0.95, 0.90 등)
    """
    oda_rates = oda_rates or {}
    log_inc = {k: np.log(1 + max(0.0, float(v))) for k, v in oda_rates.items()}
    z = {0.90:1.64, 0.95:1.96, 0.99:2.58}.get(ci, 1.96)

    horizons = [0,1,2]
    inst_est, inst_lo, inst_hi = [], [], []

    for h in horizons:
        terms = []

        # Horizon별 해당 lag의 선형결합 구성
        if h == 0:
            # lag0: SE0, IF0, ED0
            if 'SOCIAL_ENV' in log_inc: terms.append((SE0, log_inc['SOCIAL_ENV']))
            if 'INFRA'      in log_inc: terms.append((IF0, log_inc['INFRA']))
            if 'EDU'        in log_inc: terms.append((ED0, log_inc['EDU']))
            # RQ
            if rq_delta and (rq_persistent or (not rq_persistent and h==0)):
                terms.append((RQ, rq_delta))

        elif h == 1:
            # lag1: HN1, GV1
            if 'HEALTH' in log_inc: terms.append((HN1, log_inc['HEALTH']))
            if 'GOV'    in log_inc: terms.append((GV1, log_inc['GOV']))
            if rq_delta and rq_persistent:
                terms.append((RQ, rq_delta))

        elif h == 2:
            # lag2: HN2
            if 'HEALTH' in log_inc: terms.append((HN2, log_inc['HEALTH']))
            if rq_delta and rq_persistent:
                terms.append((RQ, rq_delta))

        est, se = linear_est_se(terms) if terms else (0.0, 0.0)
        inst_est.append(est)
        inst_lo.append(est - z*se)
        inst_hi.append(est + z*se)

    out = pd.DataFrame({
        'horizon': horizons,
        'instantaneous': inst_est,
        'inst_lo': inst_lo,
        'inst_hi': inst_hi
    })

    if return_cumulative:
        out['cumulative'] = out['instantaneous'].cumsum()
        # 누적 CI는 공분산 고려해야 정확하지만, 보수적으로 루트-합-분산 근사
        # (각 horizon의 선형결합을 개별로 다시 정의해 정확 계산도 가능)
        cum_lo, cum_hi = [], []
        cum_terms = []
        for h in horizons:
            # h까지의 모든 즉시효과 terms를 다시 재구성해 정확 CI 산출
            t_all = []
            for hh in range(h+1):
                if   hh==0:
                    if 'SOCIAL_ENV' in log_inc: t_all.append((SE0, log_inc['SOCIAL_ENV']))
                    if 'INFRA'      in log_inc: t_all.append((IF0, log_inc['INFRA']))
                    if 'EDU'        in log_inc: t_all.append((ED0, log_inc['EDU']))
                    if rq_delta and (rq_persistent or (not rq_persistent and hh==0)):
                        t_all.append((RQ, rq_delta))
                elif hh==1:
                    if 'HEALTH' in log_inc: t_all.append((HN1, log_inc['HEALTH']))
                    if 'GOV'    in log_inc: t_all.append((GV1, log_inc['GOV']))
                    if rq_delta and rq_persistent:
                        t_all.append((RQ, rq_delta))
                elif hh==2:
                    if 'HEALTH' in log_inc: t_all.append((HN2, log_inc['HEALTH']))
                    if rq_delta and rq_persistent:
                        t_all.append((RQ, rq_delta))
            est_c, se_c = linear_est_se(t_all) if t_all else (0.0, 0.0)
            cum_lo.append(est_c - z*se_c)
            cum_hi.append(est_c + z*se_c)

        out['cum_lo'] = cum_lo
        out['cum_hi'] = cum_hi

    return out

def plot_policy(out_df, title_suffix=''):
    fig, axes = plt.subplots(1, 2, figsize=(11,4), sharex=True)
    # Instantaneous
    ax = axes[0]
    ax.plot(out_df['horizon'], out_df['instantaneous'], marker='o')
    ax.fill_between(out_df['horizon'], out_df['inst_lo'], out_df['inst_hi'], alpha=0.25, step='mid')
    ax.axhline(0, linestyle='--', linewidth=1)
    ax.set_title(f'Instantaneous Effect {title_suffix}')
    ax.set_xlabel('Horizon (years)'); ax.set_ylabel('Δ Life Expectancy')

    # Cumulative
    ax = axes[1]
    ax.plot(out_df['horizon'], out_df['cumulative'], marker='o')
    ax.fill_between(out_df['horizon'], out_df['cum_lo'], out_df['cum_hi'], alpha=0.25, step='mid')
    ax.axhline(0, linestyle='--', linewidth=1)
    ax.set_title(f'Cumulative Effect {title_suffix}')
    ax.set_xlabel('Horizon (years)'); ax.set_ylabel('Σ Δ Life Expectancy')

    plt.tight_layout(); plt.show()

# 예: 보건 +10%, 사회/환경 +10%, (다른 부문 0%), RQ 변화 없음
scen = {'HEALTH':0.10, 'SOCIAL_ENV':0.10}
out = policy_scenario_predict(oda_rates=scen, rq_delta=0.0, rq_persistent=True, ci=0.95)
display(out)         # 수치표
plot_policy(out, title_suffix='(HEALTH +10%, SOCIAL_ENV +10%)')

**토네이도(감도) 차트 – 2년차 누적효과에 대한 민감도**

grid = [0.05, 0.10, 0.20, 0.50]
rows = []
for r in grid:
    out = policy_scenario_predict({'HEALTH':r}, rq_delta=0.0)
    rows.append(['HEALTH', r, out.loc[out['horizon']==2,'cumulative'].values[0]])
for r in grid:
    out = policy_scenario_predict({'SOCIAL_ENV':r}, rq_delta=0.0)
    rows.append(['SOCIAL_ENV', r, out.loc[out['horizon']==2,'cumulative'].values[0]])
sens = pd.DataFrame(rows, columns=['Sector','Rate','CumEffect_y2']).sort_values('CumEffect_y2', ascending=False)

plt.figure(figsize=(6,4))
colors = ['#1f77b4' if s=='HEALTH' else '#ff7f0e' for s in sens['Sector']]
plt.barh(sens['Sector'] + ' +' + (sens['Rate']*100).astype(int).astype(str) + '%',
         sens['CumEffect_y2'], color=colors)
plt.axvline(0, color='k', lw=1)
plt.xlabel('Predicted Δ Life Expectancy (2-year cumulative)')
plt.title('Policy Sensitivity: ODA Increase vs Life Expectancy (+2yr)')
plt.tight_layout()
plt.show()
