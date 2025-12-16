import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from  PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import io
import pydeck as pdk
import torch
import torch.nn as nn
import time
import joblib
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import base64
import pickle
import joblib
import os

st.set_page_config(layout="wide")

# ====== 스타일 ======
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(
            135deg,
            #000000 10%,
            #004899 50%,
            #000000 100%
        );
    }

    .stApp,
    .stApp p,
    .stApp label,
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4,
    .stApp h5 {
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div {
        min-width: 170px !important;
        max-width: 170px !important;
        background-color: rgba(255,255,255,0.15);
        backdrop-filter: blur(6px);
    }

    button[data-testid="stPopoverButton"] {
        color: black !important;       /* 아이콘 색 검정 */
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        font-size: 20px !important;    /* 아이콘 크기 */
        cursor: pointer;
        margin-top: -6px !important;
        margin-left: 0px !important;   /* 제목과 간격 */
    }

    /* ===== Tooltip icon ===== */
    .tooltip-icon {
        width: 22px;
        height: 22px;
        cursor: pointer;
        margin-left: 8px;
    }

    /* ===== Overlay ===== */
    .tooltip-overlay {
        display: none;
        position: fixed;
        inset: 0;
        background: rgba(0,0,0,0.55);
        z-index: 9998;
    }
    
    .tooltip-box {
        display: none;
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 600px;
        max-height: 75vh;
        background: #0f172a;
        color: white;
        padding: 28px;
        border-radius: 16px;
        z-index: 9999;
        overflow-y: auto;
    }

    .tooltip-close {
        position: absolute;
        top: 10px;
        right: 16px;
        font-size: 22px;
        cursor: pointer;
    }

    /* ===== Global tweaks ===== */
    # img {
    #     display: block;
    #     margin-left: auto !important;
    #     margin-right: auto !important;
    # }

    p, label, div {
        font-size: 18px;
    }

    .stButton > button {
        background-color: #2563eb !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: 600;
    }

    .sidebar-logo {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
    }
    </style>

    <script>
    function toggleTooltip(id) {
        const box = document.getElementById(id);
        const overlay = document.getElementById(id + "-overlay");
    
        const visible = box.style.display === "block";
        box.style.display = visible ? "none" : "block";
        overlay.style.display = visible ? "none" : "block";
    }
    </script>
    """,
    unsafe_allow_html=True
)

# ====== 모델 로딩 ======
@st.cache_data
def load_model(path):
    return joblib.load(path)

model = load_model("Visualization/Model/lifeexp_policy_model_HAC.pkl")

# ====== HAC 기반 선형 조합 예측 함수 ======
beta_all = model.params.drop('const', errors='ignore')
V_all    = model.cov_params().loc[beta_all.index, beta_all.index]

HN1 = 'dln_oda_health_lag1'
HN2 = 'dln_oda_health_lag2'
ED0 = 'dln_oda_edu_lag0'
IF0 = 'dln_oda_infra_lag0'
GV1 = 'dln_oda_gov_lag1'
SE0 = 'dln_oda_social_env_lag0'
RQ  = 'rq'

name2pos = {n:i for i,n in enumerate(beta_all.index)}

def L_vec(terms):
    L = np.zeros(len(beta_all))
    for n, w in terms:
        if n in name2pos:
            L[name2pos[n]] += w
    return L

def linear_est_se(terms):
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
    oda_rates = oda_rates or {}
    log_inc = {}
    for k, v in oda_rates.items():
        v = float(v)
        if v <= -1:
            raise ValueError(f"{k} 감소율은 -100% 미만일 수 없습니다.")
        log_inc[k] = np.log(1 + v)
    z = {0.90:1.64, 0.95:1.96, 0.99:2.58}.get(ci, 1.96)

    horizons = [0,1,2]
    inst_est, inst_lo, inst_hi = [], [], []

    for h in horizons:
        terms = []
        if h == 0:
            if 'SOCIAL_ENV' in log_inc: terms.append((SE0, log_inc['SOCIAL_ENV']))
            if 'INFRA'      in log_inc: terms.append((IF0, log_inc['INFRA']))
            if 'EDU'        in log_inc: terms.append((ED0, log_inc['EDU']))
            if rq_delta and (rq_persistent or (not rq_persistent and h==0)):
                terms.append((RQ, rq_delta))
        elif h == 1:
            if 'HEALTH' in log_inc: terms.append((HN1, log_inc['HEALTH']))
            if 'GOV'    in log_inc: terms.append((GV1, log_inc['GOV']))
            if rq_delta and rq_persistent: terms.append((RQ, rq_delta))
        elif h == 2:
            if 'HEALTH' in log_inc: terms.append((HN2, log_inc['HEALTH']))
            if rq_delta and rq_persistent: terms.append((RQ, rq_delta))
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
        cum_lo, cum_hi = [], []
        for h in horizons:
            t_all = []
            for hh in range(h+1):
                if hh==0:
                    if 'SOCIAL_ENV' in log_inc: t_all.append((SE0, log_inc['SOCIAL_ENV']))
                    if 'INFRA'      in log_inc: t_all.append((IF0, log_inc['INFRA']))
                    if 'EDU'        in log_inc: t_all.append((ED0, log_inc['EDU']))
                    if rq_delta and (rq_persistent or (not rq_persistent and hh==0)):
                        t_all.append((RQ, rq_delta))
                elif hh==1:
                    if 'HEALTH' in log_inc: t_all.append((HN1, log_inc['HEALTH']))
                    if 'GOV'    in log_inc: t_all.append((GV1, log_inc['GOV']))
                    if rq_delta and rq_persistent: t_all.append((RQ, rq_delta))
                elif hh==2:
                    if 'HEALTH' in log_inc: t_all.append((HN2, log_inc['HEALTH']))
                    if rq_delta and rq_persistent: t_all.append((RQ, rq_delta))
            est_c, se_c = linear_est_se(t_all) if t_all else (0.0, 0.0)
            cum_lo.append(est_c - z*se_c)
            cum_hi.append(est_c + z*se_c)
        out['cum_lo'] = cum_lo
        out['cum_hi'] = cum_hi

    return out

# --- 시나리오 함수 ---
def run_scenario_for_dashboard(
    health_rate=0.0,
    social_rate=0.0,
    edu_rate=0.0,
    infra_rate=0.0,
    gov_rate=0.0,
    rq_delta=0.0,
    rq_persistent=True,
    ci=0.95,
):
    """
    대시보드/연구용 공통 시나리오 함수
    """
    oda_rates = {
        'HEALTH': health_rate,
        'SOCIAL_ENV': social_rate,
        'EDU': edu_rate,
        'INFRA': infra_rate,
        'GOV': gov_rate,
    }

    out = policy_scenario_predict(
        oda_rates=oda_rates,
        rq_delta=rq_delta,
        rq_persistent=rq_persistent,
        ci=ci,
    )
    return out


# ====== Dashboard 페이지 ======
def dashboard_page():
    st.title("🌍 Ethiopia ODA Impact Simulator: Life Expectancy")

    col1, col2 = st.columns([1.5, 1])

    with col2:
        header_col, _ = st.columns([1, 0.001])
        with header_col:
            # 텍스트와 popover 버튼을 flex로 한 줄 정렬
            st.markdown(
                """
                <div style="display:flex; align-items:center;">
                    <h3 style="margin:0;">ODA Weight</h3>
                    <div style="margin-left:6px;">
                """,
                unsafe_allow_html=True,
            )
            with st.popover("❓"):
                st.markdown(
                    """
                    ### ODA Sliders
                    - Adjust each slider to simulate percentage changes in ODA investment.
                    - The simulation updates in real time.
                    """,
                )
            st.markdown("</div></div>", unsafe_allow_html=True)

        # --- ODA 슬라이더 (한 번만 입력) ---
        slider_health = st.slider("❤️Health ODA % change", -20, 50, 0)
        slider_social = st.slider("🌱Social & Environmental ODA % change", -20, 50, 0)
        slider_edu    = st.slider("📚Education ODA % change", -20, 50, 0)
        slider_infra  = st.slider("🏗️Infrastructure ODA % change", -20, 50, 0)
        slider_gov    = st.slider("🏛️Governance ODA % change", -20, 50, 0)

        # --- RQ 슬라이더 (절대값) ---
        slider_rq_delta = st.slider("📊Institutional Quality (absolute change)", -2.5, 2.5, 0.0, 0.01)

    
    with col1:
        # 제목과 아이콘을 같은 줄에 배치
        title_col, icon_col = st.columns([1,1])  # 비율 조정
        with title_col:
            st.subheader("Simulation Visualization")
        with icon_col:
            with st.popover("❓"):
                st.markdown("""
                ### Visualization
                This graph shows both instantaneous and cumulative effects
                across multiple time horizons.
                """)
    
        result_placeholder = st.empty()

        # --- 슬라이더 값 비율 변환 ---
        health_rate = slider_health / 100.0
        social_rate = slider_social / 100.0
        edu_rate    = slider_edu / 100.0
        infra_rate  = slider_infra / 100.0
        gov_rate    = slider_gov / 100.0
        rq_delta   = slider_rq_delta

        # --- 시나리오 호출 ---
        out_df = run_scenario_for_dashboard(
            health_rate=health_rate,
            social_rate=social_rate,
            edu_rate=edu_rate,
            infra_rate=infra_rate,
            gov_rate=gov_rate,
            rq_delta=rq_delta,
            rq_persistent=True,
            ci=0.95
        )

        # --- Plotly 그래프 (항상 표시) ---
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=out_df['horizon'],
            y=out_df['instantaneous'],
            mode='lines+markers',
            name='Instantaneous Effect',
            line=dict(color='royalblue', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=out_df['horizon'],
            y=out_df['inst_hi'],
            mode='lines',
            line=dict(color='royalblue', dash='dash'),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=out_df['horizon'],
            y=out_df['inst_lo'],
            fill='tonexty',
            mode='lines',
            line=dict(color='royalblue', dash='dash'),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=out_df['horizon'],
            y=out_df['cumulative'],
            mode='lines+markers',
            name='Cumulative Effect',
            line=dict(color='firebrick', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=out_df['horizon'],
            y=out_df['cum_hi'],
            mode='lines',
            line=dict(color='firebrick', dash='dash'),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=out_df['horizon'],
            y=out_df['cum_lo'],
            fill='tonexty',
            mode='lines',
            line=dict(color='firebrick', dash='dash'),
            showlegend=False
        ))

        fig.update_layout(
            xaxis_title="Horizon (Years)",
            yaxis_title="Δ Life Expectancy",
            template="plotly_white",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Policy Insight Summary: 슬라이더 조정 시에만 표시 ---
        if any([health_rate, social_rate, edu_rate, infra_rate, gov_rate, rq_delta]):
            delta_dict = {
                'Health ODA': health_rate,
                'Social/Env ODA': social_rate,
                'Education ODA': edu_rate,
                'Infrastructure ODA': infra_rate,
                'Governance ODA': gov_rate,
                'Institutional Quality (RQ)': rq_delta
            }

            # 단기(H0) 효과 계산
            horizon0_effects = {}
            term_map = {
                'Health ODA': HN1,
                'Social/Env ODA': SE0,
                'Education ODA': ED0,
                'Infrastructure ODA': IF0,
                'Governance ODA': GV1,
                'Institutional Quality (RQ)': RQ
            }

            for var, rate in delta_dict.items():
                if rate != 0:
                    est, _ = linear_est_se([(term_map[var], np.log(1+rate))])
                else:
                    est = 0.0
                horizon0_effects[var] = est

            # 상위 3개 변수
            top3_vars = sorted(horizon0_effects.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

            # 전체 누적 효과
            cum_effect = out_df['cumulative'].iloc[-1]

            # 상위 3개 변수 + Horizon 표시
            summary_lines = []
            for var, val in top3_vars:
                arrow = "→" if val != 0 else "→ 영향 없음"
                
                # Horizon별 효과
                horizon_effects = []
                for h in out_df['horizon']:
                    if var == 'Health ODA' and h in [1,2]:
                        est = linear_est_se([(HN1 if h==1 else HN2, np.log(1+health_rate))])[0]
                        horizon_effects.append(f"H{h}: {est:+.3f}")
                    elif var == 'Social/Env ODA' and h==0:
                        est = linear_est_se([(SE0, np.log(1+social_rate))])[0]
                        horizon_effects.append(f"H{h}: {est:+.3f}")
                    elif var == 'Education ODA' and h==0:
                        est = linear_est_se([(ED0, np.log(1+edu_rate))])[0]
                        horizon_effects.append(f"H{h}: {est:+.3f}")
                    elif var == 'Infrastructure ODA' and h==0:
                        est = linear_est_se([(IF0, np.log(1+infra_rate))])[0]
                        horizon_effects.append(f"H{h}: {est:+.3f}")
                    elif var == 'Governance ODA' and h==1:
                        est = linear_est_se([(GV1, np.log(1+gov_rate))])[0]
                        horizon_effects.append(f"H{h}: {est:+.3f}")
                    elif var == 'Institutional Quality (RQ)':
                        est = linear_est_se([(RQ, rq_delta)])[0]
                        horizon_effects.append(f"H{h}: {est:+.3f}")

                summary_lines.append(f"• {var} {delta_dict[var]*100:+.0f}% {arrow} {' | '.join(horizon_effects)} years")


            overall_summary = f"Overall Impact Summary:\nCombined changes across all ODA sectors result in a cumulative change of {cum_effect:+.3f} years.\nTop contributors are {', '.join([v[0] for v in top3_vars])}."

            # 화면에 표시
            st.markdown("💡 Policy Insight Summary:")
            for line in summary_lines:
                st.markdown(line)
            st.markdown(overall_summary)


def crop_center(img, crop_width, crop_height):
    img_width, img_height = img.size
    return img.crop((
        (img_width - crop_width) // 2,
        (img_height - crop_height) // 2,
        (img_width + crop_width) // 2,
        (img_height + crop_height) // 2
    ))

# ====== 사이드바 ======
with st.sidebar:

    logo_path = "Visualization/Design/UNDP_Logo_White_Large.png"
    image = Image.open(logo_path)

    # ---- 원하는 crop 높이 설정 ----
    desired_height = 1450      # 원하는 세로 길이
    desired_width = image.size[0]   # 가로는 그대로 유지

    # ---- 세로만 crop ----
    cropped = crop_center(image, desired_width, desired_height)

    # ---- 가로 중앙 정렬되도록 HTML로 표시 ----
    img_buffer = io.BytesIO()
    cropped.save(img_buffer, format="PNG")
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()

    st.markdown(
        f"""
        <div style="text-align:center;">
            <img src="data:image/png;base64,{img_base64}" style="width:150px; object-fit:cover;">
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- 구분선 ---    
    st.markdown(
        '<hr style="margin-top:30px; margin-bottom:20px; border:1px solid #ccc;">',
        unsafe_allow_html=True
    )

    # --- Dashboard 글자 ---
    st.markdown(
        """
        <div style='text-align:center; margin:15px 0;'>
            <span style='font-size:32px; color:#004899; font-weight:500;'>
                Dashboard
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- 구분선 ---
    st.markdown(
        '<hr style="margin-top:10px; margin-bottom:20px; border:1px solid #ccc;">',
        unsafe_allow_html=True
    )

    cyclogo_path = "Visualization/Design/cyclone_logo.png"
    cycimage = Image.open(cyclogo_path)

    st.markdown(
        f"""
        <div class="sidebar-logo">
            <img src="data:image/png;base64,{base64.b64encode(open(cyclogo_path,'rb').read()).decode()}" 
                width="150">
        </div>
        """,
        unsafe_allow_html=True
    )

# ====== Dashboard 실행 ======
dashboard_page()    









