import streamlit as st
from PIL import Image
import pydeck as pdk
import pandas as pd
import numpy as np
import plotly.express as px
import torch
import torch.nn as nn
import time
import joblib
import plotly.graph_objects as go
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# ===== 스타일 =====
st.markdown(
    """
    <style>
    .stApp { background-color: #F9FAFB; }
    section[data-testid="stSidebar"] { background-color: #ffffff; }
    section[data-testid="stSidebar"] label { font-size: 18px !important; }
    section[data-testid="stSidebar"] .stMarkdown > p { font-size: 16px !important; }
    h1, h2, h3 { font-size: inherit; }
    p, label, div, .stMarkdown, .stText, .stButton, .stSelectbox, .stSlider { font-size: 20px !important; }
    .stButton > button {
        background-color: #2563eb !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5em 1em !important;
        font-size: 16px !important;
        font-weight: 600;
    }
    .stButton > button * { color: white !important; }
    .stSidebar .stImage img { display: block; margin-left: auto; margin-right: auto; }
    </style>
    """,
    unsafe_allow_html=True
)

# ===== 초기 상태 =====
if "show_map" not in st.session_state:
    st.session_state["show_map"] = True

# ===== 데이터 =====
@st.cache_data
def load_data():
    oda = pd.read_csv(r"C:\Users\user\Desktop\CYClone_UNDP Project Streamlit\data\국가별_연도별_ODA_흐름.csv")
    coords = pd.read_csv(r"C:\Users\user\Desktop\CYClone_UNDP Project Streamlit\data\recipient_coords.csv")
    return oda, coords

oda_df, coords_df = load_data()

country_list = sorted(coords_df["country"].unique().tolist())
year_list = list(range(2014, 2023))

# ===== 사이드바 =====
with st.sidebar:
    logo_path = "C:/Users/user/Desktop/CYClone_UNDP Project Streamlit/Design/cyclone_logo.png"
    image = Image.open(logo_path)
    image = image.resize((170, int(image.size[1] * (170 / image.size[0]))), Image.LANCZOS)
    st.image(image)
    st.write("---")
    selected_country = st.selectbox("Select Country", ["View All"] + country_list)
    st.write("---")
    selected_year = st.selectbox("Select Year", year_list)
    st.write("---")
    analyze_button = st.button("🔍 Run Analysis")

# ===== 경고 메시지 placeholder =====
warning_placeholder = st.empty()

if analyze_button:
    if selected_country == "View All":
        warning_placeholder.markdown(
            """
            <div style='
                position: fixed;
                top: 30%;
                left: 50%;
                transform: translate(-50%, -50%);
                background-color: #fff3cd;
                color: #856404;
                padding: 20px 30px;
                border: 1px solid #ffeeba;
                border-radius: 10px;
                font-size: 20px;
                z-index: 9999;
                box-shadow: 0px 0px 10px rgba(0,0,0,0.2);
            '>
                ⚠️ Select a Country.
            </div>
            """, unsafe_allow_html=True
        )
        time.sleep(3)
        warning_placeholder.empty()
    else:
        st.session_state["show_map"] = False
        warning_placeholder.empty()

# ===== 제목 =====
title_placeholder = st.empty()

if st.session_state["show_map"]:
    title_html = """
        <div style='line-height:1.3; padding-bottom:10px'>
            <h1 style="font-size:45px; margin-bottom:0">Analyzing the Past, Designing the Future:</h1>
            <h2 style="font-size:32px; margin-top:0; color:gray">A Multi-target ODA Forecasting System</h2>
        </div>
    """
    title_placeholder.markdown(title_html, unsafe_allow_html=True)
else:
    title_placeholder.empty()

# ===== 지도 데이터 전처리 =====
oda_for_map = oda_df[oda_df['RecipientName'].isin(country_list)].dropna()
oda_for_map = oda_for_map.groupby(['Year', 'RecipientName'])['USD_Disbursement'].sum().reset_index()
oda_year = oda_for_map[oda_for_map["Year"] == selected_year]
oda_merged = oda_year.merge(coords_df, left_on="RecipientName", right_on="country")
oda_merged.rename(columns={"lat": "latitude", "lon": "longitude"}, inplace=True)

if selected_country != "View All":
    oda_merged = oda_merged[oda_merged["RecipientName"] == selected_country]
    center_lat = coords_df.loc[coords_df["country"] == selected_country, "lat"].values[0]
    center_lon = coords_df.loc[coords_df["country"] == selected_country, "lon"].values[0]
    zoom_level = 3.5
else:
    center_lat, center_lon, zoom_level = 10, 0, 1.2

view_state = pdk.ViewState(
    latitude=center_lat,
    longitude=center_lon,
    zoom=zoom_level,
    pitch=0,
    min_zoom=1.3,
    max_zoom=5
)

recipient_grouped = oda_year.groupby("RecipientName", as_index=False)["USD_Disbursement"].sum()
recipient_coords = coords_df.rename(columns={"country": "RecipientName"})
recipient_merged = recipient_grouped.merge(recipient_coords, on="RecipientName")
recipient_merged.rename(columns={"lat": "latitude", "lon": "longitude"}, inplace=True)
recipient_merged["radius"] = (recipient_merged["USD_Disbursement"] ** 0.6) * 5000
recipient_merged.dropna(subset=["latitude", "longitude", "USD_Disbursement"], inplace=True)

scatter_layer = pdk.Layer(
    "ScatterplotLayer",
    data=recipient_merged,
    get_position=["longitude", "latitude"],
    get_radius="radius",
    get_fill_color=[255, 100, 100, 120],
    pickable=True,
    auto_highlight=True,
)

# ===== 모델 =====
class RegressionMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        return self.model(x)

model = RegressionMLP(134, 7)
model.load_state_dict(torch.load("best_model_f.pt", map_location=torch.device("cpu")))
model.eval()

scaler_x = joblib.load("scaler_X_f.pkl")
scaler_y = joblib.load("scaler_Y_f.pkl")
X_columns = joblib.load("X_columns_134.pkl")

numerical_cols = [col for col in X_columns if not col.startswith("country_") and "_증가율" not in col and "_감소율" not in col]

# ===== 지도 화면 =====
if st.session_state["show_map"]:
    deck = pdk.Deck(
        layers=[scatter_layer],
        initial_view_state=view_state,
        map_style="light",
        tooltip={"text": "{RecipientName}\nTotal Disbursement: {USD_Disbursement} USD"}
    )
    st.pydeck_chart(deck, use_container_width=True)
else:
    # 분석 화면
    st.markdown(f"## 🎯 {selected_country} - Predicted Impact Analysis")
    st.write("Adjust ODA multipliers to see predicted outcome changes.")

    col1, col2 = st.columns([2.5, 1])

    # ===== 사용자 입력 =====
    with col2:
        with st.expander("📚 Education ODA"):
            primary_enrollment_ratee = st.slider("Primary Enrollment Rate (×)", 0.5, 3.0, 1.0, 0.5)
        with st.expander("❤️ Health ODA"):
            infant_mortality = st.slider("Infant Mortality Rate (×)", 0.5, 3.0, 1.0, 0.5)
            neonatal_mortality_rate = st.slider("Neonatal Mortality Rate (×)", 0.5, 3.0, 1.0, 0.5)
            life_expectancy = st.slider("Life Expectancy (×)", 0.5, 3.0, 1.0, 0.5)
        with st.expander("🫂 Poverty & Social Welfare ODA"):
            gni_per_capita = st.slider("GNI per Capita (×)", 0.5, 3.0, 1.0, 0.5)
        with st.expander("🏭 Productive ODA"):
            value_added_services = st.slider("Value Added in Services (×)", 0.5, 3.0, 1.0, 0.5)
            value_added_manufacturing = st.slider("Value Added in Manufacturing (×)", 0.5, 3.0, 1.0, 0.5)
            crop_production_index = st.slider("Crop Production Index (×)", 0.5, 3.0, 1.0, 0.5)
            livestock_production_index = st.slider("Livestock Production Index (×)", 0.5, 3.0, 1.0, 0.5)
        with st.expander("💰 Economic ODA"):
            gdp_per_capita = st.slider("GDP per Capita (×)", 0.5, 3.0, 1.0, 0.5)
        with st.expander("🌍 Environment/Climate ODA"):
            renewable_energy_usage = st.slider("Renewable Energy Usage (×)", 0.5, 3.0, 1.0, 0.5)
            air_pollution_index = st.slider("Air Pollution Index (×)", 0.5, 3.0, 1.0, 0.5)
            co2_emissions_per_capita = st.slider("CO₂ Emissions per Capita (×)", 0.5, 3.0, 1.0, 0.5)

    # ===== Baseline 예측 =====
    dummy_dict = {col: 0 for col in X_columns}
    dummy_dict[f"country_{selected_country}"] = 1
    dummy_dict["year"] = (selected_year - 2000) / (2022 - 2000)
    for k in dummy_dict:
        if "_ODA" in k:
            dummy_dict[k] = 1.0

    dummy_df = pd.DataFrame([dummy_dict]).reindex(columns=X_columns)
    dummy_df[numerical_cols] = scaler_x.transform(dummy_df[numerical_cols])
    dummy_tensor = torch.tensor(dummy_df.values.astype(np.float32))
    with torch.no_grad():
        dummy_output = model(dummy_tensor)
    dummy_result = scaler_y.inverse_transform(dummy_output.detach().numpy())[0]

    # ===== 사용자 입력 예측 =====
    input_dict = {col: 0 for col in X_columns}
    input_dict[f"country_{selected_country}"] = 1
    input_dict["year"] = (selected_year - 2000) / (2022 - 2000)
    input_dict.update({
        "교육_초등학교 순취학률_ODA": primary_enrollment_ratee,
        "보건_영아 사망률_ODA": infant_mortality,
        "보건_신생아 사망률_ODA": neonatal_mortality_rate,
        "보건_기대 수명_ODA": life_expectancy,
        "빈곤 및 사회복지_1인당 GNI_ODA": gni_per_capita,
        "생산_서비스업 부가가치_ODA": value_added_services,
        "생산_제조업 부가가치_ODA": value_added_manufacturing,
        "생산_농작물 생산지수_ODA": crop_production_index,
        "생산_가축 생산지수_ODA": livestock_production_index,
        "경제_1인당 GDP_ODA": gdp_per_capita,
        "환경/기후_재생 에너지 사용률_ODA": renewable_energy_usage,
        "환경/기후_대기오염 지수_ODA": air_pollution_index,
        "환경/기후_CO2 배출량_ODA": co2_emissions_per_capita,
    })

    input_df = pd.DataFrame([input_dict]).reindex(columns=X_columns)
    if input_df.isnull().sum().sum() > 0:
        st.error("❌ 누락된 입력이 있습니다.")
        st.stop()

    input_df[numerical_cols] = scaler_x.transform(input_df[numerical_cols])
    input_tensor = torch.tensor(input_df.values.astype(np.float32))
    with torch.no_grad():
        output_tensor = model(input_tensor)
    output_result = scaler_y.inverse_transform(output_tensor.detach().numpy())[0]

    # ===== 시각화 =====
    output_labels = [
        "Primary Enrollment Rate ↑ (%)",
        "Infant Mortality ↓ (%)",
        "Life Expectancy ↑ (%)",
        "GNI per Capita ↑ (%)",
        "GDP per Capita ↑ (%)",
        "Renewable Energy Usage ↑ (%)",
        "CO₂ Emissions ↓ (%)"
    ]

    delta_values = output_result - dummy_result
    delta_texts = [f"{val:.2f}% ({delta:+.2f}%)" for val, delta in zip(output_result, delta_values)]
    delta_colors = ['green' if d > 0 else 'red' if d < 0 else 'gray' for d in delta_values]

    # 막대 그래프
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=output_result,
        y=output_labels,
        orientation='h',
        marker=dict(color='#2563eb'),
        text=delta_texts,
        textfont=dict(color=delta_colors),
        textposition='outside',
    ))
    fig_bar.update_layout(
        title='Predicted Outcomes',
        xaxis=dict(title='Change (%)', range=[-30, 30], zeroline=True, zerolinecolor='black'),
        yaxis=dict(autorange="reversed"),
        height=450,
    )
    col1.plotly_chart(fig_bar, use_container_width=True)

    # 레이더 차트
    radar_labels = output_labels + [output_labels[0]]
    radar_dummy = dummy_result.tolist() + [dummy_result[0]]
    radar_user = output_result.tolist() + [output_result[0]]
    delta_texts_radar = [f"{d:+.2f}%" for d in delta_values] + [f"{delta_values[0]:+.2f}%"]
    delta_colors_radar = delta_colors + [delta_colors[0]]

    col_r1, col_r2 = col1.columns(2)
    with col_r1:
        fig_radar_baseline = go.Figure()
        fig_radar_baseline.add_trace(go.Scatterpolar(
            r=radar_dummy, theta=radar_labels, fill='toself',
            name='Baseline', line=dict(color='gray'),
            text=[f"{val:.2f}%" for val in radar_dummy]
        ))
        fig_radar_baseline.update_layout(
            polar=dict(radialaxis=dict(visible=True, showticklabels=False)),
            title="Baseline ODA Impact"
        )
        st.plotly_chart(fig_radar_baseline, use_container_width=True)

    with col_r2:
        fig_radar_user = go.Figure()
        fig_radar_user.add_trace(go.Scatterpolar(
            r=radar_user, theta=radar_labels, fill='toself',
            name='User Input', line=dict(color='royalblue')
        ))
        fig_radar_user.add_trace(go.Scatterpolar(
            r=radar_user, theta=radar_labels, mode='markers+text',
            text=delta_texts_radar, textposition='top center',
            textfont=dict(color=delta_colors_radar),
            marker=dict(size=8, color='royalblue')
        ))
        fig_radar_user.update_layout(
            polar=dict(radialaxis=dict(visible=True, showticklabels=False)),
            title="User ODA Impact"
        )
        st.plotly_chart(fig_radar_user, use_container_width=True)

    # ===== Policy Insight =====
    top_indices = np.argsort(np.abs(delta_values))[::-1][:2]
    key_changes = [(output_labels[i], delta_values[i]) for i in top_indices]

    st.markdown(
        f"""
        <div style="
            background-color: #f1f5f9;
            padding: 20px 25px;
            border-radius: 12px;
            border: 1px solid #cbd5e1;
            font-size: 18px;
            font-weight: 500;
            color: #1f2937;
            line-height: 1.6;
        ">
            <strong>{key_changes[0][0]}</strong> 변화: <strong>{key_changes[0][1]:+.2f}%</strong><br>
            <strong>{key_changes[1][0]}</strong> 변화: <strong>{key_changes[1][1]:+.2f}%</strong><br>
            해당 지표는 ODA 조정에 민감하므로 전략적 투자 우선순위로 고려할 수 있습니다.
        </div>
        """,
        unsafe_allow_html=True
    )

