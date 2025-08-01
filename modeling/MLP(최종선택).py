# --- 라이브러리 ---
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os

# ----------------------------
# 1. 데이터 로드 및 전처리
# ----------------------------
df_input = pd.read_csv("/content/drive/MyDrive/딥러닝프로젝트/UNDP Data Dive 해커톤/모델/input_final.csv")
df_y = pd.read_csv("/content/drive/MyDrive/딥러닝프로젝트/UNDP Data Dive 해커톤/모델/y.csv")

# 불필요한 열 제거
drop_y = ["교육_초등학교 이수율_증가율", "빈곤 및 사회복지_의료 접근성_증가율", "빈곤 및 사회복지_빈곤율_감소율"]
drop_x = ["교육_초등학교 이수율_ODA", "빈곤 및 사회복지_의료 접근성_ODA", "빈곤 및 사회복지_빈곤율_ODA"]
df_y = df_y.drop(columns=drop_y)
df_input = df_input.drop(columns=drop_x)

# 병합 및 이상치 처리
df_input = df_input.rename(columns={"Year": "year", "Country": "country"})
df = pd.merge(df_input, df_y, on=["year", "country"], how="inner")
df = df.replace([np.inf, -np.inf], np.nan)
df.loc[df.select_dtypes(include=[float, int]).abs() > 1e10] = np.nan

# 선형 보간
def interpolate_by_country(df):
    df = df.sort_values(["country", "year"])
    for col in df.select_dtypes(include=[float]):
        df[col] = df.groupby("country")[col].transform(lambda x: x.interpolate(method="linear", limit_direction="both"))
    return df

df = interpolate_by_country(df)

# 특정 컬럼은 국가 평균/전체 평균으로 채움
fill_cols = ["실업률", "소득 불평등", "환경적 지속가능성", "농업 부가가치 비중", "산업 부가가치 비중",
             "작물 생산 지수", "식량 생산 지수", "연간 물가상승률 (CPI 기준, %)", "GDP 대비 공교육 지출 비율 (%)",
             "인구 1,000명당 간호사 및 조산사 수", "정부예산 중 교육비 비중"]

for col in fill_cols:
    df[col] = df.groupby("country")[col].transform(lambda x: x.fillna(x.mean()))
    df[col] = df[col].fillna(df[col].mean())

# 회귀 기반 보간 (중등학교 총등록률)
def regress_fill(df, target, features):
    train = df[df[target].notnull()]
    test = df[df[target].isnull()]
    if len(train) < 100 or len(test) == 0: 
        return df
    if train[features].isnull().any().any() or test[features].isnull().any().any():
        return df
    model = LinearRegression().fit(train[features], train[target])
    df.loc[df[target].isnull(), target] = model.predict(test[features])
    return df

df = regress_fill(df, "중등학교 총등록률", ["교육_초등학교 순취학률_ODA", "GDP 대비 공교육 지출 비율 (%)", "정부예산 중 교육비 비중"])

# ----------------------------
# 2. 학습 데이터 준비
# ----------------------------
targets = [
    "교육_초등학교 순취학률_증가율",
    "보건_기대 수명_증가율",
    "생산_서비스업 부가가치_증가율",
    "생산_농작물 생산지수_증가율",
    "경제_1인당 GDP_증가율",
    "빈곤 및 사회복지_1인당 GNI_증가율",
    "환경/기후_1인당 co2 배출량_감소율"
]

df = df.dropna(subset=targets)

# 국가 원핫 인코딩 + 연도 정규화
df = pd.get_dummies(df, columns=["country"], drop_first=False)
df["year"] = (df["year"] - df["year"].min()) / (df["year"].max() - df["year"].min())

X = df.drop(columns=targets)
Y = df[targets]

# 스케일링
num_cols = [c for c in X.columns if not c.startswith("country_")]
scaler_X, scaler_Y = StandardScaler(), StandardScaler()
X[num_cols] = scaler_X.fit_transform(X[num_cols])
Y = scaler_Y.fit_transform(Y.astype(np.float32))

X_train, X_val, Y_train, Y_val = train_test_split(X.values, Y, test_size=0.2, random_state=42)

X_train, Y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32)
X_val, Y_val = torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.float32)

# ----------------------------
# 3. MLP 모델 정의 및 학습
# ----------------------------
class RegressionMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, out_dim)
        )
    def forward(self, x): return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RegressionMLP(X_train.shape[1], Y_train.shape[1]).to(device)
X_train, Y_train, X_val, Y_val = X_train.to(device), Y_train.to(device), X_val.to(device), Y_val.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS, PATIENCE = 1000, 50
best_val, counter = float("inf"), 0
save_path = "/content/drive/MyDrive/딥러닝프로젝트/UNDP Data Dive 해커톤/모델/best_model.pt"

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    loss = criterion(model(X_train), Y_train)
    loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_val), Y_val)
    if val_loss.item() < best_val:
        best_val, counter = val_loss.item(), 0
        torch.save(model.state_dict(), save_path)
    else: counter += 1
    if counter >= PATIENCE: break

# ----------------------------
# 4. 검증 및 성능 평가
# ----------------------------
y_true = scaler_Y.inverse_transform(Y_val.cpu())
y_pred = scaler_Y.inverse_transform(model(X_val).detach().cpu())

metrics = []
for i, col in enumerate(targets):
    metrics.append({
        "Target": col,
        "MAE": mean_absolute_error(y_true[:, i], y_pred[:, i]),
        "MSE": mean_squared_error(y_true[:, i], y_pred[:, i]),
        "RMSE": np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])),
        "R2": r2_score(y_true[:, i], y_pred[:, i])
    })

print(pd.DataFrame(metrics).round(4))

# ----------------------------
# 5. 모델 및 스케일러 저장
# ----------------------------
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(model.state_dict(), save_path)
joblib.dump(scaler_X, save_path.replace("best_model.pt", "scaler_X.pkl"))
joblib.dump(scaler_Y, save_path.replace("best_model.pt", "scaler_Y.pkl"))
