# --- 라이브러리 ---
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --- 데이터 불러오기 ---
df_input = pd.read_csv("/content/drive/MyDrive/딥러닝프로젝트/UNDP Data Dive 해커톤/모델/input_final.csv")
df_y = pd.read_csv("/content/drive/MyDrive/딥러닝프로젝트/UNDP Data Dive 해커톤/모델/y.csv")

# --- 불필요한 열 제거 ---
drop_y_cols = [
    "교육_초등학교 이수율_증가율",
    "빈곤 및 사회복지_의료 접근성_증가율",
    "빈곤 및 사회복지_빈곤율_감소율"
]
drop_x_cols = [
    "교육_초등학교 이수율_ODA",
    "빈곤 및 사회복지_의료 접근성_ODA",
    "빈곤 및 사회복지_빈곤율_ODA"
]
df_y = df_y.drop(columns=drop_y_cols)
df_input = df_input.drop(columns=drop_x_cols)

# --- 병합 및 이상치 처리 ---
df_input = df_input.rename(columns={"Year": "year", "Country": "country"})
df = pd.merge(df_input, df_y, on=["year", "country"], how="inner")
df = df.replace([np.inf, -np.inf], np.nan)

for col in df.select_dtypes(include=[float, int]).columns:
    df.loc[df[col].abs() > 1e10, col] = np.nan

# --- 시계열 보간 함수 ---
def interpolate_by_country(df, method="linear"):
    df = df.sort_values(by=["country", "year"])
    for col in df.select_dtypes(include=[float]).columns:
        df[col] = df.groupby("country")[col].transform(lambda x: x.interpolate(method=method, limit_direction='both'))
    return df

df = interpolate_by_country(df)

# --- 국가별 평균 보간 ---
x_mean_cols = [
    "실업률", "소득 불평등", "환경적 지속가능성", "농업 부가가치 비중", "산업 부가가치 비중",
    "작물 생산 지수", "식량 생산 지수", "연간 물가상승률 (CPI 기준, %)",
    "GDP 대비 공교육 지출 비율 (%)", "인구 1,000명당 간호사 및 조산사 수",
    "정부예산 중 교육비 비중"
]
for col in x_mean_cols:
    df[col] = df.groupby("country")[col].transform(lambda x: x.fillna(x.mean()))
    df[col] = df[col].fillna(df[col].mean())

# --- 회귀 기반 보간 ---
def regress_fill(df, target_col, feature_cols):
    non_null = df[df[target_col].notnull()]
    null = df[df[target_col].isnull()]
    if len(non_null) < 100 or len(null) == 0:
        return df
    X_train, y_train = non_null[feature_cols], non_null[target_col]
    X_pred = null[feature_cols]
    if X_train.isnull().any().any() or X_pred.isnull().any().any():
        return df
    model = LinearRegression().fit(X_train, y_train)
    df.loc[df[target_col].isnull(), target_col] = model.predict(X_pred)
    return df

df = regress_fill(
    df,
    "중등학교 총등록률",
    ["교육_초등학교 순취학률_ODA", "GDP 대비 공교육 지출 비율 (%)", "정부예산 중 교육비 비중"]
)

# --- 타겟 정의 및 결측치 제거 ---
targets = [
    "교육_초등학교 순취학률_증가율", "생산_서비스업 부가가치_증가율",
    "생산_제조업 부가가치_증가율", "생산_농작물 생산지수_증가율",
    "생산_가축 생산지수_증가율", "경제_1인당 GDP_증가율",
    "빈곤 및 사회복지_1인당 GNI_증가율", "환경/기후_1인당 co2 배출량_감소율"
]
df = df.dropna(subset=targets)

# --- 전처리 ---
df = pd.get_dummies(df, columns=["country"], drop_first=False)
df["year"] = (df["year"] - df["year"].min()) / (df["year"].max() - df["year"].min())

X = df.drop(columns=targets)
Y = df[targets]

num_cols = [col for col in X.columns if not col.startswith("country_")]
scaler_X = StandardScaler()
X[num_cols] = scaler_X.fit_transform(X[num_cols])
scaler_Y = StandardScaler()
Y = scaler_Y.fit_transform(Y.astype(np.float32))

X = X.astype(np.float32).values
Y = Y.astype(np.float32)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train)
Y_train_tensor = torch.tensor(Y_train)
X_val_tensor = torch.tensor(X_val)
Y_val_tensor = torch.tensor(Y_val)

# --- MLP 모델 ---
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

model = RegressionMLP(X_train.shape[1], Y_train.shape[1])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
X_train_tensor = X_train_tensor.to(device)
Y_train_tensor = Y_train_tensor.to(device)
X_val_tensor = X_val_tensor.to(device)
Y_val_tensor = Y_val_tensor.to(device)

# --- 학습 ---
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
EPOCHS = 1000
PATIENCE = 50
best_val_loss = float("inf")
counter = 0
save_path = "/content/drive/MyDrive/딥러닝프로젝트/UNDP Data Dive 해커톤/모델/best_model.pt"

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    loss = criterion(model(X_train_tensor), Y_train_tensor)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_val_tensor), Y_val_tensor)

    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        torch.save(model.state_dict(), save_path)
        counter = 0
    else:
        counter += 1

    if epoch % 50 == 0 or epoch == EPOCHS - 1 or counter == PATIENCE:
        print(f"[Epoch {epoch}] Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Patience: {counter}/{PATIENCE}")

    if counter >= PATIENCE:
        print(f"Early stopping at epoch {epoch}")
        break

# --- 시각화 ---
y_true_original = scaler_Y.inverse_transform(Y_val_tensor.cpu())
y_pred_original = scaler_Y.inverse_transform(model(X_val_tensor).detach().cpu())
columns = [
    "교육_초등학교 순취학률_증가율", "보건_영아 사망률_감소율", "보건_신생아 사망률_감소율",
    "보건_기대 수명_증가율", "생산_서비스업 부가가치_증가율", "생산_제조업 부가가치_증가율",
    "생산_농작물 생산지수_증가율", "생산_가축 생산지수_증가율"
]

plt.figure(figsize=(20, 12))
for i, col in enumerate(columns):
    plt.subplot(2, 4, i + 1)
    plt.scatter(y_true_original[:, i], y_pred_original[:, i], alpha=0.6, color="royalblue")
    plt.plot([y_true_original[:, i].min(), y_true_original[:, i].max()],
             [y_true_original[:, i].min(), y_true_original[:, i].max()], 'r--')
    plt.title(f"{col}\n(Actual vs Predicted)", fontsize=10)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")

plt.suptitle("Multitarget Regression Prediction Results", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
