import pandas as pd

df = pd.read_csv("/content/drive/MyDrive/딥러닝프로젝트/UNDP Data Dive 해커톤/crs_data.csv", nrows=100000)

df.info()

# 열 하나만 읽어서 메모리 최소화
df2 = pd.read_csv("/content/drive/MyDrive/딥러닝프로젝트/UNDP Data Dive 해커톤/crs_data.csv", usecols=["Year"])

print(f"총 행 수: {len(df2)}")

# 데이터 추출

file_path = "/content/drive/MyDrive/딥러닝프로젝트/UNDP Data Dive 해커톤/crs_data.csv"

# 저장 위치 기본 경로
save_dir = "/content/drive/MyDrive/딥러닝프로젝트/UNDP Data Dive 해커톤/단위별 데이터/"


# 1. 국가별 연도별 ODA 흐름
cols1 = ['Year', 'RecipientName', 'USD_Disbursement']
df1 = pd.read_csv(file_path, usecols=cols1)
df1 = df1.dropna(subset=['USD_Disbursement'])
df1.to_csv(save_dir + "국가별 연도별 ODA 흐름.csv", index=False)

# 2. 분야별 연도별 ODA 흐름
cols2 = ['Year', 'RecipientName', 'SectorName', 'USD_Disbursement']
df2 = pd.read_csv(file_path, usecols=cols2)
df2 = df2.dropna(subset=['USD_Disbursement'])
df2.to_csv(save_dir + "분야별 연도별 ODA 흐름.csv", index=False)

# 3. 목적별 연도별 ODA 흐름
cols3 = ['Year', 'RecipientName', 'PurposeName', 'USD_Disbursement']
df3 = pd.read_csv(file_path, usecols=cols3)
df3 = df3.dropna(subset=['USD_Disbursement'])
df3.to_csv(save_dir + "목적별 연도별 ODA 흐름.csv", index=False)

# 4. 국가 정보 (지역/소득그룹)
cols4 = ['RecipientName', 'RegionName', 'IncomegroupName']
df4 = pd.read_csv(file_path, usecols=cols4)
df4 = df4.drop_duplicates()
df4.to_csv(save_dir + "국가 정보_지역 및 소득그룹.csv", index=False)

# 5. 모델 입력용 데이터 (분야별 ODA + 물가보정)
cols5 = ['Year', 'RecipientName', 'SectorName', 'USD_Disbursement', 'USD_Disbursement_Defl']
df5 = pd.read_csv(file_path, usecols=cols5)
df5 = df5.dropna(subset=['USD_Disbursement'])
df5.to_csv(save_dir + "분야별 ODA + 물가보정.csv", index=False)

# 6. 전처리용 전체 핵심 버전 (전체 분석 대비)
cols6 = ['Year', 'RecipientName', 'SectorName', 'PurposeName',
         'USD_Disbursement', 'USD_Disbursement_Defl',
         'RegionName', 'IncomegroupName']
df6 = pd.read_csv(file_path, usecols=cols6)
df6 = df6.dropna(subset=['USD_Disbursement'])
df6.to_csv(save_dir + "전체 분석 대비.csv", index=False)

# 7. 국가 정보 (지역/소득그룹)
cols42 = ['RecipientName', 'RegionName', 'IncomegroupName', 'USD_Disbursement', 'PurposeName']
df42 = pd.read_csv(file_path, usecols=cols42)
df42 = df42.drop_duplicates()
df42.to_csv(save_dir + "국가 정보_지역 및 소득그룹2.csv", index=False)

from google.colab import drive
drive.mount('/content/drive')
