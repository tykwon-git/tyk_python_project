import pandas as pd

# 경로
# temper_csv_path = r'C:\Users\SBA\Desktop\tyk_python_project\data\OBS_ASOS_TIM_20260810140632.csv'
temper_csv_path = r'C:\Users\SBA\Desktop\tyk_python_project\data\OBS_ASOS_TIM_20260812143952.csv'

# 대상 월
target_months = [3, 6, 9, 12]

# 원본 df
df = pd.read_csv(temper_csv_path,encoding='cp949')
# print(df)
#       지점 지점명     일시            기온(°C)
# 0     108  서울   2025-01-01 01:00    -1.7

# --------------
# 결측치 확인 및 제거
# --------------
# print(f'전체 행: {len(df):,}')
# 전체 행: 8,736
# print('\n===== 전체 결측치 =====')
# print(df.isna().sum())
# 지점        0 / 지점명       0... 전부 0

# print('\n===== 기온 결측치 =====')
# print(df['기온(°C)'].isna().sum())

df_clean = df.dropna(subset=['일시','기온(°C)']).copy()
# print(f'빈 행 제거 후 : {len(df_clean):,}')
# 빈 행 제거 후 : 8,736

# --------------
# 날짜 변환 str > datetime
# --------------
df_clean['일시'] = pd.to_datetime(df_clean['일시'],errors='coerce')
# print(f'날짜 변환 실패 : {df_clean['일시'].isna().sum():,}') # 변환 실패 개수 출력 > 에러처리 해둘까?
# 날짜 변환 실패 : 0

# --------------
# 3,6,9,12만 남기기
# --------------
df_clean = df_clean[df_clean['일시'].dt.month.isin(target_months)].copy()
# print(df_clean)
#       지점 지점명 일시              기온(°C)
# 0     108  서울   2025-03-01 00:00 6.2

# --------------
# 날짜, 시간 추출 (6~23), 시간대 생성 > 지하철에 06이전, 24이후 때문에 6~23
# --------------
df_clean['날짜'] = df_clean['일시'].dt.normalize()
df_clean['시간'] = df_clean['일시'].dt.hour
df_clean = df_clean[df_clean['시간'].between(6,23)].copy()

df_clean['시간대'] = (df_clean['시간'].astype(str).str.zfill(2)+'-'+ (df_clean['시간']+1).astype(str).str.zfill(2))
temperature_hourly = df_clean.reset_index(drop=True)
# print(df_clean)
#       지점    지점명      일시                기온(°C)    날짜    시간    시간대
# 0     108     서울    2025-03-01 06:00:00     3.4     2025-03-01  6     06-07

# 전처리 끝난 csv를 저장
print(f'temper length : {len(temperature_hourly)}')
output_path = r'data\processed\temperature_hourly.csv'
temperature_hourly.to_csv(output_path,index=False,encoding='utf-8-sig')