import pandas as pd

# 경로
subway_csv_path = r'data\2025서울교통공사_역별 일별 시간대별 승하차인원_20251231.csv'
# 대상 월
target_months = [3, 6, 9, 12]
# 06~23까지만 사용 > 06이전, 24이후는 분리 불가능하게 통합된 값
time_cols = [
    '06-07시간대',
    '07-08시간대',
    '08-09시간대',
    '09-10시간대',
    '10-11시간대',
    '11-12시간대',
    '12-13시간대',
    '13-14시간대',
    '14-15시간대',
    '15-16시간대',
    '16-17시간대',
    '17-18시간대',
    '18-19시간대',
    '19-20시간대',
    '20-21시간대',
    '21-22시간대',
    '22-23시간대',
    '23-24시간대'
]

# 원본
df = pd.read_csv(
    subway_csv_path,
    encoding='cp949'
)

# --------------
# 결측치 확인 및 제거
# --------------
print(f'전체 행: {len(df):,}')
# print('\n===== 전체 결측치 =====')
# print(df.isna().sum())

# print('\n===== 시간대 결측치 =====')
# print(df[time_cols].isna().sum())

# print('\n===== 결측치가 있는 행 =====')
# missing_rows = df[df[time_cols].isna().any(axis=1)] # any(axis=1) > 단 하나라고 해당 조건 ture인지 (isna인지), axis=1 행 방향 검사

# print(missing_rows.head(20).to_string())

# print('\n===== 결측치 행 인덱스 =====')
# print(missing_rows.index.tolist()[:50]) # 결측치(na) 발견 행들 리스트에 담아서 앞의 50개

print(f'완전 빈 행 : {len(df.isna().all(axis=1)).sum():,}')
df_clean = df.dropna(how='all').copy() # 앞서 나온 134행 전부 drop
print(f'빈 행 제거 후 : {len(df_clean):,}')

# --------------
# 날짜 변환 str > datetime
# --------------
df_clean['수송일자'] = pd.to_datetime(
    df['수송일자'],
    errors='coerce'
)

print(f'날짜 변환 실패 : {df_clean['수송일자'].isna().sum():,}') # 변환 실패 개수 출력 > 에러처리 해둘까?

# --------------
# 대상 월만 남기고
# --------------
df_clean = df_clean[
    df_clean['수송일자'].dt.month.isin(target_months)
].copy()

# --------------
# 승차만 남겨서 이용건수 측정에 사용
# --------------
df_clean = df_clean[df_clean['승하차구분'] == '승차']

# --------------
#  날짜 컬럼 생성 (시간대는 원본에 존재)
# --------------
df_clean['날짜'] = df_clean['수송일자'].dt.normalize()
print(df_clean)

# --------------
# 시간대 기준 세로로 변환 melt()
# --------------
# df_clean = df_clean.melt(
#     id_vars=['날짜'],
# )
