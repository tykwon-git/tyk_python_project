import pandas as pd

# path
subway_csv_path = r'data\2025서울교통공사_역별 일별 시간대별 승하차인원_20251231.csv'

# 대상 월
target_months = [3, 6, 9, 12]

# 06~23까지만 사용 > 06이전, 24이후는 분리 불가능하게 통합된 값
time_cols = [
    '06-07시간대','07-08시간대','08-09시간대','09-10시간대','10-11시간대','11-12시간대','12-13시간대','13-14시간대','14-15시간대',
    '15-16시간대','16-17시간대','17-18시간대','18-19시간대','19-20시간대','20-21시간대','21-22시간대','22-23시간대','23-24시간대'
]

# 원본 df
df = pd.read_csv(subway_csv_path,encoding='cp949')

# --------------
# 결측치 확인 및 제거
# --------------
# print(f'전체 행: {len(df):,}')
# 전체 행: 199,424
# print('\n===== 전체 결측치 =====')
# print(df.isna().sum())
# 연번          134 / 수송일자        134...

# print('\n===== 시간대 결측치 =====')
# print(df[time_cols].isna().sum())
# 06-07시간대    134...

# print('\n===== 결측치가 있는 행 =====')
# missing_rows = df[df[time_cols].isna().any(axis=1)] # any(axis=1) > 단 하나라고 해당 조건 ture인지 (isna인지), axis=1 행 방향 검사
# 199290 NaN  NaN NaN ... NaN
# print(missing_rows.head(20).to_string())

# print('\n===== 결측치 행 인덱스 =====')
# print(missing_rows.index.tolist()[:50]) # 결측치(na) 발견 행들 리스트에 담아서 앞의 50개

# print(f'완전 빈 행 : {df.isna().all(axis=1).sum():,}') # 
# 완전 빈 행 : 134
df_clean = df.dropna(how='all').copy() # 앞서 나온 134행 > 파일 양식 끝에 비워둔 행 추정? 전부 drop
# print(f'빈 행 제거 후 : {len(df_clean):,}')
# 빈 행 제거 후 : 199,290

# --------------
# 날짜 변환 str > datetime
# --------------
df_clean['수송일자'] = pd.to_datetime(df['수송일자'],errors='coerce')
# print(f'날짜 변환 실패 : {df_clean['수송일자'].isna().sum():,}') # 변환 실패 개수 출력 > 에러처리 해둘까?
# 날짜 변환 실패 : 0

# --------------
# 3,6,9,12만 남기기
# --------------
df_clean = df_clean[df_clean['수송일자'].dt.month.isin(target_months)].copy()

# --------------
# 승차 건수 = 이용건수로 처리하기 위함
# --------------
df_clean = df_clean[df_clean['승하차구분'] == '승차']

# --------------
# 일자에서 뽑아서 날짜 컬럼
# --------------
df_clean['날짜'] = df_clean['수송일자'].dt.normalize() # pandas datetime64 YYYY-MM-DD hh:mm 밑으로 다 0처리
# print(df_clean)
#           연번        수송일자     호선   역번호  역명    승하차구분     06시이전  06-07시간대   07-08시간대 ...  24시이후  날짜
# 32214    32215.0      2025-03-01  1호선   150.  서울역   승차          299.0    461.0         917.0       ...  24.0     2025-03-01

# --------------
# 시간대 기준 세로로 변환 melt()
# --------------
df_clean = df_clean.melt(id_vars=['날짜'],value_vars=time_cols,var_name='시간대',value_name='승차인원')
# print(df_clean)
# [33306 rows x 27 columns]
#       날짜        시간대       승차인원
# 0     2025-03-01  06-07시간대  461.0

# --------------
# subway만 '06-07' 형태가 아니라 '06-07시간대'라서 변경
# --------------
df_clean['시간대'] = df_clean['시간대'].str.replace('시간대', '', regex=False) # 정규식 형태 변환할때는 regex true

# --------------
# 역 구분 없게 전부 통합처리
# --------------
subway_hourly = (df_clean.groupby(['날짜','시간대'])['승차인원'].sum().reset_index())
# print(subway_hourly)
# [33306 rows x 27 columns]
#       날짜        시간대          승차인원
# 0     2025-03-01  06-07시간대     55215.0

# 전처리 끝난 csv를 저장
print(f'subway length : {len(subway_hourly)}')
# output_path = r'data\processed\subway_hourly.csv'
# subway_hourly.to_csv(output_path,index=False,encoding='utf-8-sig')