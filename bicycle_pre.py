import pandas as pd


# 경로
bicycle_csv_path = r'data\서울특별시 공공자전거 대여이력 정보_2025\서울특별시 공공자전거 대여이력 정보_25'
# 대상 월
target_months = [3, 6, 9, 12]

bicycle_list = []
missing_count =0

# --------------
# 파일명 이용해서 대상 월에 해당되는 csv만 read
# --------------
for month in target_months:
    month_str = f'{month:02d}'
    # path 조합
    file_path = f'{bicycle_csv_path}{month_str}.csv'

    # 월별 결측치 합계 
    missing_count = 0
    invalid_datetime_count = 0

    # --------------
    # csv read - 파일 사이즈가 커서 chunk 이용, 10만개씩 처리
    # --------------
    print(f'\n===== {month}월 처리 =====')
    for chunk in pd.read_csv(
        file_path,
        encoding='cp949',
        usecols=['대여일시'],
        chunksize=100_000
    ):
        # 결측치
        missing_count += chunk['대여일시'].isna().sum()

        # --------------
        # 날짜 변환 str > datetime
        # --------------
        chunk['대여일시'] = pd.to_datetime(
            chunk['대여일시'],
            errors='coerce' # 잘못된 형식, 변환 불가능 등의 이상치 나오면 NaT(Not a Time) 또는 NaN(Not a Number)으로 강제변환
        )

        # 날짜 변환 실패 개수
        invalid_datetime_count += chunk['대여일시'].isna().sum()

        # 코어스(coerce) 처리해도 na면 drop 처리해서 결측치 제거
        chunk = chunk.dropna(subset=['대여일시'])

        # --------------
        # 날짜, 시간 추출
        # --------------
        chunk['날짜'] = chunk['대여일시'].dt.normalize() # pandas에서 datetime64 데이터에서 YYYY-MM-DD hh:mm 밑으로 다 0처리
        chunk['시간'] = chunk['대여일시'].dt.hour

        # --------------
        # 시간대 생성
        # 06~23까지만 사용 > 06이전, 24이후는 분리 불가능하게 통합된 값
        # --------------
        chunk = chunk[chunk['시간'].between(6,23)]
        chunk['시간대'] = (
            chunk['시간'].astype(str).str.zfill(2)
            + '-'
            + (chunk['시간']+1).astype(str).str.zfill(2)
        )
        # --------------
        # 날짜, 시간대별 대여건수 > size로 개수, 컬럼명 "따릉이대여건수"
        # 대여건수 컬럼 생성
        # --------------
        result = (
            chunk.groupby(['날짜','시간대'])
            .size()
            .reset_index(name="따릉이대여건수")
        )
        bicycle_list.append(result)

    print(f'{month}월 원본 결측치: {missing_count:,}') # 결측치 개수 출력
    print(f'{month}월 날짜 변환 후 결측치: {invalid_datetime_count:,}') # 변환 후 남은 결측치 개수

# --------------
# 1차 합산 > 3078
# --------------
bicycle_hourly = pd.concat(
    bicycle_list,
    ignore_index=True
)
print(bicycle_hourly.shape)

# --------------
# 최종합산 > 2196
# chunk 단위로 진행됐기 때문에 동일시간대도 분리되는 경우 존재
# --------------
bicycle_hourly = (
    bicycle_hourly.groupby(['날짜','시간대'],)['따릉이대여건수']
    .sum()
    .reset_index()
)
print(bicycle_hourly.shape)
print(bicycle_hourly['날짜'].min()) # 250301 00:00:00
print(bicycle_hourly['날짜'].max()) # 251231 00:00:00
print(bicycle_hourly)