import pandas as pd

bicycle_csv = r'data\서울특별시 공공자전거 대여이력 정보_2025\서울특별시 공공자전거 대여이력 정보_2501.csv'
subway_csv = r'data\2025서울교통공사_역별 일별 시간대별 승하차인원_20251231.csv'
temperature_csv = r'data\OBS_ASOS_TIM_20260810140632.csv'

def csv_count(csv_path: str, enc: str = 'cp949'):
    total_rows = 0

    for chunk in pd.read_csv(
        csv_path,
        encoding=enc,
        chunksize=100_000
    ):
        total_rows += len(chunk)

    print(f'{csv_path}')
    print(f'총 데이터 행 수: {total_rows:,}')

csv_count(bicycle_csv)
csv_count(subway_csv)
csv_count(temperature_csv)