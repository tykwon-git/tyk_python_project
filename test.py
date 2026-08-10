import pandas as pd

bicycle_csv = r'data\서울특별시 공공자전거 대여이력 정보_2025\서울특별시 공공자전거 대여이력 정보_2501.csv'
subway_csv = r'data\2025서울교통공사_역별 일별 시간대별 승하차인원_20251231.csv'
temperature_csv = r'data\OBS_ASOS_TIM_20260810140632.csv'

def csv_chk (csv_path:str, enc:str='cp949'):
    df = pd.read_csv(csv_path, encoding=enc, nrows=10)
    # print('===== 데이터 =====')
    # print(df)
    print('\n===== 데이터 크기 =====')
    print(df.shape)
    print('\n===== 컬럼명 =====')
    print(df.columns.tolist())
    print('\n===== 데이터 타입 =====')
    print(df.dtypes)
    print('='*30)

csv_chk(bicycle_csv)
csv_chk(subway_csv)
csv_chk(temperature_csv)