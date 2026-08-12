import pandas as pd

bicycle_csv = r'C:\Users\SBA\Desktop\tyk_python_project\data\서울특별시 공공자전거 대여이력 정보_2025\서울특별시 공공자전거 대여이력 정보_2501.csv'
subway_csv = r'C:\Users\SBA\Desktop\tyk_python_project\data\2025서울교통공사_역별 일별 시간대별 승하차인원_20251231.csv'
temperature_csv = r'C:\Users\SBA\Desktop\tyk_python_project\data\OBS_ASOS_TIM_20260810140632.csv'

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

def csv_check (csv_path:str, csv_name:str ,enc:str='cp949'):
    print(csv_name)
    df = pd.read_csv(csv_path, encoding=enc, nrows=10)
    print('\n===== 데이터 =====')
    print(df)
    print('데이터 크기 : ')
    print(df.shape)
    print('\n컬럼명 : ')
    print(df.columns.tolist())
    print('\n데이터 타입 : ')
    print(df.dtypes)
    print('='*30)

csv_check(bicycle_csv,'따릉이')
csv_check(subway_csv,'지하철')
csv_check(temperature_csv,'기온')