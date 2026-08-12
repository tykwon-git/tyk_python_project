import pandas as pd
import holidays 

# --------------
# 전처리 끝난 data path
# --------------
bicycle_path = r'data\processed\bicycle_hourly.csv'
subway_path = r'data\processed\subway_hourly.csv'
temperature_path = r'data\processed\temperature_hourly.csv'

bicycle = pd.read_csv(bicycle_path,encoding='utf-8-sig')
subway = pd.read_csv(subway_path,encoding='utf-8-sig')
temperature = pd.read_csv(temperature_path,encoding='utf-8-sig')

# print('===== 데이터 크기 =====')
# print(f'따릉이 : {bicycle.shape}')
# print(f'지하철 : {subway.shape}')
# print(f'기온   : {temperature.shape}')
# print('\n===== 컬럼 =====')
# print('따릉이 :', bicycle.columns.tolist())
# print('지하철 :', subway.columns.tolist())
# print('기온   :', temperature.columns.tolist())
# print('\n===== head() =====')
# print('따릉이 :\n', bicycle.head())
# print('지하철 :\n', subway.head())
# print('기온   :\n', temperature.head())
# ===== 데이터 크기 =====
# 따릉이 : (2196, 3)
# 지하철 : (2196, 3)
# 기온   : (2178, 7)
# ===== 컬럼 =====
# 따릉이 : ['날짜', '시간대', '따릉이대여건수']
# 지하철 : ['날짜', '시간대', '승차인원']
# 기온   : ['지점', '지점명', '일시', '기온(°C)', '날짜', '시간', '시간대']
# ===== head() =====
# 따릉이 :
#             날짜    시간대  따릉이대여건수
# 0  2025-03-01  06-07      787
# 지하철 :
#             날짜       시간대      승차인원
# 0  2025-03-01  06-07시간대   55215.0
# 기온   :
#      지점 지점명                   일시  기온(°C)          날짜  시간    시간대
# 0  108  서울  2025-03-01 06:00:00     3.4  2025-03-01   6  06-07

# --------------
# 필요한것만 다시 정리, 이제 날짜, 시간대, data 형태로 통일완료
# --------------
bicycle = bicycle[['날짜', '시간대', '따릉이대여건수']].copy()
subway = subway[['날짜', '시간대', '승차인원']].copy()
temperature = temperature[['날짜', '시간대', '기온(°C)']].copy()

# # --------------
# # csv 저장했다가 다시 읽으면 날짜가 str이라 datetime 처리
# # --------------
bicycle['날짜'] = pd.to_datetime(bicycle['날짜'])
subway['날짜'] = pd.to_datetime(subway['날짜'])
temperature['날짜'] = pd.to_datetime(temperature['날짜'])

# # --------------
# # 동일 문자열로 묶어야하니까 체크
# # --------------
# print('따릉이:', bicycle['시간대'].unique())
# print('지하철:', subway['시간대'].unique())
# print('기온:', temperature['시간대'].unique())
# ['06-07', '07-08', '08-09', '09-10', '10-11', '11-12', '12-13', '13-14',
#  '14-15', '15-16', '16-17', '17-18', '18-19', '19-20', '20-21', '21-22',
#  '22-23', '23-24']

# --------------
# key 중복 확인
# --------------
# key = ['날짜', '시간대']
# print(f'따릉이 중복: {bicycle.duplicated(key).sum():,}')
# print(f'지하철 중복: {subway.duplicated(key).sum():,}')
# print(f'기온 중복: {temperature.duplicated(key).sum():,}')

# #중복여부에서 안끝내고 완전히 같은 기간 범위인지 체크
# bicycle_keys = set(zip(bicycle['날짜'], bicycle['시간대']))
# subway_keys = set(zip(subway['날짜'], subway['시간대']))
# temperature_keys = set(zip(temperature['날짜'], temperature['시간대']))
# print(f'따릉이: {len(bicycle_keys):,}')
# print(f'지하철: {len(subway_keys):,}')
# print(f'기온: {len(temperature_keys):,}')
# print('따릉이 - 지하철:',len(bicycle_keys - subway_keys))
# print('따릉이 - 기온:',len(bicycle_keys - temperature_keys))
# print('지하철 - 기온:',len(subway_keys - temperature_keys))
# ~ : 2,196
# ~ - ~ : 0

# --------------
# 드디어 데이터 merge
# --------------
# bicycle에 subway merger, 공통키 날짜, 시간대 교집합 > 사실상 전부 다 
merged = bicycle.merge(subway,on=['날짜','시간대'],how='inner')
merged = merged.merge(temperature,on=['날짜','시간대'],how='inner')
merged = merged.rename(columns={'승차인원':'지하철승차인원'})
# print(merged.head(5))
#   날짜        시간대  따릉이대여건수  지하철승차인원  기온(°C)
# 0 2025-03-01  06-07   787           55215.0       3.4

# --------------
# 평일, 주말 처리 (공휴일, 대체 공휴일은 제거하려면 holidays를 import해서 처리?)
# --------------
merged['요일'] = merged['날짜'].dt.dayofweek # 월0 화1 수2 목3 금4 토5 일6 으로 변경처리
day_name = {0:'월',1:'화',2:'수',3:'목',4:'금',5:'토',6:'일'} # 요일값으로 map해서 요일명 
merged['요일명'] = merged['요일'].map(day_name) # 요일명 컬럼 추가
merged['주말여부'] = merged['요일']>=5 # 주말이면 true

kr_holidays = holidays.KR(years=2025) # 한국 법정 공휴일 및 대체 공휴일 처리 위한 lib
merged['공휴일여부'] = merged['날짜'].dt.date.isin(kr_holidays) # 공휴일이면 true
# for date, name in sorted(kr_holidays.items()):print(date, name)
# 2025-01-01 신정연휴 / 2025-01-27 임시공휴일 / 2025-01-28 설날 전날 / 2025-01-29 설날 / 2025-01-30 설날 다음날
# 2025-03-01 삼일절 / 2025-03-03 삼일절 대체 휴일 / 2025-05-05 부처님오신날; 어린이날 / 2025-05-06 부처님오신날 대체 휴일; 어린이날 대체 휴일
# 2025-06-03 대통령 선거일 / 2025-06-06 현충일 / 2025-08-15 광복절 / 2025-10-03 개천절 / 2025-10-05 추석 전날 / 2025-10-06 추석
# 2025-10-07 추석 다음날 / 2025-10-08 추석 대체 휴일 / 2025-10-09 한글날 / 2025-12-25 기독탄신일

# print(merged.head(60))
# print(merged.tail(60))

# --------------
# 따릉이, 지하철, 기온을 엮은 csv를 저장
# --------------
output_path = r'data\processed\analysis_dataset.csv'

merged = merged.reset_index(drop=True)
print(f'merged length : {len(merged)}')
merged.to_csv(output_path,index=False,encoding='utf-8-sig')