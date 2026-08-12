import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 한글폰트 처리
import koreanize_matplotlib

# --------------
# preprocessed data path
# --------------
data_path = r'data\processed\analysis_dataset.csv'
df = pd.read_csv(data_path,encoding='utf-8-sig',)

# --------------
# --------------

# --------------
# 분석 시작
# 1차적으로 평일 + 비공휴일
# --------------
analysis_df = df[(~df['주말여부'])&~df['공휴일여부']].copy()
print(analysis_df.shape)
print(analysis_df.head())

# --------------
# 
# --------------