import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# 데이터 불러오기
df = pd.read_excel("C:\\Users\\관리자\\Desktop\\데이터요약_한식.xlsx", header=None)

# 행 이름을 설정
df.index = ['traffic', 'population', 'markets','consumer price', 'sales']

# 데이터를 열 중심으로 변환
df = df.transpose()

# 데이터 표준화
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# 상관계수 계산
correlation_matrix = df_scaled[['traffic', 'population', 'markets', 'consumer price', 'sales']].corr()

# 히트맵 시각화
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix (Standardized Data)")
plt.show()