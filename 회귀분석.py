import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error  # MSE 계산을 위해 추가

# 데이터 불러오기
df = pd.read_excel("C:\\Users\\관리자\\Desktop\\데이터요약_한식.xlsx", header=None)

# 행 이름 설정
df.index = ['traffic', 'population', 'markets', 'consumer prices', 'sales']

# 데이터를 열 중심으로 변환
df = df.transpose()

# 독립 변수(X)와 종속 변수(y) 설정
X = df[['traffic', 'population', 'markets', 'consumer prices']]
y = df['sales']

# 상수항 추가 (회귀분석에서는 상수항을 추가해야 함)
X = sm.add_constant(X)

# 다중공선성 확인 (VIF 계산)
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("다중공선성 (VIF):\n", vif_data)

# 회귀모델 생성 및 적합
model = sm.OLS(y, X).fit()

# 회귀분석 결과 출력
print(model.summary())

# 예측값 생성
y_pred = model.predict(X)

# OLS의 MSE 계산
mse = mean_squared_error(y, y_pred)
print(f"OLS의 MSE: {mse:.4f}")

# 실제값 vs 예측값 시각화
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, alpha=0.7, label='Actual vs Predicted')
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--', label='Ideal Fit')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.legend()
plt.show()

# 독립 변수 vs 매출 산점도
fig, axs = plt.subplots(1, 4, figsize=(24, 6))
for i, feature in enumerate(['traffic', 'population', 'markets', 'consumer prices']):
    axs[i].scatter(df[feature], y, alpha=0.7)
    axs[i].set_xlabel(feature)
    axs[i].set_ylabel('Sales')
    axs[i].set_title(f"{feature} vs Sales")
plt.tight_layout()
plt.show()
