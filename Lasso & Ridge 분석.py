from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 데이터 불러오기
df = pd.read_excel("C:\\Users\\관리자\\Desktop\\데이터요약_한식.xlsx", header=None)

# 행 이름 설정
df.index = ['traffic', 'population', 'markets','consumer prices','sales']

# 데이터를 열 중심으로 변환
df = df.transpose()

# 독립 변수(X)와 종속 변수(y) 설정
X = df[['traffic', 'population', 'markets', 'consumer prices']]
y = df['sales']

# 상수항 추가 (회귀 분석에서는 상수항 필요)
X = sm.add_constant(X)

# 데이터를 표준화 (Lasso와 Ridge는 데이터 스케일에 민감)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Lasso 분석 (교차 검증으로 최적의 알파 선택)
lasso = LassoCV(cv=5, random_state=42).fit(X_scaled, y)
lasso_best_alpha = lasso.alpha_
print(f"Lasso 최적 알파 값: {lasso_best_alpha}")

# Ridge 분석 (교차 검증으로 최적의 알파 선택)
ridge = RidgeCV(alphas=np.logspace(-6, 6, 13), cv=5).fit(X_scaled, y)
ridge_best_alpha = ridge.alpha_
print(f"Ridge 최적 알파 값: {ridge_best_alpha}")

# Lasso 회귀 결과
lasso_coef = pd.Series(lasso.coef_, index=X.columns)
print("\nLasso 분석 변수 중요도:")
print(lasso_coef)

# Ridge 회귀 결과
ridge_coef = pd.Series(ridge.coef_, index=X.columns)
print("\nRidge 분석 변수 중요도:")
print(ridge_coef)

# 예측값 생성 및 성능 평가
y_pred_lasso = lasso.predict(X_scaled)
y_pred_ridge = ridge.predict(X_scaled)

lasso_mse = mean_squared_error(y, y_pred_lasso)
ridge_mse = mean_squared_error(y, y_pred_ridge)

print(f"\nLasso MSE: {lasso_mse}")
print(f"Ridge MSE: {ridge_mse}")

# 변수 중요도 시각화
plt.figure(figsize=(10, 6))
lasso_coef.plot(kind='bar', alpha=0.7, label='Lasso', color='blue')
ridge_coef.plot(kind='bar', alpha=0.7, label='Ridge')
plt.title('Lasso vs Ridge Variable Importance')
plt.legend()
plt.show()

# 실제값 vs 예측값 시각화
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred_lasso, alpha=0.7, label='Lasso Predicted')
plt.scatter(y, y_pred_ridge, alpha=0.7, label='Ridge Predicted', marker='x')
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--', label='Ideal Fit')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales (Lasso & Ridge)")
plt.legend()
plt.show()
