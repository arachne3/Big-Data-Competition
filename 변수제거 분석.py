import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

# 데이터 불러오기
df = pd.read_excel("C:\\Users\\관리자\\Desktop\\데이터요약_한식.xlsx", header=None)

# 행 이름 설정
df.index = ['traffic', 'population', 'markets', 'consumer prices', 'sales']

# 데이터를 열 중심으로 변환
df = df.transpose()

# 독립 변수(X)와 종속 변수(y) 설정
X = df[['traffic', 'population', 'markets', 'consumer prices']]
y = df['sales']

# 1. 데이터 정규화 (Standard Scaling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# 상수항 추가 (회귀 분석에서는 상수항 필요)
X_scaled = sm.add_constant(X_scaled)

# 2. 다중공선성 점검 (VIF 계산)
vif_data = pd.DataFrame()
vif_data["Feature"] = X_scaled.columns
vif_data["VIF"] = [variance_inflation_factor(X_scaled.values, i) for i in range(X_scaled.shape[1])]
print("\n초기 다중공선성 (VIF):")
print(vif_data)

# 3. VIF가 높은 변수 제거 (예: VIF > 10인 변수 제거)
while vif_data["VIF"].max() > 10:
    high_vif = vif_data.sort_values(by="VIF", ascending=False).iloc[0]
    print(f"\nVIF가 높은 변수 제거: {high_vif['Feature']}")
    X_scaled = X_scaled.drop(columns=high_vif['Feature'])
    
    # VIF 다시 계산
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_scaled.columns
    vif_data["VIF"] = [variance_inflation_factor(X_scaled.values, i) for i in range(X_scaled.shape[1])]

print("\n최종 다중공선성 (VIF):")
print(vif_data)

# 4. 다중 선형 회귀 모델 적합
model = sm.OLS(y, X_scaled).fit()

# 회귀 분석 결과 출력
print("\n다중 선형 회귀 분석 결과:")
print(model.summary())

# 5. 예측값 생성
y_pred = model.predict(X_scaled)

# 6. 실제값 vs 예측값 시각화
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, alpha=0.7, label='Actual vs Predicted')
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--', label='Ideal Fit')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.legend()
plt.show()

# 7. 독립 변수 vs 매출 산점도
fig, axs = plt.subplots(1, len(X.columns), figsize=(24, 6))
for i, feature in enumerate(X.columns):
    if feature in X_scaled.columns:  # 제거된 변수 제외
        axs[i].scatter(df[feature], y, alpha=0.7)
        axs[i].set_xlabel(feature)
        axs[i].set_ylabel('Sales')
        axs[i].set_title(f"{feature} vs Sales")
plt.tight_layout()
plt.show()