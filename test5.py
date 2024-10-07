# Import các thư viện cần thiết
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

# Đọc dữ liệu từ file CSV
df = pd.read_csv('Gold_Price.csv')

# Loại bỏ cột 'Date' vì không cần thiết cho mô hình dự đoán
df = df.drop(columns=['Date'])

# Chia dữ liệu thành tập huấn luyện và kiểm tra theo tỷ lệ 7:3
X = df.drop(columns=['Price'])  # Các cột dùng để dự đoán
y = df['Price']  # Cột mục tiêu (giá vàng)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Kiểm tra kích thước của tập dữ liệu sau khi chia
#print(f"Kích thước tập huấn luyện: {X_train.shape}")
#print(f"Kích thước tập kiểm tra: {X_test.shape}")

# 1. Áp dụng thuật toán hồi quy tuyến tính (Linear Regression)
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred_linear = linear_model.predict(X_test)

# Đánh giá mô hình hồi quy tuyến tính
mae_linear = mean_absolute_error(y_test, y_pred_linear)
rmse_linear = math.sqrt(mean_squared_error(y_test, y_pred_linear))
r2_linear = r2_score(y_test, y_pred_linear)

print(f"Linear Regression - MAE: {mae_linear}, RMSE: {rmse_linear}, R2: {r2_linear}")

# Kiểm tra xem có overfitting không
train_score_linear = linear_model.score(X_train, y_train)
test_score_linear = linear_model.score(X_test, y_test)
print(f"Train score (Linear Regression): {train_score_linear}")
print(f"Test score (Linear Regression): {test_score_linear}")

# 2. Áp dụng thuật toán Ridge Regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred_ridge = ridge_model.predict(X_test)

# Đánh giá mô hình Ridge Regression
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
rmse_ridge = math.sqrt(mean_squared_error(y_test, y_pred_ridge))
r2_ridge = r2_score(y_test, y_pred_ridge)

print(f"Ridge Regression - MAE: {mae_ridge}, RMSE: {rmse_ridge}, R2: {r2_ridge}")
# Kiểm tra xem có overfitting không
train_score_linear = ridge_model.score(X_train, y_train)
test_score_linear = ridge_model.score(X_test, y_test)
print(f"Train score (Ridge Regression): {train_score_linear}")
print(f"Test score (Ridge Regression): {test_score_linear}")

# 3. Áp dụng mô hình Neural Network (MLPRegressor)
nn_model = MLPRegressor(hidden_layer_sizes=(100,),  activation='relu',max_iter=500, random_state=42)
nn_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred_nn = nn_model.predict(X_test)

# Đánh giá mô hình Neural Network
mae_nn = mean_absolute_error(y_test, y_pred_nn)
rmse_nn = math.sqrt(mean_squared_error(y_test, y_pred_nn))
r2_nn = r2_score(y_test, y_pred_nn)

print(f"Neural Network - MAE: {mae_nn}, RMSE: {rmse_nn}, R2: {r2_nn}")
# Kiểm tra xem có overfitting không
train_score_linear = nn_model.score(X_train, y_train)
test_score_linear = nn_model.score(X_test, y_test)
print(f"Train score (Neural Network): {train_score_linear}")
print(f"Test score (Neural Network): {test_score_linear}")

# 4. Áp dụng mô hình Stacking
estimators = [('lr', LinearRegression()), ('ridge', Ridge())]
stacking_model = StackingRegressor(estimators=estimators, final_estimator=MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42))
stacking_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred_stacking = stacking_model.predict(X_test)

# Đánh giá mô hình Stacking
mae_stacking = mean_absolute_error(y_test, y_pred_stacking)
rmse_stacking = math.sqrt(mean_squared_error(y_test, y_pred_stacking))
r2_stacking = r2_score(y_test, y_pred_stacking)

print(f"Stacking Model - MAE: {mae_stacking}, RMSE: {rmse_stacking}, R2: {r2_stacking}")

# Vẽ biểu đồ giữa dự đoán và thực tế
plt.scatter(y_test, y_pred_linear, label="Linear Regression", alpha=0.5)
plt.scatter(y_test, y_pred_ridge, label="Ridge Regression", alpha=0.5)
plt.scatter(y_test, y_pred_nn, label="Neural Network", alpha=0.5)
plt.scatter(y_test, y_pred_stacking, label="Stacking Model", alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linewidth=2, label="Actual")
plt.xlabel("Giá thực tế")
plt.ylabel("Giá dự đoán")
plt.legend()
plt.title("Biểu đồ giữa dự đoán và thực tế của các mô hình")
plt.show()
