import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# 1. Đọc dữ liệu
df = pd.read_csv('Gold_Price.csv')

# 2. Chuẩn bị dữ liệu
X = df.drop(columns=['Price', 'Date'])  # Loại bỏ cột 'Price' và 'Date'
y = df['Price']  # Cột mục tiêu (Price)

# 3. Chia dữ liệu thành tập huấn luyện và kiểm thử (70% huấn luyện, 30% kiểm thử)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Khởi tạo và huấn luyện mô hình Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# 5. Dự đoán trên tập huấn luyện và kiểm thử
y_train_pred = lin_reg.predict(X_train)
y_test_pred = lin_reg.predict(X_test)

# 6. Tính toán R^2 cho Linear Regression
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

# 7. In kết quả cho Linear Regression
print(f'R^2 on training set (Linear Regression): {r2_train}')
print(f'R^2 on test set (Linear Regression): {r2_test}')

# 8. Huấn luyện mô hình Ridge Regression
ridge_reg = Ridge(alpha=1.0)
ridge_reg.fit(X_train, y_train)

# 9. Dự đoán với Ridge Regression
y_train_pred_ridge = ridge_reg.predict(X_train)
y_test_pred_ridge = ridge_reg.predict(X_test)

# 10. Tính toán R^2 cho Ridge Regression
r2_train_ridge = r2_score(y_train, y_train_pred_ridge)
r2_test_ridge = r2_score(y_test, y_test_pred_ridge)

# 11. In kết quả cho Ridge Regression
print(f'R^2 on training set (Ridge Regression): {r2_train_ridge}')
print(f'R^2 on test set (Ridge Regression): {r2_test_ridge}')

# 12. Biểu đồ residuals cho mô hình Linear Regression
residuals = y_test - y_test_pred
plt.scatter(y_test, residuals)
plt.hlines(y=0, xmin=y_test.min(), xmax=y_test.max(), color='red')
plt.xlabel('Actual Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Actual Values (Linear Regression)')
plt.show()

# 13. Biểu đồ residuals cho mô hình Ridge Regression
residuals_ridge = y_test - y_test_pred_ridge
plt.scatter(y_test, residuals_ridge)
plt.hlines(y=0, xmin=y_test.min(), xmax=y_test.max(), color='red')
plt.xlabel('Actual Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Actual Values (Ridge Regression)')
plt.show()
