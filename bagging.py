import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # Nhập thư viện matplotlib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor, BaggingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Đọc dữ liệu
dataframe = pd.read_csv('Gold_Price.csv')
dt_train, dt_test = train_test_split(dataframe, test_size=0.3, shuffle=False)

def NSE(y_test, y_pred):
    return (1 - (np.sum((y_pred - y_test) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)))

def MAE(y_test, y_pred):
    return mean_absolute_error(y_test, y_pred)

def RMSE(y_test, y_pred):
    return mean_squared_error(y_test, y_pred, squared=False)

# Chọn các cột đặc trưng (features)
x_train = dt_train[['Open', 'High', 'Low', 'Volume', 'Chg%']]
y_train = dt_train['Price']  # Mục tiêu là 'Price'
x_test = dt_test[['Open', 'High', 'Low', 'Volume', 'Chg%']]
y_test = dt_test['Price']

# Khởi tạo các mô hình thành phần
model1 = LinearRegression()
model2 = Ridge(alpha=1.0, max_iter=1000, tol=0.01)
model3 = MLPRegressor(alpha=0.5, hidden_layer_sizes=(20, 20), max_iter=5000)

# Tạo mô hình Bagging cho Linear Regression
bagging_model = BaggingRegressor(estimator=model1, n_estimators=10, random_state=42)

# Tạo mô hình stacking
stacking_model = StackingRegressor(
    estimators=[
        ('linear', bagging_model),  # Sử dụng mô hình Bagging cho Linear Regression
        ('ridge', model2),
        ('mlp', model3)
    ],
    final_estimator=MLPRegressor()  # Có thể thay đổi mô hình cuối
)

# Huấn luyện mô hình stacking
stacking_model.fit(x_train, y_train)
y_pred = stacking_model.predict(x_test)

# Hiển thị kết quả
print("Thực tế và Dự đoán Chênh lệch")
print("{:<15} {:<15} {:<15}".format("Giá thực tế", "Giá dự đoán", "Chênh lệch"))
for actual, predicted in zip(y_test, y_pred):
    print("{:<15} {:<15} {:<15}".format(actual, predicted, abs(actual - predicted)))

print("\nHệ số xác định (R^2) Stacking: ", r2_score(y_test, y_pred))
print("NSE Stacking: ", NSE(y_test, y_pred))
print('MAE Stacking:', MAE(y_test, y_pred))
print('RMSE Stacking:', RMSE(y_test, y_pred), '\n')

# Hàm để vẽ biểu đồ so sánh giữa giá thực tế và giá dự đoán
def plot_results(y_test, y_pred):
    plt.figure(figsize=(12, 6))
    plt.scatter(y_pred, y_test, color='blue', alpha=0.5, marker='o')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')  # Đường y=x
    plt.title('So sánh giữa Giá thực tế và Giá dự đoán (Stacking Regression)')
    plt.xlabel('Thời gian')
    plt.ylabel('Giá vàng')
    plt.legend()
    plt.grid()
    plt.show()

# Gọi hàm plot_results để vẽ đồ thị
plot_results(y_test, y_pred)
