import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # Nhập thư viện matplotlib
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

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

# Huấn luyện mô hình Ridge
clf = Ridge(alpha=1.0, max_iter=1000, tol=0.01)
rid = clf.fit(x_train, y_train)
y_pred = rid.predict(x_test)

# Hiển thị kết quả
print("Thực tế và Dự đoán Chênh lệch")
print("{:<15} {:<15} {:<15}".format("Giá thực tế", "Giá dự đoán", "Chênh lệch"))
for actual, predicted in zip(y_test, y_pred):
    print("{:<15} {:<15} {:<15}".format(actual, predicted, abs(actual - predicted)))

print("\nHệ số xác định (R^2) Ridge: ", r2_score(y_test, y_pred))
print("NSE Ridge: ", NSE(y_test, y_pred))
print('MAE Ridge:', MAE(y_test, y_pred))
print('RMSE Ridge:', RMSE(y_test, y_pred), '\n')

# Hàm để vẽ biểu đồ so sánh giữa giá thực tế và giá dự đoán
def plot_results(y_test, y_pred):
    plt.figure(figsize=(12, 6))
    plt.scatter(y_pred, y_test, color='blue', alpha=0.5, marker='o')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    plt.title('So sánh giữa Giá thực tế và Giá dự đoán (Ridge Regression)')
    plt.xlabel('Thời gian')
    plt.ylabel('Giá vàng')
    plt.legend()
    plt.grid()
    plt.show()

# Gọi hàm plot_results để vẽ đồ thị
plot_results(y_test, y_pred)
