import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # Nhập thư viện matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

def NSE(y_test, y_predict):
    return (1 - (np.sum((y_predict - y_test) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)))

def MAE(y_test, y_predict):
    return mean_absolute_error(y_test, y_predict)

def RMSE(y_test, y_predict):
    return mean_squared_error(y_test, y_predict, squared=False)

def display_results(y_test, y_predict):
    print("\nThực tế và Dự đoán:")
    print("{:<15} {:<15} {:<15}".format("Giá thực tế", "Giá dự đoán", "Chênh lệch"))
    for actual, predicted in zip(y_test, y_predict):
        print("{:<15} {:<15} {:<15}".format(actual, predicted, abs(actual - predicted)))
    
    print("\nCác chỉ số hiệu suất:")
    print("Hệ số xác định (R^2):", r2_score(y_test, y_predict))
    print("NSE:", NSE(y_test, y_predict))
    print("MAE:", MAE(y_test, y_predict))
    print("RMSE:", RMSE(y_test, y_predict))

def plot_results(y_test, y_predict):
    plt.figure(figsize=(12, 6))
    plt.scatter(y_predict, y_test, color='blue', alpha=0.5, marker='o')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    plt.title('So sánh giữa Giá thực tế và Giá dự đoán')
    plt.xlabel('Thời gian')
    plt.ylabel('Giá vàng')
    plt.legend()
    plt.grid()
    plt.show()

dataframe = pd.read_csv('Gold_Price.csv') 
dt_train, dt_test = train_test_split(dataframe, test_size=0.3, shuffle=False)

X_train = dt_train.drop(['Date', 'Price'], axis=1) 
y_train = dt_train['Price'] 
X_test = dt_test.drop(['Date', 'Price'], axis=1)
y_test = dt_test['Price']

reg = LinearRegression().fit(X_train, y_train)
y_predict = reg.predict(X_test)
y_test = np.array(y_test)

# Gọi hàm display_results để hiển thị kết quả
display_results(y_test, y_predict)

# Gọi hàm plot_results để vẽ đồ thị
plot_results(y_test, y_predict)
