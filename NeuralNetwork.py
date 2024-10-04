import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt  # Nhập thư viện matplotlib
from sklearn.model_selection import train_test_split 
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn import preprocessing 

# Đọc dữ liệu
data = pd.read_csv('Gold_Price.csv') 
le = preprocessing.LabelEncoder() 
data = data.apply(le.fit_transform) 

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
dt_Train, dt_Test = train_test_split(data, test_size=0.3, shuffle=False) 
X_train = dt_Train.drop(['Date', 'Price'], axis=1) 
y_train = dt_Train['Price'] 
X_test = dt_Test.drop(['Date', 'Price'], axis=1)
y_test = dt_Test['Price']
y_test = np.array(y_test)

# Hàm NSE
def NSE(y_test, y_pred):
    return (1 - (np.sum((y_pred - y_test) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)))

# Huấn luyện mô hình MLPRegressor
clf = MLPRegressor(alpha=0.5, hidden_layer_sizes=(20, 20), max_iter=5000).fit(X_train, y_train)
y_predict = clf.predict(X_test) 

# In các chỉ số hiệu suất
print('Hệ số xác định (R^2) Mạng nơ-ron: ', r2_score(y_test, y_predict))
print('NSE Mạng nơ-ron: ', NSE(y_test, y_predict))
print('MAE Mạng nơ-ron: ', mean_absolute_error(y_test, y_predict))
print('RMSE Mạng nơ-ron: ', np.sqrt(mean_squared_error(y_test, y_predict)), '\n')

# Hàm để vẽ biểu đồ so sánh giữa giá thực tế và giá dự đoán
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

# Gọi hàm plot_results để vẽ đồ thị
plot_results(y_test, y_predict)
