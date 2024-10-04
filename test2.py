from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

# Đọc dữ liệu từ file CSV
data = pd.read_csv('Gold_Price.csv')

# Chuyển đổi cột Date sang định dạng ngày tháng và xử lý dữ liệu
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')

# Sử dụng các cột 'Open', 'High', 'Low', 'Volume' làm đặc trưng đầu vào (features), cột 'Price' là biến mục tiêu (target)
features = data[['Open', 'High', 'Low', 'Volume']].fillna(0)  # Xử lý missing data
target = data['Price']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Xây dựng các mô hình cơ bản (base models)
base_models = [
    ('linear_reg', LinearRegression()),
    ('ridge', Ridge(alpha=1.0)),
    ('mlp', MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42))
]

# Xây dựng mô hình stacking với Hồi quy tuyến tính làm meta-model
stacking_model = StackingRegressor(estimators=base_models, final_estimator=LinearRegression())

# Huấn luyện mô hình
stacking_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = stacking_model.predict(X_test)

# Tính toán Mean Squared Error để đánh giá hiệu suất mô hình
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"RMSE: {rmse}")
