import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
import pandas as pd

# Giả sử bạn có một file CSV, thay thế đường dẫn tới file của bạn
data = pd.read_csv('Gold_Price.csv')

# Chia dữ liệu thành X và y (ví dụ: y là cột giá cần dự đoán, X là các đặc trưng đầu vào)
X = data[['Open', 'High', 'Low', 'Volume']]  # Các đặc trưng đầu vào
y = data['Price']  # Biến mục tiêu là cột giá

# Mô hình Ridge với K-Fold Cross-Validation
ridge_reg = Ridge(alpha=1.0)

# Sử dụng 5-fold Cross-Validation
scores_ridge = cross_val_score(ridge_reg, X, y, cv=5, scoring='r2')

# Hiển thị kết quả của từng fold
print(f'R^2 scores for each fold: {scores_ridge}')
print(f'Mean R^2: {np.mean(scores_ridge)}')
print(f'Standard Deviation of R^2: {np.std(scores_ridge)}')
