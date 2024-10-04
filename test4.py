import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import make_scorer, r2_score
import numpy as np

# 1. Đọc dữ liệu
df = pd.read_csv('Gold_Price.csv')

# 2. Chuẩn bị dữ liệu
X = df.drop(columns=['Price', 'Date'])  # Loại bỏ cột 'Price' và 'Date'
y = df['Price']  # Cột mục tiêu (Price)

# 3. Linear Regression với Cross-Validation
lin_reg = LinearRegression()

# Sử dụng cross_val_score với hàm đánh giá là R2 và 5 folds
scores_linear = cross_val_score(lin_reg, X, y, cv=5, scoring='r2')

# In ra kết quả của từng fold và giá trị trung bình
print(f'R^2 scores for each fold (Linear Regression): {scores_linear}')
print(f'Mean R^2 (Linear Regression): {np.mean(scores_linear)}')

# 4. Ridge Regression với Cross-Validation
ridge_reg = Ridge(alpha=1.0)

# Sử dụng cross_val_score với Ridge Regression
scores_ridge = cross_val_score(ridge_reg, X, y, cv=5, scoring='r2')

# In ra kết quả của từng fold và giá trị trung bình
print(f'R^2 scores for each fold (Ridge Regression): {scores_ridge}')
print(f'Mean R^2 (Ridge Regression): {np.mean(scores_ridge)}')
