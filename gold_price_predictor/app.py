import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import train_test_split

# Huấn luyện các mô hình
def train_models():
    df = pd.read_csv('Gold_Price.csv')  # Đường dẫn đến file CSV của bạn
    X = df.drop(columns=['Price', 'Date'])
    y = df['Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model_lr = LinearRegression().fit(X_train, y_train)
    model_ridge = Ridge().fit(X_train, y_train)
    model_nn = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500).fit(X_train, y_train)

    estimators = [
        ('lr', model_lr),
        ('ridge', model_ridge),
        ('nn', model_nn)
    ]
    model_stack = StackingRegressor(estimators=estimators, final_estimator=Ridge()).fit(X_train, y_train)

    return {
        'LinearRegression': model_lr,
        'Ridge': model_ridge,
        'NeuralNetwork': model_nn,
        'Stacking': model_stack
    }

models = train_models()

# Giao diện người dùng bằng Streamlit
st.title("Dự đoán giá vàng bằng các mô hình học máy")

# Nhập liệu từ người dùng
open_price = st.number_input("Giá mở cửa (Open)", value=0.0)
high_price = st.number_input("Giá cao nhất (High)", value=0.0)
low_price = st.number_input("Giá thấp nhất (Low)", value=0.0)
volume = st.number_input("Khối lượng giao dịch (Volume)", value=0.0)
chg = st.number_input("Phần trăm thay đổi (Chg%)", value=0.0)
actual_price = st.number_input("Giá thực tế", value=0.0)

# Dự đoán khi người dùng nhấn nút
if st.button("Dự đoán"):
    input_data = [open_price, high_price, low_price, volume, chg]

    # Dự đoán bằng các mô hình đã huấn luyện
    predictions = {name: model.predict([input_data])[0] for name, model in models.items()}
    
    # Hiển thị kết quả dự đoán và sai số
    st.subheader("Kết quả dự đoán:")
    for name, pred in predictions.items():
        absolute_error = abs(actual_price - pred)
        relative_error = (absolute_error / actual_price) * 100 if actual_price != 0 else 0
        st.write(f"{name}:")
        st.write(f"  - Dự đoán: {pred:.2f}")
        st.write(f"  - Sai số tuyệt đối: {absolute_error:.2f}")
        st.write(f"  - Sai số tương đối: {relative_error:.2f}%")
