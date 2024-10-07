import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import train_test_split

# Huấn luyện các mô hình
@st.cache_resource
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

# Tải mô hình
models = train_models()

# Giao diện người dùng
st.title("Dự đoán Giá Vàng")

# Đọc và hiển thị bảng CSV
df = pd.read_csv('Gold_Price.csv')  # Đường dẫn đến file CSV của bạn
st.subheader("Dữ liệu Giá Vàng")
st.dataframe(df)

# Nhập liệu từ người dùng
if 'input_data' not in st.session_state:
    st.session_state.input_data = {
        'Open': 29435,
        'High': 29598,
        'Low': 29340,
        'Volume': 2390,
        'Chg%': 0.25
    }

# Nhập liệu
open_price = st.number_input("Nhập giá mở cửa (Open):", value=st.session_state.input_data['Open'])
high_price = st.number_input("Nhập giá cao nhất (High):", value=st.session_state.input_data['High'])
low_price = st.number_input("Nhập giá thấp nhất (Low):", value=st.session_state.input_data['Low'])
volume = st.number_input("Nhập khối lượng (Volume):", value=st.session_state.input_data['Volume'])
chg = st.number_input("Nhập thay đổi (%) (Chg%):", value=st.session_state.input_data['Chg%'])

# Lưu số liệu nhập trước đó
st.session_state.input_data['Open'] = open_price
st.session_state.input_data['High'] = high_price
st.session_state.input_data['Low'] = low_price
st.session_state.input_data['Volume'] = volume
st.session_state.input_data['Chg%'] = chg

# Dự đoán khi nhấn nút
if st.button("Dự đoán"):
    input_data = [open_price, high_price, low_price, volume, chg]

    # Dự đoán bằng các mô hình đã huấn luyện
    predictions = {name: model.predict([input_data])[0] for name, model in models.items()}

    # Hiển thị kết quả dự đoán
    st.subheader("Kết quả dự đoán:")
    for name, pred in predictions.items():
        st.write(f"{name}: {pred:.2f}")

