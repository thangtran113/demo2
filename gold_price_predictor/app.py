from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import train_test_split

app = Flask(__name__)

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

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    actual_price = None
    errors = {}
    
    if request.method == "POST":
        input_data = [
            float(request.form['open']),
            float(request.form['high']),
            float(request.form['low']),
            float(request.form['volume']),
            float(request.form['chg'])
        ]
        actual_price = float(request.form['actual_price'])

        # Dự đoán bằng các mô hình đã huấn luyện
        predictions = {name: model.predict([input_data])[0] for name, model in models.items()}
        
        # Tính sai số
        for name, pred in predictions.items():
            absolute_error = abs(actual_price - pred)
            relative_error = (absolute_error / actual_price) * 100 if actual_price != 0 else 0
            errors[name] = {
                'pred': pred,
                'abs_error': absolute_error,
                'rel_error': relative_error
            }
        
        prediction = predictions

    return render_template("index.html", prediction=prediction, actual_price=actual_price, errors=errors)

if __name__ == "__main__":
    app.run(debug=True)

