import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox, ttk
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as RMSE
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Tạo một lớp cho ứng dụng
class GoldPricePredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("Dự đoán giá vàng")
        
        # Tạo giao diện nhập liệu
        label_title = tk.Label(root, text="Nhập thông tin :", font=("Arial Bold", 10), fg="red")
        label_title.grid(row=0, column=1, padx=40, pady=10)

        self.create_input_fields()

        self.model = self.train_models()

        button_predict = ttk.Button(root, text="Dự đoán", command=self.get_data_form)
        button_predict.grid(row=10, column=1, pady=10)

        self.result_label = tk.Label(root, text="")
        self.result_label.grid(row=11, column=1)

    def create_input_fields(self):
        # Các trường nhập liệu
        labels = ['Giá tại thời điểm mở cửa thị trường:', 
                  'Giá cao nhất trong ngày:', 
                  'Giá thấp nhất trong ngày:', 
                  'Khối lượng giao dịch:', 
                  '% Thay đổi so với giá trước đó:']
        
        self.entry_boxes = []

        for i, label in enumerate(labels):
            label_widget = tk.Label(self.root, text=label)
            label_widget.grid(row=i+1, column=1, pady=10)
            entry_box = tk.Entry(self.root)
            entry_box.grid(row=i+1, column=2, pady=10)
            self.entry_boxes.append(entry_box)

            # Gán giá trị mặc định
            if i == 0:
                entry_box.insert(0, "29678")
            elif i == 1:
                entry_box.insert(0, "30050")
            elif i == 2:
                entry_box.insert(0, "29678")
            elif i == 3:
                entry_box.insert(0, "3140")
            elif i == 4:
                entry_box.insert(0, "1.47")

    def train_models(self):
        # Tải và xử lý dữ liệu
        df = pd.read_csv('Gold_Price.csv')  # Đường dẫn đến file CSV của bạn
        X = df.drop(columns=['Price', 'Date'])  # Loại bỏ cột 'Price' và 'Date'
        y = df['Price']

        # Chia dữ liệu thành tập huấn luyện và kiểm thử (70% huấn luyện, 30% kiểm thử)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Huấn luyện các mô hình
        model_lr = LinearRegression().fit(X_train, y_train)
        model_ridge = Ridge().fit(X_train, y_train)
        model_nn = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500).fit(X_train, y_train)

        # Stacking model
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

    def get_data_form(self):
        # Nhận dữ liệu từ giao diện
        try:
            input_data = np.array([float(entry.get()) for entry in self.entry_boxes]).reshape(1, -1)
            
            # Dự đoán bằng các mô hình đã huấn luyện
            predictions = {name: model.predict(input_data)[0] for name, model in self.model.items()}
            
            # Hiển thị kết quả
            result_text = "Kết quả dự đoán:\n"
            for name, pred in predictions.items():
                result_text += f"{name}: {pred:.2f}\n"
            self.result_label.configure(text=result_text)
        except ValueError:
            messagebox.showinfo("Thông báo", "Hãy nhập đầy đủ và đúng định dạng thông tin!")

# Khởi tạo ứng dụng Tkinter
if __name__ == "__main__":
    root = tk.Tk()
    app = GoldPricePredictor(root)
    root.mainloop()
