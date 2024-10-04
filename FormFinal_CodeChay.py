import numpy as np
import pandas as pd
from tkinter import messagebox, ttk, Label, Entry, Tk

#LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

form = Tk()  
form.title("Gold Price prediction")  
form.geometry("400x600")  

def NSE(y_test, y_predict):
    return (1 - (np.sum((y_predict - y_test) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)))

def MAE(y_test, y_predict):
    return mean_absolute_error(y_test, y_predict)

def RMSE(y_test, y_predict):
    return mean_squared_error(y_test, y_predict, squared=False)

data = pd.read_csv('Gold_Price.csv') 

X = np.array(data.drop(['Date','Price'], axis=1).values)    
y = np.array(data['Price'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3 , shuffle = False)

X_train=np.array(X_train).T 
y_train=np.array(y_train)
X_test=np.array(X_test).T
y_test=np.array(y_test)

w = np.linalg.pinv(X_train@X_train.T)@X_train@y_train

#reg = LinearRegression().fit(X_train,y_train)

def get_data_form(): 
    open = Textbox_open.get()
    high = Textbox_high.get()
    low = Textbox_low.get()
    volume = Textbox_volume.get()
    Chg = Textbox_Chg.get()

    if((open == '') or (high == '') or (low == '') or (volume == '') or (Chg == '')):
        messagebox.showinfo("Thông báo", "Hãy nhập đầy đủ thông tin!")
    else:
        X_input = np.array([float(open), float(high), float(low), float(volume), float(Chg)]).T
        y_input_predict1 = X_input.T@w
        lbl_result1.configure(text= str(y_input_predict1))

        # X_input = pd.DataFrame(np.array([open, high, low, volume, Chg]).reshape(1, -1))
        # #mảng X_input sẽ được chuyển thành một mảng có 1 hàng và số cột được xác định tự động (-1)
        # y_input_predict1 = reg.predict(X_input)
        # lbl_result1.configure(text= str(y_input_predict1))


lable_ten = Label(form, text = "Nhập thông tin :", font=("Arial Bold", 10), fg="red")
lable_ten.grid(row = 1, column = 1, padx = 40, pady = 10)

label_open = Label(form, text="Giá tại thời điểm mở cửa thị trường: ")
label_open.grid(row=2, column=1, pady=10)
Textbox_open = Entry(form)
Textbox_open.insert(0, "29678")
Textbox_open.grid(row=2, column=2, pady=10)

label_high = Label(form, text="Giá cao nhất trong ngày: ")
label_high.grid(row=3, column=1, pady=10)
Textbox_high = Entry(form)
Textbox_high.insert(0, "30050")
Textbox_high.grid(row=3, column=2, pady=10)

label_low = Label(form, text="Giá thấp nhất trong ngày:")
label_low.grid(row=4, column=1, pady=10)
Textbox_low = Entry(form)
Textbox_low.insert(0, "29678")
Textbox_low.grid(row=4, column=2, pady=10)

label_volume = Label(form, text="Khối lượng giao dịch: ")
label_volume.grid(row=5, column=1, pady=10)
Textbox_volume = Entry(form)
Textbox_volume.insert(0, "3140")
Textbox_volume.grid(row=5, column=2, pady=10)

label_Chg = Label(form, text="% Thay đổi so với giá trước đó: ")
label_Chg.grid(row=6, column=1, pady=10)
Textbox_Chg = Entry(form)
Textbox_Chg.insert(0, "1.47")
Textbox_Chg.grid(row=6, column=2, pady=10)

y_predict1 = X_test.T@w
# y_predict1 = reg.predict(X_test)

lbl1 = Label(form)
lbl1.grid(column=1, row=7)
lbl1.configure(text="Tỉ lệ dự đoán đúng của LinearRegression: "+'\n'
                            +"R2: "+ str(r2_score(y_test, y_predict1)*100)+"%"+'\n'
                            +"NSE: "+str(NSE(y_test, y_predict1)*100)+"%"+'\n'
                            +"MAE: "+str(MAE(y_test, y_predict1)*100)+"%"+'\n'
                            +"RMSE: "+str(RMSE(y_test, y_predict1))+'\n')

lbl_linearregression = Label(form, text="Kết quả dự đoán theo LinearRegression: ")
lbl_linearregression.grid(row = 8, column = 1)
lbl_result1 = Label(form, text="...")
lbl_result1.grid(row = 8, column = 2)

button = ttk.Button(form, text="Dự đoán LinearRegression", command=get_data_form)
button.grid(row=10, column=1, pady=10)

form.mainloop()
