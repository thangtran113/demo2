import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import BaggingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

def NSE(y_test, y_predict):
    return (1 - (np.sum((y_predict - y_test) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)))

def MAE(y_test, y_predict):
    return mean_absolute_error(y_test, y_predict)

def RMSE(y_test, y_predict):
    return mean_squared_error(y_test, y_predict, squared=False)

dataframe = pd.read_csv('Gold_Price.csv') 
dt_train,dt_test = train_test_split(dataframe,test_size=0.3,shuffle=False)

# print("Train set:\n", dt_train)
# print("Test set:\n", dt_test)

X_train = dt_train.drop(['Date','Price'], axis = 1) 
y_train = dt_train['Price'] 
X_test= dt_test.drop(['Date','Price'], axis = 1)
y_test= dt_test['Price']

X_train=np.array(X_train).T 
y_train=np.array(y_train)
X_test=np.array(X_test).T
y_test=np.array(y_test)

# print(X_train.shape)

w = np.linalg.pinv(X_train@X_train.T)@X_train@y_train   #w = (X x X.T)+ x X x y
y_predict =  X_test.T@w


# print("Thuc te Du doan Chenh lech")
# for i in range(0,len(y_test)):
#     print(" ",y_test[i]," ",y_predict[i]," ", abs(y_test[i]-y_predict[i]))

print("Coef of determination LinearRegression chay:",r2_score(y_test,y_predict))
print("NSE LinearRegression: ", NSE(y_test,y_predict))
print('MAE LinearRegression:', MAE(y_test,y_predict))
print('RMSE LinearRegression:', RMSE(y_test,y_predict))


bagging_model = BaggingRegressor(estimator=LinearRegression(), n_estimators=10, random_state=0)
bagging_model.fit(X_train.T, y_train)
y_bagging_predict = bagging_model.predict(X_test.T)

print("\nBagging Model:")
print("Coef of determination:", r2_score(y_test, y_bagging_predict))
print("NSE:", NSE(y_test, y_bagging_predict))
print('MAE:', MAE(y_test, y_bagging_predict))
print('RMSE:', RMSE(y_test, y_bagging_predict))

# Huấn luyện mô hình Stacking
estimators = [('lr', LinearRegression()), ('dt', DecisionTreeRegressor())]
stacking_model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
stacking_model.fit(X_train.T, y_train)
y_stacking_predict = stacking_model.predict(X_test.T)

print("\nStacking Model:")
print("Coef of determination:", r2_score(y_test, y_stacking_predict))
print("NSE:", NSE(y_test, y_stacking_predict))
print('MAE:', MAE(y_test, y_stacking_predict))
print('RMSE:', RMSE(y_test, y_stacking_predict))