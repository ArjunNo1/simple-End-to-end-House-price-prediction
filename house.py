import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv('D:\codes\House_Price_Prediction_Flask-main\House_Price_Prediction_Flask-main\house_data.csv')

columns = ['bedrooms', 'bathrooms', 'floors', 'yr_built', 'price']
df = df[columns]

X = df.drop(['price'], axis = 1)
y = df['price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

reg = LinearRegression()
reg.fit(X_train, y_train)

pickle.dump(reg, open('model.pkl', 'wb'))
