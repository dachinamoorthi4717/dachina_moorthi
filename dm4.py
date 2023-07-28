import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
l= datasets.load_diabetes()
df=pd.DataFrame(data=l.data,columns=l.feature_names)
print(df)
x=df["age"].values.reshape(-1,1)
y=df["bmi"].values.reshape(-1,1)
x_train=x
y_train=y
x_test=x
y_test=y
x_train,x_test,y_train,y_test==train_test_split(x,y,test_size=0.50,random_state=0)
print(x_train)
print(y_train)
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
r2_score(y_test,y_pred)
y_pred
plt.scatter(x_test,y_test)
plt.plot(x_test,y_pred)
plt.show()