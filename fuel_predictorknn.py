import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

data= pd.read_csv("tempfuel_data.csv")
x= data[["temp"]]
y= data[["fuel"]]

model = KNeighborsRegressor(n_neighbors=3)

model.fit(x, y)
print(model.predict([[58]]))