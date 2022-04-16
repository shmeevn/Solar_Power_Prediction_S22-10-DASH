import sklearn
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model
from sklearn import datasets, svm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
le = preprocessing.LabelEncoder()

wdata = pd.read_csv('weather.csv')
wdata = wdata.drop(['Date.Full','Station.City','Station.Code','Station.Location','Data.Temperature.Max Temp','Data.Temperature.Min Temp'],axis=1)

wdata['Station.State'] = le.fit_transform(wdata['Station.State'])

wdata = (wdata-wdata.min())/(wdata.max()-wdata.min())
print(wdata)

#wdata = wdata.iloc[1:1000]

pred = 'Data.Temperature.Avg Temp'

x = wdata.drop([pred], axis = 1)
y = wdata[pred]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 50)

rfmodel = RandomForestRegressor()
rfmodel.fit(x_train, y_train)
rfpred = rfmodel.predict(x_test)

print("Random Forest R2 Score:", sklearn.metrics.r2_score(y_test, rfpred))
print("Random Forest MAE:", sklearn.metrics.mean_absolute_error(y_test, rfpred))

gbmodel = GradientBoostingRegressor()
gbmodel.fit(x_train, y_train)
gbpred = gbmodel.predict(x_test)

print("Gradient Boosting R2 Score:", sklearn.metrics.r2_score(y_test, gbpred))
print("Gradient Boosting MAE:", sklearn.metrics.mean_absolute_error(y_test, gbpred))

plt.scatter(y_test, rfpred, label="Random Forest", c='#7570b3')
plt.scatter(y_test, gbpred, label="Gradient Boosting", c='#d95f02')
plt.xlabel("Predicted value")
plt.ylabel("True value")
plt.legend()
plt.plot([0, 1], [0, 1], c='k', zorder=0)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.tight_layout()
plt.show()
