from sklearn.model_selection import train_test_split
from autosklearn.regression import AutoSklearnRegressor
from sklearn import preprocessing
import sklearn.datasets
import sklearn.metrics
import matplotlib.pyplot as plt
import pandas as pd
le = preprocessing.LabelEncoder()

wdata = pd.read_csv('weather.csv')
wdata = wdata.drop(['Date.Full','Station.City','Station.Code','Station.Location','Data.Temperature.Max Temp','Data.Temperature.Min Temp'],axis=1)
wdata = wdata[wdata["Station.State"] == "Illinois"]
wdata['Station.State'] = le.fit_transform(wdata['Station.State'])

wdata = (wdata-wdata.min())/(wdata.max()-wdata.min())
print(wdata)

#wdata = wdata.iloc[1:1000]

pred = 'Data.Temperature.Avg Temp'

x = wdata.drop([pred], axis = 1)
y = wdata[pred]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 50)

automl = AutoSklearnRegressor(time_left_for_this_task = 240, per_run_time_limit = 30)
automl.fit(x_train, y_train)

print(automl.leaderboard())

y_hat = automl.predict(x_test)

print('Mean Absolute Error: ', sklearn.metrics.mean_absolute_error(y_test, y_hat))

train_predictions = automl.predict(x_train)
print("Train R2 score:", sklearn.metrics.r2_score(y_train, train_predictions))
test_predictions = automl.predict(x_test)
print("Test R2 score:", sklearn.metrics.r2_score(y_test, test_predictions))

#train_predictions = train_predictions[::111]
#test_predictions = test_predictions[::111]
#y_train = y_train[::111]
#y_test = y_test[::111]

plt.scatter(y_train, train_predictions, label="Train samples", c='#d95f02')
plt.scatter(y_test, test_predictions, label="Test samples", c='#7570b3')
plt.xlabel("Predicted value")
plt.ylabel("True value")
plt.legend()
plt.plot([0, 0.8], [0, 0.8], c='k', zorder=0)
plt.xlim([0, 0.8])
plt.ylim([0, 0.8])
plt.tight_layout()
plt.show()
