import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import datetime
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

df = quandl.get('WIKI/GOOGL') #updating dataframe from quandl server
#print(df)
df = df[['Adj. Open','Adj. High', 'Adj. Low', 'Adj. Close','Adj. Volume',]]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100
df['PCT_Change'] = (df['Adj. Open'] - df['Adj. Close']) / df['Adj. Open'] * 100
df = df[['Adj. Close','HL_PCT','PCT_Change','Adj. Volume']]
print(df.head())

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True) #check for empty data and assign some default value

forecast_out = int(math.ceil(0.01*len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)
#print(df.head())

X = np.array(df.drop(['label'],1)) #defining features with every column excluding label

X = preprocessing.scale(X) #scaling feature values (0 t0 1)
X_lately = X[-forecast_out:]
X = X[:-forecast_out:]
df.dropna(inplace=True)
y = np.array(df['label'])
y = np.array(df['label'])

X_train, X_test, y_train, y_test =  cross_validation.train_test_split(X,y, test_size=0.2) # shuffle the training set and get 0.2% of data as training data
#clf = svm.SVR() # svm algo
clf = LinearRegression() # linear regression algo
clf.fit(X_train,y_train) #train the data
accuracy = clf.score(X_test,y_test) #compare result
print(accuracy)
forecast_set = clf.predict(X_lately)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
#last_unix = last_date.Timestamp()
last_unix_temp = pd.to_datetime(last_date)
last_unix = last_unix_temp.value/ 1000000
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
































