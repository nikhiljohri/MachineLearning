import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import datetime
import time
import matplotlib.pyplot as plt
import pickle
from matplotlib import style

style.use('ggplot')

df = quandl.get('WIKI/GOOGL') #updating dataframe from quandl server
df = df[['Adj. Open','Adj. High', 'Adj. Low', 'Adj. Close','Adj. Volume',]]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100
df['PCT_Change'] = (df['Adj. Open'] - df['Adj. Close']) / df['Adj. Open'] * 100
df = df[['Adj. Close','HL_PCT','PCT_Change','Adj. Volume']]

forecast_col = 'Adj. Close'
#df.fillna(-99999, inplace=True) #check for empty data and assign some default value
df.dropna(inplace=True)
df['Adj. Close'].plot()

forecast_out = int(math.ceil(0.01*len(df))) #33
df['label'] = df[forecast_col].shift(-forecast_out) #shift df['label'] 33 days up as we will predict for these 33 days


X = np.array(df.drop(['label'],1)) #defining features with every column excluding label
X = preprocessing.scale(X) #scaling feature values (0 t0 1)
X_lately = X[-forecast_out:] #x_lately is a period for which we will predict i.e last 33 days
X = X[:-forecast_out] #shifting x 33 days up as we will predict for these 33 days
df.dropna(inplace=True)
y = np.array(df['label'])


X_train, X_test, y_train, y_test =  cross_validation.train_test_split(X,y, test_size=0.2) # shuffle the training set and get 0.2% of data as training data
#clf = svm.SVR() # svm algo
clf = LinearRegression() # linear regression algo
clf.fit(X_train,y_train) #train the data

with open('linearregression.pickle','wb') as f: #pickle save the train in a file for future, so that it doesn't have to train again.
    pickle.dump(clf,f)

pickle_in = open('linearregression.pickle','rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test,y_test) #compare result
print(accuracy)
forecast_set = clf.predict(X_lately)

#everything below it is for plotting the graph with date as x axis.
df['Forecast'] = np.nan
print(df)
last_date = df.iloc[-1].name # getting last date of df. It will be 33 days back as we have removed last 33 days from df to make predictions of
print(last_date)
#last_unix = last_date.Timestamp()
last_unix = time.mktime(last_date.timetuple()) # converting date into seconds
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]
#df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
































