# import neccessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# upload csv file
df = pd.read_csv("municipality_bus_utilization.csv")
# convert the timestamp column to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# seperate municipaly id in order to use
municipaly_id = df.iloc[:,1:2].values
municipaly_id = pd.DataFrame(data=municipaly_id, columns = ['municipaly_id'])

remain = df.iloc[:,2:4].values
remain = pd.DataFrame(data=remain, columns = ['usage', 'total_capacity'])

# concatenate processed id with remain data
data = pd.concat([df['timestamp'], municipaly_id], axis=1)
data = pd.concat([data, remain], axis=1)

# seperate train and test data
traindata = data.iloc[:10390,:]
testdata = data.iloc[10390:,:]

# sort the id form 0 to 9 regarding to data
traindata = traindata.sort_values(["municipaly_id","timestamp"], ascending = (True,True))
testdata = testdata.sort_values(["municipaly_id","timestamp"], ascending = (True,True))

# create the timestamp as an index
indexed_traindata = traindata.set_index(['timestamp'])
indexed_testdata = testdata.set_index(['timestamp'])

## normalize the dataset
scaler = MinMaxScaler()
train = scaler.fit_transform(indexed_traindata)
test = scaler.fit_transform(indexed_testdata)
print('TrainData:\n',train)
print('TestData:\n',test)

# combine last 3 lags and current time
# take recursively data
from keras.preprocessing.sequence import TimeseriesGenerator

n_input = 3
n_features = 3

generatorTrain = TimeseriesGenerator(train,train,length=n_input,batch_size=1)
generatorTest = TimeseriesGenerator(test,test,length=n_input,batch_size=1)

batch_0 = generatorTrain[0]
x,y = batch_0

# LSTM Model
model =Sequential()
model.add(LSTM(50, activation='tanh', input_shape=(n_input,n_features)))
model.add(Dense(n_features))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(generatorTrain,epochs=20,batch_size=72,shuffle=False)

predictions = []
first_batch = train[-n_input:]
current_batch =first_batch.reshape((1,n_input,n_features))
for i in range(len(test)):
  # get the prediction value for first batch
  current_pred = model.predict(current_batch)[0]
  # append the predition into the array
  predictions.append(current_pred)
  # remove the first value
  current_batch_rmv_first = current_batch[:,1:,:]
  # update the batch
  current_batch = np.append(current_batch_rmv_first,[[current_pred]],axis=1)

# invert the transformation to use original values
predictions_actual_scale = scaler.inverse_transform(predictions)
testdata_actual_scale = scaler.inverse_transform(test)
plt.plot(predictions_actual_scale[:,1])
plt.plot(testdata_actual_scale[:,1])

# calculate root mean squared error
from math import sqrt
from sklearn.metrics import mean_squared_error
testScore = sqrt(mean_squared_error(testdata_actual_scale[:,1],predictions_actual_scale[:,1]))
print('Test Score: %.2f RMSE' % (testScore))