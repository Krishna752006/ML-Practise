import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

apple_df = pd.read_csv('AAPL.csv')
apple_df = apple_df[apple_df['Volume'] != 0]

apple_df['Date'] = pd.to_datetime(apple_df['Date'])
apple_df.set_index('Date', inplace = True)
apple_df.sort_index(inplace = True)

apple_df['Log_Volume'] = np.log(apple_df['Volume'])
apple_df.drop(columns = ['Volume'], inplace = True)

scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(apple_df[apple_df.columns])
apple_scaled_df = pd.DataFrame(scaled_values, columns = apple_df.columns, index = apple_df.index)

'''
# To See their Plots
for i in apple_scaled_df.columns:
    plt.rcParams['figure.figsize'] = (15, 15)
    plt.plot(apple_scaled_df.index,apple_scaled_df[i])
    plt.title(i)
    plt.show()
'''


def create_sequence(data, window_size):
    x = []
    y = []
    for i in range(window_size, len(data)):
        x.append(data.iloc[i-window_size:i].values)
        y.append(data.iloc[i].values)
    return np.array(x), np.array(y)

x, y = create_sequence(apple_scaled_df, window_size = 60)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

model = keras.Sequential([
    
    keras.layers.Input(shape = (x_train.shape[1], x_train.shape[2])),
    keras.layers.Dropout(0.4),

    keras.layers.LSTM(50, return_sequences=True),
    keras.layers.Dropout(0.4),

    keras.layers.LSTM(50, return_sequences=False),
    keras.layers.Dropout(0.4),

    keras.layers.Dense(y_train.shape[1])
])

model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['RootMeanSquaredError'])

early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10, restore_best_weights = True)
model.fit(x_train, y_train,epochs = 100,validation_split = 0.2,batch_size = 5,callbacks = [early_stopping])

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
y_test_rescaled = scaler.inverse_transform(y_test)