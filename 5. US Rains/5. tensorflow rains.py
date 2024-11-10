import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow import keras
import tensorflow as tf
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder

data = pd.read_csv('5. us rains.csv')
data['Date'] = pd.to_datetime(data['Date'])
current_date = pd.to_datetime(datetime.now().date())
data = data[data['Date']<=current_date]
data.set_index('Date', inplace=True)

encoder = LabelEncoder()
data['Location'] = encoder.fit_transform(data['Location'])

x = data.drop(['Rain Tomorrow'], axis=1).values
y = data['Rain Tomorrow'].values.reshape(-1, 1)

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

model = keras.models.Sequential([
    keras.Input(shape=(7,)), 
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid') 
]) 

loss = keras.losses.BinaryCrossentropy()
optim = keras.optimizers.Adam(learning_rate=0.01)
metrics = ['accuracy']
model.compile(loss=loss, optimizer=optim, metrics=metrics)

batch_size = 64
epochs = 5
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True,verbose = 1)

model.evaluate(x_test, y_test, batch_size=batch_size, verbose = 1)

predictions = model(x_test)
#print(predictions[5][0])
y_pred = tf.cast(predictions > 0.5, tf.int32)
print(y_pred[9][0])
print(y_test[9][0])

'''
model.save("5.keras")
new_model = keras.models.load_model("5.keras")
new_model.evaluate(x_test,y_test,verbose = True)
'''