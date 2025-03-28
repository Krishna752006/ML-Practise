import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow import keras
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('1. b_c_d.csv')
x = data.drop(['diagnosis'], axis=1).values
y = data['diagnosis'].map({'M':1,'B':0}).values.reshape(-1, 1)

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

model = keras.models.Sequential([
    keras.Input(shape=(13,)), 
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid') 
])

loss = keras.losses.BinaryCrossentropy()
optim = keras.optimizers.Adam(learning_rate=0.01)
metrics = ['accuracy']
model.compile(loss=loss, optimizer=optim, metrics=metrics)

batch_size = 64
epochs = 20
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2)

model.evaluate(x_test, y_test, batch_size=batch_size, verbose=2)

predictions = model(x_test)
print(predictions[5][0])
y_pred = tf.cast(predictions > 0.5, tf.int32)
print(y_pred[5][0])
print(y_test[5][0])

'''model.save("1.keras")
new_model = keras.models.load_model("1.keras")
new_model.evaluate(x_test,y_test,verbose = True)'''