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

data = pd.read_csv('6. student performance.csv')
data['Extracurricular Activities'] = data['Extracurricular Activities'].map({'Yes':1,'No':0})

x = data.drop(['Performance Index'], axis=1).values
y = data['Performance Index'].values.reshape(-1, 1)

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

model = keras.models.Sequential([
    keras.Input(shape=(5,)), 
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='linear') 
]) 

loss = keras.losses.MeanSquaredError()
optim = keras.optimizers.Adam(learning_rate=0.01)
metrics = [keras.metrics.MeanAbsoluteError()]
model.compile(loss=loss, optimizer=optim, metrics=metrics)

batch_size = 64
epochs = 10
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=1)

model.evaluate(x_test, y_test, batch_size=batch_size, verbose = 1)

predictions = model(x_test)
print(predictions[9][0])
print(y_test[9][0])

'''
model.save("6.keras")
new_model = keras.models.load_model("6.keras")
new_model.evaluate(x_test,y_test,verbose = True)
'''