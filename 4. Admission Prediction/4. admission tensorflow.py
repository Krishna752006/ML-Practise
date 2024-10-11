import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('4. Admission prediction.csv')
data = data.drop(['Serial No.'], axis=1)
x = data.drop(['Chance of Admit '], axis=1).values
y = data['Chance of Admit '].values.reshape(-1, 1)

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)


x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

model = keras.models.Sequential([
    keras.Input(shape=(7,)), 
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='linear') 
])

loss = keras.losses.MeanSquaredError()
optim = keras.optimizers.Adam(learning_rate=0.01)
metrics = [keras.metrics.MeanAbsoluteError()]
model.compile(loss=loss, optimizer=optim, metrics=metrics)

batch_size = 64
epochs = 20
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2)

model.evaluate(x_test, y_test, batch_size=batch_size, verbose=2)

predictions = model(x_test)
prediction_value = np.round(predictions[0],2)
print(prediction_value[0])
print(y_test[0][0])

'''
model.save("4.keras")
new_model = keras.models.load_model("4.keras")
new_model.evaluate(x_test,y_test,verbose = True)
'''