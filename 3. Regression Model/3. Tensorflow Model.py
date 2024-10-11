import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('3. house price.csv')
x = data.drop(['House_Price'], axis=1).values
y = data['House_Price'].values.reshape(-1, 1)

scalerx = StandardScaler()
scalery = StandardScaler()
x_scaled = scalerx.fit_transform(x)
y_scaled = scalery.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.2, random_state=42)

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
pred = scalery.inverse_transform(predictions)
y_actual = scalery.inverse_transform(y_test)
print(pred[1][0])
print(y_actual[1][0])

'''model.save("3.keras")
new_model = keras.models.load_model("3.keras")
new_model.evaluate(x_test,y_test,verbose = True)'''