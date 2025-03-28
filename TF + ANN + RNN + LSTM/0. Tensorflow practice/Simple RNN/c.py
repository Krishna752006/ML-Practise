import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

mnist = keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()
# print(x_train.shape,y_train.shape)

#to normalise: 0,255 -> 0,1
x_train, x_test = x_train/255.0, x_test/255.0

#model
model = keras.models.Sequential()
model.add(keras.Input(shape=(28,28)))
#model.add(layers.SimpleRNN(128,return_sequences = True,activation = 'relu'))
model.add(layers.SimpleRNN(128,return_sequences = False,activation = 'relu'))
model.add(layers.Dense(10))

loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True)
optim = keras.optimizers.Adam(learning_rate = 0.001)
metrics = ['accuracy']

model.compile(loss = loss, optimizer = optim, metrics = metrics)

batch_size = 64
epochs = 10
model.fit(x_train,y_train,batch_size = batch_size,epochs=epochs,shuffle = True,verbose = 2)

#model evaluation
model.evaluate(x_test,y_test,batch_size=batch_size,verbose = 2)

probability_model = keras.models.Sequential([
    model,
    keras.layers.Softmax()
])

predictions = probability_model(x_test)
pred0 = predictions[0]
print(pred0)
label0 = np.argmax(pred0)
print(label0)

model.save("c1.keras")
new_model = keras.models.load_model("c1.keras")
new_model.evaluate(x_test,y_test,verbose = True)