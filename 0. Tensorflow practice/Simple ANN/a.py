import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

mnist = keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()
# print(x_train.shape,y_train.shape)

#to normalise: 0,255 -> 0,1
x_train, x_test = x_train/255.0, x_test/255.0

'''
# To see 6 digits

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(x_train[i], cmap = 'gray')
plt.show()

'''

#model
model = keras.models.Sequential([
    keras.Input(shape = (28,28)),
    keras.layers.Flatten(),# Reduce the dimentions by one i.e (28,28) --> 784 which is 28*28
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dense(10)
])

'''
# Alternate method
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape = (28,28)))
model.add(keras.layers.Dense(128, activation = 'relu'))
model.add(keras.layers.Dense(10))
'''
# print(model.summary()) 
loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True)
optim = keras.optimizers.Adam(learning_rate = 0.001)
metrics = ['accuracy']

model.compile(loss = loss, optimizer = optim, metrics = metrics)

batch_size = 64
epochs = 5
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

model.save("a1.keras")
new_model = keras.models.load_model("a1.keras")
new_model.evaluate(x_test,y_test,verbose = True)