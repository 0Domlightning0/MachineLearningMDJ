import tensorflow as tf

#py -m pip install tensorflow

import numpy as np
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

celsius_q    = np.array([1,2,3,4,5,6,7,8,9],  dtype=float)
fahrenheit_a = np.array([2,4,6,8,10,12,14,16,18],  dtype=float)

for i,c in enumerate(celsius_q):
  print("{} degrees Celsius = {} degrees Fahrenheit".format(c, fahrenheit_a[i]))

l0 = tf.keras.layers.Dense(units=1, input_shape=[1])

model = tf.keras.Sequential([l0])

model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1))

history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Finished training the model")

# Epoch = number of times the A.I reviews the data set.

# units=1 means 1 variable

#input Shape =  

import matplotlib.pyplot as plt
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])   

print(model.predict([100.0]))

print("These are the layer variables: {}".format(l0.get_weights()))
