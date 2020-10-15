# Handwritten Letter Recognition

## Importing Libraries


```python
# General purpose libraries
import os
import matplotlib.pyplot as plt
import numpy as np

# Import Tensorflow model and layers
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.backend import clear_session
from tensorflow.keras.optimizers import Adam

# Evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
```

## Loading and Scaling Images


```python
letters = 'abcdefghijklmnopqrstuvwxyz'

def prepare_data(letters, path = './data', oversampling=1):
    
    # Image data and classes
    mnist_x = []
    mnist_y = []

    for number, letter in enumerate(list(letters)):
        # Open a directory with current letter
        for image in os.listdir(path+'/'+letter):
            # read png only
            if image[-3:] == 'png': 
                # Reading image into array, taking only Alpha channel in RGBA
                mnist_x.append(plt.imread(path+'/'+letter+'/'+image)[:,:,3])
                # Adding class according to the letter, like so: 0 => 'A', 1 => 'B'...
                mnist_y.append(number)

    # Saving both as numpy array
    mnist_x = np.array(mnist_x)
    mnist_y = np.array(mnist_y)

    return mnist_x, mnist_y
```


```python
def scale(arr):
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    return arr
```


```python
mnist_x, mnist_y = prepare_data(letters)
mnist_x = scale(mnist_x)

print('Input shape:', mnist_x.shape)
print('Output shape:', mnist_y.shape)
```

    Input shape: (1732, 28, 28)
    Output shape: (1732,)


## Train Test Split


```python
X_train, X_test, y_train, y_test = train_test_split(mnist_x, mnist_y, test_size=0.1, random_state=42)
```

## Keras Model


```python
clear_session()

model = Sequential()

model.add(Conv1D(filters=28, kernel_size=2, strides=1, activation='relu', input_shape=(28, 28)))
model.add(MaxPooling1D(1))

model.add(Conv1D(filters=28, kernel_size=1, strides=1, activation='relu'))
model.add(MaxPooling1D(1))

model.add(Conv1D(filters=28, kernel_size=1, strides=1, activation='relu'))
model.add(MaxPooling1D(1))

model.add(Conv1D(filters=28, kernel_size=1, strides=1, activation='relu'))
model.add(MaxPooling1D(1))

model.add(Conv1D(filters=28, kernel_size=1, strides=1, activation='relu'))
model.add(MaxPooling1D(1))


model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(letters), activation='softmax'))

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(0.001),
    metrics=['accuracy'],
)
```


```python
#model.summary()
```

## Training and Evaluation


```python
history = model.fit(X_train, y_train, epochs=15, verbose=1)

plt.style.use('seaborn-whitegrid')
plt.plot(history.history['loss'])
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

y_pred = np.argmax(model.predict(X_train), axis=-1)
print('Model accuracy on Train:',accuracy_score(y_train, y_pred))

y_pred = np.argmax(model.predict(X_test), axis=-1)
print('Model accuracy on Test:',accuracy_score(y_test, y_pred))
```

    Epoch 1/15
    49/49 [==============================] - 0s 3ms/step - loss: 3.1758 - accuracy: 0.0982
    Epoch 2/15
    49/49 [==============================] - 0s 3ms/step - loss: 2.4654 - accuracy: 0.3030
    Epoch 3/15
    49/49 [==============================] - 0s 3ms/step - loss: 1.6955 - accuracy: 0.5205
    Epoch 4/15
    49/49 [==============================] - 0s 3ms/step - loss: 1.2028 - accuracy: 0.6547
    Epoch 5/15
    49/49 [==============================] - 0s 3ms/step - loss: 0.8360 - accuracy: 0.7471
    Epoch 6/15
    49/49 [==============================] - 0s 3ms/step - loss: 0.5632 - accuracy: 0.8363
    Epoch 7/15
    49/49 [==============================] - 0s 3ms/step - loss: 0.3894 - accuracy: 0.8883
    Epoch 8/15
    49/49 [==============================] - 0s 3ms/step - loss: 0.2831 - accuracy: 0.9178
    Epoch 9/15
    49/49 [==============================] - 0s 3ms/step - loss: 0.1767 - accuracy: 0.9506
    Epoch 10/15
    49/49 [==============================] - 0s 3ms/step - loss: 0.0976 - accuracy: 0.9775
    Epoch 11/15
    49/49 [==============================] - 0s 3ms/step - loss: 0.0579 - accuracy: 0.9891
    Epoch 12/15
    49/49 [==============================] - 0s 3ms/step - loss: 0.0402 - accuracy: 0.9917
    Epoch 13/15
    49/49 [==============================] - 0s 3ms/step - loss: 0.0698 - accuracy: 0.9782
    Epoch 14/15
    49/49 [==============================] - 0s 3ms/step - loss: 0.0894 - accuracy: 0.9711
    Epoch 15/15
    49/49 [==============================] - 0s 3ms/step - loss: 0.0435 - accuracy: 0.9917



![png](images/1.png)


    Model accuracy on Train: 0.9948652118100129
    Model accuracy on Test: 0.8390804597701149



```python
#confusion_matrix(y_test, y_pred)
```

## Saving


```python
model.save('model.h5')
```

## Exporting to TensorflowJS

Converter will create `model.json` and `group1-shard1of1.bin` files


```python
# pip install tensorflowjs
```


```python
# tensorflowjs_converter --input_format=keras --output_format=tfjs_layers_model ./model.h5 ./www/
```

## Load TensorflowJS


```python
const model = null;
async function loadModel() {
    model = await tf.loadLayersModel('./model.json');
}

loadModel();
```
