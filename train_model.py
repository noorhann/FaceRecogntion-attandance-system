from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle
from joblib import dump, load
import numpy as np
from matplotlib import pyplot as plt

model = Sequential([Dense(128, input_shape = (128,), activation = 'relu'),
                Dense(64, activation = 'relu', kernel_initializer = 'he_uniform'),
                Dense(64, activation = 'relu',  kernel_initializer = 'he_uniform'),
                Dense(32, activation = 'relu',  kernel_initializer = 'he_uniform'),
                Dense(32, activation = 'relu',  kernel_initializer = 'he_uniform'),
                Dense(27, activation = 'softmax')])


model.compile(loss='sparse_categorical_crossentropy', optimizer = 'adam' ,metrics=['accuracy'])

# load the face embeddings
print("[INFO] loading face embeddings...")
data = load("output/embeddings.joblib")

# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

embeddings = data["embeddings"]
embeddings = np.array(embeddings)
h = model.fit(embeddings, labels, epochs = 50)

plt.plot(h.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

plt.plot(h.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()


model.save('recognizer.h5')
with open('output/le.joblib', 'wb') as f:  
        dump(le, f)
f.close()

