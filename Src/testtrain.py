import os
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle



X_train = []
Y_train = []

def getHammingHashing(img):
    img = img.astype('float32')
    return img / 255


for i in range(0,21):
    finger = cv2.imread("Dataset/User"+str(i)+"/finger.bmp")
    finger = cv2.resize(finger, (64,64))
    
    vein = cv2.imread("Dataset/User"+str(i)+"/vein.bmp")
    vein = cv2.resize(vein, (64,64))
    
    combine = cv2.hconcat([finger, vein])
    hashing = getHammingHashing(combine)
    X_train.append(hashing)
    Y_train.append(i)
    print(i)

        
X_train = np.asarray(X_train)
Y_train = np.asarray(Y_train)
print(Y_train)

test = X_train[3]
cv2.imshow("aa",test)
cv2.waitKey(0)
indices = np.arange(X_train.shape[0])
np.random.shuffle(indices)
X_train = X_train[indices]
Y_train = Y_train[indices]
Y_train = to_categorical(Y_train)

print(Y_train)
if os.path.exists('model/model.json'):
    with open('model/model.json', "r") as json_file:
        loaded_model_json = json_file.read()
        classifier = model_from_json(loaded_model_json)
    classifier.load_weights("model/model_weights.h5")
    classifier._make_predict_function()   
    print(classifier.summary())
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    acc = data['accuracy']
    accuracy = acc[9] * 100
    print("Training Model Accuracy = "+str(accuracy))
else:
    vgg19 = Sequential()
    vgg19.add(Convolution2D(32, 3, 3, input_shape = (64, 128, 3), activation = 'relu'))
    vgg19.add(MaxPooling2D(pool_size = (2, 2)))
    vgg19.add(Convolution2D(32, 3, 3, activation = 'relu'))
    vgg19.add(MaxPooling2D(pool_size = (2, 2)))
    vgg19.add(Flatten())
    vgg19.add(Dense(output_dim = 256, activation = 'relu'))
    vgg19.add(Dense(output_dim = Y_train.shape[1], activation = 'softmax'))
    print(vgg19.summary())
    vgg19.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    hist = vgg19.fit(X_train, Y_train, batch_size=16, epochs=30, shuffle=True, verbose=2)
    vgg19.save_weights('model/model_weights.h5')            
    model_json = vgg19.to_json()
    with open("model/model.json", "w") as json_file:
        json_file.write(model_json)
    f = open('model/history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    acc = data['accuracy']
    accuracy = acc[29] * 100
    print("Training Model Accuracy = "+str(accuracy))

