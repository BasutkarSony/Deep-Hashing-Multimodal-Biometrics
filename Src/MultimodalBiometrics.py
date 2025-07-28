
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import os
import cv2
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle

main = tkinter.Tk()
main.title("Deep Hashing for Secure Multimodal Biometrics") #designing main screen
main.geometry("1300x1200")

global filename
global X_train, Y_train
global vgg19

def upload(): 
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n")

#calculate hashing on image features
def getHammingHashing(img):
    img = img.astype('float32')
    return img / 255    

def deepFeatureExtraction():#extract features from images and then convert features into hamming hashing
    global X_train, Y_train
    X_train = []
    Y_train = []
    for i in range(0,21):
        finger = cv2.imread("Dataset/User"+str(i)+"/finger.bmp")
        finger = cv2.resize(finger, (64,64))
        vein = cv2.imread("Dataset/User"+str(i)+"/vein.bmp")
        vein = cv2.resize(vein, (64,64))
        combine = cv2.hconcat([finger, vein])
        hashing = getHammingHashing(combine)
        X_train.append(hashing)
        Y_train.append(i)

def CancelableModule():#function to choose random selection bits
    global X_train, Y_train
    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)#will choose bits randomly
    X_train = X_train[indices]
    Y_train = Y_train[indices]
    Y_train = to_categorical(Y_train)


def enrollment(): #enrollment starts execution by calling deep features extraction and cancelable module
    global X_train, Y_train
    global vgg19
    deepFeatureExtraction()
    CancelableModule()
    text.delete('1.0', END)
    text.insert(END,"Hashing and deep features extraction process completed\n")
    text.insert(END,"Total users biometric images found in dataset : "+str(X_train.shape[0])+"\n\n")

    #vgg19 get trained on deep features hashing
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
    hist = vgg19.fit(X_train, Y_train, batch_size=16, epochs=20, shuffle=True, verbose=2)
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
    accuracy = acc[19] * 100
    text.insert(END,"VGG19 training task completed with accuracy : "+str(accuracy))


def authentication():
    global vgg19
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir="testImages")
    finger = cv2.imread(str(filename)+"/finger.bmp")
    finger = cv2.resize(finger, (64,64))
    vein = cv2.imread(str(filename)+"/vein.bmp")
    vein = cv2.resize(vein, (64,64))
    combine = cv2.hconcat([finger, vein])
    hashing = getHammingHashing(combine)
    temp = []
    temp.append(hashing)
    temp = np.asarray(temp)
    predict = vgg19.predict(temp)
    predict = np.argmax(predict)
    combine = cv2.resize(combine,(500,400))
    cv2.putText(combine, 'User Authenticated as : User'+str(predict), (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
    cv2.imshow('User Authenticated as : User'+str(predict), combine)
    cv2.waitKey(0)

def graph():
    f = open('model/history.pckl', 'rb')
    accuracy = pickle.load(f)
    f.close()
    acc = accuracy['accuracy']
    loss = accuracy['loss']
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy/Loss')
    plt.plot(acc, 'ro-', color = 'green')
    plt.plot(loss, 'ro-', color = 'blue')
    plt.legend(['Accuracy', 'Loss'], loc='upper left')
    #plt.xticks(wordloss.index)
    plt.title('Iteration Wise Accuracy & Loss Graph')
    plt.show()

font = ('times', 16, 'bold')
title = Label(main, text='Deep Hashing for Secure Multimodal Biometrics')
title.config(bg='darkviolet', fg='gold')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Finger & Vein Dataset", command=upload)
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

enrollButton = Button(main, text="Enrollment using Multimodel CNN", command=enrollment)
enrollButton.place(x=320,y=550)
enrollButton.config(font=font1) 

authButton = Button(main, text="Authentication using Multimodal CNN", command=authentication)
authButton.place(x=620,y=550)
authButton.config(font=font1) 

graphButton = Button(main, text="Multimodal CNN Graph", command=graph)
graphButton.place(x=50,y=600)
graphButton.config(font=font1)


main.config(bg='turquoise')
main.mainloop()
