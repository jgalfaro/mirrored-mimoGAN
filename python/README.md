Supplementary Material for Pilot Contamination Attack Detection in Massive
MIMO Using Generative Adversarial Networks [Python code]
===


```python
# Import libraries and layers
import os
import numpy as np
from numpy import expand_dims
from numpy import zeros,ones
from numpy.random import randn
from numpy.random import randint
from keras.optimizers import Adam, RMSprop
from keras.layers.advanced_activations import LeakyReLU,ReLU
from keras.models import Sequential
from keras.layers import Dense,Reshape,Flatten,Conv2D,MaxPooling2D,Dropout,BatchNormalization,Conv2DTranspose
import matplotlib.pyplot as plt
import scipy.io
from scipy.io import loadmat
from keras.initializers import RandomNormal
```


```python
os.environ["KERAS_BACKEND"] = "tensorflow"
np.random.seed(10)
Y_noisy=scipy.io.loadmat('In_Data.mat');
Y_noisy = np.array(list(Y_noisy.values())[3])
Channels=scipy.io.loadmat('Out_Data.mat');
Channels = np.array(list(Channels.values())[3])
print(Y_noisy.shape)
print(Channels.shape)
```

```python
# design discriminator
# The input size if discriminator
Input_shape=(64,8,2)

# Discriminator network structure
def define_discriminator(Input_shape):
    init = RandomNormal(mean=0, stddev=0.02)
    model=Sequential()

    model.add(Conv2D(64,kernel_size=4,strides=2,input_shape=Input_shape,
                             padding='same',kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Dropout(rate = 0.4))

    model.add(Conv2D(128,kernel_size=4,strides=2,padding='same',kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Dropout(rate = 0.5))

    model.add(Conv2D(256,kernel_size=4,strides=2,padding='same',kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Dropout(rate = 0.4))

    model.add(Flatten())
    model.add(Dense(1,activation='linear'))

    model.compile(loss='mean_squared_error',optimizer=Adam(lr=0.0002, beta_1=0.5),
                          metrics=['accuracy'])
    return model
Figure1=define_discriminator(Input_shape)
Figure1.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    conv2d (Conv2D)              (None, 32, 4, 64)         2112
    _________________________________________________________________
    batch_normalization (BatchNo (None, 32, 4, 64)         256
    _________________________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 32, 4, 64)         0
    _________________________________________________________________
    dropout (Dropout)            (None, 32, 4, 64)         0
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 16, 2, 128)        131200
    _________________________________________________________________
    batch_normalization_1 (Batch (None, 16, 2, 128)        512
    _________________________________________________________________
    leaky_re_lu_1 (LeakyReLU)    (None, 16, 2, 128)        0
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 16, 2, 128)        0
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 8, 1, 256)         524544
    _________________________________________________________________
    batch_normalization_2 (Batch (None, 8, 1, 256)         1024
    _________________________________________________________________
    leaky_re_lu_2 (LeakyReLU)    (None, 8, 1, 256)         0
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 8, 1, 256)         0
    _________________________________________________________________
    flatten (Flatten)            (None, 2048)              0
    _________________________________________________________________
    dense (Dense)                (None, 1)                 2049
    =================================================================
    Total params: 661,697
    Trainable params: 660,801
    Non-trainable params: 896
    _________________________________________________________________



```python
# design autoencoder as generator
Input = (64,8,2)

# Generator network structure
def define_generator(Input):
    init = RandomNormal(mean=0, stddev=0.02)
    model=Sequential()
    # Encoder
    model.add(Conv2D(128,kernel_size=4,strides=2,input_shape=Input_shape,padding='same',kernel_initializer=init))
    model.add(ReLU(0.2))
    model.add(BatchNormalization())

    model.add(Conv2D(128,kernel_size=4,strides=1,input_shape=Input_shape,padding='same',kernel_initializer=init))
    model.add(ReLU(0.2))
    model.add(BatchNormalization())

    model.add(Conv2D(64,kernel_size=4,strides=2,padding='same',kernel_initializer=init))
    model.add(ReLU(0.2))
    model.add(BatchNormalization())

    model.add(Conv2D(32,kernel_size=4,strides=2,padding='same',kernel_initializer=init))
    model.add(ReLU(0.2))
    model.add(BatchNormalization())


    # Decoder
    model.add(Conv2DTranspose(32,kernel_size=4,strides=2,padding='same',kernel_initializer=init))
    model.add(ReLU(0.2))
    model.add(BatchNormalization())

    model.add(Conv2DTranspose(64,kernel_size=4,strides=1,padding='same',kernel_initializer=init))
    model.add(ReLU(0.2))
    model.add(BatchNormalization())

    model.add(Conv2DTranspose(128,kernel_size=4,strides=2,padding='same',kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(ReLU(0.2))

    model.add(Conv2DTranspose(2,kernel_size=4,strides=2,activation='tanh',padding='same',kernel_initializer=init))
    #generator.compile(loss='binary_crossentropy',optimizer=Adam(lr=0.0002,beta_1=0.5))

    return model
Figure2=define_generator(Input)
Figure2.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    conv2d_3 (Conv2D)            (None, 32, 4, 128)        4224
    _________________________________________________________________
    re_lu (ReLU)                 (None, 32, 4, 128)        0
    _________________________________________________________________
    batch_normalization_3 (Batch (None, 32, 4, 128)        512
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 32, 4, 128)        262272
    _________________________________________________________________
    re_lu_1 (ReLU)               (None, 32, 4, 128)        0
    _________________________________________________________________
    batch_normalization_4 (Batch (None, 32, 4, 128)        512
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 16, 2, 64)         131136
    _________________________________________________________________
    re_lu_2 (ReLU)               (None, 16, 2, 64)         0
    _________________________________________________________________
    batch_normalization_5 (Batch (None, 16, 2, 64)         256
    _________________________________________________________________
    conv2d_6 (Conv2D)            (None, 8, 1, 32)          32800
    _________________________________________________________________
    re_lu_3 (ReLU)               (None, 8, 1, 32)          0
    _________________________________________________________________
    batch_normalization_6 (Batch (None, 8, 1, 32)          128
    _________________________________________________________________
    conv2d_transpose (Conv2DTran (None, 16, 2, 32)         16416
    _________________________________________________________________
    re_lu_4 (ReLU)               (None, 16, 2, 32)         0
    _________________________________________________________________
    batch_normalization_7 (Batch (None, 16, 2, 32)         128
    _________________________________________________________________
    conv2d_transpose_1 (Conv2DTr (None, 16, 2, 64)         32832
    _________________________________________________________________
    re_lu_5 (ReLU)               (None, 16, 2, 64)         0
    _________________________________________________________________
    batch_normalization_8 (Batch (None, 16, 2, 64)         256
    _________________________________________________________________
    conv2d_transpose_2 (Conv2DTr (None, 32, 4, 128)        131200
    _________________________________________________________________
    batch_normalization_9 (Batch (None, 32, 4, 128)        512
    _________________________________________________________________
    re_lu_6 (ReLU)               (None, 32, 4, 128)        0
    _________________________________________________________________
    conv2d_transpose_3 (Conv2DTr (None, 64, 8, 2)          4098
    =================================================================
    Total params: 617,282
    Trainable params: 616,130
    Non-trainable params: 1,152
    _________________________________________________________________



```python
# GAN design
Generator=define_generator(Input)
Discriminator=define_discriminator(Input_shape)
def define_GAN(Generator,Discriminator):
    Discriminator.tainable=False
    model=Sequential()
    model.add(Generator)
    model.add(Discriminator)
    model.compile(loss='mean_squared_error',optimizer=Adam(lr=0.0002, beta_1=0.5))
    return model
Figure3=define_GAN(Generator,Discriminator)
Figure3.summary()
```

    Model: "sequential_4"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    sequential_2 (Sequential)    (None, 64, 8, 2)          617282
    _________________________________________________________________
    sequential_3 (Sequential)    (None, 1)                 661697
    =================================================================
    Total params: 1,278,979
    Trainable params: 1,276,931
    Non-trainable params: 2,048
    _________________________________________________________________



```python
# training
def Train(Discriminator, Generator, GAN, Epoch, batch_size):
    # load data
    #Xtrain = np.array([np.array(TrainData[4][ii][0]) for ii in range(len(TrainData[4]))])
    #YTrain = np.array([np.array(TrainData[5][ii][0]) for ii in range(len(TrainData[5]))])
    # add one size to Xtrain
    Y=Y_noisy.reshape(Y_noisy.shape[0],64,8,2)
    H=Channels.reshape(Channels.shape[0],64,8,2)

    #Xtrain=Xtrain.reshape(Xtrain.shape[0],128,400,1)
    Num_Batch=int(Y.shape[0]/batch_size)
    print("Num_Batch in each epoch:",Num_Batch)


    # prepare to save results
    #Rloss=[]
    #Floss=[]
    Raccuracy=list()
    Faccuracy=list()
    Dloss=list()
    #Accuracy=[]
    Gloss=list()


    for i in range(Epoch):
        for j in range (Num_Batch):

            # generate label 1 and 0 for real and fake data respectively
            Real_label=np.ones((batch_size,1))
            Fake_label=np.zeros((batch_size,1))

            # calculating loss and accuracy of discriminator on real and fake data
            Ch_index=np.random.randint(0,H.shape[0],batch_size)
            Real_data=H[Ch_index]

            Y_index=np.random.randint(0,Y.shape[0],batch_size)
            Fake=Y[Y_index]
            Fake_data=Generator.predict(Fake)


            Rloss,Racc=Discriminator.train_on_batch(Real_data,Real_label)
            #_,Racc_temp=discriminator.evaluate(Real_data,Real_label, verbose=0)

            Floss,Facc=Discriminator.train_on_batch(Fake_data,Fake_label)
            #_,Facc_temp=discriminator.evaluate(Fake_data,Fake_label, verbose=0)

            d_loss=0.5*np.add(Rloss,Floss)

            # calculating loss of generator and its training
            #discriminator.trainable=False
            Y_index=np.random.randint(0,Y.shape[0],batch_size)
            Fake=Y[Y_index]
            g_loss=GAN.train_on_batch(Fake,Real_label)

            #discriminator.trainable=True

            # show results for each batch size
            print("%d [Dloss: %f , Racc: %.2f, Facc: %.2f ] [Gloss: %f]" %
                   (j,d_loss,Racc,Facc,g_loss))

            Dloss.append(d_loss)
            #Accuracy.append((accuracy_temp))
            #Floss.append((Floss_temp))
            Raccuracy.append(Racc)
            Faccuracy.append(Facc)
            Gloss.append(g_loss)

        # save weights of both networks
        Generator.save('generator.h5')
        Discriminator.save('discriminator.h5')
        Generator.save_weights('weightG.h5')
        Discriminator.save_weights('weightsD.h5')

    plt.figure(figsize=(18,4))
    plt.subplot(1,2,1)
    plt.plot(range(Epoch*Num_Batch),Dloss,label='Discriminator')
    plt.plot(range(Epoch*Num_Batch),Gloss,label='Generator')
    #plt.plot(range(Epoch*Num_Batch),Gloss,label='Generator')
    plt.xlabel('Batch')
    plt.xlim(0,Epoch*Num_Batch,10)
    plt.ylabel('Loss')
    plt.title('Loss of discriminator and generator')
    plt.legend(prop={"size":18})
    plt.savefig('Loss.png')
    plt.show()

    plt.figure(figsize=(18,4))
    plt.subplot(1,2,2)
    #plt.plot(range(Epoch*Num_Batch),Accuracy,label='Discriminator')
    plt.plot(range(Epoch*Num_Batch),Raccuracy,label='Real')
    plt.plot(range(Epoch*Num_Batch),Faccuracy,label='Fake')
    plt.xlabel('Batch')
    plt.xlim(0,Epoch*Num_Batch,10)
    plt.ylabel('Accuracy')
    plt.title('Accuracy of discriminator on real and fake data')
    plt.legend(prop={"size":18})
    plt.savefig('Accuracy.png')
    plt.show()


Epoch=200
batch_size=64
Generator=define_generator(Input)
Discriminator=define_discriminator(Input_shape)
GAN=define_GAN(Generator,Discriminator)

Train(Discriminator, Generator, GAN, Epoch, batch_size)


```

    [Streaming output truncated to the last lines]
    1 [Dloss: 0.286162 , Racc: 0.92, Facc: 0.33 ] [Gloss: 0.317062]
    2 [Dloss: 0.263764 , Racc: 0.91, Facc: 0.41 ] [Gloss: 0.326628]
    3 [Dloss: 0.310396 , Racc: 0.95, Facc: 0.27 ] [Gloss: 0.316457]
    4 [Dloss: 0.259184 , Racc: 0.94, Facc: 0.34 ] [Gloss: 0.280944]
    5 [Dloss: 0.249212 , Racc: 0.92, Facc: 0.31 ] [Gloss: 0.295670]
    6 [Dloss: 0.287362 , Racc: 0.91, Facc: 0.33 ] [Gloss: 0.254748]
    7 [Dloss: 0.303926 , Racc: 0.92, Facc: 0.30 ] [Gloss: 0.279799]
    8 [Dloss: 0.267876 , Racc: 0.88, Facc: 0.42 ] [Gloss: 0.233002]
    9 [Dloss: 0.251394 , Racc: 0.83, Facc: 0.41 ] [Gloss: 0.310391]
    10 [Dloss: 0.280684 , Racc: 0.89, Facc: 0.28 ] [Gloss: 0.264651]
    11 [Dloss: 0.277831 , Racc: 0.81, Facc: 0.34 ] [Gloss: 0.250370]
    12 [Dloss: 0.286297 , Racc: 0.86, Facc: 0.39 ] [Gloss: 0.302438]
    13 [Dloss: 0.227025 , Racc: 0.91, Facc: 0.44 ] [Gloss: 0.272932]
    14 [Dloss: 0.329719 , Racc: 0.81, Facc: 0.25 ] [Gloss: 0.251994]
    15 [Dloss: 0.290339 , Racc: 0.88, Facc: 0.27 ] [Gloss: 0.274222]
    16 [Dloss: 0.267608 , Racc: 0.88, Facc: 0.36 ] [Gloss: 0.304878]
    17 [Dloss: 0.238372 , Racc: 0.86, Facc: 0.45 ] [Gloss: 0.298761]
    18 [Dloss: 0.263402 , Racc: 0.89, Facc: 0.34 ] [Gloss: 0.223732]
    19 [Dloss: 0.239629 , Racc: 0.91, Facc: 0.41 ] [Gloss: 0.275957]
    20 [Dloss: 0.293232 , Racc: 0.83, Facc: 0.28 ] [Gloss: 0.314177]
    21 [Dloss: 0.263066 , Racc: 0.94, Facc: 0.33 ] [Gloss: 0.277499]
    22 [Dloss: 0.238531 , Racc: 0.88, Facc: 0.44 ] [Gloss: 0.337812]
    23 [Dloss: 0.243028 , Racc: 0.86, Facc: 0.39 ] [Gloss: 0.308273]
    24 [Dloss: 0.284176 , Racc: 0.81, Facc: 0.33 ] [Gloss: 0.384925]
    25 [Dloss: 0.323969 , Racc: 0.89, Facc: 0.27 ] [Gloss: 0.248487]
    26 [Dloss: 0.291248 , Racc: 0.91, Facc: 0.27 ] [Gloss: 0.191670]
    27 [Dloss: 0.292860 , Racc: 0.92, Facc: 0.25 ] [Gloss: 0.307716]
    28 [Dloss: 0.261962 , Racc: 0.88, Facc: 0.38 ] [Gloss: 0.332461]
    29 [Dloss: 0.282566 , Racc: 0.89, Facc: 0.30 ] [Gloss: 0.298612]
    30 [Dloss: 0.265727 , Racc: 0.88, Facc: 0.34 ] [Gloss: 0.329520]
    31 [Dloss: 0.235253 , Racc: 0.89, Facc: 0.36 ] [Gloss: 0.285646]
    32 [Dloss: 0.284099 , Racc: 0.91, Facc: 0.30 ] [Gloss: 0.311845]
    33 [Dloss: 0.266527 , Racc: 0.95, Facc: 0.34 ] [Gloss: 0.328265]
    34 [Dloss: 0.287836 , Racc: 0.94, Facc: 0.31 ] [Gloss: 0.347010]
    35 [Dloss: 0.276176 , Racc: 0.91, Facc: 0.31 ] [Gloss: 0.325810]
    36 [Dloss: 0.278968 , Racc: 0.89, Facc: 0.31 ] [Gloss: 0.266439]
    37 [Dloss: 0.290556 , Racc: 0.78, Facc: 0.33 ] [Gloss: 0.207656]
    38 [Dloss: 0.269976 , Racc: 0.89, Facc: 0.33 ] [Gloss: 0.256243]
    39 [Dloss: 0.308555 , Racc: 0.88, Facc: 0.20 ] [Gloss: 0.238571]
    40 [Dloss: 0.265504 , Racc: 0.92, Facc: 0.30 ] [Gloss: 0.296756]
    41 [Dloss: 0.245455 , Racc: 0.91, Facc: 0.34 ] [Gloss: 0.317585]
    42 [Dloss: 0.222755 , Racc: 0.95, Facc: 0.48 ] [Gloss: 0.297113]
    43 [Dloss: 0.241186 , Racc: 0.92, Facc: 0.34 ] [Gloss: 0.263093]
    44 [Dloss: 0.301592 , Racc: 0.88, Facc: 0.28 ] [Gloss: 0.317357]
    45 [Dloss: 0.267987 , Racc: 0.83, Facc: 0.42 ] [Gloss: 0.305630]
    46 [Dloss: 0.255435 , Racc: 0.81, Facc: 0.33 ] [Gloss: 0.318768]
    47 [Dloss: 0.246620 , Racc: 0.88, Facc: 0.44 ] [Gloss: 0.327373]
    48 [Dloss: 0.308877 , Racc: 0.88, Facc: 0.31 ] [Gloss: 0.301185]
    49 [Dloss: 0.334929 , Racc: 0.89, Facc: 0.25 ] [Gloss: 0.227285]
    50 [Dloss: 0.261533 , Racc: 0.89, Facc: 0.47 ] [Gloss: 0.272019]
    51 [Dloss: 0.315578 , Racc: 0.95, Facc: 0.28 ] [Gloss: 0.305839]
    52 [Dloss: 0.292897 , Racc: 0.84, Facc: 0.33 ] [Gloss: 0.350356]
    53 [Dloss: 0.272475 , Racc: 0.91, Facc: 0.33 ] [Gloss: 0.309540]
    54 [Dloss: 0.255990 , Racc: 0.95, Facc: 0.38 ] [Gloss: 0.242184]
    55 [Dloss: 0.241786 , Racc: 0.97, Facc: 0.33 ] [Gloss: 0.334644]
    56 [Dloss: 0.229570 , Racc: 0.91, Facc: 0.39 ] [Gloss: 0.249060]
    57 [Dloss: 0.270615 , Racc: 0.83, Facc: 0.30 ] [Gloss: 0.325878]
    58 [Dloss: 0.268262 , Racc: 0.86, Facc: 0.30 ] [Gloss: 0.269157]
    59 [Dloss: 0.301215 , Racc: 0.92, Facc: 0.23 ] [Gloss: 0.420019]
    60 [Dloss: 0.249229 , Racc: 0.89, Facc: 0.41 ] [Gloss: 0.232897]
    61 [Dloss: 0.300320 , Racc: 0.88, Facc: 0.23 ] [Gloss: 0.263652]




![png](../figs/output_5_1.png)





![png](../figs/output_5_2.png)




```python
# uploading trained networks
os.environ["KERAS_BACKEND"] = "tensorflow"
np.random.seed(10)
Attack_Data=scipy.io.loadmat('Test_Data.mat');
Attack_Data = np.array(list(Attack_Data.values())[3])
print(Attack_Data.shape)
Channel_real=scipy.io.loadmat('Test_Channel.mat');
Channel_real = np.array(list(Channel_real.values())[3])
print(Channel_real.shape)
```


```python
Seed = 0.1
# Load trained networks
from sklearn.metrics import roc_curve, auc, confusion_matrix,accuracy_score
from sklearn.metrics import precision_recall_curve
from keras.models import load_model
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
import random
import math
import seaborn as sns
#Discriminator=define_discriminator(Input_shape)
#Discriminator=load_model('discriminator.h5')
#Generator=define_generator(Input)
#Generator=load_model('generator.h5')
Estimated_H=Generator.predict(Attack_Data)
realDis=Discriminator.predict(Channel_real)
fakeDis=Discriminator.predict(Estimated_H)

plt.figure(figsize=(4,4))
#plt.subplot(1,2,1)
sns.distplot(realDis, hist=True, kde=True,
             color = 'darkblue',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4},label='Normal Data')
plt.ylabel('Density')
plt.xlabel('Output')
#plt.xlim(-0.1,0.1,0.01)
plt.legend()
plt.figure(figsize=(4,4))
sns.distplot(fakeDis, hist=True, kde=True,
             color = '#E69F00',
             hist_kws={'edgecolor':'#D55E00'},
             kde_kws={'linewidth': 4},label='Abnormal Data')
plt.ylabel('Density')
plt.xlabel('Output')
plt.legend()
#plt.xlim(-0.01,0.01,0.01)
#plt.xlim('')
#plt.hist(realDis['realDis'], color = 'blue', edgecolor = 'black')
#plt.plot(realDis,'o',label='real')
#plt.ylim(-1,1,0.005)
#plt.show()
#plt.figure(figsize=(18,4))
#plt.subplot(1,2,2)
#plt.plot(fakeDis,'o',label='fake')
#plt.ylim(-0.04,0.02,0.005)
#plt.show()
print(fakeDis)
print(realDis)

```


    [[8.3677031e-07]
     [4.6080493e-07]
     [3.2140542e-07]
     [6.3176327e-07]
     [...]
     [ 2.03373376e-02]
     [-2.26563914e-03]
     [ 8.92654806e-03]]




![png](../figs/output_7_2.png)





![png](../figs/output_7_3.png)


```python
YPredLabel = abs(np.array([realDis, fakeDis]))
YPredLabel[YPredLabel<0.001] = 0
YPredLabel[YPredLabel>=0.001] = 1
YPredLabel.resize(2000,1)
print(YPredLabel)
Ylabel = np.array([ones([1000]), zeros([1000])])
Ylabel.resize(2000,1)
print(Ylabel)
# confusion matrix
Accuracy= confusion_matrix(Ylabel, YPredLabel)
precision = precision_score(Ylabel, YPredLabel, average='binary')
recall = recall_score(Ylabel, YPredLabel, average='binary')
score = f1_score(Ylabel, YPredLabel, average='binary')
print(precision)
print(recall)
print(score)
print(accuracy_score(Ylabel, YPredLabel))
print(Accuracy)
fpr, tpr, threshold = roc_curve(Ylabel, YPredLabel)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(10, 5))
plt.subplot(221)
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim(-0.05,1,0.2)
plt.ylim(-0.05,1,0.2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
```

    [[1.]
     [1.]
     [1.]
     ...
     [0.]
     [0.]
     [0.]]
    [[1.]
     [1.]
     [1.]
     ...
     [0.]
     [0.]
     [0.]]
    1.0
    0.953
    0.9759344598054276
    0.9765
    [[1000    0]
     [  47  953]]


    <matplotlib.legend.Legend at 0x7f0e223b4ed0>

![png](../figs/output_8_2.png)


```python
YPredLabel = abs(np.array([realDis, fakeDis]))
YPredLabel[YPredLabel<0.002] = 0
YPredLabel[YPredLabel>=0.002] = 1
YPredLabel.resize(2000,1)
print(YPredLabel)
Ylabel = np.array([ones([1000]), zeros([1000])])
Ylabel.resize(2000,1)
print(Ylabel)
# confusion matrix
Accuracy= confusion_matrix(Ylabel, YPredLabel)
precision = precision_score(Ylabel, YPredLabel, average='binary')
recall = recall_score(Ylabel, YPredLabel, average='binary')
score = f1_score(Ylabel, YPredLabel, average='binary')
print(precision)
print(recall)
print(score)
print(Accuracy)
print(accuracy_score(Ylabel, YPredLabel))
fpr, tpr, threshold = roc_curve(Ylabel, YPredLabel)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(10, 5))
plt.subplot(221)
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim(-0.05,1,0.2)
plt.ylim(-0.05,1,0.2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
```

    [[1.]
     [1.]
     [0.]
     ...
     [0.]
     [0.]
     [0.]]
    [[1.]
     [1.]
     [1.]
     ...
     [0.]
     [0.]
     [0.]]
    1.0
    0.898
    0.946259220231823
    [[1000    0]
     [ 102  898]]
    0.949


    <matplotlib.legend.Legend at 0x7f0e22244ed0>


![png](../figs/output_9_2.png)


```python
YPredLabel = abs(np.array([realDis, fakeDis]))
YPredLabel[YPredLabel<0.003] = 0
YPredLabel[YPredLabel>=0.003] = 1
YPredLabel.resize(2000,1)
print(YPredLabel)
Ylabel = np.array([ones([1000]), zeros([1000])])
Ylabel.resize(2000,1)
print(Ylabel)
# confusion matrix
Accuracy= confusion_matrix(Ylabel, YPredLabel)
precision = precision_score(Ylabel, YPredLabel, average='binary')
recall = recall_score(Ylabel, YPredLabel, average='binary')
score = f1_score(Ylabel, YPredLabel, average='binary')
print(precision)
print(recall)
print(score)
print(print(accuracy_score(Ylabel, YPredLabel)))
print(Accuracy)
fpr, tpr, threshold = roc_curve(Ylabel, YPredLabel)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 5))
plt.subplot(221)
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim(-0.05,1,0.2)
plt.ylim(-0.05,1,0.2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
```

    [[1.]
     [1.]
     [0.]
     ...
     [0.]
     [0.]
     [0.]]
    [[1.]
     [1.]
     [1.]
     ...
     [0.]
     [0.]
     [0.]]
    1.0
    0.84
    0.9130434782608696
    0.92
    None
    [[1000    0]
     [ 160  840]]


    <matplotlib.legend.Legend at 0x7f0e221f4250>


![png](..(/figs/output_10_2.png)


```python
YPredLabel = abs(np.array([realDis, fakeDis]))
YPredLabel[YPredLabel<0.004] = 0
YPredLabel[YPredLabel>=0.004] = 1
YPredLabel.resize(2000,1)
print(YPredLabel)
Ylabel = np.array([ones([1000]), zeros([1000])])
Ylabel.resize(2000,1)
print(Ylabel)
# confusion matrix
Accuracy= confusion_matrix(Ylabel, YPredLabel)
precision = precision_score(Ylabel, YPredLabel, average='binary')
recall = recall_score(Ylabel, YPredLabel, average='binary')
score = f1_score(Ylabel, YPredLabel, average='binary')
print(precision)
print(recall)
print(score)
print(print(accuracy_score(Ylabel, YPredLabel)))
print(Accuracy)
fpr, tpr, threshold = roc_curve(Ylabel, YPredLabel)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 5))
plt.subplot(221)
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim(-0.05,1,0.2)
plt.ylim(-0.05,1,0.2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
```

    [[1.]
     [1.]
     [0.]
     ...
     [0.]
     [0.]
     [0.]]
    [[1.]
     [1.]
     [1.]
     ...
     [0.]
     [0.]
     [0.]]
    1.0
    0.779
    0.8757729061270377
    0.8895
    None
    [[1000    0]
     [ 221  779]]



    <matplotlib.legend.Legend at 0x7f0e22194dd0>



![png](../figs/output_11_2.png)


```python
YPredLabel = abs(np.array([realDis, fakeDis]))
YPredLabel[YPredLabel<0.005] = 0
YPredLabel[YPredLabel>=0.005] = 1
YPredLabel.resize(2000,1)
print(YPredLabel)
Ylabel = np.array([ones([1000]), zeros([1000])])
Ylabel.resize(2000,1)
print(Ylabel)
# confusion matrix
Accuracy= confusion_matrix(Ylabel, YPredLabel)
precision = precision_score(Ylabel, YPredLabel, average='binary')
recall = recall_score(Ylabel, YPredLabel, average='binary')
score = f1_score(Ylabel, YPredLabel, average='binary')
print(precision)
print(recall)
print(score)
print(Accuracy)
print(print(accuracy_score(Ylabel, YPredLabel)))
fpr, tpr, threshold = roc_curve(Ylabel, YPredLabel)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 4))
plt.subplot(221)
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim(-0.05,1,0.2)
plt.ylim(-0.05,1,0.2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
```

    [[1.]
     [1.]
     [0.]
     ...
     [0.]
     [0.]
     [0.]]
    [[1.]
     [1.]
     [1.]
     ...
     [0.]
     [0.]
     [0.]]
    1.0
    0.724
    0.839907192575406
    [[1000    0]
     [ 276  724]]
    0.862
    None


    <matplotlib.legend.Legend at 0x7f0e220e2790>


![png](../figs/output_12_2.png)


