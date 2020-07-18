from keras.datasets import mnist 
import numpy as np

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

from keras import models 
from keras import layers 
 
network = models.Sequential() 
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,))) 
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])


train_images=train_images.reshape((60000,28*28))
train_images=train_images.astype('float32')/255

test_images=test_images.reshape((10000,28*28))
test_images=test_images.astype('float32')/255

from keras.utils import to_categorical

train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)

network.fit(train_images,train_labels,epochs=5,batch_size=128)

test_loss,test_acc=network.evaluate(test_images,test_labels)


#my work
print(train_images.shape)
my_data=np.random.randint(0,high=11,size=(4,3,2))
my_data2=my_data.reshape(4,3*2)
my_data3=my_data.reshape(4,3,2,1)
my_data4=my_data3[0]#look variable explorer
my_data5=my_data2.reshape(4,6,1)