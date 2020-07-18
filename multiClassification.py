import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import reuters
(train_data,train_label),(test_data,test_label)=reuters.load_data(num_words=10000)

#vectorize_data
def vectorize_data(data,dimension=10000):
    results = np.zeros((len(data),dimension))
    for i,sequence in enumerate(data):
        results[i,sequence]=1
    return results

x_train=vectorize_data(train_data)
x_test=vectorize_data(test_data)

#vectorize/encode labels
def one_hot_label(label,dimension=46):
    results = np.zeros((len(label),dimension))
    for i,sequence in enumerate(label):
        results[i,sequence]=1
    return results

y_train=one_hot_label(train_label)
y_test=one_hot_label(test_label)

#or integer tensor
y_train2=np.array(train_label)
y_test2=np.array(test_label)


#building network1

from keras import models
from keras import layers
model=models.Sequential()
model.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(46,activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

x_val=x_train[:1000]
partial_x_train=x_train[1000:]
y_val=y_train[:1000]
partial_y_train=y_train[1000:]

history=model.fit(partial_x_train,partial_y_train,
                  epochs=20,batch_size=512,
                  validation_data=(x_val,y_val))


#building network2
from keras import models
from keras import layers
model2=models.Sequential()
model2.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
model2.add(layers.Dense(64,activation='relu'))
model2.add(layers.Dense(46,activation='softmax'))

model2.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

x_val=x_train[:1000]
partial_x_train=x_train[1000:]
y_val=y_train2[:1000]
partial_y_train2=y_train2[1000:]

history2=model2.fit(partial_x_train,partial_y_train2,
                  epochs=20,batch_size=512,
                  validation_data=(x_val,y_val))


#visulaisation
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()


plt.clf()
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

results=model.evaluate(x_test,y_test)
results2=model2.evaluate(x_test,y_test2)

print(results)

predict=model.predict(x_test)
predict2=model2.predict(x_test)


#the vector in predict tensor  has 46 probabilities
# the index with high probability shows the category(0-45) and output of np.argmax()

for i in predict:
    print(np.argmax(i))
    
print(np.argmax(predict[0]))













