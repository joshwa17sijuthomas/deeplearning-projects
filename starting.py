from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import imdb


(train_data,train_label),(test_data,test_labels)=imdb.load_data(num_words=10000)


word_index = imdb.get_word_index()
word_index_items=word_index.items()

max_charcter=max([max(sequence) for sequence in train_data])

for sequence in train_data:
    print(sequence)
    
reverse_word_index = dict([(value, key) for (key, value) in word_index .items()])

decoded_review =' '.join([reverse_word_index.get(i-3, '?') for i in train_data[0]])    
    

def vectorize_sequences(sequences,dimension=10000):
    results = np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence]=1
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

#vectorize labels
y_train = np.asarray(train_label).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

from keras import models
from keras import layers
from keras import regularizers
#model1
model=models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])

#validating
x_val=x_train[:10000]
partial_x_train=x_train[10000:]
y_val=y_train[:10000]
partial_y_train=y_train[10000:]

history=model.fit(partial_x_train,partial_y_train,epochs=20,batch_size=512,
                  validation_data=(x_val,y_val))

#model2(regulaization and dropout)
model2=models.Sequential()
model2.add(layers.Dense(16,kernel_regularizer=regularizers.l2(0.001),activation='relu',input_shape=(10000,)))
model2.add(layers.Dropout(0.5))
model2.add(layers.Dense(16,kernel_regularizer=regularizers.l2(0.001),activation='relu'))
model2.add(layers.Dropout(0.5))
model2.add(layers.Dense(1,activation='sigmoid'))

model2.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])

#validating
history2=model2.fit(partial_x_train,partial_y_train,epochs=20,batch_size=512,
                  validation_data=(x_val,y_val))


#not necessary
history_dict=history.history
history_dict.keys()

#visualisation1

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss(normal)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

plt.clf()
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy(normal)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#visualization2(regularization and dropout)
history_dict = history2.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss(regukarisation and dropout)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

plt.clf()
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy(regukarisation and dropout)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()




#final evaluation and prediction
result=model.evaluate(x_test,y_test)
predict=model.predict(x_test)


result2=model2.evaluate(x_test,y_test)
predict2=model2.predict(x_test)

#my experiments on this projects

from keras.datasets import next_imdb
(train_data_new,train_label_new),(test_data_new,test_labels_new)=next_imdb.load_data(num_words=10000)

learn=dict([(1,'joshwa',)])
lean3=dict([('the',1),('hero',2),('of',3),('epics',4),('mammootty',5),('mollywood',6),('mohanlal',7),(' ',8)])
learn2=dict(word_index.items())
new_imdb=[[1,2,3,4,5],[1,2,3,6,7]]
t_data=new_imdb
lean3_item=lean3.items()
reverse_lean=dict([(value,key) for (key,value) in lean3.items()])
actor=[]
for j in t_data:
    k=' '.join([reverse_lean.get(i,'?') for i in j])
    actor.append(k)
t2_data=[[4,5,6,7,8],[4,5,6,9,10]]#offset data values by adding 3
actor_new=[]
for j in t2_data:
    m=' '.join([reverse_lean.get(i-3,'?') for i in j])
    actor_new.append(m)
    

def vectorize_mysequences(sequences,dimension=10):
    results = np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence]=1
    return results

enum=list(enumerate(t_data))
demo=np.zeros((2,10))
demo[1,[1,2,3]]=1

my_result=vectorize_mysequences(t_data)
print(imdb)
y=10

decoded_review0 =' '.join([reverse_word_index.get(i, '?') for i in train_data[0]])
decoded_review1 =' '.join([reverse_word_index.get(i-1, '?') for i in train_data[0]])
decoded_review2 =' '.join([reverse_word_index.get(i-2, '?') for i in train_data[0]])
decoded_review3 =' '.join([reverse_word_index.get(i-3, '?') for i in train_data[0]])
decoded_review4 =' '.join([reverse_word_index.get(i-4, '?') for i in train_data[0]])


#dropout
layer_output=np.array([[1.1,9.1,17.1],
                       [0.1,0.2,0.3]])
    
#shape of layer_output is ()
layer_output2=np.array(2)

#produce a random array of shape (2,3) with values 0 or 1
my_output=np.random.randint(0,high=2,size=(2,3))

#make it multiplied by 0 or 1 but not 50%
layer_output*=np.random.randint(0,high=2,size=layer_output.shape)

layer_output2*=np.random.randint(0,high=2,size=layer_output2.shape)

# * operation simply multiply correspondig values in two matrices



