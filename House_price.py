#Boston house rpricing
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import boston_housing
(train_data,train_target),(test_data,test_target)=boston_housing.load_data()

#feature wise normalisation/scaling
mean=train_data.mean(axis=0)
train_data-=mean

std = train_data.std(axis=0)
train_data/=std

test_data-=mean
test_data/=std

#building newtwork
from keras import models
from keras import layers

def build_model():
    model=models.Sequential()
    model.add(layers.Dense(64,activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
    return model

#k_fold validation
    
k=4
num_val_samples=len(train_data)//k
#num_epochs=100
num_epochs=500
#all_scores=[]
all_mae_histories=[]
for i in range(k):
    print("processing fold",i)
    val_data= train_data[i*num_val_samples:(i+1)*num_val_samples]
    val_target=train_target[i*num_val_samples:(i+1)*num_val_samples]
    
    partial_train_data=np.concatenate([train_data[:i*num_val_samples],
                                       train_data[(i+1)*num_val_samples:]],axis=0)
    
    partial_train_target=np.concatenate([train_target[:i*num_val_samples],
                                       train_target[(i+1)*num_val_samples:]],axis=0)
    
    model=build_model()
#    model.fit(partial_train_data,partial_train_target,
#              epochs=num_epochs,batch_size=1,verbose=0)
#    result=model.evaluate(val_data,val_target,verbose=0)
#    val_mse,val_mae=model.evaluate(val_data,val_target,verbose=0)
    
#    all_scores.append(val_mae)
    history=model.fit(partial_train_data,partial_train_target,
                      validation_data=(val_data,val_target),
                      epochs=num_epochs,batch_size=1,verbose=0)
    mae_history=history.history['val_mae']#val_mae_mean_error not present
    print(mae_history)
    all_mae_histories.append(mae_history)#check this list
    
#np.mean(all_scores)    

average_mae_history=[np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
# in the above statement 2 things needed to be considered
#1.range is 500 not 4
#2.outside [] is very importent and beautiful - first for i in range get exceuted to i=0

#visualisation
plt.plot(range(1,len(average_mae_history)+1),average_mae_history)
plt.xlabel("epochs")
plt.ylabel('average_mae') 
plt.show()
plt.clf   

#smoothening the points
def smooth_curve(points,factor=0.9):
    smoothed_points=[]
    for point in points:
        if smoothed_points:
            previous=smoothed_points[-1]
            smoothed_points.append(previous*factor+point*(1-factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

smoothed_mae_history=smooth_curve(average_mae_history[10:])
plt.plot(range(1,len(smoothed_mae_history)+1),smoothed_mae_history)
plt.xlabel('epochs')
plt.ylabel('_average_smoothed_mae')
plt.show()

#real_training
model=build_model()
model.fit(train_data,train_target,epochs=50,batch_size=16)
test_mse,test_mae=model.evaluate(test_data,test_target)
#2.809

    
#my works
mylist=[[1,2,3,4],[2,4,6,8],[3,6,9,12],[4,8,12,16]]

demo_avg1=[np.mean([x[0] for x in mylist])]
average_mylist=[np.mean([x[i] for x in mylist]) for  i in range(len(mylist[0]))] 

#or
demo_avg2=[]
for i in range(len(mylist[0])):
    demo_avg2.append(np.mean([x[i] for x in mylist]))
    
    
    
#if mypoints=empty it print false .else true
mypoints=[2.3]
new_point=4
if mypoints:
    print(True)
else:
    print(False)
    
prev=mypoints[-1]

mypoints.append(prev*0.9+new_point*(1-0.9))
#smoothened point is more close to prev than current point
#Now mypoint=[2.3,2.46999999]


    
    