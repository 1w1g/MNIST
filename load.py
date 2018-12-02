from matplotlib import pyplot as plt 
from mnist import MNIST
import numpy as np
def decimal_to_bit(position):
    output=np.zeros([1,10])
    output[0,position]='1'
    return output
def reshape_image(orginal_img):
    reshaped_image=np.reshape(orginal_img,(1,784))
    return reshaped_image
def reshape_label(orginal_lbl):
    reshaped_label=np.reshape(orginal_lbl,(1,1))
    return reshaped_label
def load_training_data():
    mndata=MNIST('./data/')
    mndata.gz=True
    images,label=mndata.load_training()
    x=[]
    y=[]
    i=0
    print('***Loading Training Data***')
    for img,lbl in zip(images,label):
        y.append(reshape_label(lbl))
        x.append(reshape_image(img))
        i+=1
        if i%1000==0:
            print('loading image ',i)
    return x,y
def load_test_data():
    mndata=MNIST('./data/')
    mndata.gz=True
    images,label=mndata.load_testing()
    test_x=[]
    test_y=[]
    i=0
    print('***Loading Testing Data***')
    for img,lbl in zip(images,label):
        test_y.append(lbl)
        test_x.append(reshape_image(img))
        i+=1
        if i%1000==0:
            print('loading image ',i)
    return test_x,test_y


if __name__=='__main__':
    x,y=load_training_data()
    print(y[57])
    test_x,test_y=load_test_data()

