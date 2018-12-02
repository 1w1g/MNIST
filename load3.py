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
    x,y=mndata.load_training()
    return x,y
def load_test_data():
    mndata=MNIST('./data/')
    mndata.gz=True
    x,y=mndata.load_testing()
    return x,y


if __name__=='__main__':
    x,y=load_training_data()
    print(y[57])
    test_x,test_y=load_test_data()

