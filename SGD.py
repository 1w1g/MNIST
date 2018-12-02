#import training data (load.py) from MNIST data located in ./data
import load
#import numpy for calculations
import numpy as np
#import this for time mesuring
import timeit
class NeuralNetwork():
    def __init__(self,net):
        self.size=len(net)
        self.weights=[np.random.randn(x,y) for x,y in zip(net[:-1],net[1:])]
        self.biases=[np.random.randn(1,y) for y in net[1:]]
    def feed_forward(self,a):
        for w,b in zip(self.weights,self.biases):
            a=self.sigmoid(np.dot(a,w)+b)
        return a
    def sigmoid(self,z):
        return(1/(1+np.exp(-z)))
    def sigmoid_prime(self,z):
        return self.sigmoid(z)*(1-self.sigmoid(z)) 
    def backpropogation(self,x,y):
        nabla_w=[np.zeros(w.shape) for w in self.weights]
        nabla_b=[np.zeros(b.shape) for b in self.biases]
        zs=[]
        activation=x
        activations=[]
        activations.append(activation)
        for w,b in zip(self.weights,self.biases):
            z=(np.dot(activation,w)+b)
            zs.append(z)
            activation=self.sigmoid(z)
            activations.append(activation)
        delta=(activations[-1]-y)*self.sigmoid_prime(zs[-1])
        nabla_b[-1]=delta
        nabla_w[-1]=np.dot(activations[-2].T,delta)
        for h in range(2,self.size):
            sp=self.sigmoid_prime(zs[-h])
            delta=np.dot(delta,self.weights[-h+1].T)*sp
            nabla_b[-h]=delta
            nabla_w[-h]=np.dot(activations[-h-1].T,delta)
        return nabla_w,nabla_b
    def update_weights(self,x,y,eta,epoch,test_x,test_y,batch_size=1,):
        for j in range(1,epoch+1):
            idx=np.random.choice(len(x),replace=False)
            np.random.shuffle(x[idx])
            np.random.shuffle(y[idx])
            # Divid data into mini batches and update weights
            for i in range(0,len(x),batch_size):
                x_mini=x[i:i+batch_size]
                y_mini=y[i:i+batch_size]
                nabla_w=[np.zeros(w.shape) for w in self.weights]
                nabla_b=[np.zeros(b.shape) for b in self.biases]
                for x_one,y_one in zip(x_mini,y_mini):
                    delta_nabla_w,delta_nabla_b=self.backpropogation(x_one,y_one)
                    nabla_w=[nw+dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]
                    nabla_b=[nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
                self.weights=[w-(eta*dnw)/len(x_mini) for w,dnw in zip(self.weights,nabla_w)]
                self.biases=[b-(eta*dnb)/len(x_mini) for b,dnb in zip(self.biases,nabla_b)]
            print ("epch=%r %r/%r" %(j,self.evaluate_test(test_x,test_y),len(test_x)))
    def evaluate_test(self,test_data_x,test_data_y):
        result=[(np.argmax(self.feed_forward(x)),y) for x,y in zip(test_data_x,test_data_y) ]
        return sum((int(x==y)for x,y in result))

        

            
start_time=timeit.default_timer()
x,y=load.load_training_data()
test_x,test_y=load.load_test_data()
net=NeuralNetwork([784,30,10])
print('before training')
print(net.feed_forward(x[0]))
print ("epch=00 %r/%r" %(net.evaluate_test(test_x,test_y),len(test_x)))
net.update_weights(x,y,3,30,test_x,test_y,10)
print('after training')
print(net.feed_forward(x[0]))
stop_time=timeit.default_timer()
progress_time=stop_time-start_time
print('Time=',progress_time)
