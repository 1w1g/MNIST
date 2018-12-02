#import training data (load.py) from MNIST data located in ./data
import load1 as load
#import numpy for calculations
import numpy as np
#import this for time mesuring
import timeit
class NeuralNetwork():
    def __init__(self,net):
        self.size=len(net)
        self.weights=[np.random.randn(y,x) for x,y in zip(net[:-1],net[1:])]
        self.biases=[np.random.randn(y,1) for y in net[1:]]
    def feed_forward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
    def backpropogation(self,x,y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.size):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_w, nabla_b)   
    def backpropogation1(self,x,y):
        nabla_w=[np.zeros(w.shape) for w in self.weights]
        nabla_b=[np.zeros(b.shape) for b in self.biases]
        zs=[]
        activation=x
        activations=[]
        activations.append(activation)
        for w,b in zip(self.weights,self.biases):
            z=(np.dot(w,activation)+b)
            zs.append(z)
            activation=sigmoid(z)
            activations.append(activation)
        delta=(activations[-1]-y)*sigmoid_prime(zs[-1])
        nabla_b[-1]=delta
        nabla_w[-1]=np.dot(delta,activations[-2].T)
        for h in range(2,self.size):
            sp=sigmoid_prime(zs[-h])
            delta=np.dot(self.weights[-h+1].T,delta)*sp
            nabla_b[-h]=delta
            nabla_w[-h]=np.dot(delta,activations[-h-1].T)
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
    def evaluate_test1(self,test_data_x,test_data_y):
        result=[(np.argmax(self.feed_forward(x)),y) for x,y in zip(test_data_x,test_data_y) ]
        return sum((int(x==y)for x,y in result))

    def evaluate_test(self, test_data_x,test_data_y):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feed_forward(x)), y) for (x, y) in zip(test_data_x,test_data_y)]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

    def SGD(self, x,y, eta,epochs,test_x,test_y,mini_batch_size):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        n_test = len(test_x)
        n = len(x)
        for j in range(epochs):
            #random.shuffle(training_data)
            x_minis=[x[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
            y_minis=[y[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
                
            for x_mini,y_mini in zip(x_minis,y_minis):
                self.update_mini_batch(x_mini,y_mini, eta)

            print("Epoch %r: %r / %r" %(j, self.evaluate_test(test_x,test_y), n_test))


    def update_mini_batch(self, x_mini,y_mini, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in zip(x_mini,y_mini):
            delta_nabla_w,delta_nabla_b = self.backpropogation(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(x_mini))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(x_mini))*nb
                       for b, nb in zip(self.biases, nabla_b)]
#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

        

            
start_time=timeit.default_timer()
x,y=load.load_training_data()
test_x,test_y=load.load_test_data()
net=NeuralNetwork([784,60,10])
print('before training')
print(net.feed_forward(x[0]))
net.SGD(x,y,3,30,test_x,test_y,10)
print('after training')
print(net.feed_forward(x[0]))
stop_time=timeit.default_timer()
progress_time=stop_time-start_time
print('Time=',progress_time)
