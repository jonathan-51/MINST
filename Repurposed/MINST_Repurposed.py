import numpy
import matplotlib.pyplot as plt
import idx2numpy
import time

#Converts the MNIST Binary dataset into a grey scale format
images = idx2numpy.convert_from_file("train-images.idx3-ubyte")
labels = idx2numpy.convert_from_file("train-labels.idx1-ubyte")
#ensures that the output stays on one line in the terminal (only if it should)
numpy.set_printoptions(linewidth=numpy.inf)
#Suppresses scientific notion when printing out floating point numbers
numpy.set_printoptions(suppress=True)

class Model:
    def __init__(self,images,labels):
        self.images = images
        self.labels = labels
    
    def train(self,train_images):
        sample = 0
        parameters = self.getParameters()
        train_images, _ = self.getTrainingDataset(train_images,train_labels)

        OHE = self.getOneHotEncoding(train_labels,sample)
        loss = self.forward(train_images,parameters,OHE,sample)

        return loss

#======================================================================================================
# 1. INITIALISATION
#======================================================================================================

    def getTrainingDataset(self,images,labels):
        """
        Prepares the training dataset from the MNIST dataset.

        Selects the first 50,000 samples from training dataset for training,
        reshapes each 28x28 image into a 784-dimensional vector,
        and normalizes pixel intensities to the [0, 1] range.
        """

        train_labels = labels[:50000] 
        train_images = images[:50000]

        train_images = numpy.array(train_images.reshape(-1,784),dtype=float)

        train_images = train_images/255

        return train_images,train_labels

    def getValidationDataset(self,images,labels):
        """
        Prepares the validation dataset from the MNIST dataset.

        Selects the last 10000 samples from training dataset for validation,
        reshapes each 28x28 image into a 784-dimensional vector,
        and normalizes pixel intensities to the [0, 1] range.
        """
        val_labels = labels[50000:60000]
        val_images = images[50000:60000]

        val_images = numpy.array(val_images.reshape(-1,784),dtype=float)

        val_images = val_images/255

        return val_images,val_labels
    
    def getParameters(self):
        """Loads pre-trained model weights and biases from file."""
       
        parameters = numpy.load("Repurposed/test_parameters.npz")

        parameters_dict = {"weight_k_i":parameters['wki'], # weights: input -> hidden
                           "bias_k":parameters['bk'],      # biases:  hidden
                           "weight_j_k":parameters['wjk'], # weights: hidden -> output
                           "bias_j":parameters['bj']}      # weights: output
        return parameters_dict
    
    def getOneHotEncoding(self,labels,sample):
        """Prepares a binary 1 Dimensional array with 10 elements, where each index
        represents its corresponding class. The class of current sample will index
        a 1 for its corresponding index.
        """
        OHE = numpy.zeros(10,dtype=int)

        OHE[labels[sample]] = 1 # Indexing 1 to the index that corresponds to the current digit

        return OHE

#======================================================================================================
# 2. FORWARD PROPAGATION
#======================================================================================================
    def forward(self,images,parameters,OHE,sample): 
        """Runs a forward pass for one sample."""
        
        Ak = self.Activation_k(images,parameters,sample) # Calculates Activation Neurons in 2nd Layer
        Aj = self.Activation_j(Ak,parameters)            # Calculates Activation Neurons in Output Layer
        loss = self.CrossEntropyLoss(Aj,OHE)             # Calculates Loss Value for entire network

        return loss
    
    def ReLU(self,Zk):
        """
        Introduces non-linearity to network.

        Takes all values of pre activation neurons:
        if value is zero or negative, it will return a 0;
        if value is positive, it will return itself
        """

        # Return a boolean array where values will return true if satisfy RHS of equation, or else it will return false
        Zk_Boolean = Zk > 0 

        #Will apply a mask on Zk, where all indices that are False(zero or negative) will return 0.
        Zk = numpy.where(Zk_Boolean,Zk,0)

        return Zk
    
    def Activation_k(self,images,parameters,sample):
        """
        Computes value of each 64 activation neurons in the 2nd layer of network in 64 element array.

        Calculates the weighted sum of one pre-activation neuron in the 2nd layer,
        stores it in Zk array with respective index, 
        non-linearility is introduced by applying ReLU function on preactivation neurons
        to compute activation neurons

        Preactivation neurons in hidden layer (2nd layer) --> Zk
        Activation neurons in hidden layer (2nd layer) --> Ak
        """        
        Zk = numpy.zeros(64) 

        for i in range(64):
            Zk[i] = parameters["weight_k_i"][i]@images[sample] # Matrix Multiplication to calculate weighted sum
        Zk = Zk + parameters["bias_k"]                         # Matrix addition to calculate preactivation neuron

        Ak = self.ReLU(Zk)

        return Ak
    
    def SoftMax(self, Zj):
        """
        Applying SoftMax function to Activation Neurons in output layer.

        Returns a 1 Dimensional array with 10 elements representing the model's
        confidence for each class, in probability format, 
        where each index represents its corresponding class
        """
        Zj_nominator = numpy.exp(Zj - numpy.max(Zj))   # Computing nominator 
        Zj_denominator = numpy.sum(Zj_nominator)       # Computing denominator

        Aj = Zj_nominator/Zj_denominator   # Calculates the probability for each possible class

        return Aj
        
        
    def Activation_j(self,Ak,parameters):
        """
        Computes value of each 10 activation neurons in the output layer of network in a 10 element array.

        Calculates the weighted sum of one pre-activation neuron in the output layer,
        stores it in Zj array with respective index, 
        Softmax function is applied to Zj to compute activation neurons ,
        which returns model's confidence for each class in probability format
     

        Preactivation neurons in output layer (Final layer) --> Zj
        Activation neurons in output layer (Final layer) --> Aj
        """
        Zj = numpy.zeros(10)

        for i in range(10):
            Zj[i] = parameters["weight_j_k"][i]@Ak  # Matrix Multiplication to calculate weighted sum
            Zj = Zj + parameters["bias_j"]          # Matrix addition to calculate preactivation neuron

        Aj = self.SoftMax(Zj)

        return Aj

    def CrossEntropyLoss(self,Aj,OHE):
        """
        Returns loss value for entire network for current sample, which is essentially the
        negative natural log of the model's probability for the True class.
        OHE is in binary format, so only the index of True Class has a value of 1.

        Loss of Network = Sum(OHE[class] * Aj[class]),
        which is the same as:
        Loss of Network = 1 * Aj[True class]

        """
        loss = -numpy.log(Aj[numpy.where(OHE == 1)])

        return loss[0]
 
    

    
Test = Model(images,labels)
train_images,train_labels = Test.getTrainingDataset(images,labels)

Loss = Test.train(train_images)
print(Loss)
