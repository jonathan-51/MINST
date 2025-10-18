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
        pass

    def train(self,train_images,train_labels,LR):
        """Runs one full training epoch with a sample size of 50000 units.
        Initialisation --> Front Propagation --> Back Propagation
        Updates Parameters after each iteration"""

        # Getting training dataset (First 50000 samples from main dataset)
        train_images, train_labels = self.getTrainingDataset(train_images,train_labels)

        Aj_Epoch = numpy.zeros((50,10))
        OHE_Epoch = numpy.zeros((50,10))

        # Loops front and back propagation
        for sample in range(50):
            train_images_sample = train_images[sample].reshape(784,1)                                   # Reshape from (784,) to (784,1)
            parameters = self.getParameters()                                                           # Loading Parameters from File
            OHE,OHE_Epoch = self.getOneHotEncoding(train_labels,sample,OHE_Epoch)                       # Getting One Hot Encoding (10,1)
            Zk,Ak,Aj,loss,Aj_Epoch = self.Forward(train_images_sample,parameters,OHE,sample,Aj_Epoch)   # One Forward Pass
            updated_parameters = self.Backward(Aj,Ak,OHE,parameters,Zk,train_images_sample,LR)          # One Backward Pass

        return loss,Aj_Epoch,OHE_Epoch
    
    def val(self,val_images,vak_labels):
        """Runs one full validation epoch with a sampel size of 10000 units.
        Initialisation --> Front Propagation. No back propagation. 
        Tests the generalisation of the model."""

        # Getting Validation dataset (Last 10000 Units from main dataset)
        val_images, vak_labels = self.getValidationDataset(val_images,vak_labels)

        Aj_Epoch = numpy.zeros((50,10))
        OHE_Epoch = numpy.zeros((50,10))

        #Loops front propagation
        for sample in range(50):

            val_images_sample = val_images[sample].reshape(784,1)                                   # Reshape from (784,) to (784,1)
            parameters = self.getParameters()                                                       # Loading Parameters from File
            OHE,OHE_Epoch = self.getOneHotEncoding(vak_labels,sample,OHE_Epoch)                     #Getting One Hot Encoding (10,1)
            _,_,_, loss,Aj_Epoch = self.Forward(val_images_sample,parameters,OHE,sample,Aj_Epoch)   # One Forward Pass


        return loss,Aj_Epoch,OHE_Epoch

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

        parameters_dict = {"Weight_k_i":parameters['wki'], # weights: input -> hidden
                           "Bias_k":parameters['bk'],      # biases:  hidden
                           "Weight_j_k":parameters['wjk'], # weights: hidden -> output
                           "Bias_j":parameters['bj']}      # weights: output
        return parameters_dict
    
    def getOneHotEncoding(self,labels,sample,OHE_Epoch):
        """Prepares a binary 1 Dimensional array with 10 elements, where each index
        represents its corresponding class. The class of current sample will index
        a 1 for its corresponding index.
        """
        OHE = numpy.zeros((10,1),dtype=int)

        # Indexing 1 to the index that corresponds to the current digit
        # Returns a 2D array of size (10,1)
        OHE[labels[sample]] = 1 
        OHE_Epoch[sample] = OHE.T
        return OHE,OHE_Epoch

#======================================================================================================
# 2. FORWARD PROPAGATION
#======================================================================================================

    def Forward(self,image_sample,parameters,OHE,sample,Aj_Epoch): 
        """Runs a forward pass for one sample."""
        
        Ak, Zk = self.Activation_k(image_sample,parameters,sample)       # Calculates Activation Neurons in 2nd Layer
        Aj,Aj_Epoch = self.Activation_j(Ak,parameters,sample,Aj_Epoch)   # Calculates Activation Neurons in Output Layer
        loss = self.CrossEntropyLoss(Aj,OHE)                             # Calculates Loss Value for entire network

        return Zk,Ak,Aj,loss,Aj_Epoch
    
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
    
    def Activation_k(self,image_sample,parameters,sample):
        """
        Computes value of each 64 activation neurons in the 2nd layer of network in 64 element array.

        Calculates the weighted sum of one pre-activation neuron in the 2nd layer,
        stores it in Zk array with respective index, 
        non-linearility is introduced by applying ReLU function on preactivation neurons
        to compute activation neurons

        Preactivation neurons in hidden layer (2nd layer) --> Zk
        Activation neurons in hidden layer (2nd layer) --> Ak
        """        

        # Matrix Multiplication to calculate weighted sum (64,784)@(784,1)
        # Matrix addition to calculate preactivation neuron (64,1) + (64,1)
        Zk = (parameters["Weight_k_i"]@image_sample) + parameters["Bias_k"]  

        Ak = self.ReLU(Zk) #(64,1)

        return Ak, Zk
    
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
        
    def Activation_j(self,Ak,parameters,sample,Aj_Epoch):
        """
        Computes value of each 10 activation neurons in the output layer of network in a 10 element array.

        Calculates the weighted sum of one pre-activation neuron in the output layer,
        stores it in Zj array with respective index, 
        Softmax function is applied to Zj to compute activation neurons ,
        which returns model's confidence for each class in probability format
     

        Preactivation neurons in output layer (Final layer) --> Zj
        Activation neurons in output layer (Final layer) --> Aj
        """

        # Matrix Multiplication to calculate weighted sum (10,64)@(64,1)
        # Matrix addition to calculate preactivation neuron (64,1)+(64,1)
        Zj = parameters["Weight_j_k"]@Ak
        Zj = Zj + parameters["Bias_j"]

        Aj = self.SoftMax(Zj) # (10,1)

        # Storing all Activating neurons (50000,10)
        Aj_Epoch[sample] = Aj.T # (1,10)

        return Aj,Aj_Epoch

    def CrossEntropyLoss(self,Aj,OHE):
        """
        Returns loss value for entire network for current sample, which is essentially the
        negative natural log of the model's probability for the True class.
        OHE is in binary format, so only the index of True Class has a value of 1.

        Loss of Network = Sum(OHE[class] * Aj[class]),
        which is the same as:
        Loss of Network = 1 * Aj[True class]

        """

        loss = -numpy.log(Aj[numpy.where(OHE == 1)]) # Returns a 1D array of size (1,)

        return loss[0]
 
#======================================================================================================
# 3. BACKWARD PROPAGATION
#======================================================================================================    

    def Backward(self,Aj,Ak,OHE,parameters,Zk,train_images_sample,LR):
        """Runs a Backward Pass for one sample"""

        gradients = self.getGradients(Aj,Ak,OHE,parameters,Zk,train_images_sample)  #Computes gradient values for all parameters
        updated_parameters = self.ParametersUpdate(parameters,gradients,LR)         #Returns updated parameters

        return updated_parameters
    
    def dReLU(self,Zk):
        """Returns the derivatives of all Ak values w.r.t their respective Zk values"""

        indices = Zk > 0               #Returns all indices where its values are positive as Boolean Type
        Zk = numpy.where(indices,1,0)  #Applies index mask, where all True Values become 1, and all False values become 0

        return Zk
    
    def getGradients(self,Aj,Ak,OHE,parameters,Zk,train_images_sample):
        """Calculates gradients for all parameters.
        Returns a dictionary of all 4 types of parameters, where each gradient value
        represents the value of the same index"""
        
        dZ_j = Aj - OHE                                             #Preactivation Neurons in Output Layer
        dZ_k = ((parameters["Weight_j_k"].T)@dZ_j)*self.dReLU(Zk)   #Preactivation Neurons in Hidden Layer

        db_j = dZ_j                                                 #Biases in Output Layer
        dW_jk = dZ_j@Ak.T                                           #Weights in Hidden Layer

        db_k = dZ_k                                                 #Biases in Hidden Layer
        dw_ki = dZ_k@train_images_sample.T                          #Weights in Input Layer

        gradients = {"Weight_k_i":dw_ki,
                     "Bias_k":db_k,
                     "Weight_j_k":dW_jk,
                     "Bias_j":db_j}
        
        return gradients

    def ParametersUpdate(self,parameters,gradients,LR):
        """Updating parameters by taking the difference between its value and 
        its gradient multiplied by factor (Learning Rate).
        Storing the updated parameters in a dictionary."""

        weight_k_i_updated = parameters["Weight_k_i"] - LR * gradients["Weight_k_i"]    #Weights in Input Layer
        bias_k_updated = parameters["Bias_k"] - LR * gradients["Bias_k"]                #Biases in Hidden Layer

        weight_j_k_updated = parameters["Weight_j_k"] - LR * gradients["Weight_j_k"]    #Weights in Hidden Layer Layer
        bias_j_updated = parameters["Bias_j"] - LR * gradients["Bias_j"]                #Biases in Output Layer

        updated_parameters = {"Weight_k_i":weight_k_i_updated,
                              "Bias_k":bias_k_updated,
                              "Weight_j_k":weight_j_k_updated,
                              "Bias_j":bias_j_updated}

        return updated_parameters

#======================================================================================================
# 4. Evaluation
#======================================================================================================    

    def getData(self,train_loss,val_loss,train_Aj_Epoch,val_Aj_Epoch,train_OHE_Epoch,val_OHE_Epoch):
        self.TL = train_loss
        self.VL = val_loss
        self.train_Aj_Epoch = train_Aj_Epoch
        self.val_Aj_Epoch = val_Aj_Epoch
        self.train_OHE_Epoch = train_OHE_Epoch
        self.val_OHE_Epoch = val_OHE_Epoch

        return
class Evaluation:
    def __init__(self,model):
        self.model = model

        pass
    
    def Logging(self,train_loss,val_loss):

        with open("Test_3_LR0-005/epoch_summary_T3.csv","a") as f:
            f.write(f"\n{e:.1f},{train_loss:.4f},{val_loss:.4f},{train_acc}%,{val_acc}%,{train_max_prob_avg},{val_max_prob_avg},{train_time}s,{val_time}s,{lr},{gradient_mean},{tcacc[0]},{tcacc[1]},{tcacc[2]},{tcacc[3]},{tcacc[4]},{tcacc[5]},{tcacc[6]},{tcacc[7]},{tcacc[8]},{tcacc[9]},{vcacc[0]},{vcacc[1]},{vcacc[2]},{vcacc[3]},{vcacc[4]},{vcacc[5]},{vcacc[6]},{vcacc[7]},{vcacc[8]},{vcacc[9]}")

        return 
     
    def Accuracy(self):
        """Handles calculating the accuracy of model for 1 epoch from both Training and Validation.
        Model's highest predicted probability for an iteration is considered the model's prediction.
        Compares model's highest predicted probability with true labels of its corresponding index.
        Counts how many model has gotten correct,
        computes the accuracy."""


        train_predictions_total = self.model.train_Aj_Epoch     # Getting model's training predictions for all classes from each iteration
        val_predictions_total = self.model.val_Aj_Epoch         # Getting model's validation predictions for all classes from each iteration
        train_true_labels_total = self.model.train_OHE_Epoch    # Getting True labels for training dataset
        val_true_labels_total = self.model.val_OHE_Epoch        # Getting True labels for validation dataset

        train_prediction_indices = numpy.argmax(train_predictions_total,axis=1)                      # Finds the indices of model's prediction for all iterations 
        train_sample_indices = numpy.arange(len(train_true_labels_total))                            # Gets the indices of length of training dataset
        train_correct = sum(train_true_labels_total[train_sample_indices,train_prediction_indices])  # Computes total amount model got correct from 1 training epoch

        val_prediction_indices = numpy.argmax(val_predictions_total,axis=1)                          # Finds the indices of models prediction for all iterations
        val_sample_indices = numpy.arange(len(val_true_labels_total))                                # Gets the indices of length of validation dataset
        val_correct = sum(val_true_labels_total[val_sample_indices,val_prediction_indices])          # Computes the total amount model got correct from 1 validation epoch

        train_accuracy = train_correct/len(train_true_labels_total) # Computes accuracy of model from training
        val_accuracy = val_correct/len(val_true_labels_total)       # Computes accuracy of model from validation
        
        return train_accuracy,val_accuracy
    
    def Time(self):

        return
    
    def GradientNorm(self):

        return
    
    def ClassAccuracy(self):

        return
    
    def Calibration(self):

        return
    
    def ConfusionMatrix(self):

        return


Test = Model(images,labels)

#train_images,train_labels = Test.getTrainingDataset(images,labels)
train_loss = Test.train(images,labels,LR=0.01)
val_loss = Test.val(images,labels)
Results = Evaluation(Test)
train_accuracy,val_accuracy = Results.Accuracy()
print(train_accuracy)
print(val_accuracy)
