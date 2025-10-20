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
    def __init__(self,images,labels,train_samples,val_samples):
        self.images = images
        self.labels = labels
        self.tsamples = train_samples
        self.vsamples = val_samples
        pass

    def train(self,images,labels,LR,e):
        """Runs one full training epoch with a sample size of 50000 units.
        Initialisation --> Front Propagation --> Back Propagation
        Updates Parameters after each iteration"""

        # Getting training dataset (First 50000 samples from main dataset)
        train_images, train_labels = self.getTrainingDataset(images,labels)

        sample_number = self.tsamples
        Aj_Epoch = numpy.zeros((sample_number,10))
        OHE_Epoch = numpy.zeros((sample_number,10))
        norm_grad = 0
        loss_epoch = 0
        start_time = time.time()  # Logs start time for 1 training epoch

        parameters = self.getTrainingParameters(e)  # Loading pretrained Parameters from File

        # Loops front and back propagation
        for sample in range(sample_number):
            train_images_sample = train_images[sample].reshape(784,1)                                       # Reshape from (784,) to (784,1)
            
            OHE,OHE_Epoch = self.getOneHotEncoding(train_labels,sample,OHE_Epoch)                           # Getting One Hot Encoding (10,1)
            
            Zk,Ak,Aj,loss,Aj_Epoch,loss_epoch = self.Forward(train_images_sample,parameters,OHE,sample,Aj_Epoch,loss_epoch)       # One Forward Pass
            
            parameters,norm_grad = self.Backward(Aj,Ak,OHE,parameters,Zk,train_images_sample,LR,norm_grad)  # One Backward Pass
        
        end_time = time.time()  # Logs end time for 1 training epoch

        self.getTrainingData(loss_epoch,Aj_Epoch,OHE_Epoch,(end_time-start_time),norm_grad/sample_number)
        return parameters
    
    def val(self,images,labels,updated_parameters):
        """Runs one full validation epoch with a sampel size of 10000 units.
        Initialisation --> Front Propagation. No back propagation. 
        Tests the generalisation of the model."""

        # Getting Validation dataset (Last 10000 Units from main dataset)
        val_images, val_labels = self.getValidationDataset(images,labels)

        sample_number = self.vsamples

        Aj_Epoch = numpy.zeros((sample_number,10))
        OHE_Epoch = numpy.zeros((sample_number,10))
        loss_epoch = 0

        start_time = time.time()  # Logs start time for 1 validation epoch

        #Loops front propagation
        for sample in range(sample_number):

            
            val_images_sample = val_images[sample].reshape(784,1)                                   # Reshape from (784,) to (784,1)
            
            #parameters = self.getValidationParameters()                                             # Loading Parameters from File
            
            OHE,OHE_Epoch = self.getOneHotEncoding(val_labels,sample,OHE_Epoch)                     #Getting One Hot Encoding (10,1)
            
            _,_,_, loss,Aj_Epoch,loss_epoch = self.Forward(val_images_sample,updated_parameters,OHE,sample,Aj_Epoch,loss_epoch)   # One Forward Pass

        end_time = time.time()  # Logs end time for 1 validation epoch

        self.getValidationData(loss_epoch,Aj_Epoch,OHE_Epoch,(end_time-start_time))
        return

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
    
    def getTrainingParameters(self,e):
        """Loads pre-trained model weights and biases from file."""
       
        parameters = numpy.load(f"Test_4.1_LR0-00005/parameters_T4.1/parameters_e{e-1}_T4.1.npz")
        #parameters = numpy.load(f"initial_parameters.npz")
        parameters_dict = {"Weight_k_i":parameters['wki'],               # weights: input -> hidden
                           "Bias_k":parameters['bk'].reshape(64,1),      # biases:  hidden
                           "Weight_j_k":parameters['wjk'],               # weights: hidden -> output
                           "Bias_j":parameters['bj'].reshape(10,1)}      # weights: output
        
        print(parameters_dict["Weight_k_i"].shape)
        print((parameters_dict["Bias_k"]).shape)
        print(parameters_dict["Weight_j_k"].shape)
        print((parameters_dict["Bias_j"]).shape)
        # print("-------------------")
        return parameters_dict
    
    #def getValidationParameters(self):
        """Loads pre-trained model weights and biases from file."""
       
        parameters = numpy.load("Repurposed/test_parameters5.npz")

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

    def Forward(self,image_sample,parameters,OHE,sample,Aj_Epoch,loss_epoch): 
        """Runs a forward pass for one sample."""
        
        Ak, Zk = self.Activation_k(image_sample,parameters,sample)       # Calculates Activation Neurons in 2nd Layer
        Aj,Aj_Epoch = self.Activation_j(Ak,parameters,sample,Aj_Epoch)   # Calculates Activation Neurons in Output Layer
        loss,loss_epoch = self.CrossEntropyLoss(Aj,OHE,loss_epoch)                             # Calculates Loss Value for entire network

        return Zk,Ak,Aj,loss,Aj_Epoch,loss_epoch
    
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
        Zk = (parameters["Weight_k_i"]@image_sample) 

        Zk = Zk + parameters["Bias_k"]  
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
        # Matrix addition to calculate preactivation neuron (10,1)+(10,1)
        Zj = parameters["Weight_j_k"]@Ak
        Zj = Zj + parameters["Bias_j"]
        Aj = self.SoftMax(Zj) # (10,1)
        # Storing all Activating neurons (50000,10)
        Aj_Epoch[sample] = Aj.T # (1,10)

        return Aj,Aj_Epoch

    def CrossEntropyLoss(self,Aj,OHE,loss_epoch):
        """
        Returns loss value for entire network for current sample, which is essentially the
        negative natural log of the model's probability for the True class.
        OHE is in binary format, so only the index of True Class has a value of 1.

        Loss of Network = Sum(OHE[class] * Aj[class]),
        which is the same as:
        Loss of Network = 1 * Aj[True class]

        """


        loss = -numpy.log(Aj[numpy.where(OHE == 1)]) # Returns a 1D array of size (1,)
        loss_epoch += loss
        return loss[0],loss_epoch
 
#======================================================================================================
# 3. BACKWARD PROPAGATION
#======================================================================================================    

    def Backward(self,Aj,Ak,OHE,parameters,Zk,train_images_sample,LR,norm_grad):
        """Runs a Backward Pass for one sample"""

        gradients,norm_grad = self.getGradients(Aj,Ak,OHE,parameters,Zk,train_images_sample,norm_grad)  #Computes gradient values for all parameters
        updated_parameters = self.ParametersUpdate(parameters,gradients,LR)         #Returns updated parameters

        return updated_parameters,norm_grad
    
    def dReLU(self,Zk):
        """Returns the derivatives of all Ak values w.r.t their respective Zk values"""

        indices = Zk > 0               #Returns all indices where its values are positive as Boolean Type
        Zk = numpy.where(indices,1,0)  #Applies index mask, where all True Values become 1, and all False values become 0

        return Zk
    
    def getGradients(self,Aj,Ak,OHE,parameters,Zk,train_images_sample,norm_grad):
        """Calculates gradients for all parameters.
        Returns a dictionary of all 4 types of parameters, where each gradient value
        represents the value of the same index"""
        
        dZ_j = Aj - OHE                                             #Preactivation Neurons in Output Layer
        dZ_k = ((parameters["Weight_j_k"].T)@dZ_j)*self.dReLU(Zk)   #Preactivation Neurons in Hidden Layer
        

        db_j = dZ_j
        dW_jk = dZ_j@Ak.T                                           #Weights in Hidden Layer
        
        
        db_k = dZ_k                                                 #Biases in Hidden Layer
        dw_ki = dZ_k@train_images_sample.T                          #Weights in Input Layer
        gradients = {"Weight_k_i":dw_ki,
                     "Bias_k":db_k,
                     "Weight_j_k":dW_jk,
                     "Bias_j":db_j}
 

        norm_grad += numpy.sqrt((numpy.sum(dW_jk**2)) + (numpy.sum(db_j**2)) + (numpy.sum(dw_ki**2)) + (numpy.sum(db_k**2)))
        return gradients,norm_grad

    def ParametersUpdate(self,parameters,gradients,LR):
        """Updating parameters by taking the difference between its value and 
        its gradient multiplied by factor (Learning Rate).
        Storing the updated parameters in a dictionary."""

        # print(parameters["Weight_k_i"].shape)
        # print(gradients["Weight_k_i"].shape)
        # print(parameters["Bias_k"].shape)
        # print(gradients["Bias_k"].shape)
        # print(parameters["Weight_j_k"].shape)
        # print(gradients["Weight_j_k"].shape)
        # print(parameters["Bias_j"].shape)
        # print(gradients["Bias_j"].shape)

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
# 3.5 PREPARING EVALUATION
#======================================================================================================    

    def getTrainingData(self,train_loss,train_Aj_Epoch,train_OHE_Epoch,train_time,norm_grad_mean):
        self.trainloss = train_loss
        self.train_predicted_probability = train_Aj_Epoch
        self.train_true_label = train_OHE_Epoch
        self.traintime = train_time
        self.normgrad = norm_grad_mean

        return
    
    def getValidationData(self,val_loss,val_Aj_Epoch,val_OHE_Epoch,val_time):
        self.valloss = val_loss
        self.val_predicted_probability = val_Aj_Epoch
        self.val_true_label = val_OHE_Epoch
        self.valtime = val_time

        return

#======================================================================================================
# 4. EVALUATION
#======================================================================================================    
class Evaluation:
    def __init__(self,model):
        self.model = model

        pass
    
    def getEpochStats(self):
        """Aggregate all data required to log for epoch summary."""

        train_loss_avg,val_loss_avg = self.getLoss()

        train_acc,val_acc = self.getAccuracy()  #gets model's correct predictions for both training and validation

        train_time = self.model.traintime       #gets model's training time and validation time for one epoch
        val_time = self.model.valtime

        norm_grad_mean = self.model.normgrad    #gets normalized gradient for one epoch (Average magnitude of all parameters gradient from one iteration)

        return train_loss_avg,val_loss_avg,train_acc,val_acc,train_time,val_time,norm_grad_mean
    
    def getLoss(self):

        train_loss = self.model.trainloss       #gets summation of epoch's training loss value
        val_loss = self.model.valloss           #gets summation of epoch's validation loss value
        train_loss_avg = train_loss/self.model.tsamples
        
        val_loss_avg = val_loss/self.model.vsamples

        return train_loss_avg[0],val_loss_avg[0]
    def getAccuracy(self):
        """Handles calculating the accuracy of model for 1 epoch from both Training and Validation.
        Model's highest predicted probability for an iteration is considered the model's prediction.
        Compares model's highest predicted probability with true labels of its corresponding index.
        Counts how many model has gotten correct,
        computes the accuracy."""


        train_predictions_total = self.model.train_predicted_probability    # Getting model's training predictions for all classes from each iteration
        val_predictions_total = self.model.val_predicted_probability        # Getting model's validation predictions for all classes from each iteration
        train_true_labels_total = self.model.train_true_label               # Getting True labels for training dataset
        val_true_labels_total = self.model.val_true_label                   # Getting True labels for validation dataset

        train_prediction_indices = numpy.argmax(train_predictions_total,axis=1)                      # Finds the indices of model's prediction for all iterations, 1D
        train_sample_indices = numpy.arange(len(train_true_labels_total))                            # Gets the indices of length of training dataset, 1D               
        train_correct = sum(train_true_labels_total[train_sample_indices,train_prediction_indices])  # Computes total amount model got correct from 1 training epoch

        val_prediction_indices = numpy.argmax(val_predictions_total,axis=1)                          # Finds the indices of models prediction for all iterations
        val_sample_indices = numpy.arange(len(val_true_labels_total))                                # Gets the indices of length of validation dataset
        val_correct = sum(val_true_labels_total[val_sample_indices,val_prediction_indices])          # Computes the total amount model got correct from 1 validation epoch

        train_accuracy = (train_correct/len(train_true_labels_total)) * 100     # Computes accuracy of model from training
        val_accuracy = (val_correct/len(val_true_labels_total)) * 100           # Computes accuracy of model from validation
        
        return train_accuracy,val_accuracy
    
    def getCalibration(self):
        """Returns models average predicted probability in each bin and model's total correct predictions in each bin. 
        Array of 10 bins, each bin representing a specific predicted probability's range, 
        e.g bin 1 represents predicted probability between 10%-20%,etc. """


        val_predictions_total = self.model.val_predicted_probability        # Getting model's validation predictions for all classes from each iteration
        val_true_labels_total = self.model.val_true_label                   # Getting True labels for validation dataset
                     
        pred_prob_bin = numpy.zeros((1,10))
        total_bin = numpy.zeros((1,10))
        correct_bin = numpy.zeros((1,10))


        max_pred = numpy.max(val_predictions_total,axis=1)  # Gets all the model's max predictions from each sample, 1D
        bin_index = numpy.int64(numpy.floor(max_pred*10))   # Gets each predictions bin index. 1D array where bin index represents its corresponding prediction

        max_pred_indices = numpy.argmax(val_predictions_total,axis=1)                   # Computing index for model's prediction for each interation
        correct_index = numpy.array(numpy.where(val_true_labels_total == 1)[1])         # Computing index for true label
        correct_bin_indices = bin_index[numpy.where(max_pred_indices == correct_index)] # Computes model's correct predictions in the form of bin number in an array

        #Looping through every bin number
        for bin in range(10):
            pred_prob_bin[0,bin] = numpy.sum(numpy.where(bin_index == bin,max_pred,0))  # Storing the summation of model's predicted probability for each bin
            total_bin[0,bin] = numpy.sum(numpy.where(bin_index == bin,1,0))             # Storing the summation of total number of samples in each bin
            correct_bin[0,bin] = numpy.sum(correct_bin_indices == bin)                  # Storing the summation of model's correct predictions for each bin

        
        predicted_accuracy = numpy.divide(pred_prob_bin , total_bin, out=numpy.zeros_like(pred_prob_bin) , where = total_bin != 0)  #Computes the average predicted accuracy for each bin
        true_accuracy = numpy.divide(correct_bin , total_bin, out=numpy.zeros_like(correct_bin) , where = total_bin != 0)           #Computes the average true accuracy for each bin

        return predicted_accuracy, true_accuracy
    
    def getConfusionMatrix(self):
        """Returns a 10x10 matrix that determines each class' accuracy, aswell as
        what class the model predicted when an incorrect prediction is made.
        True Labels represents the row number respective. Digit 0 = Index 0
        Prediction represents the column number respectively. Digit 0 = Index 0"""

        val_predictions_total = self.model.val_predicted_probability        # Getting model's validation predictions for all classes from each iteration
        val_true_labels_total = self.model.val_true_label                   # Getting True labels for validation dataset

        conf_matrix = numpy.zeros((10,10))

        i_p = numpy.argmax(val_predictions_total,axis=1)    # Getting index of model's prediction for all iterations in an epoch
        i_t = numpy.where(val_true_labels_total==1)[1]      # Getting index of correct class for all interations in an epoch
        numpy.add.at(conf_matrix,(i_t,i_p),1) 
        return conf_matrix

#======================================================================================================
# 5. LOGGING
#======================================================================================================    
class Logging:
    def __init__(self,model,evaluation):
        self.model = model
        self.evaluation = evaluation
        pass

    def savingParameters(self,updated_parameters,e):
        numpy.savez(f"Test_4.1_LR0-00005/parameters_T4.1/parameters_e{e}_T4.1.npz",wki = updated_parameters['Weight_k_i'], bk = updated_parameters['Bias_k'], wjk = updated_parameters['Weight_j_k'], bj = updated_parameters['Bias_j'])

        return

    def EpochSummary(self,epoch,LR):

        train_loss,val_loss,train_acc,val_acc,train_time,val_time,norm_grad_mean= self.evaluation.getEpochStats()
        with open("Test_4.1_LR0-00005/epoch_summary_T4.1.csv","a") as f:
            f.write(f"\n{epoch},{LR},{train_loss:.4f},{val_loss:.4f},{train_acc:.2f},{val_acc:.2f},{train_time:.0f},{val_time:.0f},{norm_grad_mean:.3f}")

        return
    
    def CalibrationCurve(self,e):
        p_acc, t_acc = self.evaluation.getCalibration()
        with open("Test_4.1_LR0-00005/CC_V_T4.1.csv", "a") as f:
            f.write(f"\n{e}")

            for idx in range(10):
                f.write(f",{p_acc[0,idx]:.4f}")
            for idx in range(10):
                f.write(f",{t_acc[0,idx]:.4f}")        
        return
    
    def ConfusionMatrix(self,e):
        conf_matrx = self.evaluation.getConfusionMatrix()
        numpy.savez(f"Test_4.1_LR0-00005/CM_T4.1/CM_e{e}_T4.1.npz",confusion_matrx = conf_matrx)
        return

class LearnRateFinder:
    def __init__(self,model,eval,log):
        self.model = model
        self.evaluation = eval
        self.logging = log
        pass
    
    def Learn(self,image,label):
        """Runs one full training epoch with a sample size of 50000 units.
        Initialisation --> Front Propagation --> Back Propagation
        Updates Parameters after each iteration"""

        parameters = self.initializeParameters()  # Loading pretrained Parameters from File

        Aj_Epoch,OHE_Epoch,loss_batch,sample_counter,gradients = self.initializeVariables()
        batch_counter = 1
        LR = 1e-7
        # Loops front and back propagation
        for sample in range(60000):
            sample_counter += 1

            image_sample = image[sample].reshape(784,1)                                       # Reshape from (784,) to (784,1)
            
            OHE,_ = self.model.getOneHotEncoding(label,sample,OHE_Epoch)                           # Getting One Hot Encoding (10,1)
            
            Zk,Ak,Aj,_,_,loss_batch = self.model.Forward(image_sample,parameters,OHE,sample,Aj_Epoch,loss_batch)       # One Forward Pass
            
            gradients = self.BackwardLearnRate(Aj,Ak,OHE,parameters,Zk,image_sample,gradients)  # One Backward Pass

            if sample_counter == 150:
                batch_counter += 1

                #Calculate average loss for each batch. Need batch_loss
                loss_batch = self.BatchLoss(loss_batch)

                #Calculate average gradient for each parameter for each batch.
                gradients = self.GradientAvg(gradients)

                #Calculate new parameters
                parameters = self.ParametersUpdateLearnRate(parameters,gradients,LR)

                #Log to file
                self.LogtoFile(LR,loss_batch)

                #Update Learning Rate 1e-7 --> 10
                LR = self.LRUpdated(LR)

                #Reinitializing
                Aj_Epoch,OHE_Epoch,loss_batch,sample_counter,gradients = self.initializeVariables()

        return         

    def initializeParameters(self):
        #initilizes a 10x64 array
        weight_jk = numpy.random.normal(loc=0,scale=numpy.sqrt((2/784)),size=(10,64))
        #initlizaes a 10 element 1D array
        bias_j = numpy.zeros((10,1))
        #initilizes a 64x784 array
        weight_ki = numpy.random.normal(loc=0,scale=numpy.sqrt((2/784)),size=(64,784))
        #initlizaes a 64 element 1D array
        bias_k = numpy.zeros((64,1))

        parameters = {"Weight_k_i":weight_ki,               # weights: input -> hidden
                      "Bias_k":bias_k,                      # biases:  hidden
                      "Weight_j_k":weight_jk,               # weights: hidden -> output
                      "Bias_j":bias_j}                      # weights: output

        return parameters
    
    def initializeVariables(self):
        Aj_Epoch = numpy.zeros((60000,10))
        OHE_Epoch = numpy.zeros((60000,10))
        loss_batch = 0
        sample_counter = 0
        gradients = {"Weight_k_i":0,
                     "Bias_k":0,
                     "Weight_j_k":0,
                     "Bias_j":0}        
        
        return Aj_Epoch,OHE_Epoch,loss_batch,sample_counter,gradients

    def BatchLoss(self,loss_batch):
        loss_batch_avg = loss_batch/150
        return loss_batch_avg

    def GradientAvg(self,gradients):
        gradients["Weight_k_i"] = gradients["Weight_k_i"]/150
        gradients["Bias_k"] = gradients["Bias_k"]/150
        gradients["Weight_j_k"] = gradients["Weight_j_k"]/150
        gradients["Bias_j"] = gradients["Bias_j"]/150
        return gradients

    def BackwardLearnRate(self,Aj,Ak,OHE,parameters,Zk,train_images_sample,gradients):
        """Runs a Backward Pass for one sample"""

        gradients = self.getGradients(Aj,Ak,OHE,parameters,Zk,train_images_sample,gradients)  #Computes gradient values for all parameters

        return gradients
    
    def dReLU(self,Zk):
        """Returns the derivatives of all Ak values w.r.t their respective Zk values"""

        indices = Zk > 0               #Returns all indices where its values are positive as Boolean Type
        Zk = numpy.where(indices,1,0)  #Applies index mask, where all True Values become 1, and all False values become 0

        return Zk
    
    def getGradients(self,Aj,Ak,OHE,parameters,Zk,train_images_sample,gradients):
        """Calculates gradients for all parameters.
        Returns a dictionary of all 4 types of parameters, where each gradient value
        represents the value of the same index"""
        
        dZ_j = Aj - OHE                                             #Preactivation Neurons in Output Layer
        dZ_k = ((parameters["Weight_j_k"].T)@dZ_j)*self.dReLU(Zk)   #Preactivation Neurons in Hidden Layer
        

        db_j = dZ_j
        dW_jk = dZ_j@Ak.T                                           #Weights in Hidden Layer
        
        
        db_k = dZ_k                                                 #Biases in Hidden Layer
        dw_ki = dZ_k@train_images_sample.T                          #Weights in Input Layer
        
        gradients["Weight_k_i"] += dw_ki
        gradients["Bias_k"] += db_k
        gradients["Weight_j_k"] += dW_jk
        gradients["Bias_j"] += db_j
 
        return gradients
    
    def ParametersUpdateLearnRate(self,parameters,gradients,LR):
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

    def LogtoFile(self,LR,loss_batch):
        with open("LR_Optimizer.csv", "a") as f:
            f.write(f"\n{LR},{loss_batch[0]}")
        return

    def LRUpdated(self,LR):
        start = 1e-7
        end = 1
        batch_number = 400
        lr_multiplier = (end/start)**(1/(batch_number-1))
        lr_updated = LR*lr_multiplier
        return lr_updated

train_sample = 50000
val_sample = 10000

Test = Model(images,labels,train_sample,val_sample)
Results = Evaluation(Test)
Recording = Logging(Test,Results)
Learning = LearnRateFinder(Test,Results,Recording)

LR = 0.00005

# for epoch in range(2,51):
#     updated_parameters = Test.train(images,labels,LR,epoch)    
#     Recording.savingParameters(updated_parameters,epoch)
#     Test.val(images,labels,updated_parameters)
#     Recording.EpochSummary(epoch,LR)
#     Recording.CalibrationCurve(epoch)
#     Recording.ConfusionMatrix(epoch)

Learning.Learn(images,labels)