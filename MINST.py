import numpy
import matplotlib.pyplot as plt
import idx2numpy
import time
#Converts the MNIST Binary dataset into a grey scale format
train_images = idx2numpy.convert_from_file("train-images.idx3-ubyte")
train_labels_total = idx2numpy.convert_from_file("train-labels.idx1-ubyte")
#ensures that the output stays on one line in the terminal (only if it should)
numpy.set_printoptions(linewidth=numpy.inf)
#Suppresses scientific notion when printing out floating point numbers
numpy.set_printoptions(suppress=True)

#reshapes the 3D arraay into a 2D array by compressing the 28x28 matrix inside into a 1D 784 element array
def reshape(train_images):
    #Compresses the 3D (60000,28,28) array into a 2D (60000,784) array
    new_train_images = train_images.reshape(-1,784)
    return numpy.array(new_train_images,dtype=float)

#encodes all the greyscale values (0-255) into a range between 0-1
def normalize(new_train_images):
    for i in range(len(new_train_images)):
        for j in range(len(new_train_images[i])):
            new_train_images[i,j] = new_train_images[i,j]/255
    return new_train_images

#Generates a binary array (1 = correct, 0 = incorrect) where the index of the array represents the digit
def one_hot_encoding(labels, n):
    #Generate a 1D array (,10) of zeros
    one_hot = numpy.zeros(10,dtype=int)
    #assigns the correct index as 1
    one_hot[labels[n]] = 1 
    return one_hot

#ReLU Function
def ReLU(FHL):
    for i in range(64):
        if FHL[i] <= 0:
            FHL[i] = 0
    return FHL

#Calculates the activation ouput for Hidden Layer
def activation(train_images_normalized,weight_ki,bias_k,n):
    #initilizes a 64 element 1D array of zeros
    FHL = numpy.zeros(64)
    #Calculates the weighted sum of one pixel in digit and indexes it in FHL array.
    #After loop, will end up with a 64 element 1D array that represents the value of each neuron in the 1st hidden layer.
    for i in range(64):
        FHL[i] = weight_ki[i]@train_images_normalized[n]
    FHL = FHL + bias_k
    #Applies the non-linearity to the neurons.
    FHL = ReLU(FHL)
    #returns the 64 element array representing the 64 neurons in the first hidden layer for a single digit
    return FHL

#Calculates the raw output neurons in final layer
def activation2(FHL,weight_jk,bias_j):
    #initilizes a 10 element 1D array
    output = numpy.zeros(10)
    #Takes the weighted sum of weights and activation neurons in the hidden layer via matrix multiplication. Add bias via matrix addition.
    #Output will be the output layers' neuron's raw values
    for i in range(10):
        #output[i] = weight_jk_current[i]@FHL_total[digit]
        output[i] = weight_jk[i]@FHL
    output = output + bias_j
    return output

#Softmax function
def softmax(output_local):
    #Calculates the activation neuron's values for the output layer
    output_current = numpy.exp(output_local- numpy.max(output_local))
    #Calculates the sum of these activation neurons
    sum = numpy.sum(output_current)

    #Calculates the probability for each possible digit
    output = output_current/sum

    #returns a 1D array that consists of the probability for each possible digit.
    return output

#function that determines the index of both predicted and true label within the 60000 sames.
def incorrect(probability,one_hot,e,n,category):
    #Computes the index of the digit that the model is most confident is correct
    i_f = numpy.where(probability == numpy.max(probability))
    i_f = i_f[0]
    i_f = i_f[0]
    #computes the index of the true digit
    i_t = numpy.where(one_hot == 1)
    i_t = i_t[0]
    i_t = i_t[0]
    #if the model made an incorrect prediction
    if one_hot[i_f] == 0 :
        #Stores probability value of predicted digit
        false_predicted_probability = round(probability[i_f]*100,2)
        #Stores index of predicted digit
        false_label = i_f
        
        #stores probability of correct digit
        true_predicted_probability = round(probability[i_t]*100,2)
        #stores index of correct digit
        true_label = i_t

        #Logs the statistics in a csv file
        with open("incorrect.csv","a") as f:
            f.write(f"\n{category},{e},{n},{true_label},{true_predicted_probability}%,{false_label},{false_predicted_probability}%")
        return 
    else:
        return

#Cross Entropy Loss function
def loss_entropy(probability,labels,n):
    #initilize an array length of number of initial inputs.
    loss = numpy.zeros(len(probability))

    #returns a 1D array where the index of the digit will be the only value that is > 0, all other values will be 0 due to the binary nature of the algorithm.
    loss[labels[n]] = -numpy.log(probability[labels[n]])
    return loss

#Counts how many times the algorithm predicted correctly
def accuracy_num(correct,correct_total,one_hot,probability):
    #computes index of prediction
    index = numpy.where(probability == numpy.max(probability))
    #checks if the prediction is correct
    if one_hot[index[0]] == 1 :
        #Increase 1 for both batch and epoch
        correct += 1
        correct_total += 1
        return correct, correct_total
    else:
        return correct, correct_total

#Calculates the derivative for activation w.r.t raw value. Two equations generalized into 1
def dReLU(FHL,k):
    if FHL[k] > 0:
        return 1
    else:
        return 0

#Calculates all derivatives for all weights and biases in the network for each input sample, and stores it in an array.
def backprop(probability,weight_jk,train_images_normalized,FHL,n,one_hot):
    #initilizing an empty list that will hold all the derivatives of the loss value w.r.t the raw output value. 
    # Each column will compute out the derivative for that specific raw output value, for a total of 10 values per row.
    dC_dzL = numpy.zeros((10))
    #initilizing an empty list that will hold all the derivatives of the loss value w.r.t the raw neuron value in the hidden layer.
    #Each column will compute out the derivative for that specific raw neuron value for its respective output neuron.
    dC_dzL_1k_unsumed = numpy.zeros((64,10))
    dC_dzL_1k = numpy.zeros((64))

    #initializing a list that will hold all the derivatives of the loss value w.r.t the weight values in the final layer. Each row will represent an individual activation value
    #that is tied to 64 unique weight values. The columns will represent all 64 unique weight values tied to each activation value. This is in a 3D array, where the length of the
    #3rd vector represents the size of the input batch.
    dC_dwLjk = numpy.zeros((10,64))

    #initializing a list that will hold all the derivatives of the loss value w.r.t the bias values in the final layer. Each row will represent an individual input sample. Each 
    #column will compute out the derivative for that specific bias value, for a total of 10 values per row
    dC_dbLj = numpy.zeros((10))
    
    #initializing a list that will hold all the derivatives of the loss value w.r.t the weight values in the final layer - 1. Each row will represent an individual activation layer
    # in the first hidden layer that is tied to 784 unique weight values. The columns will represent all 784 unique weight values tied to each 64 activation values. This is a 3D array,
    #where the length of the 3rd vecto r represents the size of the input batch.
    dC_dwL_1ki = numpy.zeros((64,784))

    #intializing a list that will hold all the derivatives of the loss value w.r.t the bias values in the final layer - 1. Each row will represent an individual input sample.
    #each column will compute out the derivative for that specfiic bias value, for a total of 64 bias values per row
    dC_dbL_1k = numpy.zeros((64))

    #Gradients for the preactivation neurons in the final layer
    for j in range(10):
        #dC/dz[L,j] = a[L,j] - 1
        dC_dzL[j] = probability[j] - one_hot[j]

    #Gradients for the biases in the final layer
    #dC/db[L,j] = a[L,j] - 1
    dC_dbLj = dC_dzL

    #Gradients for the biases in the hidden layer
    for k in range(64):
        for j in range(10):
            #dC/dz[L-1,k] = sum(w[L,jk](a[L,j]-1)*da[L-1,k]/dz[L-1,k])
            dC_dzL_1k_unsumed[k,j] = weight_jk[j,k]*dC_dzL[j]*dReLU(FHL,k)
    for k in range(64):
        dC_dzL_1k[k] = numpy.sum(dC_dzL_1k_unsumed[k])
    #dC/db[L-1,k] = sum(w[L,jk](a[L,j]-1)*da[L-1,k]/dz[L-1,k])
    dC_dbL_1k = dC_dzL_1k
    
    #Gradients for the weights in the input layer
    for k in range(64):
        for i in range(784):
            dC_dwL_1ki[k,i] = dC_dzL_1k[k]*train_images_normalized[n,i]

    #Gradients for the weights in the hidden layer
    for j in range(10):
        for k in range(64):
            #dC/dW[L,jk] = (a[L,j] -1)(a[L-1,k])
            dC_dwLjk[j,k] = dC_dzL[j]*FHL[k]
    
    return dC_dwLjk, dC_dbLj, dC_dbL_1k, dC_dwL_1ki, dC_dzL

#calculates the mean gradient for 1 batch
def norm_gradients(dw_FL_batch, db_FL_batch, db_FL_1_batch, dw_FL_1_batch):
    dbias_j_norm = round(numpy.sqrt(numpy.sum(db_FL_batch**2)),4)
    dbias_k_norm = round(numpy.sqrt(numpy.sum(db_FL_1_batch**2)),4)
    dweight_jk_norm = round(numpy.sqrt(numpy.sum(dw_FL_batch**2)),4)
    dweight_ki_norm = round(numpy.sqrt(numpy.sum(dw_FL_1_batch**2)),4)

    return dbias_j_norm,dbias_k_norm,dweight_jk_norm,dweight_ki_norm

#Update parameters based on learning rate of 0.01 and gradients.
def learning(bias_j,dbias_FL,bias_k,dbias_FL_1,weight_jk,dweight_FL,weight_ki,dweight_FL_1,lr):
    
    #Updates biases in output layer
    bias_j_new = bias_j - lr*dbias_FL
    #Updates biases in hidden layer
    bias_k_new = bias_k - lr*dbias_FL_1
    #updates weights in hidden layer
    weight_jk_new = weight_jk - lr*dweight_FL
    #updates weights in input layer
    weight_ki_new = weight_ki - lr*dweight_FL_1

    return bias_j_new, bias_k_new, weight_jk_new, weight_ki_new

#calculates the mean predicted probability, loss, accuracy for a single batch.
def batch_log_calc(probability_avg_batch,batch_num,loss_avg_batch,correct):

    #Calculates the average probability for how confident the model is when predicting the correct digit, rounded to 2 d.p, in percetange
    probability_avg_batch = round((probability_avg_batch/batch_num)*100,2)
    #Calculates the average loss value for the batch, rounded to 4 d.p
    loss_avg_batch = round(loss_avg_batch/batch_num,4)
    #calculates how accurate the model is at in predicting the correct digit, in percentage, rounded to 2 d.p
    accuracy_batch = round((correct/batch_num)*100,2)

    return  probability_avg_batch,loss_avg_batch, accuracy_batch

#Determines if the model predicted correctly for the one sample
def incorrect_correct(probability_total,one_hot_total,n):
    #Finds the index of what the model predicted
    i_p = numpy.where(probability_total[n] == numpy.max(probability_total[n]))
    if one_hot_total[n,i_p] == 1:
        return True
    else:
        return False

#Returns the cumulative total for the model's correct predictions from one training epoch every 10000 samples
def class_accuracy(class_total,probability_total,one_hot_total,n):
    if n == 9999:
        #from index 0 to 9999 (10000 samples)
        for i in range(10000):
            #Determines if model made a correct prediction
            if incorrect_correct(probability_total,one_hot_total,i) == True:
                #computing index of correct prediction; essentially represents the class it predicted right
                index = numpy.where(probability_total[i] == numpy.max(probability_total[i]))
                #Increase that index by 1
                class_total[index] += 1
    elif n == 19999:
        #from index 10000 to 19999 (10000 samples)
        for i in range(10000,20000):
            #Determines if model made a correct prediction
            if incorrect_correct(probability_total,one_hot_total,i) == True:
                #computing index of correct prediction; essentially represents the class it predicted right
                index = numpy.where(probability_total[i] == numpy.max(probability_total[i]))
                #Increase that index by 1
                class_total[index] += 1
    elif n == 29999:
        #from index 20000 to 29999 (10000 samples)
        for i in range(20000,30000):
            #Determines if model made a correct prediction
            if incorrect_correct(probability_total,one_hot_total,i) == True:
                #computing index of correct prediction; essentially represents the class it predicted right
                index = numpy.where(probability_total[i] == numpy.max(probability_total[i]))
                #Increase that index by 1
                class_total[index] += 1
    elif n == 39999:
        #from index 30000 to 39999 (10000 samples)
        for i in range(30000,40000):
            #Determines if model made a correct prediction
            if incorrect_correct(probability_total,one_hot_total,i) == True:
                #computing index of correct prediction; essentially represents the class it predicted right
                index = numpy.where(probability_total[i] == numpy.max(probability_total[i]))
                #Increase that index by 1
                class_total[index] += 1
    elif n == 49999:
        #from index 40000 to 49999 (10000 samples)
        for i in range(40000,50000):
            #Determines if model made a correct prediction
            if incorrect_correct(probability_total,one_hot_total,i) == True:
                #computing index of correct prediction; essentially represents the class it predicted right
                index = numpy.where(probability_total[i] == numpy.max(probability_total[i]))
                #Increase that index by 1
                class_total[index] += 1
    return class_total

#Returns the total cumulative of each class in one training epoch every 10000 samples.
def labels_accuracy(labels_total,train_labels,n):
    if n == 9999:
        #from index 0 to 9999 (10000 samples)
        for i in range(10000):
            #Increase the index of class by 1
            labels_total[train_labels[i]] += 1

    if n == 19999:
        #from index 10000 to 19999 (10000 samples)
        for i in range(10000,20000):
            #Increase the index of class by 1
            labels_total[train_labels[i]] += 1

    if n == 29999:
        #from index 20000 to 29999 (10000 samples)
        for i in range(20000,30000):
            #Increase the index of class by 1
            labels_total[train_labels[i]] += 1

    if n == 39999:
        #from index 30000 to 39999 (10000 samples)
        for i in range(30000,40000):
            #Increase the index of class by 1
            labels_total[train_labels[i]] += 1

    if n == 49999:
        #from index 40000 to 49999 (10000 samples)
        for i in range(40000,50000):
            #Increase the index of class by 1
            labels_total[train_labels[i]] += 1

    return labels_total

#returns the total correct predictions by model for 1 validation epoch
def val_class_accuracy(class_total,probability_total,one_hot_total,n):
    for i in range(n+1):
        #Determines if the model predicted correctly for one sample
        if incorrect_correct(probability_total,one_hot_total,i) == True:
            #Computing index of correctly predicted class
            index = numpy.where(probability_total[i] == numpy.max(probability_total[i]))
            #Add one to the predicted class
            class_total[index] += 1
    return class_total

#Returns the total number of each class for one validation epoch
def val_labels_accuracy(labels_total,val_labels,n):
    for i in range(n+1):
        #Adds one to the index of class
        labels_total[val_labels[i]] += 1
    return labels_total
    

#calculates the time and the mean loss, accuracy, and gradient for an epoch. Gradient is normalized gradient of each iteration averaged out over an epoch
def epoch_log_calc(n,loss_avg_epoch,correct_total,end_time,start_time,gradient_mean):
    #calculates the average loss for 1 epoch
    loss_avg_epoch = round(loss_avg_epoch/(n+1),4)
    #calculates the probability of a correct prediction
    correct_total = round(correct_total/(n+1)*100,2)
    #calculates the time taken to run 1 epoch
    time_epoch = round(end_time - start_time)
    #calculates the average mean gradient for one iteration over 1 epoch
    gradient_mean = round(gradient_mean/(n+1),4)
    return loss_avg_epoch,correct_total,time_epoch,gradient_mean


def train_epoch(bias_j, bias_k, weight_jk, weight_ki,train_dataset,e,batch_num,val_dataset,train_labels,val_labels,lr):

    val_e, count,batch,correct,correct_total,probability_avg_batch,loss_avg_batch,loss_avg_epoch,gradient_mean,dw_FL_batch,dw_FL_1_batch,db_FL_batch,db_FL_1_batch = 0,0,0,0,0,0,0,0,0,0,0,0,0

    probability_total = numpy.zeros((len(train_dataset),10))
    one_hot_total = numpy.zeros((len(train_dataset),10))
    class_total = numpy.zeros(10)
    labels_total = numpy.zeros(10)
    #logs start time of 1 epoch
    start_time = time.time()
    
    #Loops each iteration for a total of 60000 iterations
    for n in range(len(train_dataset)):
        if count == 0:
            #logs start time for 1 batch
            batch_start = time.time()
        
        #Generates a binary array (1 = correct, 0 = incorrect) where the index of the array represents the digit
        one_hot = one_hot_encoding(train_labels,n)
        #stores 'nth' one hot in an array
        one_hot_total[n] = one_hot

        #returns the activation value for the 64 neurons in the FHL
        FHL = activation(train_dataset,weight_ki,bias_k,n)

        #returns the raw output values for the 10 neurons in the output layer
        output = activation2(FHL,weight_jk,bias_j)

        #Applies the softmax value to the 10 raw neuron values in the final layer to transform it into a probability format
        probability = softmax(output)
        #stores 'nth' probability in an array
        probability_total[n] = probability

        #determines if the model made an incorrect prediction.
        #incorrect(probability,one_hot,e,n,category = "train")

        # #Calculates the loss entropy for each 10 output neurons.
        loss = loss_entropy(probability,train_labels,n)
        # #Calulates the loss value for the entire network. 
        loss_network = numpy.sum(loss)

        #Counts how many times the algorithm predicted correctly
        correct,correct_total = accuracy_num(correct,correct_total,one_hot,probability)

        #Calculates all derivatives for all weights and biases in the network for each input sample, and stores it in an array.
        dweight_FL, dbias_FL, dbias_FL_1, dweight_FL_1, dC_dzL = backprop(probability,weight_jk,train_dataset,FHL,n,one_hot)

        #calculates the average magnitude of the gradient for the whole network
        gradient_mean += numpy.sqrt((numpy.sum(dweight_FL**2)) + (numpy.sum(dbias_FL**2)) + (numpy.sum(dweight_FL_1**2)) + (numpy.sum(dbias_FL_1**2)))

        #Update parameters based on learning rate of 0.01 and gradients.
        bias_j, bias_k, weight_jk, weight_ki = learning(bias_j,dbias_FL,bias_k,dbias_FL_1,weight_jk,dweight_FL,weight_ki,dweight_FL_1,lr) 

        #Calculates sum of how confident the model is when predicting for the correct digit, for the current batch
        probability_avg_batch +=  probability[train_labels[n]]
        #Calculates the sum of the network's loss value, for current batch
        loss_avg_batch +=  loss_network
        #Calculates the sum of the network's loss value, for the whole epoch
        loss_avg_epoch += loss_network

        #calculates the sum of each batch's parameter's gradients.
        dw_FL_batch += dweight_FL
        dw_FL_1_batch += dweight_FL_1
        db_FL_batch += dbias_FL
        db_FL_1_batch += dbias_FL_1

        #tracking number of iterations in order to log stats for each batch
        count += 1
        #Logging stats after each batch
        if count % batch_num == 0:
            #tracking number of batches
            batch += 1
            #reinitializes tracking number for number of iterations to determine subsequent batch
            count = 0

            #function that calculates the stats for each batch
            probability_avg_batch,loss_avg_batch, accuracy_batch = batch_log_calc(probability_avg_batch,batch_num,loss_avg_batch,correct)
            
            #Calculates the magnitude of each parameter's gradient for 1 batch
            dbias_j_norm,dbias_k_norm,dweight_jk_norm,dweight_ki_norm = norm_gradients(dw_FL_batch, db_FL_batch, db_FL_1_batch, dw_FL_1_batch)

            #logs the end time for one batch
            batch_end = time.time()
            
            #recording a batch's statistics into a csv file
            with open("Test_1_LR0-01/training_report_T1.csv","a") as f:
                f.write(f"\n{e},{batch},{lr},{loss_avg_batch},{probability_avg_batch}%,{accuracy_batch}%,{round((batch_end - batch_start),3)}s,{dweight_ki_norm},{dbias_k_norm},{dweight_jk_norm},{dbias_j_norm}")
            
            #Reintialize the variables to prep for next batch
            probability_avg_batch,correct,loss_avg_batch,dw_FL_batch, db_FL_batch, db_FL_1_batch, dw_FL_1_batch = 0,0,0,0,0,0,0

        if n == 9999 or n == 19999 or n == 29999 or n == 39999:
            print (f"n = {n}")
            print(f"val_e = {val_e}")
            val_e += 1

            #Gets the total cumulative correct predictions by model every 10000 training samples.
            class_total = class_accuracy(class_total,probability_total,one_hot_total,n)
            print(f"Train_class: {class_total}")
            #Gets the total cumulative true labels in the training dataset, every 10000 samples
            labels_total = labels_accuracy(labels_total,train_labels,n)
            print(f"Train_labels: {labels_total}")
            #Calculates the percentage of models correct predictions for each class, arranged in a 1D array with 10 elements,
            tcacc = numpy.round((class_total/labels_total)*100,2)

            #Function to run 1 epoch for validation
            val_loss, val_acc, val_time,val_class_total, val_labels_total = val_epoch(bias_j, bias_k, weight_jk, weight_ki,val_dataset,val_labels)

            #Calculates the percentage of models correct predictions for each class, arranged in a 1D array with 10 elements. From one validation epoch
            vcacc = numpy.round((val_class_total/val_labels_total)*100,2)

            time_val = time.time()
            with open("Test_1_LR0-01/epoch_summary_T1.csv","a") as f:
                f.write(f"\n{(e-1)+(val_e/5)},{round(loss_avg_epoch/(n+1),4)},{val_loss},{round(correct_total/(n+1)*100,2)}%,{val_acc}%,{round(time_val - start_time)}s,{val_time}s,{lr},{round(gradient_mean/(n+1),4)},{tcacc[0]},{tcacc[1]},{tcacc[2]},{tcacc[3]},{tcacc[4]},{tcacc[5]},{tcacc[6]},{tcacc[7]},{tcacc[8]},{tcacc[9]},{vcacc[0]},{vcacc[1]},{vcacc[2]},{vcacc[3]},{vcacc[4]},{vcacc[5]},{vcacc[6]},{vcacc[7]},{vcacc[8]},{vcacc[9]}")


    #Gets the total cumulative correct predictions by model every 10000 training samples.
    class_total = class_accuracy(class_total,probability_total,one_hot_total,n)
    print(f"Train_class: {class_total}")
    #Gets the total cumulative true labels in the training dataset, every 10000 samples
    labels_total = labels_accuracy(labels_total,train_labels,n)
    print(f"Train_labels: {labels_total}")
    #Calculates the percentage of models correct predictions for each class, arranged in a 1D array with 10 elements,
    tcacc = numpy.round((class_total/labels_total)*100,2)

    #logging end time for 1 epoch
    end_time = time.time()

    #Stores the most recent iteration's parameters into a npz file
    numpy.savez(f"Test_1_LR0-01/epoch_{e}_parameters_T1.npz",wki = weight_ki, bk = bias_k, wjk = weight_jk, bj = bias_j)

    #calculates the statistics for 1 epoch
    loss_avg_epoch,correct_total,time_epoch,gradient_mean = epoch_log_calc(n,loss_avg_epoch,correct_total,end_time,start_time,gradient_mean)

    return loss_avg_epoch, correct_total,time_epoch,gradient_mean,bias_j, bias_k, weight_jk, weight_ki,class_total,labels_total,tcacc

def val_epoch(bias_j, bias_k, weight_jk, weight_ki,val_dataset,val_labels):
    
    correct = 0
    correct_total = 0
    loss_avg_epoch = 0
    probability_total = numpy.zeros((len(val_dataset),10))
    one_hot_total = numpy.zeros((len(val_dataset),10))
    class_total = numpy.zeros(10)
    labels_total = numpy.zeros(10)

    #logs start time of 1 epoch
    start_time = time.time()

    for n in range(len(val_dataset)):
        #print(f"val:{n}")
        one_hot = one_hot_encoding(val_labels,n)
        #stores 'nth' one hot in an array
        one_hot_total[n] = one_hot

        #returns the activation value for the 64 neurons in the FHL for one digit
        FHL = activation(val_dataset,weight_ki,bias_k,n)

        #returns the raw output values for the 10 neurons in the output layer
        output = activation2(FHL,weight_jk,bias_j)

        #Applies the softmax value to the 10 raw neuron values in the final layer to transform it into a probability format
        probability = softmax(output)
        #stores 'nth' probability in an array
        probability_total[n] = probability
        
        #determines if the model made an incorrect prediction.
        #incorrect(probability,one_hot,e,n,category = "validation")

        # #Calculates the loss entropy for each 10 output neurons. Ultimately,
        loss = loss_entropy(probability,val_labels,n)

        # #Calulates the loss value for the entire network
        loss_network = numpy.sum(loss)
        #Calculates the sum of the network's loss value, for the whole epoch
        loss_avg_epoch += loss_network

        #Counts how many times the algorithm predicted correctly
        correct,correct_total = accuracy_num(correct,correct_total,one_hot,probability)

    #logging end time for 1 epoch
    end_time = time.time()

    #calculates the average loss for 1 epoch
    val_loss = round(loss_avg_epoch/(n+1),4)
    #calculates the probability of a correct prediction
    val_acc = round(correct_total/(n+1)*100,2)
    #calculates the time taken to run 1 epoch
    val_time = round(end_time - start_time)

    #Gets total correct predictions for each class in a 1D array with 10 elements
    val_class_total = val_class_accuracy(class_total,probability_total,one_hot_total,n)
    print(f"Val_class: {val_class_total}")
    #Gets total number of each class, stored in a 1D array with 10 elements
    val_labels_total = val_labels_accuracy(labels_total,val_labels,n)
    print(f"Val_labels: {val_labels_total}")
    return val_loss, val_acc, val_time, val_class_total, val_labels_total

#reshapes the 3D arraay into a 2D array by compressing the 28x28 matrix inside into a 1D 784 element array
new_train_images = reshape(train_images)

#encodes all the greyscale values (0-255) into a range between 0-1
train_images_normalized = normalize(new_train_images)

#Splits the training data into training and validation data. Just number of rows has changed, each row still contains 784 elements
train_dataset = train_images_normalized[:50000,:]
val_dataset = train_images_normalized[50000:60000,:]
train_labels = train_labels_total[:50000]
val_labels = train_labels_total[50000:60000]

#Sets number of iterations per batch
batch_num = 250
#epoch number
e = 10
#Learning Rate
lr = 0.01

#Loads the stored parameters
#data = numpy.load(f"epoch_{e-1}_parameters.npz")
data = numpy.load(f"Test_1_LR0-01/epoch_9_parameters_T1.npz")
weight_ki = data["wki"]
bias_k = data["bk"]
weight_jk = data["wjk"]
bias_j = data["bj"]

#Function to run 1 epoch for training
train_loss,train_acc,train_time,gradient_mean,bias_j, bias_k, weight_jk, weight_ki, class_total, labels_total, tcacc = train_epoch(bias_j, bias_k, weight_jk, weight_ki,train_dataset,e,batch_num,val_dataset,train_labels,val_labels,lr)


#Function to run 1 epoch for validation
val_loss, val_acc, val_time, val_class_total, val_labels_total = val_epoch(bias_j, bias_k, weight_jk, weight_ki,val_dataset,val_labels)

#Calculates the percentage of models correct predictions for each class, arranged in a 1D array with 10 elements. From one validation epoch
vcacc = numpy.round((val_class_total/val_labels_total)*100,2)

#Record statistics onto a csv file after each epoch
with open("Test_1_LR0-01/epoch_summary_T1.csv","a") as f:
    f.write(f"\n{e},{train_loss},{val_loss},{train_acc}%,{val_acc}%,{train_time}s,{val_time}s,{lr},{gradient_mean},{tcacc[0]},{tcacc[1]},{tcacc[2]},{tcacc[3]},{tcacc[4]},{tcacc[5]},{tcacc[6]},{tcacc[7]},{tcacc[8]},{tcacc[9]},{vcacc[0]},{vcacc[1]},{vcacc[2]},{vcacc[3]},{vcacc[4]},{vcacc[5]},{vcacc[6]},{vcacc[7]},{vcacc[8]},{vcacc[9]}")

