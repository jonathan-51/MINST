import numpy
import matplotlib.pyplot as plt
import idx2numpy
import time
#Converts the MNIST Binary dataset into a grey scale format
train_images = idx2numpy.convert_from_file("train-images.idx3-ubyte")
train_labels = idx2numpy.convert_from_file("train-labels.idx1-ubyte")
#ensures that the output stays on one line in the terminal (only if it should)
numpy.set_printoptions(linewidth=numpy.inf)
numpy.set_printoptions(suppress=True)

#reshapes the 3D arraay into a 2D array by compressing the 28x28 matrix inside into a 1D 784 element array
def reshape(train_images):
    # new_train_images = numpy.array(train_images[0],dtype=float).reshape(1,784)
    # for i in range(1,90):
    #     new_train_images = numpy.append(new_train_images,train_images[i].reshape(1,784),axis=0)
    # return new_train_images

    new_train_images = train_images.reshape(-1,784)
    return numpy.array(new_train_images,dtype=float)

#encodes all the greyscale values (0-255) into a range between 0-1
def normalize(new_train_images):
    for i in range(len(new_train_images)):
        for j in range(len(new_train_images[i])):
            new_train_images[i,j] = new_train_images[i,j]/255
    return new_train_images

def one_hot_encoding(train_labels, n):
    one_hot = numpy.zeros(10,dtype=int)
    one_hot[train_labels[n]] = 1 
    return one_hot

#ReLU Function
def ReLU(FHL):
    for i in range(64):
        if FHL[i] <= 0:
            FHL[i] = 0
    return FHL

#Calculates the activation ouput for Hidden Layer
def activation(train_images_normalized,weight_ki,bias_k,n):
    #initilizes a 64 element 1D array
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
    #initilizes a 64 element 1D array
    output = numpy.zeros(10)
    for i in range(10):
        #output[i] = weight_jk_current[i]@FHL_total[digit]
        output[i] = weight_jk[i]@FHL
    output = output + bias_j
    return output

#Softmax function
def softmax(output_local):
    #Calculates the values of e^(output value)
    output_current = numpy.exp(output_local- numpy.max(output_local))
    #Calculates the sum of these output values after taking the e^
    sum = numpy.sum(output_current)

    #print(f"output_local = {output_local}")
    #print(f"output_current = {output_current}")
    #print(f"sum = {sum}")

    #Calculates the probability
    output = output_current/sum

    #print(f"output_local{i} = {output_local}")

    #returns an array where each row consists of the probability of the computer's decision for the digit.
    return output

#Cross Entropy Loss function
def loss_entropy(probability,train_labels,n):
    #initilize an array length of number of initial inputs.
    loss = numpy.zeros(len(probability))

    loss[train_labels[n]] = -numpy.log(probability[train_labels[n]])
    return loss

def dReLU(FHL,k):
    if FHL[k] > 0:
        return 1
    else:
        return 0
    
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

def learning(bias_j,dbias_FL,bias_k,dbias_FL_1,weight_jk,dweight_FL,weight_ki,dweight_FL_1):
    lr = 0.01
    #Updates biases in output layer
    bias_j_new = bias_j - lr*dbias_FL
    #Updates biases in hidden layer
    bias_k_new = bias_k - lr*dbias_FL_1
    #updates weights in hidden layer
    weight_jk_new = weight_jk - lr*dweight_FL
    #updates weights in input layer
    weight_ki_new = weight_ki - lr*dweight_FL_1

    return bias_j_new, bias_k_new, weight_jk_new, weight_ki_new

#reshapes the 3D arraay into a 2D array by compressing the 28x28 matrix inside into a 1D 784 element array
new_train_images = reshape(train_images)

#encodes all the greyscale values (0-255) into a range between 0-1
train_images_normalized = normalize(new_train_images)

#initilizes a 10x64 array
weight_jk = numpy.random.normal(loc=0,scale=numpy.sqrt((2/784)),size=(10,64))
#initlizaes a 10 element 1D array
bias_j = numpy.zeros((10))
#initilizes a 64x784 array
weight_ki = numpy.random.normal(loc=0,scale=numpy.sqrt((2/784)),size=(64,784))
#initlizaes a 64 element 1D array
bias_k = numpy.zeros((64))


# data = numpy.load("epoch_1_parameters")
# weight_ki = data["wki"]
# bias_k = data["bk"]
# weight_jk = data["wjk"]
# bias_j = data["bj"]

count = 0
batch = 0
probability_avg = 0
start_time = time.time()
for n in range(60000):

    one_hot = one_hot_encoding(train_labels,n)

    #returns the activation value for the 64 neurons in the FHL for one digit
    FHL = activation(train_images_normalized,weight_ki,bias_k,n)

    #returns the raw output values for the 10 neurons in the output layer
    output = activation2(FHL,weight_jk,bias_j)

    #Applies the softmax value to the 10 raw neuron values in the final layer to transform it into a probability format
    #print(f"output{n}: {output}")
    probability = softmax(output)
    print(f"probability{n}: {probability}, {train_labels[n]}")
    # #Calculates the loss entropy for each 10 output neurons. Ultimately,
    loss = loss_entropy(probability,train_labels,n)

    # #Calulates the loss value for the entire network
    loss_network = numpy.sum(loss)

    #Calculates all derivatives for all weights and biases in the network for each input sample, and stores it in an array.
    dweight_FL, dbias_FL, dbias_FL_1, dweight_FL_1, dC_dzL = backprop(probability,weight_jk,train_images_normalized,FHL,n,one_hot)
    # print(f"gradient{n}: {dweight_FL[0]}")
    # print(f"weight{n}: {weight_jk[0]}")
    # print(f"gradient{n}: {dbias_FL} {train_labels[n]}")
    # print(f"bias{n}: {bias_j}")
    # print(f"gradient{n}: {dC_dzL} {train_labels[n]}")
    # print(f"bias{n}: {output}")    
    #Update parameters based on learning rate of 0.01 and gradients.
    bias_j, bias_k, weight_jk, weight_ki = learning(bias_j,dbias_FL,bias_k,dbias_FL_1,weight_jk,dweight_FL,weight_ki,dweight_FL_1) 

    probability_avg = probability_avg + probability[train_labels[n]]

    count += 1
    if count % 300 == 0:
        batch += 1
        count = 0
        probability_avg = (probability_avg/300)*100
        with open("training_report.csv","a") as f:
            f.write(f"\n1,{batch},{loss_network},0.01,{probability_avg}%")
        probability_avg = 0


numpy.savez("epoch_1_parameters.npz",wki = weight_ki, bk = bias_k, wjk = weight_jk, bj = bias_j)
end_time = time.time()
print(end_time - start_time)

