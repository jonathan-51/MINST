import numpy
import matplotlib.pyplot as plt
import idx2numpy

#Converts the MNIST Binary dataset into a grey scale format
train_images = idx2numpy.convert_from_file("train-images.idx3-ubyte")
train_labels = idx2numpy.convert_from_file("train-labels.idx1-ubyte")
#ensures that the output stays on one line in the terminal (only if it should)
numpy.set_printoptions(linewidth=numpy.inf)
numpy.set_printoptions(suppress=True)

#reshapes the 3D arraay into a 2D array by compressing the 28x28 matrix inside into a 1D 784 element array
def reshape(train_images):
    new_train_images = numpy.array(train_images[0],dtype=float).reshape(1,784)
    for i in range(1,100):
        new_train_images = numpy.append(new_train_images,train_images[i].reshape(1,784),axis=0)
    return new_train_images

    #new_train_images = train_images.reshape(1000,784)
    #return numpy.array(new_train_images,dtype=float)

#encodes all the greyscale values (0-255) into a range between 0-1
def normalize(new_train_images):
    for i in range(len(new_train_images)):
        for j in range(len(new_train_images[i])):
            new_train_images[i,j] = new_train_images[i,j]/255
    return new_train_images

def one_hot_encoding(digit):
    vector_output = numpy.zeros(10,dtype=int)
    vector_output[digit] = 1 
    return vector_output


def ReLU(FHL):
    for i in range(64):
        if FHL[i] <= 0:
            FHL[i] = 0
    return FHL

#Calculates the activation ouput
def activation(train_images_normalized,digit):
    #initilizes a 64x784 array
    weight_ki = numpy.random.normal(loc=0,scale=numpy.sqrt((2/784)),size=(64,784))
    #initlizaes a 64 element 1D array
    bias_k = numpy.zeros((64))
    #initilizes a 64 element 1D array
    FHL = numpy.zeros(64)
    #Calculates the weighted sum of one pixel in digit and indexes it in FHL array.
    #After loop, will end up with a 64 element 1D array that represents the value of each neuron in the 1st hidden layer.
    for i in range(64):
        FHL[i] = weight_ki[i]@train_images_normalized[digit]
    FHL = FHL + bias_k
    #Applies the non-linearity to the neurons.
    FHL = ReLU(FHL)
    #returns the 64 element array representing the 64 neurons in the first hidden layer for a single digit
    return FHL,weight_ki

def activation2(FHL_total,digit):

    #initilizes a 10x64 array
    weight_jk_current = numpy.random.normal(loc=0,scale=numpy.sqrt((2/784)),size=(10,64))
    #initlizaes a 10 element 1D array
    bias_j = numpy.zeros((10))
    #initilizes a 64 element 1D array
    output = numpy.zeros(10)
    for i in range(10):
        output[i] = weight_jk_current[i]@FHL_total[digit]
    output = output + bias_j
    return output,weight_jk_current

def softmax(output_local):
    for i in range(len(output_local)):
        #Calculates the values of e^(output value)
        output_current = numpy.exp(output_local[i])
        #Calculates the sum of these output values after taking the e^
        sum = numpy.sum(output_current)

        #print(f"output_local = {output_local}")
        #print(f"output_current = {output_current}")
        #print(f"sum = {sum}")

        #Calculates the probability
        probability_current = output_current/sum
        output_local[i] = probability_current

        #print(f"output_local{i} = {output_local}")

    #returns an array where each row consists of the probability of the computer's decision for the digit.
    return output_local

def loss_entropy(probability,train_labels):
    #initilize an array length of number of initial inputs.
    loss = numpy.zeros(len(probability))

    #for each initial input, calculates the loss entropy for that individual input
    for i in range(len(probability)):
        
        loss[i] = -numpy.log(probability[i,train_labels[i]])
    #sums up the batch of inputs and calculates the average

    loss_avg = numpy.sum(loss)/len(probability)
    return loss_avg

def backprop(probability,train_labels,weight_jk,train_images_normalized):
    #initilizing an empty list that will hold all the derivatives of the loss value w.r.t the raw output value. Each row represents an individual input sample. Each column
    #will compute out the derivative for that specific raw output value, for a total of 10 values per row.
    dC_dzL = numpy.zeros((len(probability),10))

    #initializing a list that will hold all the derivatives of the loss value w.r.t the weight values in the final layer. Each row will represent an individual activation value
    #that is tied to 64 unique weight values. The columns will represent all 64 unique weight values tied to each activation value. This is in a 3D array, where the length of the
    #3rd vector represents the size of the input batch.
    dC_dwLjk = numpy.zeros((len(probability),10,64))

    #initializing a list that will hold all the derivatives of the loss value w.r.t the bias values in the final layer. Each row will represent an individual input sample. Each 
    #column will compute out the derivative for that specific bias value, for a total of 10 values per row
    dC_dbLj = numpy.zeros((len(probability),10))
    
    #initializing a list that will hold all the derivatives of the loss value w.r.t the weight values in the final layer - 1. Each row will represent an individual activation layer
    # in the first hidden layer that is tied to 784 unique weight values. The columns will represent all 784 unique weight values tied to each 64 activation values. This is a 3D array,
    #where the length of the 3rd vecto r represents the size of the input batch.
    dC_dwL_1ki = numpy.zeros((len(probability),64,784))

    #intializing a list that will hold all the derivatives of the loss value w.r.t the bias values in the final layer - 1. Each row will represent an individual input sample.
    #each column will compute out the derivative for that specfiic bias value, for a total of 64 bias values per row
    dC_dbL_1k = numpy.zeros((len(probability),64))

    for n in range(len(probability)):
        for j in range(10):
            #dC/dz[L,j] = a[L,j] - 1
            dC_dzL[n,j] = probability[n,train_labels[n]] - 1

            #dC/db[L,j] = a[L,j] - 1
            dC_dbLj[n,j] = dC_dzL[n,j]

            for k in range(64):
                #dC/dW[L,jk] = (a[L,j] -1)(a[L-1,k])
                dC_dwLjk[n,j,k] = dC_dzL[n,j]*FHL_total[n,k]

                #dC/db[L-1,k] = w[L,jk](a[L,j] - 1)
                dC_dbL_1k[n,k] = dC_dzL[n,j]* weight_jk[n,j,k]

                for i in range(784):
                    #dC/dw[L-1,ki] = w[L,jk](a[L,j] - 1)a[L-2,i]
                    dC_dwL_1ki[n,k,i] = dC_dzL[n,j]*weight_jk[n,j,k]*train_images_normalized[n,i]

    return dC_dwLjk, dC_dbLj, dC_dbL_1k, dC_dwL_1ki

number = 2

#reshapes the 3D arraay into a 2D array by compressing the 28x28 matrix inside into a 1D 784 element array
new_train_images = reshape(train_images)

#encodes all the greyscale values (0-255) into a range between 0-1
train_images_normalized = normalize(new_train_images)

FHL_total = []
for digit in range(number):
    #returns the activation value for the 64 neurons in the FHL for one digit
    FHL,weight_ki = activation(train_images_normalized,digit)
    #Appends the value for each neuron in an array
    FHL_total = numpy.append(FHL_total,FHL)

#Transforms the array in a way that each row represents the 64 neurons that encode the specific digit.
FHL_total = FHL_total.reshape(-1,64)
#actual = one_hot_encoding(train_labels[0])

weight_jk = numpy.zeros((number,10,64))
output_total = []
for digit in range(number):
    #returns the value of the 10 neurons that encodes the 10 possible digits.
    output,weight_jk_current = activation2(FHL_total,digit)
    #appends the value for each neuron in a 1D array
    output_total = numpy.append(output_total,output)
    #appends the weights_jk of the current input sample to an array
    weight_jk[digit] = weight_jk_current

#transforms the 1D array in such a way that each row represents the 10 neurons that encode a specific digit.
output_total = output_total.reshape(-1,10)

#returns the values encoded in a probability format.
probability = softmax(output_total)


#Calculates the average loss entropy
loss_average = loss_entropy(probability,train_labels)

#Calculates all derivatives for all weights and biases in the network for each input sample, and stores it in an array.
dweight_FL, dbias_FL, dbias_FL_1, dweight_FL_1 = backprop(probability,train_labels,weight_jk,train_images_normalized)

print(dweight_FL_1.shape)
print(dweight_FL_1)