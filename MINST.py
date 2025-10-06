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
    weight = numpy.random.normal(loc=0,scale=numpy.sqrt((2/784)),size=(64,784))
    #initlizaes a 64 element 1D array
    bias = numpy.zeros((64))
    #initilizes a 64 element 1D array
    FHL = numpy.zeros(64)
    #Calculates the weighted sum of one pixel in digit and indexes it in FHL array.
    #After loop, will end up with a 64 element 1D array that represents the value of each neuron in the 1st hidden layer.
    for i in range(64):
        FHL[i] = weight[i]@train_images_normalized[digit]
    FHL = FHL + bias
    #Applies the non-linearity to the neurons.
    FHL = ReLU(FHL)
    #returns the 64 element array representing the 64 neurons in the first hidden layer for a single digit
    return FHL

def activation2(FHL_total,digit):

    #initilizes a 10x64 array
    weight = numpy.random.normal(loc=0,scale=numpy.sqrt((2/784)),size=(10,64))
    #initlizaes a 10 element 1D array
    bias = numpy.zeros((10))
    #initilizes a 64 element 1D array
    output = numpy.zeros(10)
    for i in range(10):
        output[i] = weight[i]@FHL_total[digit]
    output = output + bias
    return output

def softmax(output_local):
    for i in range(len(output_local)):
        #Calculates the values of e^(output value)
        output_current = numpy.exp(output_local[i])
        #Calculates the sum of these output values after taking the e^
        sum = numpy.copy(numpy.sum(output_current))
        #Calculates the probability
        probability_current = output_current/sum
        output_local[i] = probability_current
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

#reshapes the 3D arraay into a 2D array by compressing the 28x28 matrix inside into a 1D 784 element array
new_train_images = reshape(train_images)
#encodes all the greyscale values (0-255) into a range between 0-1
train_images_normalized = normalize(new_train_images)
FHL_total = []
for digit in range(15):
    #returns the activation value for the 64 neurons in the FHL for one digit
    FHL = activation(train_images_normalized,digit)
    #Appends the value for each neuron in an array
    FHL_total = numpy.append(FHL_total,FHL)
#Transforms the array in a way that each row represents the 64 neurons that encode the specific digit.
FHL_total = FHL_total.reshape(-1,64)
#actual = one_hot_encoding(train_labels[0])

output_total = []
for digit in range(15):
    #returns the value of the 10 neurons that encodes the 10 possible digits.
    output = activation2(FHL_total,digit)
    #appends the value for each neuron in a 1D array
    output_total = numpy.append(output_total,output)
#transforms the 1D array in such a way that each row represents the 10 neurons that encode a specific digit.
output_total = output_total.reshape(-1,10)
#returns the values encoded in a probability format.
probability = softmax(output_total)
print(probability.shape)
#Calculates the average loss entropy
loss_average = loss_entropy(probability,train_labels)

print(loss_average)
