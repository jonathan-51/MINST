import numpy
import matplotlib.pyplot as plt
import idx2numpy

#Converts the MNIST Binary dataset into a grey scale format
train_images = idx2numpy.convert_from_file("train-images.idx3-ubyte")
train_labels = idx2numpy.convert_from_file("train-labels.idx1-ubyte")
#ensures that the output stays on one line in the terminal (only if it should)
numpy.set_printoptions(linewidth=numpy.inf)

#initilizes array
new_train_images = numpy.array(train_images[0]).reshape(1,784)

#From the 2nd iteration, flatten the 28x28 matrix into a 2 Dimensional 1x784 matrix, and append it.
for i in range(1,5000):
    new_train_images = numpy.append(new_train_images,train_images[i].reshape(1,784),axis=0)
    print(i)

print(new_train_images.shape)


