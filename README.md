# BabyIO


## This repository is designed to allow myself ot learn nural netowrk architecture and build one myself with a decent explaination on every small thing i did not understand.

## Definitions:
### Sigmoid
- The sigmoid function introduces non-linearity into the model
- The sigmoid function is a mathematical function that maps any real-valued number into a value between 0 and 1. It is commonly used as an activation function in neural networks, particularly in binary classification tasks.
- σ(x)=1+e−x1​ where e is the base of the natural logarithm (approximately equal to 2.71828). x is the input to the function.
- Other activation functions, such as ReLU (Rectified Linear Unit), are often preferred for hidden layers.

### weights:
- Weight Initialization: Randomly initializes weights for the connections between layers. This is crucial for breaking symmetry and allowing the network to learn.

### numpy things
- np.random.uniform
- T is for the transpose of the matrix

     Purpose of Transposing: When calculating the error for the hidden layer during backpropagation, you need to propagate the error from the output layer back to the hidden layer. The dot product of the derivative of the predicted output (d_predicted_output) with the transposed weights matrix allows you to compute the contribution of each output neuron’s error to the hidden layer neurons. This operation is essential for determining how much each hidden neuron contributed to the error in the output, which is then used to update the weights connecting the hidden layer to the input layer.
