import numpy as np
import nnfs
from nnfs.datasets import spiral_data

# nnfs.init()

class Layer_Dense:

    # Layer initialization : He initialization
    def __init__(self, n_inputs, n_neurons):
        he = np.sqrt(2 / n_inputs)
        self.weights = he * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        #parameters for batch normalization
        self.gamma = np.ones((1, n_neurons))
        self.beta = np.zeros((1, n_neurons))
        self.epsilon = 1e-5  # small constant to avoid division by zero in batch normalization

    
    # Forward pass
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward pass
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU:
    # Forward pass
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    
    # Backward pass
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
    
class Activation_Softmax:
    # Forward pass
    def forward(self, inputs):
        self.inputs = inputs
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    # Backward pass
    def backward(self, dvalues):
        #Create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)

            # Calculate Jacobian matrix of the output and
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            # Calculate sample-wise gradient and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

class Loss:
    #common loss calculation
    def calculate(self, output, y):
        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        batch_loss = np.mean(sample_losses)

        return batch_loss

class Loss_CategoricalCrossentropy(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):
        #clip data to prevent division by 0
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        n = len(y_pred_clipped)

        # Probabilities for target values -
        # for categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(n), y_true]
        
        # for one hot encoding
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        
        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
    # Backward pass
    def backward(self, dvalues, y_true):
        num_samples = len(dvalues)
        num_labels = len(dvalues[0])

        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(num_labels)[y_true]
        
        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / num_samples


# Softmax classifier - combined Softmax activation and cross-entropy loss for faster backward step

class Activation_Softmax_Loss_CategoricalCrossentropy:

    #Creates activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()
    
    #Forward pass
    def forward(self, inputs, y_true):
        #Output layer's activation function
        self.activation.forward(inputs)
        #Set the output
        self.output = self.activation.output
        #Calculate loss value
        losses =  self.loss.calculate(self.output, y_true)
        #Return losses
        return losses
    
    #Backward pass
    def backward(self, dvalues, y_true):
        #Number of samples
        num_samples = len(dvalues)
        #If labels are one-hot encoded, turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        
        #Copy so we can safely modify
        self.dinputs = dvalues.copy()
        #Calculate gradient
        self.dinputs[range(num_samples), y_true] -= 1
        #Normalize gradient
        self.dinputs = self.dinputs / num_samples 


class Optimizer_Adam:

    def __init__(self, learning_rate = 0.001, decay = 0, epsilon = 1e-7, beta_1 = 0.9, beta_2 = 0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
    
    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1 / (1 + self.decay * self.iterations))
    
    # Update parameters
    def update_params(self, layer):
        # If layer does not contain momentum arrays, create them filled with zeros
        if not hasattr(layer, 'weight_momentums'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)

            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        # Update momentum with current gradients
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        # Get corrected momentum
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        #update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases ** 2

        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))


        # Update weights and biases
        layer.weights += -self.current_learning_rate * weight_momentums_corrected/ (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected/ (np.sqrt(bias_cache_corrected) + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

class Layer_Dropout:

    def __init__(self, rate):
        #Store rate, we invert it as for example for dropout of 0.1 we need success rate of 0.9
        self.rate = 1 - rate

    #Forward pass
    def forward(self, inputs):
        #Save input values
        self.inputs = inputs
        #Generate and save scaled mask
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        #Apply mask to output values
        self.output = inputs * self.binary_mask

    #Backward pass
    def backward(self, dvalues):
        #Gradient on values
        self.dinputs = dvalues * self.binary_mask

#Load Emnist dataset from data folder
import torchvision.datasets as ds
from torchvision import transforms
import torch.utils.data


train_validation_dataset = ds.EMNIST(root='./data', split='letters',
                              train=True,
                              transform=transforms.ToTensor(),
                              download=True)


independent_test_dataset = ds.EMNIST(root='./data', split='letters',
                             train=False,
                             transform=transforms.ToTensor())


# separate the training and validation datasets with a 85/15 split
train_dataset, validation_dataset = torch.utils.data.random_split(train_validation_dataset, [int(0.85 * len(train_validation_dataset)), int(0.15 * len(train_validation_dataset))])

#separate the classes from the images
train_images = train_dataset.dataset.data
train_labels = train_dataset.dataset.targets

# Solve the AttributeError: Tensor object has no attribute astype
train_images = train_images.numpy()
train_labels = train_labels.numpy()

X = train_images.reshape(train_images.shape[0], -1)
y = train_labels.reshape(train_labels.shape[0], -1)
y = y.flatten()

X = X / 255.0
#reduce the values of labels by 1
y = y - 1



# Create Dense layer with 784 input features and 1024 output values
dense1 = Layer_Dense(784, 1024)

# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()

# Create Dropout layer
dropout1 = Layer_Dropout(0.3)

# Create second Dense layer with 1024 input features (as we take output
# of previous layer here) and 26 output values (output values)
dense2 = Layer_Dense(1024, 26)

# Create Softmax classifier's combined loss and activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# Create optimizer
optimizer = Optimizer_Adam(learning_rate=0.05, decay=5e-7)

# Train in loop with batch size 1024
for epoch in range(100):
    batch_loss = 0
    batch_accuracy = 0
    batch_size = 1248
    num_of_batches = len(X) // batch_size
    # Shuffle indices
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    for mini_batch in range(num_of_batches):
        # Get random set of samples
        X_batch = X[mini_batch * batch_size:(mini_batch + 1) * batch_size]
        y_batch = y[mini_batch * batch_size:(mini_batch + 1) * batch_size]

        # Forward pass
        dense1.forward(X_batch)
        activation1.forward(dense1.output)
        dropout1.forward(activation1.output)
        dense2.forward(activation1.output)

        # Calculate loss from output of dense2 (softmax activation)
        loss = loss_activation.forward(dense2.output, y_batch)

        # Calculate accuracy from output of dense2 and targets
        predictions = np.argmax(loss_activation.output, axis=1)
        if len(y_batch.shape) == 2:
            y_batch = np.argmax(y_batch, axis=1)
        
        #print predictions and targets
        # for i in range(len(predictions)):
        #     print(predictions[i], y_batch[i])
        accuracy = np.mean(predictions == y_batch)

        # Backward pass
        loss_activation.backward(loss_activation.output, y_batch)
        dense2.backward(loss_activation.dinputs)
        dropout1.backward(dense2.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        # Update weights and biases
        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()

        batch_accuracy += accuracy
        batch_loss += loss
    
    batch_accuracy /= num_of_batches
    batch_loss /= num_of_batches
    #print the results in result.txt file
    with open("result.txt", "a") as f:
        f.write("Epoch: "+str(epoch) + " Accuracy: " + str(batch_accuracy) + " Loss:" + str(batch_loss) + "\n")

# Validation

X = validation_dataset.dataset.data
y = validation_dataset.dataset.targets

# Solve the AttributeError: Tensor object has no attribute astype
X = X.numpy()
y = y.numpy()

X = X.reshape(train_images.shape[0], -1)
y = y.reshape(train_images.shape[0], -1)
y = y.flatten()
X = X / 255.0
y = y - 1

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)

# Calculate loss from output of dense2 (softmax activation)
loss = loss_activation.forward(dense2.output, y)

# Calculate accuracy from output of dense2 and targets
predictions = np.argmax(loss_activation.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)

accuracy = np.mean(predictions == y)

with open("result.txt", "a") as f:
    f.write("Validation Accuracy: " + str(accuracy) + "\n")
    f.write("Validation Loss:" + str(loss) + "\n")



