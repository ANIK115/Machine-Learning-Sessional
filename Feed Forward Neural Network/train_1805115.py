import numpy as np
import sklearn as sk
from sklearn.metrics import f1_score
import pickle

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


import csv
import numpy as np
import matplotlib.pyplot as plt

def generate_graph(filename, graph_name):
    # read the data from the csv file
    graph_epoch = []
    graph_validation_accuracy = []
    graph_validation_loss = []
    graph_validation_f1_macro_score = []
    graph_train_accuracy = []
    graph_train_loss = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the first row (header)
        for row in reader:
            graph_epoch.append(int(row[0]))  # Convert epoch to integer
            graph_validation_accuracy.append(float(row[1]))  # Convert to float
            graph_validation_loss.append(float(row[2]))  # Convert to float
            graph_validation_f1_macro_score.append(float(row[3]))  # Convert to float
            graph_train_accuracy.append(float(row[4]))  # Convert to float
            graph_train_loss.append(float(row[5]))  # Convert to float



    import matplotlib.pyplot as plt

    # Create a figure and a set of subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

    # Plot Train & Validation Loss
    axs[0].plot(graph_epoch, graph_train_loss, color='blue', label='train_loss')
    axs[0].plot(graph_epoch, graph_validation_loss, color='green', label='validation_loss')
    axs[0].set_title('Train & Validation Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    # Set y-axis from 0 to 5 with a step of 0.5
    axs[0].set_ylim(0, 5)  # Set the limits of the y-axis
    axs[0].set_yticks(np.arange(0, 5.5, 0.5))  # Set the ticks from 0 to 5 with 0.5 step


    # Plot Train & Validation Accuracy
    axs[1].plot(graph_epoch, graph_train_accuracy, color='blue', label='train_accuracy')
    axs[1].plot(graph_epoch, graph_validation_accuracy, color='green', label='validation_accuracy')
    axs[1].set_title('Train & Validation Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()

    # Set y-axis from 0 to 1 with a step of 0.1
    axs[1].set_ylim(0, 1)  # Set the limits of the y-axis
    axs[1].set_yticks(np.arange(0, 1.1, 0.1))  # Set the ticks from 0 to 1 with 0.1 step


    # Plot F1 Macro Score
    axs[2].plot(graph_epoch, graph_validation_f1_macro_score, color='green', label='validation_f1_macro_score')
    axs[2].set_title('F1 Macro Score')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('F1 Score')
    axs[2].legend()

    # Set y-axis from 0 to 1 with a step of 0.1
    axs[2].set_ylim(0, 1)  # Set the limits of the y-axis
    axs[2].set_yticks(np.arange(0, 1.1, 0.1))  # Set the ticks from 0 to 1 with 0.1 step

    # Adjust the layout
    plt.tight_layout()

    # Save the figure
    plt.savefig(graph_name)

    # Show the plot
    plt.show()


class Dense_Fixed:
    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases
    
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

# Dense layer
class Layer_Dense:
    
    # Layer initialization : He initialization
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1=0, weight_regularizer_l2=0, bias_regularizer_l1=0, bias_regularizer_l2=0):
        he = np.sqrt(2 / n_inputs)
        self.weights = he * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2


    def l1_regularizer(self):
        dL1 = np.ones_like(self.weights)
        dL1[self.weights < 0] = -1
        self.dweights += self.weight_regularizer_l1 * dL1

        dL1 = np.ones_like(self.biases)
        dL1[self.biases < 0] = -1
        self.dbiases += self.bias_regularizer_l1 * dL1

    def l2_regularizer(self):
        self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

    # Forward pass
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward pass
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        #Gradients on regularization
        # L1 regularization
        if self.weight_regularizer_l1 > 0 or self.bias_regularizer_l1 > 0:
            self.l1_regularizer()
        
        # L2 regularization 
        if self.weight_regularizer_l2 > 0 or self.bias_regularizer_l2 > 0:
            self.l2_regularizer()

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
        # To avoid exploding values, we are subtracting the maximum value from the inputs. This will keep the range from (-infinity, 0) to (0,1)
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize them for each sample
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    # Backward pass
    def backward(self, dvalues):
        #uninitialized array instead of initializing it with zeros
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)

            diagonal_matrix = np.diagflat(single_output)    # S_ij * delta_jk
            S_ij_Dot_S_ik = np.dot(single_output, single_output.T)
            jacobian_matrix = diagonal_matrix - S_ij_Dot_S_ik

            # Calculate sample-wise gradient and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


#This is a base class for the loss function
class Loss:
    #common loss calculation
    def calculate(self, output, y):
        # Calculate sample losses
        sample_losses = self.forward(output, y) #Whichever loss class inherits from this class, it will have to implement the forward method

        # Calculate mean loss
        batch_loss = np.mean(sample_losses)

        return batch_loss
    
    def regularization_loss(self, layer):
        regularization_loss = 0
        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)
        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)
        
        return regularization_loss


class Loss_CategoricalCrossentropy(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):
        #clip data to prevent division by 0
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values -
        # for categorical labels: Here the y_pred has predicted probabilities for each class but we need to get the probability of the correct class only. Therefore we're indexing the y_pred_clipped with the correct class index using range(len(y_pred_clipped)) and y_true
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(len(y_pred_clipped)), y_true]
        
        # for one hot encoding
        # elif len(y_true.shape) == 2:
        #     correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        
        # return cross entropy Losses
        return -np.log(correct_confidences)
    
    # Backward pass
    def backward(self, dvalues, y_true):
        num_labels = len(dvalues[0])

        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(num_labels)[y_true]
        
        # Calculate normalized gradient
        self.dinputs = (-y_true / dvalues) / len(dvalues)



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
        #If labels are one-hot encoded, turn them into discrete values
        # if len(y_true.shape) == 2:
        #     y_true = np.argmax(y_true, axis=1)
        
        #Copy so we can safely modify
        self.dinputs = dvalues.copy()
        #Calculate gradient
        self.dinputs[range(len(dvalues)), y_true] -= 1
        #Normalize gradient
        self.dinputs = self.dinputs / len(dvalues) 


class Optimizer_Stochastic_Gradient_Descent:
    def __init__(self, learning_rate=0.01, decay=0.0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0

    def update_learning_rate(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1 / (1 + self.decay * self.iterations))
    
    def update_params(self, layer):
        layer.weights += -self.current_learning_rate * layer.dweights
        layer.biases += -self.current_learning_rate * layer.dbiases

    def update_iteration_count(self):
        self.iterations += 1

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
    def update_learning_rate(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1 / (1 + self.decay * self.iterations))
    
    # Update parameters
    def update_params(self, layer):

        # If layer does not contain momentum arrays, create them filled with zeros
        if not hasattr(layer, 'weight_m'):
            layer.weight_m = np.zeros_like(layer.weights)
            layer.bias_m = np.zeros_like(layer.biases)

            layer.weight_v = np.zeros_like(layer.weights)
            layer.bias_v = np.zeros_like(layer.biases)
        
        layer.weight_m = self.beta_1 * layer.weight_m + (1 - self.beta_1) * layer.dweights
        layer.bias_m = self.beta_1 * layer.bias_m + (1 - self.beta_1) * layer.dbiases
        layer.weight_v = self.beta_2 * layer.weight_v + (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_v = self.beta_2 * layer.bias_v + (1 - self.beta_2) * layer.dbiases ** 2


        weight_m_corrected = layer.weight_m / (1 - self.beta_1 ** (self.iterations + 1))
        bias_m_corrected = layer.bias_m / (1 - self.beta_1 ** (self.iterations + 1))
        weight_v_corrected = layer.weight_v / (1 - self.beta_2 ** (self.iterations + 1))
        bias_v_corrected = layer.bias_v / (1 - self.beta_2 ** (self.iterations + 1))

        layer.weights += -self.current_learning_rate * weight_m_corrected/ (np.sqrt(weight_v_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_m_corrected/ (np.sqrt(bias_v_corrected) + self.epsilon)

    # Call once after any parameter updates
    def update_iteration_count(self):
        self.iterations += 1


class Layer_Dropout:
    
    def __init__(self, rate):
        #Store rate, we invert it as for example for dropout of 0.1 we need success rate of 0.9
        self.rate = 1 - rate

    #Forward pass
    def forward(self, inputs):
        self.inputs = inputs
        self.masking = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.masking

    #Backward pass
    def backward(self, dvalues):
        #Gradient on values
        self.dinputs = dvalues * self.masking


class Model:
    def __init__(self):
        self.layers = []
        self.loss_activation = None
        self.optimizer = None
    
    def create_model(self, layers, loss_activation, optimizer = None):
        self.layers = layers
        self.loss_activation = loss_activation
        self.optimizer = optimizer

    def forward(self, X, type = 'train'):
        for layer in self.layers:
            if type != 'train' and isinstance(layer, Layer_Dropout):
                continue
            layer.forward(X)
            X = layer.output
        self.output = X
    
    def backward(self, y_pred, y_true):
        self.loss_activation.backward(y_pred, y_true)
        dinputs = self.loss_activation.dinputs
        for layer in reversed(self.layers):
            layer.backward(dinputs)
            dinputs = layer.dinputs
        self.dinputs = dinputs
    
    def train_and_validate_model(self, X, y, X_validation, y_validation, iterations, batch_size, model_name, csv_file_name, result_file_name):

        max_f1_macro_score = 0
        best_epoch = 0
        best_accuracy = 0
        best_loss = 0
        graph_epoch = []
        graph_train_accuracy = []
        graph_validation_accuracy = []
        graph_train_loss = []
        graph_validation_loss = []
        graph_validation_f1_macro_score = []
        new_model = Model()

        for epoch in range(iterations):
            batch_loss = 0
            batch_accuracy = 0
            # batch_size = np.random.randint(low=512, high=2048)
            num_of_batches = len(X) // batch_size
            # Shuffle indices
            indices = np.arange(len(X))
            np.random.shuffle(indices)

            for mini_batch in range(num_of_batches):
                X_batch = X[mini_batch * batch_size:(mini_batch + 1) * batch_size]
                y_batch = y[mini_batch * batch_size:(mini_batch + 1) * batch_size]
                self.forward(X_batch)
                loss = self.loss_activation.forward(self.output, y_batch)
                regularization_loss = 0
                for layer in self.layers:
                    if isinstance(layer, Layer_Dense):
                        regularization_loss += self.loss_activation.loss.regularization_loss(layer)
                loss += regularization_loss

                predictions = np.argmax(self.loss_activation.output, axis=1)
                if len(y_batch.shape) == 2:
                    y_batch = np.argmax(y_batch, axis=1)
                
                accuracy = np.mean(predictions == y_batch)

                # Backward pass
                self.backward(self.loss_activation.output, y_batch)

                # Update weights and biases
                self.optimizer.update_learning_rate()
                for layer in self.layers:
                    if isinstance(layer, Layer_Dense):
                        self.optimizer.update_params(layer)
                
                self.optimizer.update_iteration_count()

                batch_loss += loss
                batch_accuracy += accuracy
            
            #Validation
            self.forward(X_validation.copy(), type='validation')
            validation_loss = self.loss_activation.forward(self.output, y_validation)
            validation_predictions = np.argmax(self.loss_activation.output, axis=1)
            if len(y_validation.shape) == 2:
                y_validation = np.argmax(y_validation, axis=1)
            validation_accuracy = np.mean(validation_predictions == y_validation)
            validation_f1_macro_score = f1_score(y_validation, validation_predictions, average='macro')

            #write epoch, validation loss, validation accuracy and validation f1 macro score to file


            if validation_f1_macro_score > max_f1_macro_score:
                max_f1_macro_score = validation_f1_macro_score
                #save the model
                new_model.create_model(self.layers, self.loss_activation)


            graph_epoch.append(epoch)
            graph_train_loss.append(batch_loss/num_of_batches)
            graph_train_accuracy.append(batch_accuracy/num_of_batches)
            graph_validation_loss.append(validation_loss)
            graph_validation_accuracy.append(validation_accuracy)
            graph_validation_f1_macro_score.append(validation_f1_macro_score)

            with open(result_file_name+'.txt', 'a') as f:
                f.write("Epoch: " + str(epoch) + " ")
                f.write("Validation Accuracy: " + str(validation_accuracy) + " ")
                f.write("Validation F1 Score: " + str(validation_f1_macro_score) + " ")
                f.write("Validation Loss:" + str(validation_loss) + "\n")
                #write train accuracy and loss
                f.write("Train Accuracy: " + str(batch_accuracy/num_of_batches) + " ")
                f.write("Train Loss: " + str(batch_loss/num_of_batches) + "\n\n")

            
        
        #clear all the data except the weights and biases
        self = new_model
        new_layers = []
        for layer in self.layers:
            if isinstance(layer, Layer_Dense):
                new_layers.append(Dense_Fixed(layer.weights, layer.biases))
            elif isinstance(layer, Activation_ReLU):
                new_layers.append(Activation_ReLU())
        



        #save the model
        with open(model_name+".pickle", 'wb') as f:
            pickle.dump(new_layers, f)

        # write the data in a csv file
        import csv
        with open(csv_file_name+'.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Validation Accuracy", "Validation Loss", "Validation F1 Score", "Train Accuracy", "Train Loss"])
            for i in range(len(graph_epoch)):
                writer.writerow([graph_epoch[i], graph_validation_accuracy[i], graph_validation_loss[i], graph_validation_f1_macro_score[i], graph_train_accuracy[i], graph_train_loss[i]])

        #generate the graph
        generate_graph(csv_file_name+'.csv', csv_file_name+'.png')

      


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
torch.manual_seed(RANDOM_SEED)
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

#Take 768 features from the images
# X = X[:, :768]

X = X / 255.0
#reduce the values of labels by 1
y = y - 1


#Prepare validation data

X_validation = validation_dataset.dataset.data
y_validation = validation_dataset.dataset.targets

# Solve the AttributeError: Tensor object has no attribute astype
X_validation = X_validation.numpy()
y_validation = y_validation.numpy()

X_validation = X_validation.reshape(X_validation.shape[0], -1)
y_validation = y_validation.reshape(y_validation.shape[0], -1)

# X_validation = X_validation[:, :768]
y_validation = y_validation.flatten()
X_validation = X_validation / 255.0
y_validation = y_validation - 1





# learning_rates = [0.0005, 0.0001, 0.005, 0.001]
# folder = "outputs/"
# model_name = folder + "model_"
# csv_file_name = folder + "result_"
# result_file_name = folder + "result_"
# for i in range(4):

#     dense1 = Layer_Dense(784, 784, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)
#     activation1 = Activation_ReLU()
#     dropout1 = Layer_Dropout(0.3)
#     dense2 = Layer_Dense(784, 256, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)
#     activation2 = Activation_ReLU()
#     dropout2 = Layer_Dropout(0.2)
#     dense3 = Layer_Dense(256, 26)
#     loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

#     optimizer = Optimizer_Adam(learning_rate= learning_rates[i], decay=5e-7)

#     layers = []
#     layers.append(dense1)
#     layers.append(activation1)
#     layers.append(dropout1)
#     layers.append(dense2)
#     layers.append(activation2)
#     layers.append(dropout2)
#     layers.append(dense3)


#     model = Model()
#     model.create_model(layers, loss_activation, optimizer)
#     model.train_and_validate_model(X, y, X_validation, y_validation, iterations=100, batch_size=624, model_name=model_name + str(i), csv_file_name=csv_file_name + str(i), result_file_name=result_file_name + str(i))








# learning_rates = [0.005, 0.001, 0.0005, 0.0001]
# folder = "outputs/"
# model_name = folder + "model_"
# csv_file_name = folder + "result_"
# result_file_name = folder + "result_"
# index = 0
# for i in range(4, 8):
#     dense1 = Layer_Dense(784, 1024, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)
#     activation1 = Activation_ReLU()
#     dropout1 = Layer_Dropout(0.4)
#     dense2 = Layer_Dense(1024, 26, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)
#     loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
#     optimizer = Optimizer_Adam(learning_rate= learning_rates[index], decay=5e-7)
#     index += 1

#     layers = []
#     layers.append(dense1)
#     layers.append(activation1)
#     layers.append(dropout1)
#     layers.append(dense2)

#     model = Model()
#     model.create_model(layers, loss_activation, optimizer)
#     model.train_and_validate_model(X, y, X_validation, y_validation, iterations=100, batch_size=624, model_name=model_name + str(i), csv_file_name=csv_file_name + str(i), result_file_name=result_file_name + str(i))






# learning_rates = [0.1, 0.05, 0.001, 0.005]
# folder = "outputs/"
# model_name = folder + "model_"
# csv_file_name = folder + "result_"
# result_file_name = folder + "result_"
# index = 0
# for i in range(8, 9):
#     dense1 = Layer_Dense(784, 784, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)
#     activation1 = Activation_ReLU()
#     dropout1 = Layer_Dropout(0.3)
#     dense2 = Layer_Dense(784, 256, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)
#     activation2 = Activation_ReLU()
#     dropout2 = Layer_Dropout(0.2)
#     dense3 = Layer_Dense(256, 26)

#     loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
#     optimizer = Optimizer_Stochastic_Gradient_Descent(learning_rate= learning_rates[index], decay=5e-7)
#     index += 1

#     layers = []
#     layers.append(dense1)
#     layers.append(activation1)
#     layers.append(dropout1)
#     layers.append(dense2)
#     layers.append(activation2)
#     layers.append(dropout2)
#     layers.append(dense3)


#     model = Model()
#     model.create_model(layers, loss_activation, optimizer)
#     model.train_and_validate_model(X, y, X_validation, y_validation, iterations=100, batch_size=624, model_name=model_name + str(i), csv_file_name=csv_file_name + str(i), result_file_name=result_file_name + str(i))



#choose best model
model_numbers = 0
best_model = None
max_f1_macro_score = 0
best_model_number = 0

for model_numbers in range(12):
    layers = []
    with open("outputs/model_"+str(model_numbers)+".pickle", 'rb') as f:
        layers = pickle.load(f)
    
    model = Model()
    model.create_model(layers, Activation_Softmax_Loss_CategoricalCrossentropy(), None)
    model.forward(X_validation.copy(), type='validation')
    validation_loss = model.loss_activation.forward(model.output, y_validation)
    validation_predictions = np.argmax(model.loss_activation.output, axis=1)
    if len(y_validation.shape) == 2:
        y_validation = np.argmax(y_validation, axis=1)
    validation_accuracy = np.mean(validation_predictions == y_validation)
    validation_f1_macro_score = f1_score(y_validation, validation_predictions, average='macro')
    if validation_f1_macro_score > max_f1_macro_score:
        max_f1_macro_score = validation_f1_macro_score
        best_model = model
        best_model_number = model_numbers

print("Best model number: ", best_model_number)
print("Best model f1 macro score: ", max_f1_macro_score)
#save the best model as model_1805115.pickle

best_layers =  []
for layer in best_model.layers:
    if isinstance(layer, Dense_Fixed):
        best_layers.append(Dense_Fixed(layer.weights, layer.biases))
    elif isinstance(layer, Activation_ReLU):
        best_layers.append(Activation_ReLU())


# with open("outputs/model_1805115.pickle", 'wb') as f:
#     pickle.dump(best_layers, f)