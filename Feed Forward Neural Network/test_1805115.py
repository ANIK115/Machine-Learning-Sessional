import numpy as np
import sklearn as sk
from sklearn.metrics import f1_score
import torchvision.datasets as ds
from torchvision import transforms

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

class Dense_Fixed:
    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases
    
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    # Forward pass
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    


class Activation_Softmax:
    # Forward pass
    def forward(self, inputs):
        self.inputs = inputs
        # To avoid exploding values, we are subtracting the maximum value from the inputs. This will keep the range from (-infinity, 0) to (0,1)
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize them for each sample
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)


#This is a base class for the loss function
class Loss:
    #common loss calculation
    def calculate(self, output, y):
        # Calculate sample losses
        sample_losses = self.forward(output, y) #Whichever loss class inherits from this class, it will have to implement the forward method

        # Calculate mean loss
        batch_loss = np.mean(sample_losses)

        return batch_loss
    


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
    



class Model:
    def __init__(self):
        self.layers = []
        self.loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
    
    def create_model(self, layers):
        self.layers = layers

    def forward(self, X):
        for layer in self.layers:
            layer.forward(X)
            X = layer.output
        self.output = X

    def test(self, X, y):
        # Perform an inference on a given input
        self.forward(X)
        # Calculate the loss
        data_loss = self.loss_activation.forward(self.output, y)
        # Get test accuracy
        predictions = np.argmax(self.output, axis=1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == y)
        f1_macro_score = f1_score(y, predictions, average='macro')
        return accuracy, f1_macro_score, data_loss, predictions


independent_test_dataset = ds.EMNIST(root='./data', split='letters',
                             train=False,
                             transform=transforms.ToTensor())

model = Model()
layers = []
# Load the model
import pickle
with open('outputs/model_1805115.pickle', 'rb') as f:
    layers = pickle.load(f)

model.create_model(layers)

X_test = independent_test_dataset.data
y_test = independent_test_dataset.targets

# Solve the AttributeError: Tensor object has no attribute astype
X_test = X_test.numpy()
y_test = y_test.numpy()

X_test = X_test.reshape(X_test.shape[0], -1)
y_test = y_test.reshape(y_test.shape[0], -1)

y_test = y_test.flatten()
X_test = X_test / 255.0
y_test = y_test - 1

accuracy, f1_macro_score, loss, predictions = model.test(X_test, y_test)

with open("outputs/TestResult2.txt", "a") as f:
    f.write("Test Accuracy: " + str(accuracy) + " ")
    f.write("Test F1 Score: " + str(f1_macro_score) + " ")
    f.write("Test Loss:" + str(loss) + "\n")

# confusion matrix using sklearn
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, predictions)

#find TP, TN, FP, FN
TP = np.diag(confusion_matrix)
FP = np.sum(confusion_matrix, axis=0) - TP
FN = np.sum(confusion_matrix, axis=1) - TP
TN = []
for i in range(confusion_matrix.shape[0]):
    temp = np.delete(confusion_matrix, i, 0)    # delete ith row
    temp = np.delete(temp, i, 1)                # delete ith column
    TN.append(sum(sum(temp)))

# write the results in a tabular form where each row represents a class labeled from 0 to 25
# each column represents the TP, TN, FP, FN for each class

with open("outputs/TestResult2.txt", "a") as f:
    #write the TP, TN, FP, FN for each class
    f.write("label\t\t TP:\t\t TN:\t\t FP:\t\t FN:\n")
    for i in range(confusion_matrix.shape[0]):
        f.write(str(i) + "\t\t" + str(TP[i]) + "\t\t" + str(TN[i]) + "\t\t" + str(FP[i]) + "\t\t\t" + str(FN[i]) + "\n")

print("Test Accuracy: ", accuracy)
print("Test F1 Score: ", f1_macro_score)
print("Test Loss: ", loss)
# print("Recall: ", TP/(TP+FN))
# print("Precision: ", TP/(TP+FP))
# print("Senstivity: ", TP/(TP+FN))
# print("Specificity: ", TN/(TN+FP))