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

X = train_images.reshape(train_images.shape[0], -1)
y = train_labels.reshape(train_labels.shape[0], -1)
print(y.shape)
print(len(y.shape))

#make y a 1D array
y = y.flatten()
print(y.shape)
print(len(y.shape))
y = y-1

# Perform one-hot encoding on the labels
# num_classes = 27
# one_hot_labels = torch.zeros(len(y), num_classes)
# one_hot_labels[torch.arange(len(y)), y] = 1

# # Print the shape of the one-hot encoded labels
# print(one_hot_labels.shape)

#print the labels in labels.txt file
with open("labels.txt", "w") as f:
    for s in y:
        f.write(str(s) +"\n")


