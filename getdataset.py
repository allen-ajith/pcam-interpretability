import torchvision.datasets as datasets

# Define the root directory where the dataset will be stored
root_dir = './data'

# Download the training dataset
train_dataset = datasets.PCAM(root=root_dir, split='train', download=True)

# Download the validation dataset
val_dataset = datasets.PCAM(root=root_dir, split='val', download=True)

# Download the test dataset
test_dataset = datasets.PCAM(root=root_dir, split='test', download=True)
