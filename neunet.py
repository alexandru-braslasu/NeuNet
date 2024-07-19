import sys
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
from torchvision.models import resnet18
from torch.utils.data import Dataset, DataLoader
from PIL import Image

def process_image(image_path):
    # Load the image file
    image = Image.open(image_path)
    # Resize the image and transform it
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Ensure the image is resized to 32x32
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image_transformed = transform(image).unsqueeze(0)  # Add batch dimension
    return image_transformed

def predict_image(model, image_path, device):
    # Process the image
    image = process_image(image_path)
    image = image.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Disable gradient computation for inference
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_class = classes[predicted.item()]

    return predicted_class

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=256,
                                          shuffle=True)

testset = datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = DataLoader(testset, batch_size=256,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(model, dataloader, device, optimizer, criterion, epoch):
    model.train()

    total_train_loss = 0.0
    dataset_size = 0

    bar = tqdm(enumerate(dataloader), total=len(dataloader), colour='cyan', file=sys.stdout)
    for step, (images, labels) in bar:
        images = images.to(device)
        labels = labels.to(device)

        batch_size = images.shape[0]

        optimizer.zero_grad()
        pred = model(images)
        loss = criterion(pred, labels)

        loss.backward()
        optimizer.step()

        total_train_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = np.round(total_train_loss / dataset_size, 2)
        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss)


    return epoch_loss

def valid_epoch(model, dataloader, device, criterion, epoch):
    model.eval()

    total_val_loss = 0.0
    dataset_size = 0

    correct = 0

    bar = tqdm(enumerate(dataloader), total=len(dataloader), colour='cyan', file=sys.stdout)
    for step, (images, labels) in bar:
        images = images.to(device)
        labels = labels.to(device)

        batch_size = images.shape[0]

        pred = model(images)
        loss = criterion(pred, labels)

        _, predicted = torch.max(pred, 1)
        correct += (predicted == labels).sum().item()

        total_val_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = np.round(total_val_loss / dataset_size, 2)

        accuracy = np.round(100 * correct / dataset_size, 2)

        bar.set_postfix(Epoch=epoch, Valid_Acc=accuracy, Valid_Loss=epoch_loss)

    return accuracy, epoch_loss

def run_training(model, criterion, optimizer, num_epochs):
    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))

    top_accuracy = 0.0

    for epoch in range(num_epochs):

        train_loss = train_epoch(model, trainloader, device, optimizer, criterion, epoch)
        with torch.no_grad():
            val_accuracy, val_loss = valid_epoch(model, testloader, device, criterion, epoch)
            if val_accuracy > top_accuracy:
                print(f"Validation Accuracy Improved ({top_accuracy} ---> {val_accuracy})")
                top_accuracy = val_accuracy
        print()
        
model = resnet18(pretrained=True)

learning_rate = 0.001
epochs = 2
criterion = nn.CrossEntropyLoss()

# ResNet is trained on ImageNet - 1000 classes. We modify the last layer to have only 10 classes
in_features = model.fc.in_features
model.fc = nn.Linear(in_features=in_features, out_features=10, bias=True)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

run_training(model, criterion, optimizer, epochs)

# Example image path
image_path = 'imag2.jpg'

# Prediction
predicted_class = predict_image(model, image_path, device)
print(f'The predicted class of the image is: {predicted_class}')
