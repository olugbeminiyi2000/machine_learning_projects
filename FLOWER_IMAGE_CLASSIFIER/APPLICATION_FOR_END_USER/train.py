import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from get_input_args import get_input_args

train_dir = get_input_args().dir1
valid_dir = get_input_args().dir2
network = get_input_args().arch
epochs = get_input_args().epochs
hu = get_input_args().hu
lr1 = get_input_args().lr1
lr2 = get_input_args().lr2
lr3 = get_input_args().lr3
train_on = get_input_args().gpu

def train_process(train,valid,network,train_on,epochs,hu,lr1,lr2,lr3):
    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    validation_transforms =  transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])


    


    # TODO: Load the datasets with ImageFolder
    train_data_set = datasets.ImageFolder(train, transform=train_transforms)
    validation_data_set = datasets.ImageFolder(valid,transform=validation_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data_set, batch_size=64,       shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_data_set, batch_size=64, shuffle=True)


    if network == 'densenet121':
        model = models.densenet121('DenseNet121_Weights.IMAGENET1K_V1')
        model.classifier = nn.Sequential(nn.Linear(1024, hu),
                                   nn.ReLU(),
                                   nn.Dropout(0.2),
                                   nn.Linear(hu, 102),
                                   nn.LogSoftmax(dim=1))
        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False
    
        device = torch.device(train_on if torch.cuda.is_available() else "cpu")
        criterion = nn.NLLLoss()
    

        # Only train the classifier parameters, feature parameters are frozen
        optimizer = optim.Adam(model.classifier.parameters(), lr=lr1)

        model.to(device);
        epochs = epochs
        steps = 0
        running_loss = 0
        print_every = 5
        for epoch in range(epochs):
            for inputs, labels in trainloader:
                steps += 1
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(device), labels.to(device)
        
                logps = model.forward(inputs)
                loss = criterion(logps, labels)
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
        
                if steps % print_every == 0:
                    test_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for inputs, labels in validationloader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            logps = model.forward(inputs)
                            batch_loss = criterion(logps, labels)
                    
                            test_loss += batch_loss.item()
                    
                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Test loss: {test_loss/len(validationloader):.3f}.. "
                          f"Test accuracy: {accuracy/len(validationloader):.3f}")
                    running_loss = 0
                    model.train()
            
        
    
    elif network == 'densenet169':
        model = models.densenet169('DenseNet169_Weights.IMAGENET1K_V1')
        model.classifier = nn.Sequential(nn.Linear(1664, hu),
                                   nn.ReLU(),
                                   nn.Dropout(0.2),
                                   nn.Linear(hu, 102),
                                   nn.LogSoftmax(dim=1))
        
        device = torch.device(train_on if torch.cuda.is_available() else "cpu")
        criterion = nn.NLLLoss()

        # Only train the classifier parameters, feature parameters are frozen
        optimizer = optim.Adam(model.classifier.parameters(), lr=lr2)

        model.to(device);
        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False
            
        for param in model.classifier.parameters():
            param.requires_grad = True
           
        epochs = epochs
        steps = 0
        running_loss = 0
        print_every = 5
        for epoch in range(epochs):
            for inputs, labels in trainloader:
                steps += 1
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(device), labels.to(device)
        
                logps = model.forward(inputs)
                loss = criterion(logps, labels)
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
        
                if steps % print_every == 0:
                    test_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for inputs, labels in validationloader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            logps = model.forward(inputs)
                            batch_loss = criterion(logps, labels)
                    
                            test_loss += batch_loss.item()
                    
                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Test loss: {test_loss/len(validationloader):.3f}.. "
                          f"Test accuracy: {accuracy/len(validationloader):.3f}")
                    running_loss = 0
                    model.train()
                
            
    elif network == 'resnet152':
        model = models.resnet152('ResNet152_Weights.IMAGENET1K_V2')
        model.fc = nn.Sequential(nn.Linear(2048, hu),
                                   nn.ReLU(),
                                   nn.Dropout(0.2),
                                   nn.Linear(hu, 102),
                                   nn.LogSoftmax(dim=1))
        device = torch.device(train_on if torch.cuda.is_available() else "cpu")
        criterion = nn.NLLLoss()

        # Only train the classifier parameters, feature parameters are frozen
        optimizer = optim.Adam(model.fc.parameters(), lr=lr3)

        model.to(device);
        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
            
        epochs = epochs
        steps = 0
        running_loss = 0
        print_every = 5
        for epoch in range(epochs):
            for inputs, labels in trainloader:
                steps += 1
                 # Move input and label tensors to the default device
                inputs, labels = inputs.to(device), labels.to(device)
        
                logps = model.forward(inputs)
                loss = criterion(logps, labels)
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
        
                if steps % print_every == 0:
                    test_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for inputs, labels in validationloader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            logps = model.forward(inputs)
                            batch_loss = criterion(logps, labels)
                    
                            test_loss += batch_loss.item()
                    
                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Test loss: {test_loss/len(validationloader):.3f}.. "
                          f"Test accuracy: {accuracy/len(validationloader):.3f}")
                    running_loss = 0
                    model.train()
    
    
    return model

model = train_process(train_dir,valid_dir,network,train_on,epochs,hu,lr1,lr2,lr3)
checkpoint = {'model_name': model,
              'state_dict':model.state_dict()
             }

torch.save(checkpoint, 'checkpoint.pth')


            
        
            
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


            
        
            
           
        
        
        
        
        
    
    
                

                
                
                
    
    





















