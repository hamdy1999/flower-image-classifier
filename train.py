import argparse
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms , models
from collections import OrderedDict
import os

def main():
    
    #get the prefered options
    in_args = get_input_args()

    model = build_model(in_args.arch, in_args.hidden_units)
    print("model built")
    train_dataloader, valid_dataloader, test_dataloader, mapping = data_loaders(in_args.data_directory)
    print("data loaded")
    model, optimizer = training(model, train_dataloader, valid_dataloader,test_dataloader, in_args.learning_rate, in_args.gpu, in_args.epochs)
    print("model tarined")
    checkpoint = {'model_state_dict': model.state_dict(),
                  'input_size': 25088,
                  'hidden_units': model.classifier.fc1.out_features,   
                  'dropout': 0.5,
                  'output_size': 102,
                  'map_classes': mapping,
                  'epochs': 3,
                  'optimizer': optimizer.state_dict(),
                  'lr': 0.001,
                  'arch': in_args.arch,
                  'gpu': in_args.gpu }
    print('checkpoint dict is succeccfully created')
    save_checkpoint(checkpoint, in_args.save_dir)
    print("train.py running is succeccfuly done")
       
def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object. 
     7 command line arguments are created:
       data_directory - the directory that has the data for training, validating, and testing images (defualt = "flower")
       save_dir - Set directory to save checkpoints (default = '') 
       arch - CNN model architecture to use for image classification(default = 'vgg')
       learning_rate - step that th model learn bu(default- .001)
       epochs - number of epochs training (default = 4)
       hidden_units - units that hidden layer composite of (default = 1000)
       gpu - the device that model training on (default = 'cpu')
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory',type=str, default = 'flowers', help='the directory that has the data')
    parser.add_argument('--save_dir', type=str, default='', help='path to save the checkpoint')
    parser.add_argument('--arch', type=str, default='vgg19', help='chosen models')
    parser.add_argument('--learning_rate', type=float, default = 0.001, help='the rate that model learn by')
    parser.add_argument('--hidden_units', type=int, default=1000,help='hyperparameter of hidden layer')
    parser.add_argument('--gpu', dest='gpu', action='store_true', default = False)
    parser.add_argument('--epochs',type=int, default = 3, help='the number of epochs that model learn by')
    return parser.parse_args()
                        
def data_loaders(data_dir):
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # TODO: Define your transforms for the training, validation, and testing sets

    train_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomRotation(30),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

    #Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)
    #Using the image datasets and the trainforms, define the dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size = 32, shuffle= True)
    valid_dataloader = torch.utils.data.DataLoader(valid_datasets, batch_size = 32, shuffle= True)
    test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size = 32, shuffle= True)
    
    return train_dataloader, valid_dataloader, test_dataloader, train_datasets.class_to_idx
                        
def build_model(arch, hidden_units):
    if arch == 'vgg13':
        model = models.vgg19(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([
                          ('Dropout1', nn.Dropout(p=.5)),
                          ('fc1', nn.Linear(25088, hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('Dropout2', nn.Dropout(p=.5)),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        model.classifier = classifier 
        return model
    
    else:
        arch == 'vgg19'
        model = models.vgg19(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([
                          ('Dropout1', nn.Dropout(p=.5)),
                          ('fc1', nn.Linear(25088, hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('Dropout2', nn.Dropout(p=.5)),
                          ('fc2', nn.Linear(hidden_units,102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        model.classifier = classifier
        return model
    
def validation(model, test_dataloader, criterion, device):
    test_loss = 0
    accuracy = 0
    with torch.no_grad():
        for images, labels in iter(test_dataloader):

            images, labels = images.to(device), labels.to(device)        

            output = model.forward(images)
            test_loss += criterion(output, labels).item()

            ps = torch.exp(output)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()
    
    
    return test_loss, accuracy

def training(model, train_dataloader, valid_dataloader, test_dataloader, l_rate, gpu, epochs):
    if gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = 'cpu'
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=l_rate)
    model.to(device)
       
    steps = 0
    running_loss = 0
    print_every = 200
    for e in range(epochs):
        model.train()
        for images, labels in iter(train_dataloader):
            steps += 1
            images, labels = images.to(device), labels.to(device)        
        
            optimizer.zero_grad()
        
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval()
            
                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    test_loss, accuracy = validation(model, valid_dataloader, criterion, device)
                
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(test_dataloader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(test_dataloader)))
            
                running_loss = 0
            
            # Make sure training is back on
                model.train()
    print("the model is succeccfuly trained")
    
    print("the final test :")
    testing(model, test_dataloader,criterion, device)
    
    return model, optimizer

def testing(model, test_dataloader, criterion, device):
    model.eval()            
    with torch.no_grad():
        test_loss, accuracy = validation(model, test_dataloader, criterion, device)            
        print("Test Loss: {:.3f}.. ".format(test_loss/len(test_dataloader)),
              "Test Accuracy: {:.3f}".format(accuracy/len(test_dataloader)))

def save_checkpoint(checkpoint, path):
    #creating and checking the dirctory

    # Create target Directory if don't exist
    if path == '':
        torch.save(checkpoint, 'checkpoint.pth')
    elif not os.path.exists(path):
        os.mkdir(path)
        torch.save(checkpoint, path + '/checkpoint.pth')
    else:
        torch.save(checkpoint, path + '/checkpoint.pth')
    print("checkpoint is cucceccfuly saved")
    
if __name__ == "__main__":
    main()
