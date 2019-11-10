import numpy as np
import torch
from torch import nn
from collections import OrderedDict
from PIL import Image
import argparse
import json
from torchvision import models


def main():
    
    #get the prefered options
    in_args = get_input_args()
    
    model = load_checkpoint(in_args.checkpoint)
    print("model is loaded")
    probs, classes = predict(in_args.image_path, model, in_args.top_k, in_args.gpu)
    print("image is succeccfuly predicted")
    with open(in_args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    class_names = [cat_to_name[i] for i in classes]
    
    print("Probabilities are", probs)
    print("Classes are", class_names)
    print("done")
   
def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object. 
     5 command line arguments are created:
       data_directory - the directory that has the data for training, validating, and testing images (defualt = "flower")
       gpu - the device that model training on (default = 'cpu')
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help= "Path to image", type=str)
    parser.add_argument("checkpoint", help= "path to checkpoint", default="checkpoint.pth", type=str)
    parser.add_argument("--top_k", help= "top k classes of image", default=5, type=int)
    parser.add_argument('--category_names', type=str, default='cat_to_name.json')
    parser.add_argument('--gpu', dest='gpu', action='store_true', default = False)
    
    return parser.parse_args()
                        
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    dropout = checkpoint['dropout']
    input_size = checkpoint['input_size']
    hidden_units = checkpoint['hidden_units']
    output_size = checkpoint['output_size']
    arch = 'vgg19'
    
    if arch == 'vgg19':
        model = models.vgg19(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        classifier = nn.Sequential(OrderedDict([
                                    ('fc1', nn.Linear(input_size, hidden_units)),
                                    ('relu', nn.ReLU()),
                                    ('dropout', nn.Dropout(dropout)),
                                    ('fc2', nn.Linear(hidden_units, output_size)),
                                    ('output', nn.LogSoftmax(dim=1))
        ]))
        model.classifier = classifier
        model.load_state_dict(checkpoint['model_state_dict'])
        model.class_to_idx = checkpoint['map_classes']
        return model
    else:
        model = models.vgg13(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        classifier = nn.Sequential(OrderedDict([
                                    ('fc1', nn.Linear(input_size, hidden_units)),
                                    ('relu', nn.ReLU()),
                                    ('dropout', nn.Dropout(dropout)),
                                    ('fc2', nn.Linear(hidden_units, output_size)),
                                    ('output', nn.LogSoftmax(dim=1))
        ]))
        model.classifier = classifier
        model.load_state_dict(checkpoint['model_state_dict'])
        model.class_to_idx = checkpoint['map_classes']
        return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    current_width, current_height = image.size
    if current_width < current_height :
        new_height = int(current_height * 256 / current_width)
        image = image.resize((256, new_height))
    else:
        new_width = int(current_width * 256 / current_height)
        image.resize((new_width, 256))
    
    precrop_width, precrop_height = image.size
    left = (precrop_width - 224)/2
    top = (precrop_height - 224)/2
    right = (precrop_width + 224)/2
    bottom = (precrop_height + 224)/2
    
    img_croped = image.crop((left, top, right, bottom))
    np_image = np.array(img_croped)
    np_image = np_image / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    np_image = ( np_image - mean )/std
    
    transposed_img = np_image.transpose((2,0,1))
    return transposed_img

def predict(image_path, model, topk, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if gpu :
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = 'cpu'
    
    model.to(device)

    img = Image.open(image_path)
    img = process_image(img)
    img = torch.from_numpy(img).unsqueeze_(0).type(torch.FloatTensor).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(img.float()) 

    probability = torch.exp(outputs)
    probs, indices = probability.topk(topk)
    probs = probs.cpu().numpy()[0]
    indices  = indices.cpu().numpy()[0]

    idx_to_class = {v:k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[x] for x in indices]
    return probs, classes
    
if __name__ == "__main__":
    main()
