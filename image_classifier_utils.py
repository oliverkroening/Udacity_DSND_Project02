import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

def load_datasets(data_dir):
    '''
    Load training, test and validation datasets and create dataloaders

    INPUT:  data_dir - directory of datasets

    OUTPUT: train_loader - dataloader for training dataset
            test_loader - dataloader for test dataset
            valid_loader - dataloader for validation dataset
    '''
    # define directories
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define transforms for the training set
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                           transforms.RandomRotation(40),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    # Define transforms for the validation set
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    # Define your transforms for the testing set
    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_dataset  = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Using the image datasets and the transforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,batch_size=32, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_dataset,batch_size=32, shuffle=True)

    return train_loader, test_loader, valid_loader, train_dataset

def nn_create(network = "vgg11", dropout = 0.2, hidden_layer1 = 512, hidden_layer2 = 256, hidden_layer3 = 128, num_outputs = 102, lr = 0.001, gpu = "gpu"):
    '''
    This function creates a neural network by using the feauteres of a pretrained network,
    which are fed into a defined classifier. The network consists of three hidden layers and a variable amount of output classes.

    INPUT:  network - string of pretrained model (vggXX)
            dropout - dropout rate
            hidden_layer1 - nodes of first hidden layer
            hidden_layer2 - nodes of second hidden layer
            hidden_layer3 - nodes of third hidden layer
            lr - learning rate

    OUTPUT: model - classifier
            optimizer - optimizer of model parameters
            criterion - defined loss
    '''
    model_creatable = True

    if network == 'vgg11':
        network = models.vgg11(pretrained = True)
    elif network == 'vgg13':
        network = models.vgg13(pretrained = True)
    elif network == 'vgg16':
        network = models.vgg16(pretrained = True)
    elif network == 'vgg19':
        network = models.vgg19(pretrained = True)
    else:
        print("Please enter an existing VGG network (vgg11, vgg13, vgg16, vgg19)")
        model_creatable = False

    if model_creatable:
        # get in_features of model and define output classes
        num_features = network.classifier[0].in_features

         # turn off gradients, since they are not required
        for param in network.parameters():
            param.requires_grad = False

        # define classifier
        classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(num_features, hidden_layer1)),
                                                ('relu1', nn.ReLU()),
                                                ('fc2', nn.Linear(hidden_layer1, hidden_layer2)),
                                                ('relu2', nn.ReLU()),
                                                ('fc3', nn.Linear(hidden_layer2, hidden_layer3)),
                                                ('relu3', nn.ReLU()),
                                                ('fc4', nn.Linear(hidden_layer3, num_outputs)),
                                                ('dropout', nn.Dropout(dropout)),
                                                ('output', nn.LogSoftmax(dim=1))
                                               ]))

        # replace classifier
        network.classifier = classifier

        # move model to GPU/CPU
        if gpu == "gpu":
            network.cuda()

        # define optimizer and criterion
        optimizer = optim.Adam(network.classifier.parameters(), lr = lr)
        criterion = nn.NLLLoss()

        return network, optimizer, criterion

def train_model(model, optimizer, criterion, epochs, train_loader, valid_loader, gpu):
    '''
    train the deep learning model.

    INPUT:  model - deep learning model supposed to be pretrained
            optimizer - defined training optimizer
            criterion - definied training criterion
            epochs - amount of training epochs
            train_loader - loaded training set
            valid_loader - loaded validation set
            gpu - set to "gpu" if the model has to be moved to CUDA/GPU

    OUTPUT: model - trained model
    '''
    # define training parameters
    steps = 0
    print_every = 5
    running_loss = 0
    if gpu == "gpu":
        model.to('cuda')
    train_losses, valid_losses = [], []

    for epoch in range(epochs):
        for inputs, labels in train_loader:
            steps += 1

            # move inputs and labels tensors to default devices
            if gpu == "gpu":
                inputs,labels = inputs.to('cuda'), labels.to('cuda')

            # clear gradient
            optimizer.zero_grad()

            # Forward pass
            outputs = model.forward(inputs)

            # calculate loss
            loss = criterion(outputs, labels)

            # backward pass
            loss.backward()

            # perform optimization step
            optimizer.step()

            # cumulate loss
            running_loss += loss.item()

            # validation pass
            if steps % print_every == 0:
                model.eval()
                valid_loss = 0
                accuracy=0

                # turn off gradients, since no back propagation on validation pass required
                with torch.no_grad():
                    for inputs_valid,labels_valid in valid_loader:
                        # move everything to GPU
                        inputs_valid, labels_valid = inputs_valid.to('cuda:0') , labels_valid.to('cuda:0')
                        model.to('cuda:0')

                        # Forward pass
                        outputs_valid = model.forward(inputs_valid)

                        # calculate validation loss
                        valid_loss = criterion(outputs_valid,labels_valid)

                        # calculate accuracy by checking if the predicted classes match the labels
                        ps = torch.exp(outputs_valid)
                        top_p, top_class = ps.topk(1,dim=1)
                        equals = top_class == labels_valid.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                train_losses.append(running_loss/len(train_loader))
                valid_losses.append(valid_loss/len(valid_loader))

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Training Loss: {running_loss/len(train_loader):.3f}.. "
                      f"Validation loss: {valid_loss/len(valid_loader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(valid_loader):.3f}")
                running_loss = 0
                model.train()

    return model

def save_checkpoint(model, network, hidden_layer1, hidden_layer2, hidden_layer3, output_classes, dropout, lr, filepath, train_dataset):
    '''
    save the checkpoint of a deep learning model

    INPUT:  model - deep learning model
            network - structure of deep learning model
            hidden_layer1 - nodes in first hidden layer
            hidden_layer2 - nodes in second hidden layer
            hidden_layer3 - nodes in third hidden layer
            dropout - dropout rate
            lr - learning rate
            filepath - destination of the checkpoint
            train_dataset - training dataset
    '''
    model.class_to_idx = train_dataset.class_to_idx
    model.cpu
    torch.save({'network' : network,
                'hidden_layer1' : hidden_layer1,
                'hidden_layer2' : hidden_layer2,
                'hidden_layer3' : hidden_layer3,
                'outputs' : output_classes,
                'dropout' : dropout,
                'learning_rate' : lr,
                'state_dict' : model.state_dict(),
                'class_to_idx':model.class_to_idx},
                filepath)

def load_checkpoint(filepath, gpu):
    '''
    Load previously saved checkpoint

    INPUT:  filepath - path of checkpoint file

    OUTPUT: loaded_model - model created from loaded checkpoint data
    '''

    # load checkpoint data (use CPU)
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)

    # read out model properties
    network = checkpoint['network']
    hidden_layer1 = checkpoint['hidden_layer1']
    hidden_layer2 = checkpoint['hidden_layer2']
    hidden_layer3 = checkpoint['hidden_layer3']
    output_classes = checkpoint['outputs']
    dropout = checkpoint['dropout']
    lr = checkpoint['learning_rate']

    # create model from properties using nn_create and pass state_dict
    loaded_model,_,_ = nn_create(network, dropout, hidden_layer1, hidden_layer2, hidden_layer3, output_classes, lr, gpu)
    loaded_model.class_to_idx = checkpoint['class_to_idx']
    loaded_model.load_state_dict(checkpoint['state_dict'])

    return loaded_model

def calc_accuracy_test(model, test_loader, criterion):
    '''
    Function calculates the accuracy of the trained model according the test dataset loaded into the test_loader

    INPUT:  test_loader - test dataset loaded into the test_loader

    OUTPUT: test_accuracy - calculated accuracy
    '''
    # initialize accuracy
    accuracy = 0
    # move model to GPU
    model.to('cuda:0')

    # turn off gradients, since only forward pass is required
    with torch.no_grad():
        # loop images and labels of test dataset
        for images, labels in test_loader:
            # move everything to GPU
            images, labels = images.to('cuda:0') , labels.to('cuda:0')

            # Forward pass
            outputs = model(images)

            # calculate validation loss
            valid_loss = criterion(outputs,labels)

            # calculate accuracy by checking if the predicted classes match the labels
            ps = torch.exp(outputs)
            top_p, top_class = ps.topk(1,dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        print('Accuracy on Test dataset: {:.2f}%'.format(accuracy/len(test_loader)*100))

    return accuracy

def predict(image_path, model, top_k, gpu):
    '''
    Predict the class (or classes) of an image using a trained deep learning model.

        INPUT:  image_path - path to image file
                model - deep learning model defined by checkpoint file
                topk - top k classes of the prediction

        OUTPUT: probs_topk_list - list of probabilities of top k classes
                classes_topk_list - list of classes with the k highest probabilites
    '''
    # process image
    img = process_image(image_path)

    # load deep learning model and move to CPU
    model = load_checkpoint(model,gpu)

    if gpu == "gpu":
        model.to('cuda:0')
        img_tensor = torch.from_numpy(img).type(torch.cuda.FloatTensor)
    else:
        model.cpu()
        img_tensor = torch.from_numpy(img).type(torch.FloatTensor)

    img_tensor = img_tensor.unsqueeze_(0)

    # set model to evaluation mode
    model.eval()

    # turn off gradients - not required for predicting
    with torch.no_grad():
        # forward pass for predictions
        outputs = model.forward(img_tensor)

    # calculate output probabilities and get the top k classes with indices - save as lists
    probs = torch.exp(outputs)
    probs_topk = probs.topk(top_k)[0]
    idx_topk = probs.topk(top_k)[1]
    probs_topk_list = np.array(probs_topk)[0]
    idx_topk_list = np.array(idx_topk[0])

    # map indices to classes
    idx_to_class = {x: y for y, x in model.class_to_idx.items()}

    # create class list
    classes_topk_list = []
    for i in idx_topk_list:
        classes_topk_list += [idx_to_class[i]]

    return probs_topk_list, classes_topk_list

def process_image(image):
    '''
    Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array

        INPUT: image - path to image file

        OUTPUT: img_np - image as Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model

    # open image as PIL image
    img_pil = Image.open(image)

    # define transformations
    transform = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    # perform transformation
    img_tensor = transform(img_pil)

    # convert PyTorch Tensor to Numpy array and return array
    return img_tensor.numpy()
