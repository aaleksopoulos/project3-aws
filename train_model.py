#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse

from torchvision.datasets import ImageFolder
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
import logging, sys
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

#TODO: Import dependencies for Debugging andd Profiling
import smdebug.pytorch as smd
from smdebug import modes
from smdebug.pytorch import get_hook


def test(model, test_loader, criterion, hook, device):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    
    logger.info('testing model')
    model.eval()        
    hook.set_mode(modes.EVAL)
    test_loss = 0      
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()* data.size(0)
            pred = output.max(1, keepdim = True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()


        test_loss = test_loss /  len(test_loader.dataset)
        logger.info("Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                    test_loss, 
                    correct, 
                    len(test_loader.dataset), 
                    100.0 * correct / len(test_loader.dataset)
        ))
    

def train(model, train_loader, validation_loader, epochs, criterion, optimizer, hook, device):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    for epoch in range(1, epochs+1):
        
        logger.info('training model, epoch:{}'.format(epoch))
        
        hook.set_mode(smd.modes.TRAIN)
        model.train()
        running_loss = 0
        train_correct = 0 
        
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            pred = model(data)
            loss = criterion(pred, target)
            running_loss += loss.item() 
            loss.backward()
            optimizer.step()
            pred = pred.argmax(dim = 1, keepdim= True)
            train_correct += pred.eq(target.view_as(pred)).sum().item()
            
       
        logger.info('validating model')
        hook.set_mode(smd.modes.EVAL)
        model.eval()
        validate_loss = 0
        val_correct = 0
        with torch.no_grad():
            for data, target in validation_loader:
                data = data.to(device)
                target = target.to(device)
                pred = model(data)
                loss = criterion(pred, target)
                validate_loss += loss
                pred = pred.argmax(1, keepdim = True)
                val_correct += pred.eq(target.view_as(pred)).sum().item()



        logger.info("Epoch : {} training_loss : {} training accuracy : {}% validating_loss : {} validating accuracy : {}%".format(
            epoch,
            running_loss / len(train_loader.dataset),
            (100 * (train_correct / len(train_loader.dataset))),
            validate_loss / len(validation_loader.dataset),
            (100 * (val_correct / len(validation_loader.dataset)))

        ))
        
    return model
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False   
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 254),
        nn.ReLU(),
        nn.Linear(254, 133))
    
    return model

def create_data_loaders(train_data_path, test_data_path, validation_data_path, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    train_transform = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.RandomHorizontalFlip(p=0.1),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor()])
                                                            
    test_transform = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor()])

    logger.info("getting train data from {}".format(train_data_path))
    train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=train_transform)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    logger.info("getting test data from {}".format(test_data_path))
    test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=test_transform)
    test_data_loader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    logger.info("getting validation data from {}".format(validation_data_path))
    validation_data = torchvision.datasets.ImageFolder(root=validation_data_path, transform=test_transform)
    validation_data_loader  = torch.utils.data.DataLoader(validation_data, batch_size=batch_size) 
    
    return train_data_loader, test_data_loader, validation_data_loader

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    
    if args.device != 'cpu':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = 'cpu'

    model=model.to(device)

    logger.info("Running on device : {}".format(device))

    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr = args.lr)


    hook = smd.Hook.create_from_json_file()
    hook.register_module(model)
    hook.register_loss(loss_criterion)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    train_loader, test_loader, validation_loader=create_data_loaders(train_data_path=args.train,
                                                                     test_data_path=args.test,
                                                                     validation_data_path=args.validation,
                                                                     batch_size=args.batch_size)      
    model = train(model=model, 
                  train_loader=train_loader, 
                  validation_loader=validation_loader, 
                  epochs=args.epochs, 
                  criterion=loss_criterion, 
                  optimizer=optimizer, 
                  hook=hook, 
                  device=device)
   
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model=model,
         test_loader=test_loader,
         criterion=loss_criterion, 
         hook=hook, 
         device=device)
    
    '''
    TODO: Save the trained model
    '''
    logger.info("Saving model")
    path = os.path.join(args.model_dir, "model.pth")
    torch.save(model, path)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need
    '''
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="batch size for training - defaul = 16",
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default = 5,
        help="number of epochs - default = 5",
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=0.005, 
        help="learning rate - default = 0.005"
    )

    parser.add_argument(
        "--device", 
        type=str, 
        default="cpu", 
        help="device to run trainings - default = cpu"
    )
    
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test", type=str, default=os.environ["SM_CHANNEL_TEST"])
    parser.add_argument("--validation", type=str, default=os.environ["SM_CHANNEL_VALIDATION"])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    
    args=parser.parse_args()
    
    main(args)
