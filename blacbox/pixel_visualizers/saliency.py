import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
import logging
import numpy
class Saliency:
    def __init__(
        self, 
        model, 
        device = 'cpu'):
        '''
            Description:
            1. Pass the model to the saliency visualizer
            2. set the device to CPU/GPU

            Args:
            model -> nn.Module
            device -> cpu/cuda
        '''
        self.device = device
        self.model = model.to(self.device)

    def reveal(
        self,
        images = None,
        path = None,
        class_idx = 'keepmax'):
        '''
            Description:
            The class takes in images of type tensor or a path and returns the saliency for the same

            Args:
            images -> type = torch.Tensor, shape = (B,C,H,W)
            path -> str(required if images is None)
            class_idx -> int(class to visualize)/str
            

            Types of class_idx:
            'keepmax' -> Visualize the max score from classification
            'keepmin' -> Visualize the min score from classification
            '0-len(output)' -> For visualizing a certain output

        '''

        # Raise error if both path and images are provided
        if(path!=None and images!=None):
            raise ValueError("Image batches cannot be passed when path is provided")

        # If path is provided
        elif(path!=None):
            images = cv2.imread(path)
            images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
            images = self.preprocess_image(images)

        # If batches of image is provided
        if(images!=None):
            saliencies = self.retrieve_saliencies(images, class_idx)
            return np.array(saliencies)

        # If None then raise errors
        else:
            raise AttributeError("Either path or images need to be provided to reveal Saliency visualization.")
    
    def preprocess_image(self, image):
        '''
            Description:
            Takes in an image and applies some transformations to it

            Args:
            image -> np.ndarray

        '''
        if(isinstance(image, numpy.ndarray)):
            transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            image = transform(image)
            image = image.unsqueeze(0)
            return image

        else:
            raise ValueError("Preprocessing requires type np.ndarray")

    def retrieve_saliencies(self, images, class_idx):
        saliencies = []
        for image in images:
            self.image = image.unsqueeze(0).to(self.device)
            self.image.requires_grad = True
            ''' 
            Calculating the gradients and computing the 
            max across each channel and returning the saliency map
            '''
            self.output = self.model(self.image)
            if(isinstance(class_idx, str)):
                if(class_idx == 'keepmax'):
                    class_idx = self.output.argmax(dim = 1)

                elif(class_idx == 'keepmin'):
                    class_idx = self.output.argmin(dim = 1)

                else:
                    error = "class_idx only supports two str arguments\n.\
                    1. keepmax\n \
                    2. keepmin\n \
                    "
                    raise ValueError(error)
            self.output[0, class_idx].backward()
            saliency, _ = torch.max(self.image.grad.data.abs(), dim=1)
            saliency = saliency.reshape(224,224,1)
            saliency = saliency.cpu().numpy()
            saliencies.append(saliency)
        return saliencies



    
        

    

        