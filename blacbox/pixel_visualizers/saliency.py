import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from blacbox.utils.load_image import load_image
import torch.nn.functional as F
import torch.nn as nn
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
        images = load_image(images, path)
        saliencies = self.retrieve_saliencies(images, class_idx)
        return np.array(saliencies)
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



    
        

    

        