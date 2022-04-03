import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from blacbox.utils.load_image import load_image
import torch.nn.functional as F
import torch.nn as nn
import cv2
class _BaseWrapper(object):
    
    def __init__(self, model, device = 'cpu'):
        '''
            Description: BaseWrapper for BackProp and Guided BackProp

            Args:

                model -> nn.Module
                device -> cpu/cuda
        '''
        super(_BaseWrapper, self).__init__()
        self.device = device
        self.model = model.to(self.device)
        self.handlers = []  # a set of hook function handlers
    
    def _encode_one_hot(self, ids):
        '''
            Description: One hot encoder for the model outputs

            Args: 

                ids -> int (id to encode)
        '''
        one_hot = torch.zeros_like(self.logits).to(self.device)
        one_hot[0, ids] = 1.0
        #print(one_hot)
        return one_hot
    def forward(self, image):
        '''
            Description: Method for feed-forward process

            Args:
                image -> torch.Tensor(image to be processed through the neural network)
        '''
        self.image_shape = image.shape[2:]
        self.logits = self.model(image)
        self.probs = F.softmax(self.logits, dim=1)
        return self.probs.sort(dim=1, descending=True)

    def backward(self, ids):
        '''
            Description: Backward prop for the neural network

            Args:
                ids -> int(id to calculate the gradient for)

        '''
        one_hot = self._encode_one_hot(ids)
        self.model.zero_grad()
        self.logits.backward(gradient=one_hot, retain_graph=True)

    def get_gradients(self):
        raise NotImplementedError

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()

class BackPropagation(_BaseWrapper):
    def forward(self, image):
        '''
            Description: Method for feed-forward process

            Args:
                image -> torch.Tensor(image to be processed through the neural network)
        '''
        self.image = image.requires_grad_()
        return super(BackPropagation, self).forward(self.image)

    def get_gradients(self):
        """
        Generating the gradients for the image
        """
        gradient = self.image.grad.clone()
        self.image.grad.zero_()
        return gradient

class GuidedBackPropagation(BackPropagation):
    """
    "Striving for Simplicity: the All Convolutional Net"
    https://arxiv.org/pdf/1412.6806.pdf
    Look at Figure 1 on page 8.
    """
    def backward_hook(self,module, grad_in, grad_out):   
            """
            PyTorch hook function for zeroing out the negative gradients during backprop
            """
            # Cut off negative gradients
            if isinstance(module, nn.ReLU):
                #print(module)
                grad_in[0][grad_in[0] > 0] = 1
                return (F.relu(grad_in[0]),)

    def __init__(self, model, device):
        '''
            Description: GBP module for CNN visualization

            Args:

                model -> nn.Module
                device -> cpu/cuda

            Example:
                gbp = GuidedBackPropagation(model = model, device = 'cuda')
        '''
        super(GuidedBackPropagation, self).__init__(model, device)
        

    def reveal(self, images = None, path = None):
        '''
            Description: function for forwarding the images and getting the gradients to visualize

            Args:
                images -> type = torch.Tensor, shape = (B,C,H,W)
                path -> str(required if images is None)

            Example:
                gbp = GuidedBackPropagation(model = model, device = 'cuda')
                gradients = gbp.reveal(images = images)
        '''  
        for module in self.model.named_modules():
            self.handlers.append(module[1].register_backward_hook(self.backward_hook))
        
        images = load_image(images, path)
        grads = self.retrieve_gradients(images)
        return np.array(grads)

    def retrieve_gradients(self, images):
        """
            Generating the gradients
        """
        grads = []
        for image in images:
            image = image.unsqueeze(0).to(self.device)
            ids  = self.forward(image)
            self.backward(ids=ids.indices[0,0])
            gradient = self.get_gradients()
            gradient = gradient.cpu().squeeze().permute(1,2,0).detach().numpy()
            gradient -= gradient.min()
            gradient /= gradient.max()
            gradient *= 255.0
            gradient = gradient.astype(int)
            grads.append(gradient)

        self.remove_hook()
        self.handlers = []
        return grads




