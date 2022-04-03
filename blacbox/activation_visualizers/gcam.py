import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import cv2
class GCAM:
    def __init__(self, model, interpolate = 'bilinear', device = 'cpu'):
        '''
            Description:  Method to visualize CNN layers w.r.t output
            
            Args:
                model -> nn.Module
                interpolate -> interpolation types (refer to https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html)
                device -> 'cpu'/'cuda' (device to set model to)

            Example:
                gcam = GCAM(model, interpolate = 'bilinear', device = 'cuda')

        '''
        self.device = device
        self.model = model.to(self.device)
        self.interpolate = interpolate
        self.activations = {}
        self.gradients = {}
        self.colormap_dict = {
            "autumn": cv2.COLORMAP_AUTUMN,
            "bone": cv2.COLORMAP_BONE,
            "jet" : cv2.COLORMAP_JET,
            "winter" : cv2.COLORMAP_WINTER,
            "rainbow" : cv2.COLORMAP_RAINBOW,
            "ocean" : cv2.COLORMAP_OCEAN,
            "summer" : cv2.COLORMAP_SUMMER,
            "spring" : cv2.COLORMAP_SPRING,
            "cool" : cv2.COLORMAP_COOL,
            "hsv" : cv2.COLORMAP_HSV,
            "pink" : cv2.COLORMAP_PINK,
            "hot" : cv2.COLORMAP_HOT

        }

    def get_activations(self,key):
        '''
            Description: Method to get activations for a particular layer

            Args: 
                key -> nn.Module(the module to calculate activations for)

        '''
        def hook(module, input, out):
            self.activations[key] = out.detach()
        return hook
    def get_gradients(self,key):
        '''
            Description: Method to get gradients for a particular layer

            Args: 
                key -> nn.Module(the module to calculate gradients for)

        '''
        def hook(module, grad_in, grad_out):
            self.gradients[key] = grad_out[0]
        return hook
    def reveal(
        self, 
        images = None, 
        module = None, 
        class_idx = 0, 
        path = None, 
        colormap = "jet"
        ):
        '''
            Description: Where it all takes place and we get the output visualize a CNN layer.

            Args:
                images -> type = torch.Tensor, shape = (B,C,H,W)
                path -> str(required if images is None)
                module -> nn.Module (to perform Grad-CAM on)
                class_idx -> int (class_idx to calculate gradients on)
                colormap -> colormap to apply on heatmap

            Types of colormaps:
                "autumn": cv2.COLORMAP_AUTUMN,
                "bone": cv2.COLORMAP_BONE,
                "jet" : cv2.COLORMAP_JET,
                "winter" : cv2.COLORMAP_WINTER,
                "rainbow" : cv2.COLORMAP_RAINBOW,
                "ocean" : cv2.COLORMAP_OCEAN,
                "summer" : cv2.COLORMAP_SUMMER,
                "spring" : cv2.COLORMAP_SPRING,
                "cool" : cv2.COLORMAP_COOL,
                "hsv" : cv2.COLORMAP_HSV,
                "pink" : cv2.COLORMAP_PINK,
                "hot" : cv2.COLORMAP_HOT

            Types of class_idx:
                'keepmax' -> Visualize the max score from classification
                'keepmin' -> Visualize the min score from classification
                '0-len(output)' -> For visualizing a certain output

            Example:
                from blacbox import GCAM
                gcam = GCAM(model, interpolate = True)
                output_viz = gcam.reveal(
                    images = images,
                    module = model.resnet.layer[0].conv1,
                    class_idx = 'keepmax',
                    colormap = 'hot'

                )
        '''
        # Raise error if both path and images are provided
        if(path!=None and images!=None):
            raise ValueError("Image batches cannot be passed when path is provided")

        # If path is provided
        elif(path!=None):
            images = cv2.imread(path)
            images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
            images = self.preprocess_image(images).unsqueeze(0)

        # If batches of image is provided
        if(images!=None):
            if(isinstance(images, torch.Tensor) == False):
                raise ValueError("reveal() expects images to be of type tensor with shape (B,C,H,W)")
            if(isinstance(module, nn.Module)):
                key = str(module)
                self.ac_handler = module.register_forward_hook(self.get_activations(key))
                self.grad_handler = module.register_backward_hook(self.get_gradients(key))
                gcams = self.retrieve_gcams(images, class_idx, key, colormap)
                return gcams
            else:
                error = "Module argument expects variable of type nn.Module"
                raise ValueError(error)
        
        # If None then raise errors
        else:
            raise ValueError("Either path or images need to be provided to reveal GCAM visualization.")

    @staticmethod
    def overlay(heatmaps, images, influence = 0.3, is_np = True):
        if(is_np):
            images = images.cpu().detach().permute(0,2,3,1).numpy()
            images = cv2.normalize(
                images, 
                None,
                alpha = 0,
                beta = 255,
                norm_type = cv2.NORM_MINMAX,
                dtype = cv2.CV_32F
            )
        superimposed_img = heatmaps*(influence) + images
        return superimposed_img.astype(int)

    def preprocess_image(self, images):
        '''
            Description:
            Takes in an images and applies some transformations to it

            Args:
            images -> np.ndarray

        '''
        if(isinstance(images, np.ndarray)):
            transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            images = transform(images)
            return images

        else:
            raise AttributeError("Preprocessing requires a np.ndarray")

    def retrieve_gcams(self, images, class_idx, key, colormap, apply_cmap = True):
        """
        Retrieving gcams from the images provided
        """
        gcams = []            
        # Selecting the index 
       
        for image in images:
            # Taking the images
            self.image = image.unsqueeze(0).to(self.device)
            self.image.requires_grad = True
            output = self.model(self.image)
            if(class_idx == "keepmax"):
                class_idx = output.argmax(dim = 1)
            elif(class_idx == "keepmin"):
                class_idx = output.argmin(dim = 1)

            # Calculating gradients w.r.t to the idx selected
            class_required = output[0,class_idx]
            class_required.backward()


            # Retrieving gradients and activation maps
            fmaps = self.activations[key]
            weights = self.gradients[key]

            # Freeing up the gradients from the images
            self.image.grad.zero_()


            # Avg pooling as in the paper
            weights = F.adaptive_avg_pool2d(weights, 1)

            # Retrieving the gcam outputs
            gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
            gcam = F.relu(gcam)
            gcam = F.interpolate(
                gcam, self.image.shape[2:4], mode=self.interpolate, align_corners=False
            )
            gcam = gcam.cpu().squeeze(0).detach().permute(1,2,0).numpy()
            gcam = cv2.normalize(
                gcam, 
                None,
                alpha = 0,
                beta = 255,
                norm_type = cv2.NORM_MINMAX,
                dtype = cv2.CV_8UC3
            )
            gcam = cv2.applyColorMap(gcam, self.colormap_dict[colormap])
            #gcam = (gcam*255).astype(np.uint8)
            gcams.append(gcam)
        
        # Removing the hooks, activations, gradients stored
        self.ac_handler.remove()
        self.grad_handler.remove()
        self.activations = {}
        self.gradients = {}
        return np.array(gcams)


