import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import cv2
from .gcam import GCAM
class GCAM_plus(GCAM):
    def __init__(self, model, interpolate = 'bilinear', device = 'cpu'):
        super().__init__(model, interpolate, device)

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
            output = F.softmax(output)
            if(class_idx == "keepmax"):
                class_idx = output.argmax(dim = 1)
            elif(class_idx == "keepmin"):
                class_idx = output.argmin(dim = 1)

            # Calculating gradients w.r.t to the idx selected
            cr = output[0,class_idx]
            cr.backward()
                

            # Retrieving gradients and activation maps
            fmaps = self.activations[key]
            weights = self.gradients[key]
            weights = weights/cr
            
            # Freeing up the gradients from the images
            self.image.grad.zero_()
            weights = weights.squeeze().unsqueeze(0)
            b, k, u, v = weights.size()
            a_num = weights.pow(2)
            a_dm = weights.pow(2).mul(2) + \
                fmaps.mul(weights.pow(3)).view(b, k, u*v).sum(-1, keepdim=True).view(b, k, 1, 1)
            a_dm = torch.where(a_dm != 0.0, a_dm, torch.ones_like(a_dm))
            alpha = a_num.div(a_dm)
            weights = (alpha*weights).view(b, k, u*v).sum(-1).view(b, k, 1, 1)

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