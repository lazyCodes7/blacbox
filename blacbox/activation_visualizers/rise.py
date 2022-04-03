import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from blacbox.utils.load_image import load_image
class RISE:
    def __init__(
        self,
        model,
        n_masks = 1000,
        p1=0.1,
        size=(224, 224),
        mask_size=(7, 7),
        batch_size=128
    ):
        self.model = model
        self.n_masks = n_masks
        self.p1 = p1
        self.size = size
        self.mask_size = mask_size
        self.batch_size = batch_size
        self.masks = self.generate()

    def load_masks(self, filepath):
        masks = torch.load(filepath)
        return masks

    def generate(self):
        # cell size in the upsampled mask
        H = int(self.size[0] / self.mask_size[0])
        W = int(self.size[1] / self.mask_size[1])

        resize_h = int((self.mask_size[0] + 1) * H)
        resize_w = int((self.mask_size[1] + 1) * W)

        masks = []

        for _ in range(self.n_masks):
            # generate binary mask
            binary_mask = torch.randn(
                1, 1, self.mask_size[0], self.mask_size[1])
            binary_mask = (binary_mask < self.p1).float()

            # upsampling mask
            mask = F.interpolate(
                binary_mask, (resize_h, resize_w), mode='bilinear', align_corners=False)

            # random cropping
            i = np.random.randint(0, H)
            j = np.random.randint(0, W)
            mask = mask[:, :, i:i+self.size[0], j:j+self.size[1]]

            masks.append(mask)

        masks = torch.cat(masks, dim=0)   # (N_masks, 1, H, W)

        return masks

    def reveal(        
        self, 
        images = None, 
        module = None, 
        class_idx = 0, 
        path = None
    ):
        images = load_image(images, path)
        saliencies = self.retrieve_saliencies(images)
        return saliencies

    def retrieve_saliencies(
        self, 
        x = None,
        class_idx = 'keepmax'
    ):

        # x: input image. (1, 3, H, W)
        device = x.device

        # keep probabilities of each class
        probs = []
        # shape (n_masks, 3, H, W)
        masked_x = torch.mul(self.masks, x.to('cpu').data)

        out = self.model(x)

        if(class_idx == 'keepmax'):
            class_idx = out.argmax()
        
        if(class_idx == 'argmin'):
            class_idx = out.argmin()
            
        for i in range(0, self.n_masks, self.batch_size):
            input = masked_x[i:min(i + self.batch_size, self.n_masks)].to(device)
            out = self.model(input)
            probs.append(torch.softmax(out, dim=1).to('cpu').data)

        probs = torch.cat(probs)    # shape => (n_masks, n_classes)
        n_classes = probs.shape[1]

        # caluculate saliency map using probability scores as weights
        saliency = torch.matmul(
            probs.data.transpose(0, 1),
            self.masks.view(self.n_masks, -1)
        )
        saliency = saliency.view(
            (n_classes, self.size[0], self.size[1]))
        saliency = saliency / (self.n_masks * self.p1)

        # normalize
        m, _ = torch.min(saliency.view(n_classes, -1), dim=1)
        saliency -= m.view(n_classes, 1, 1)
        M, _ = torch.max(saliency.view(n_classes, -1), dim=1)
        saliency /= M.view(n_classes, 1, 1)
        return saliency.data[class_idx]