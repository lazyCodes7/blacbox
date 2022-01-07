import unittest
import math
from blacbox import GCAM
from blacbox import Saliency
from blacbox import GuidedBackPropagation
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import torch
class Tester(unittest.TestCase):
    def setUp(self):
        image1 = cv2.imread('blacbox/architectures/images/black.png')
        image1= cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.imread('blacbox/architectures/images/dog.png')
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        image1 = self.preprocess_image(image1)
        image2 = self.preprocess_image(image2)
        self.images = torch.stack((image1, image2), axis = 0)
        self.model_test1 = models.resnet18(pretrained = True)
        self.model_test2 = models.resnet50(pretrained=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def test_gcam(self):
        gcam = GCAM(
            model = self.model_test2,
            interpolate = 'bilinear',
            device = self.device
        )
        heatmap = gcam.reveal(
            images = self.images, 
            module = self.model_test2.layer4[0].conv1, 
            class_idx = 'keepmax', 
            colormap = 'hot'
        )
        self.assertEqual(type(heatmap), np.ndarray)
    
    def test_gbp(self):
        gbp = GuidedBackPropagation(
            model = self.model_test1,
            device = self.device
        )
        generated_img = gbp.reveal(
            path = 'blacbox/architectures/images/black.png' , 
        )
        self.assertEqual(type(generated_img), np.ndarray)

    def test_saliency(self):
        sal = Saliency(
            model = self.model_test1,
            device=self.device
        )
        maps = sal.reveal(images=self.images, class_idx = 'keepmax')
        self.assertEqual(type(maps), np.ndarray)
    
    def test_out_size(self):
        sal = Saliency(
            self.model_test1
        )
        maps = sal.reveal(images=self.images, class_idx = 'keepmax')
        self.assertEqual(len(maps.shape), 4)
    


    def preprocess_image(self, image):
        transform = transforms.Compose([
                        
                        transforms.ToPILImage(),
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),

        ])

        image = transform(image)
        return image
    

if __name__ == '__main__':
    unittest.main()
