from __future__ import print_function, division
import warnings
warnings.filterwarnings("ignore")
import os.path
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os
import argparse
import cv2
class RaceClassifier:
    def __init__(self,model_path = 'blacbox/pretrained_models/checkpoint.pt'):
        ## Loading the Resnet34 pretrained model
        self.model = torchvision.models.resnet34(pretrained=True)
        # Selecting the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Output has lots of features like the age, race which sum upto 18
        self.model.fc = nn.Linear(self.model.fc.in_features, 18)
        #model_fair_7.load_state_dict(torch.load('fair_face_models/fairface_alldata_20191111.pt'))

        # Loading the pretrained model trained on the Fairface dataset
        self.model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
        self.model = self.model.to(self.device)

        # Setting eval mode to not compute gradients
        self.model.eval()

    def preprocess_image(self, image):
        transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = transform(image)
        image = image.unsqueeze(0)
        return image

    
    # Predict function used to predict the race of the face detected
    def predict(self,image):
        self.image = image.to(self.device)
        self.image = self.image.requires_grad_()
        
        # fair
        self.outputs = self.model(self.image)
        self.grad_outputs = self.outputs
        self.outputs = self.outputs.cpu().detach().numpy()
        #self.outputs = np.squeeze(self.outputs)

        '''
        self.race_outputs = self.outputs[:7]
        self.gender_outputs = self.outputs[7:9]
        self.age_outputs = self.outputs[9:18]
        race_score = np.exp(self.race_outputs) / np.sum(np.exp(self.race_outputs))
        gender_score = np.exp(self.gender_outputs) / np.sum(np.exp(self.gender_outputs))
        age_score = np.exp(self.age_outputs) / np.sum(np.exp(self.age_outputs))
        race_pred = np.argmax(race_score, dim = 1)
        gender_pred = np.argmax(gender_score, dim = 1)
        age_pred = np.argmax(age_score)
        return (race_pred,gender_pred,age_pred)
        '''
    
    def __call__(self, image):
        preds = self.predict(image)
        return self.grad_outputs