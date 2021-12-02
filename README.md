# blacbox

Making CNNs interpretable. Well because accuracy is not the best anymore

## Summary of Contents
<details open="open">
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li><a href="#new">New!</a></li>
    <li><a href="#coming-up">Coming up</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#examples">Examples</a></li>
    <li><a href="#watch-a-demo">Watch a demo</a>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
   
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project
Consider a scenario where a military commander wants to identify enemy tanks and he uses an object-detetection algorithm. So based on some data he reports an accuracy of 99%. Naturally he would be happy(enemies definetely wouldn't). But when he puts this algorithm to use he fails terribly. Turns out the data used had images of tanks only at night but not during day so instead of learning about the tank the algorithm learned to distinguish day from night. He would have lost but thanks to blacbox he uses our techniques to save his day! 


### Built With
* [OpenCV](https://opencv.org/)
* [PyTorch](https://pytorch.org/)
* [Numpy, Pandas](https://pandas.pydata.org/)

## New!
- The project is fresh so the whole thing is new right now:p

## Coming up
- Improved version of Grad-CAM (Score-CAM, Grad-CAM++)
- A relatively new method just for object-detection algorithms. The DRISE method! (Paper: https://arxiv.org/abs/2006.03204)
- Added support for Image-Captioning and VQA models

<!-- GETTING STARTED -->
## Getting Started
1. Install the latest version from github
```
pip install git+https://github.com/lazyCodes7/blacbox.git
```
2. Install from testPyPi(Working towards a stable version to release on PyPi)
```
pip install -i https://test.pypi.org/simple/ blacbox==0.1.0
```
## Some example usages.
### 1. Saliency Maps
A saliency map is a way to measure the spatial support of a particular class in each image. It is the oldest and most frequently used explanation method for interpreting the predictions of convolutional neural networks. The saliency map is built using gradients of the output over the input.
Paper link: https://arxiv.org/abs/1911.11293

```python

from blacbox import Saliency
import matplotlib.pyplot as plt

# Load a model
model = models.resnet50(pretrained = True)

# Pass the model to Saliency generator
maps = Saliency(
  model, 
  device = 'cuda'
)

# Images.shape = (B,C,H,W), class_idx = used to specify the class to calculate gradients against
# saliencie return_type = np.ndarray(B,H,W,C)
saliencie = maps.reveal(
  images = images, 
  class_idx = "keepmax"
)

```
### Input image




