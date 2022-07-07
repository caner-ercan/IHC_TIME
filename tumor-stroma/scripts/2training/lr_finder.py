#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastai.vision.all import *


# In[2]:


import fastai
print(fastai.__version__)


# from fastai.vision.all import *Below you will find the exact imports for everything we use today

# In[3]:


from fastcore.xtras import Path

from fastai.callback.hook import summary
from fastai.callback.progress import ProgressCallback
from fastai.callback.schedule import lr_find, fit_flat_cos

from fastai.data.block import DataBlock
from fastai.data.external import untar_data, URLs
from fastai.data.transforms import get_image_files, FuncSplitter, Normalize

from fastai.layers import Mish
from fastai.losses import BaseLoss
from fastai.optimizer import ranger

from fastai.torch_core import tensor

from fastai.vision.augment import *
from fastai.vision.core import PILImage, PILMask
from fastai.vision.data import ImageBlock, MaskBlock, imagenet_stats
from fastai.vision.learner import unet_learner

from PIL import Image
import numpy as np

from torch import nn
from torchvision.models.resnet import resnet34

import torch
import torch.nn.functional as F

from fastai.data.all import *
from fastai.vision.core import *
from fastai.vision.data import *


# In[4]:


import torch
from fastai.vision import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fastai.callback.tensorboard import *
#from fastai.callbacks import CSVLogger
#from fastai.callbacks.tensorboard import LearnerTensorboardWriter
#from tensorboardX import SummaryWriter
import pandas as pd
import torch
from fastai.vision import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#from fastai.callbacks import CSVLogger
import sys

from os.path import join as opj


# In[5]:


#get_ipython().run_line_magic('matplotlib', 'inline')
# BS = sys.argv[1]
bs = sys.argv[1]
# In[6]:

projectDir = '/scicore/home/pissal00/ercan0000/220310_training'
projectPath = Path(projectDir)

path  = projectPath/'GT'

def clear_pyplot_memory():
    plt.clf()
    plt.cla()
    plt.close()
    
# Our validation set is inside a text document called `valid.txt` and split by new lines. Let's read it in:
valid_fnames = (path/'valid.txt').read_text().split('\n')
# Let's look at an image and see how everything aligns up



path_im = path/'images'
path_lbl = path/'labels'




fnames = get_image_files(path_im)
lbl_names = get_image_files(path_lbl)




img_fn = fnames[1]
img = PILImage.create(img_fn)
#img.show(figsize=(5,5))


# Now let's grab our y's. They live in the `labels` folder and are denoted by a `_P`



get_msk = lambda o: path_lbl/f'{o.stem}-labelled.png'


# The stem and suffix grab everything before and after the period respectively.

# Our masks are of type `PILMask` and we will make our gradient percentage (alpha) equal to 1 as we are not overlaying this on anything yet



msk = PILMask.create(get_msk(img_fn))
#msk.show(figsize=(5,5), alpha=0.5)



codes = np.loadtxt(path/'codes.txt', dtype=str)#; codes


# We need a split function that will split from our list of valid filenames we grabbed earlier. Let's try making our own.


def FileSplitter(fname):
    "Split `items` depending on the value of `mask`."
    valid = Path(fname).read_text().split('\n') 
    def _func(x): return x.name in valid
    def _inner(o, **kwargs): return FuncSplitter(_func)(o)
    return _inner


# This takes in our filenames, and checks for all of our filenames in all of our items in our validation filenames

xtra_tfms = (Brightness(p=0.75), Contrast(p=0.75),Saturation(max_lighting=0.2, p=0.75))
 

stroma = DataBlock(blocks=(ImageBlock, MaskBlock(codes)),
                get_items=get_image_files,
                splitter=FileSplitter(path/'valid.txt'),
                #splitter=TrainTestSplitter(test_size=0.2),
                get_y=get_msk,
                #item_tfms=aug_transforms(do_flip=True, flip_vert=True),
                #item_tfms=[SegmentationAlbumentationsTransform(transformPipeline)],
                batch_tfms=[*aug_transforms(do_flip=True, flip_vert=True, max_lighting=0.2, max_warp=0.2, p_affine=0.25, p_lighting=0.75, xtra_tfms=xtra_tfms), Normalize.from_stats(*imagenet_stats)])    



dls = stroma.dataloaders(path/'images', bs=eval(bs))

dls.vocab = codes


# Now we need a methodology for grabbing that particular code from our output of numbers. Let's make everything into a dictionary


name2id = {v:k for k,v in enumerate(codes)}


name2id




void_code = name2id['Void']
bkg_dbrs_code = name2id['Debris_Background']


# For segmentation, we want to squeeze all the outputted values to have it as a matrix of digits for our segmentation mask. From there, we want to match their argmax to the target's mask for each pixel and take the average

# In[25]:


def Accuracy_Overall(inp, targ, void_idx=void_code, axis=1):
    "Computes non-background accuracy for multiclass segmentation"
    #https://github.com/fastai/fastai/blob/aeb5317d11d986420fde85fc9749aac1f53eefd6/fastai/metrics.py#L323
    targ = cast(targ.squeeze(1), TensorBase)
    mask = targ != void_idx
    return (inp.argmax(dim=axis)[mask]==targ[mask]).float().mean()

def Accuracy_nonBackground(inp, targ, void_idx=void_code, bkg_dbrs_idx=bkg_dbrs_code, axis=1):
    "Computes non-background accuracy for multiclass segmentation"
    #https://github.com/fastai/fastai/blob/aeb5317d11d986420fde85fc9749aac1f53eefd6/fastai/metrics.py#L323
    targ = cast(targ.squeeze(1), TensorBase)
    mask = targ != void_idx & bkg_dbrs_idx 
    return (inp.argmax(dim=axis)[mask]==targ[mask]).float().mean()


'''
def segment_acc (input, target):
    target =target.squeeze(1)
    return (input.argmax(dim=1)==target).float().mean()'''


# ## The Dynamic Unet

# ![](http://tuatini.me/content/images/2017/09/u-net-architecture.png)

# [Source](https://r.search.yahoo.com/_ylt=AwrExdqB0utfdJIAiU.jzbkF;_ylu=c2VjA2ZwLWF0dHJpYgRzbGsDcnVybA--/RV=2/RE=1609319169/RO=11/RU=https%3a%2f%2ftuatini.me%2fpractical-image-segmentation-with-unet%2f/RK=2/RS=ZXDe1xIW7NgnEcZwMcj9.YgKmG4-)

# U-Net allows us to look at pixel-wise representations of our images through sizing it down and then blowing it bck up into a high resolution image. The first part we call an "encoder" and the second a "decoder"
# 
# On the image, the authors of the UNET paper describe the arrows as "denotions of different operations"

# We have a special `unet_learner`. Something new is we can pass in some model configurations where we can declare a few things to customize it with!
# 
# * Blur/blur final: avoid checkerboard artifacts
# * Self attention: A self-attention layer
# * y_range: Last activations go through a sigmoid for rescaling
# * Last cross - Cross-connection with the direct model input
# * Bottle - Bottlenck or not on that cross
# * Activation function
# * Norm type

# Let's make a `unet_learner` that uses some of the new state of the art techniques. Specifically:
# 
# * Self-attention layers: `self_attention = True`
# * Mish activation function: `act_cls = Mish`

# Along with this we will use the `Ranger` as optimizer function.

# In[26]:


opt = ranger


'''class CombinedLoss:
    "Dice and Focal combined"
    def __init__(self, axis=1, smooth=1., alpha=1.):
        store_attr()
        self.focal_loss = FocalLossFlat(axis=axis)
        self.dice_loss =  DiceLoss(axis, smooth)
        
    def __call__(self, pred, targ):
        return self.focal_loss(pred, targ) + self.alpha * self.dice_loss(pred, targ)
    
    def decodes(self, x):    return x.argmax(dim=self.axis)
    def activation(self, x): return F.softmax(x, dim=self.axis)
'''
class IOU(AvgMetric):
    "Intersection over Union Metric"
    def __init__(self, class_index, class_label, axis): store_attr('axis,class_index,class_label')
    def accumulate(self, learn):
        pred, targ = learn.pred.argmax(dim=self.axis), learn.y
        intersec = ((pred == targ) & (targ == self.class_index)).sum().item()
        union = (((pred == self.class_index) | (targ == self.class_index))).sum().item()
        if union: self.total += intersec
        self.count += union
  
    @property
    def name(self): return self.class_label

metrics = [
    #MIOU(3, axis=1), 
    Accuracy_Overall,Accuracy_nonBackground,DiceMulti(axis=1), JaccardCoeff(axis=1)]

for x in range(1,4): metrics.append(IOU(x, codes[x], axis=1))
# In[32]:


learn = unet_learner(dls,
                     resnet34,
                     metrics=metrics,
                     self_attention=True,
                     act_cls=Mish,
                     loss_func=CrossEntropyLossFlat(axis=1),
                     wd=1e-1,
                     opt_func=opt).to_fp16()


# In[30]:


learn.path=projectPath/'output'



# If we do a `learn.summary` we can see this blow-up trend, and see that our model came in frozen. Let's find a learning rate

# In[33]:


'''
# In[ ]:


#fname="unet-after-unfreeze-WD-2-best"
#https://github.com/WaterKnight1998/Deep-Tumour-Spheroid/blob/develop/notebooks/UNet.ipynb

callbacksFitAfterUnfreeze=[CSVLogger,ShowGraphCallback(),
    EarlyStoppingCallback(monitor='valid_loss', min_delta=0.1, patience=3),
    SaveModelCallback(monitor='valid_loss', comp=np.less, min_delta=0.1, fname=fname, every_epoch=False)],


# In[40]:


max_lr = 1e-2
wd = 1e-2
# 1cycle policy
learn.fit_one_cycle(cyc_len=8, max_lr=max_lr, wd=wd)
'''

# In[ ]:

import time
timeStamp=time.strftime("%Y%m%d_%H%M")

outputDir = opj(projectDir,'output')
modelName = "lrfind_bs{}_{}".format(bs,timeStamp)

output = opj(outputDir,'reports',modelName+'.txt')
Path(output).touch()
orig_stdout = sys.stdout
f = open(output, 'w')
sys.stdout = f
print('Current time: '+timeStamp )
print(learn.summary())

learn.lr_find(show_plot=True)
fig_path = opj(outputDir,'plots',modelName+'.png')

plt.savefig(fig_path)
clear_pyplot_memory()

sys.stdout = orig_stdout
f.close()

savepath = opj(outputDir,'models',modelName)
learn.save(savepath)




'''
lr = 1e-3
# With our new optimizer, we will also want to use a different fit function, called `fit_flat_cos`

# In[42]:


learn.fit_one_cycle(1
                    #,cbs=cbs
                   )


# In[ ]:
'''



