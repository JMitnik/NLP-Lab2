# %%
import re
import random
import time
import math
import numpy as np
import nltk
import matplotlib.pyplot as plt
import sys
import os
from abc import ABC, abstractmethod
plt.style.use('default')

# %%
class MutePrint:

    def __init__(self):
        self.stdout_restore = sys.stdout
    # Disable

    def blockPrint(self):
        sys.stdout = open(os.devnull, 'w')

    # Restore
    def enablePrint(self):
        sys.stdout = self.stdout_restore
mute = MutePrint()
# %%
!wget http:
    //nlp.stanford.edu / sentiment / trainDevTestTrees_PTB.zip - O trainDevTestTrees_PTB.zip
!unzip trainDevTestTrees_PTB.zip

# %%
from os.path import exists
from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())
cuda_output = !ldconfig - p | grep cudart.so | sed - e 's/.*\.\([0-9]*\)\.\([0-9]*\)$/cu\1\2/'
accelerator = cuda_output[0] if exists('/dev/nvidia0') else 'cpu'

!pip install - q http:
    //download.pytorch.org / whl / {accelerator} / torch - 1.0.0 - {platform} - linux_x86_64.whl torchvision

# %%
from google.colab import drive
drive.mount('/gdrive')
!cp "/gdrive/My Drive/glove.840B.300d.sst.txt" .

# %%
!wget - q https:
    //github.com / JMitnik / NLP - Lab2 / raw / cg / main.py - O . / main.py

# %%
mute.blockPrint()
from main import *
mute.enablePrint()

# %%
class Experiment(ABC):

    @abstractmethod
    def __init__(self, *args, **xargs):
        self.args = args
        self.xargs = xargs
        self.model = args[0]
    @abstractmethod
    def train(self):
        path = "{}.pt".format(xargs['exp_name'] if 'exp_name' in xargs or self.model.__class__.__name__)
        if os.path.exists(path):
            ckpt= torch.load(path)
            self.model.load_state_dict(ckpt["state_dict"])
            return
    @abstractmethod
    def eval(eval_fn=None):
        if eval_fn is None:
            if not ('eval_fn' in xargs):
                eval_fn = simple_evaluate
            else:
                eval_fn = xargs['eval_fn']
    @abstractmethod
    def get_accuracy():
        pass
    @abstractmethod
    def get_losses():
        pass
    results = do_train(tree_model)
    acc, loss = results
    plt.plot(acc)

# %%
plt.plot(loss)
