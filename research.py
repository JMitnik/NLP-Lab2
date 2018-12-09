# %%
import re
import random
import time
import math
import numpy as np
import nltk
import matplotlib.pyplot as plt
import sys, os
plt.style.use('default')

# %%
!wget http://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip
!unzip trainDevTestTrees_PTB.zip

# %%
from os.path import exists
from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())
cuda_output = !ldconfig -p|grep cudart.so|sed -e 's/.*\.\([0-9]*\)\.\([0-9]*\)$/cu\1\2/'
accelerator = cuda_output[0] if exists('/dev/nvidia0') else 'cpu'

!pip install -q http://download.pytorch.org/whl/{accelerator}/torch-1.0.0-{platform}-linux_x86_64.whl torchvision

# %%
from google.colab import drive
drive.mount('/gdrive')
!cp "/gdrive/My Drive/glove.840B.300d.sst.txt" .

# %%
!wget https://github.com/JMitnik/NLP-Lab2/raw/cg/main.py

# %%
from main import *

# %%
results = do_train(tree_model)
acc, loss = results
plt.plot(acc)

# %%
plt.plot(loss)