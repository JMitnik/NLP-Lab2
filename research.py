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
!pip install humanize
import psutil, humanize
def printm():
  process = psutil.Process(os.getpid())
  print("Gen RAM Free:" + humanize.naturalsize( psutil.virtual_memory().available ), "| Proc size:" + humanize.naturalsize( process.memory_info().rss))

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
!wget http://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip -O trainDevTestTrees_PTB.zip
!unzip trainDevTestTrees_PTB.zip

# %%
!pip install pytreebank
import pytreebank
# %%
from os.path import exists
from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())
cuda_output = !ldconfig - p | grep cudart.so | sed - e 's/.*\.\([0-9]*\)\.\([0-9]*\)$/cu\1\2/'
accelerator = cuda_output[0] if exists('/dev/nvidia0') else 'cpu'

!pip install -q http://download.pytorch.org/whl/{accelerator}/torch-1.0.0-{platform}-linux_x86_64.whl torchvision

# %%
from google.colab import drive
drive.mount('/gdrive')
!cp "/gdrive/My Drive/glove.840B.300d.sst.txt" .

# %%
!wget -q https://github.com/JMitnik/NLP-Lab2/raw/cg/main.py -O ./main.py
!wget -q https://github.com/JMitnik/NLP-Lab2/raw/cg/utils.py -O ./utils.py

# %%
mute.blockPrint()
from main import *
from utils import *
mute.enablePrint()

# %%
def get_subtree_dataset():
    '''
    extract all subtrees together to the exact form of the `train_data` used in last ipynb
    args: None
    returns: a list contains three list of Examples, each of them corresponds to one of 'train', 'test', 'dev' set
    '''
    dataset = pytreebank.load_sst("./trees")
    datasets = dataset.values()
    print(dataset.keys())
    results = []
    for D in datasets:
        result = []
        for tree in D:
            tree.lowercase()
            for c in tree.all_children():
                sc = str(c)
                trans = transitions_from_treestring(sc)
                label = c.label
                tree = c
                tokens = tokens_from_treestring(sc)
                result.append(Example(tokens=tokens, tree=tree, label=label, transitions=trans))
        results.append(result)
    return results

# %%
subtree_train_data, subtree_test_data, subtree_dev_data = get_subtree_dataset()

# %%
class Experiment():

    def __init__(self, model, optimizer, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.model = model
        self.optimizer = optimizer
    def train(self):
        path = "{}.pt".format(self.kwargs['exp_name'] if 'exp_name' in self.kwargs else self.model.__class__.__name__)
        if os.path.exists(path):
            ckpt= torch.load(path)
            self.model.load_state_dict(ckpt["state_dict"])
            return
        self.losses, self.accs = train_model(self.model, self.optimizer, *self.args, **self.kwargs)
    def eval(self, eval_fn=None, data=test_data, **kwargs):
        if eval_fn is None:
            if not ('eval_fn' in kwargs):
                eval_fn = simple_evaluate
            else:
                eval_fn = self.kwargs['eval_fn'] 
        return eval_fn(self.model, data, **kwargs)

    def plot(self):
        plt.plot(self.losses)
        plt.plot(self.accs)
        return
    def get_accuracy():
        return self.accs

    def get_losses():
        return self.losses



# %%
def prepare_subtreelstm_minibatch(mb, vocab):
  """
  Returns sentences reversed (last word first)
  Returns transitions together with the sentences.  
  """
  batch_size = len(mb)
  maxlen = max([len(ex.tokens) for ex in mb])
    
  # vocab returns 0 if the word is not there
  # NOTE: reversed sequence!
  x = [pad([vocab.w2i.get(t, 0) for t in ex.tokens], maxlen)[::-1] for ex in mb]
  
  x = torch.LongTensor(x)
  x = x.to(device)
  
  y = [ex.label for ex in mb]
  y = torch.LongTensor(y)
  y = y.to(device)
  
  maxlen_t = max([len(ex.transitions) for ex in mb])
  transitions = [pad(ex.transitions, maxlen_t, pad_value=2) for ex in mb]
  transitions = np.array(transitions)
  transitions = transitions.T  # time-major
  
  return (x, transitions), y


# %%
# build all the experiments by feeding corresponding parameters
# cant think of cleaner way to do it :(
xargs_bow = dict(num_iterations=30000, print_every=1000, eval_every=1000)
optimizer = optim.Adam(bow_model.parameters(), lr=0.0005)
bow_exp = Experiment(bow_model, optimizer, **xargs_bow)

optimizer = optim.Adam(cbow_model.parameters(), lr=0.0005)
cbow_exp = Experiment(cbow_model, optimizer, **xargs_bow)

optimizer = optim.Adam(deep_cbow_model.parameters(), lr=0.0005)
deep_cbow_exp = Experiment(deep_cbow_model, optimizer, **xargs_bow)

optimizer = optim.Adam(pt_deep_cbow_model.parameters(), lr=0.0005)
deep_cbow_exp = Experiment(pt_deep_cbow_model, optimizer, num_iterations=30000,
      print_every=1000, eval_every=1000)

optimizer = optim.Adam(lstm_model.parameters(), lr=3e-4)
lstm_exp = Experiment(lstm_model, optimizer, num_iterations=25000, print_every=250, eval_every=1000)

optimizer = optim.Adam(tree_model.parameters(), lr=2e-4)
tree_lstm_exp = Experiment(tree_model, optimizer, num_iterations=30000, 
      print_every=250, eval_every=250,
      prep_fn=prepare_treelstm_minibatch,
      eval_fn=evaluate,
      batch_fn=get_minibatch,
      batch_size=25, eval_batch_size=25)

# build a new tree lstm for feeding subtree
sub_tree_model = TreeLSTMClassifier(
    len(v.w2i), 300, 150, len(t2i), v)

with torch.no_grad():
  sub_tree_model.embed.weight.data.copy_(torch.from_numpy(vectors))
  sub_tree_model.embed.weight.requires_grad = False
optimizer = optim.Adam(sub_tree_model.parameters(), lr=2e-4)
sub_tree_lstm_exp = Experiment(sub_tree_model, optimizer, num_iterations=30000,
                           print_every=250, eval_every=250,
                           prep_fn=prepare_subtreelstm_minibatch,
                           eval_fn=evaluate,
                           batch_fn=get_minibatch,
                           batch_size=25, eval_batch_size=25, exp_name='subtree_lstm', train_data=subtree_train_data)


# %%
sub_tree_lstm_exp.train()

# %%
