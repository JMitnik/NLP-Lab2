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
cuda_output = !ldconfig -p | grep cudart.so | sed -e 's/.*\.\([0-9]*\)\.\([0-9]*\)$/cu\1\2/'
accelerator = cuda_output[0] if exists('/dev/nvidia0') else 'cpu'
!pip install -q http://download.pytorch.org/whl/{accelerator}/torch-0.4.1-{platform}-linux_x86_64.whl torchvision
# if accelerator == 'cpu':
#   raise InvalidArgumentError('should run this notebook under gpu enviroment')
# !pip install torch torchvision

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

name2class = {'bow': BOW, 'cbow': CBOW, 'deep_cbow': DeepCBOW, 'pt_deep_cbow': DeepCBOW, 'lstm': LSTMClassifier, 'mini_lstm': LSTMClassifier,
              'tree_lstm': TreeLSTMClassifier, 'subtree_lstm': TreeLSTMClassifier}
name2lr = {'bow': 5e-4, 'cbow': 5e-4, 'deep_cbow': 5e-4, 'pt_deep_cbow': 5e-4,
           'lstm': 3e-4, 'mini_lstm':2e-4, 'tree_lstm': 2e-4, 'subtree_lstm': 2e-4}
xargs_bow = dict(num_iterations=30000, print_every=1000, eval_every=1000)
xargs_lstm = dict(num_iterations=25000, print_every=250, eval_every=1000)
xargs_mini_lstm = dict(num_iterations=30000,
                  print_every=250, eval_every=250,
                  batch_size=batch_size,
                  batch_fn=get_minibatch,
                  prep_fn=prepare_minibatch,
                  eval_fn=evaluate)
xargs_tree_lstm = dict(num_iterations=30000,
                       print_every=250, eval_every=250,
                       prep_fn=prepare_treelstm_minibatch,
                       eval_fn=evaluate,
                       batch_fn=get_minibatch,
                       batch_size=25, eval_batch_size=25)
xargs_subtree_lstm = dict(num_iterations=30000,
                          print_every=250, eval_every=250,
                          prep_fn=prepare_subtreelstm_minibatch,
                          eval_fn=evaluate,
                          batch_fn=get_minibatch,
                          batch_size=25, eval_batch_size=25, train_data=subtree_train_data)
name2xargs = {'bow': xargs_bow, 'cbow': xargs_bow, 'deep_cbow': xargs_bow, 'pt_deep_cbow': xargs_bow,
              'lstm': xargs_lstm, 'mini_lstm': xargs_mini_lstm, 'tree_lstm': xargs_tree_lstm, 'subtree_lstm': xargs_subtree_lstm}

bow_p = [vocab_size, n_classes, v]
cbow_p = [len(v.w2i), embedding_dim, len(t2i), v]
deep_cbow_p = [len(v.w2i), embedding_dim, hidden_dim, len(t2i), v]
pt_deep_cbow_p = [len(nv.w2i), embedding_dim, hidden_dim, len(t2i), nv]
lstm_p = [len(nv.w2i), 300, 168, len(t2i), nv]
tree_lstm_p = [len(nv.w2i), 300, 150, len(t2i), nv]
name2model_p = {'bow': bow_p, 'cbow': cbow_p, 'deep_cbow': deep_cbow_p, 'pt_deep_cbow': pt_deep_cbow_p,
                'lstm': lstm_p, 'mini_lstm':lstm_p, 'tree_lstm': tree_lstm_p, 'subtree_lstm': tree_lstm_p}


def do_experiment(rd_seed, exp_name_li=list(name2class.keys())):
    torch.cuda.manual_seed(rd_seed)
    np.random.seed(rd_seed)
    for n in exp_name_li:
        # build a new tree lstm for feeding subtree
        model = name2class[n](*name2model_p[n])
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=name2lr[n])
        if n.startswith('pt') or n.endswith('lstm'):
            with torch.no_grad():
                model.embed.weight.data.copy_(torch.from_numpy(vectors))
                model.embed.weight.requires_grad = False
        exp = Experiment(model, optimizer, exp_name='{}_rd_seed_{}'.format(
            n, rd_seed), **name2xargs[n])
        exp.train()
        yield exp

# %%
rd_s_li = [7, 42, 1984]
exp_li = []
for rs_s in rd_s_li:
    exp_li.append(list(do_experiment(rs_s)))

# %%

!cp ./*.pt "/gdrive/My Drive/pts"

# %%

