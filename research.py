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
        self.pred, _, _, self.acc = eval_fn(self.model, data, **kwargs)
        return 

    def plot(self):
        plt.plot(self.losses)
        plt.plot(self.accs)
        return

    def get_accuracy():
        return self.accs

    def get_losses():
        return self.losses

# %%
# build all the experiments by feeding corresponding parameters
# cant think of cleaner way to do it :(
from collections import namedtuple
ExperimentModel = namedtuple("ExperimentModel", ["name", "modelclass" ,"lr", "options", "parameters"])

models = [
#     ExperimentModel("bow", BOW, 5e-4,
#         dict(num_iterations=30000, print_every=1000, eval_every=1000),
#         [vocab_size, n_classes, v]
#     ),
#     ExperimentModel("cbow", CBOW, 5e-4, 
#         dict(num_iterations=30000, print_every=1000, eval_every=1000),
#         [len(v.w2i), embedding_dim, len(t2i), v]
#     ),
#     ExperimentModel("deep_cbow", DeepCBOW, 5e-4,
#         dict(num_iterations=30000, print_every=1000, eval_every=1000),
#         [len(v.w2i), embedding_dim, hidden_dim, len(t2i), v]
#    ),
#     ExperimentModel("pt_deep_cbow", DeepCBOW, 5e-4,
#         dict(num_iterations=30000, print_every=1000, eval_every=1000),
#         [len(nv.w2i), embedding_dim, hidden_dim, len(t2i), nv]
#    ),
#     ExperimentModel("lstm", LSTMClassifier, 3e-4,
#         dict(num_iterations=25000, print_every=250, eval_every=1000),
#         [len(nv.w2i), 300, 168, len(t2i), nv]
#    ),
#     ExperimentModel("mini_lstm", LSTMClassifier, 2e-4,
#         dict(num_iterations=30000,
#                   print_every=250, eval_every=250,
#                   batch_size=batch_size,
#                   batch_fn=get_minibatch,
#                   prep_fn=prepare_minibatch,
#                   eval_fn=evaluate
#         ),
#         [len(nv.w2i), 300, 168, len(t2i), nv]
#     ),
    ExperimentModel("tree_lstm", TreeLSTMClassifier, 2e-4,
       dict(num_iterations=30000,
           print_every=250, eval_every=250,
           prep_fn=prepare_treelstm_minibatch,
           eval_fn=evaluate,
           batch_fn=get_minibatch,
           batch_size=25, eval_batch_size=25),
        [len(nv.w2i), 300, 150, len(t2i), nv]
   )
#     ExperimentModel("subtree_lstm", TreeLSTMClassifier, 2e-4,
#        dict(num_iterations=30000,
#               print_every=250, eval_every=250,
#               prep_fn=prepare_treelstm_minibatch,
#               eval_fn=evaluate,
#               batch_fn=get_minibatch,
#               batch_size=25, eval_batch_size=25, train_data=subtree_train_data),
#         [len(nv.w2i), 300, 150, len(t2i), nv]
#    )
]
!cp /gdrive/My\ Drive/pts/*.pt ./

def do_experiment(rd_seed, experiment_models, train_embed=False):
    torch.cuda.manual_seed(rd_seed)
    np.random.seed(rd_seed)
    for exp_model in experiment_models:
        # build a new tree lstm for feeding subtree
        model = exp_model.modelclass(*exp_model.parameters)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=exp_model.lr)
        if exp_model.name.startswith('pt') or exp_model.name.endswith('lstm'):
            with torch.no_grad():
                model.embed.weight.data.copy_(torch.from_numpy(vectors))

                if not train_embed:
                    model.embed.weight.requires_grad = False
        
        path_embed_string = "_train_embed" if train_embed else ""
        exp = Experiment(model, optimizer, exp_name='{}_rd_seed_{}{}'.format(
            exp_model.name, rd_seed, path_embed_string), **exp_model.options)
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
!pip install prettytable
import prettytable as pt


# %%

def generate_tables():
    models_eval_results = {}
    for rd, el in zip(rd_s_li, *exp_li):
        for n, exp in zip(model_name_li, el):
            prep_func = prepare_treelstm_minibatch if n.startswith(
                'tree') or n.startswith('subtree') else prepare_minibatch
            exp.eval(eval_fn=evaluate_with_results, batch_fn=get_minibatch,
                    prep_fn=prep_func)
            pred_and_acc = [exp.pred, exp.acc]
            models_eval_results[n] = [pred_and_acc] if not models_eval_results.get(
                n) else models_eval_results[n].append([pred_and_acc])
    acc_table = pt.PrettyTable()
    acc_table.field_names = model_name_li
    def helper(x):
        y = [e[1] for e in x]
        m = np.mean(y)
        v = np.std(y)
        s = '{}+-{}'.format(m, v)
        return s
    acc_table.add_row([helper(models_eval_results[n]) for n in model_name_li])
    sig_table = pt.PrettyTable()
    sig_table.field_names = ['models']+model_name_li
    for n1 in model_name_li:
        row = [n1]
        for n2 in model_name_li:
            sig = sign_test(models_eval_results[n1][0][0], models_eval_results[n2][0][0])
            row.append(sig)
        sig_table.add_row(row)
    return acc_table, sig_table

# %%

acc_table, sig_table = generate_tables()

# %%
print(acc_table)

# %%
print(sig_table)

# %%
!pip install pydrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)


def load_shared_files():
  file_id = "1az0y5LU2T7vrhc3AVpVgfZthp7kAJUqH"
  files = drive.ListFile({'q': "'%s' in parents and trashed=false" % file_id}).GetList()
  for f in files:
    print('title: %s, id: %s' % (f['title'], f['id']))
    fname = os.path.join('', f['title'])
    print('downloading to {}'.format(fname))
    f_ = drive.CreateFile({'id': f['id']})
    f_.GetContentFile(fname)
  
load_shared_files()

# %%
from itertools import groupby

def prep_bin(data, bin_size):
    """ Returns a bin """
    max_line_length = max([len(example.tokens) for example in data])
    sorted_data = sorted(data, key=lambda x: len(x.tokens))
    bins = []
    
    for key, group in groupby(sorted_data, lambda x: len(x.tokens)):
        bins.append(list(group))
        
    return bins

def split_data_on_sentlen(data, sent_len_split=20):
    """Splits data into two parts, based on `sent_len_split`"""
    result_l = []
    result_r = []

    for example in data:
        if len(example.tokens) < sent_len_split:
            result_l.append(example)
        else:
            result_r.append(example)
    
    return [result_l, result_r]


# BE SURE TO MAKE SURE TO COPY AL .ct files!
# %%
def sent_len_evaluate(model, data,  batch_fn=get_minibatch, bin_size=5 , prep_fn=prepare_minibatch, **kwargs):
    """Evaluates a model for different sentence sizes.
        - Data: Example[]
        - Model: Trained Model
        - prep_fn: Method to turn Example into tensor
    """
    set_trace()
    binned_data = split_data_on_sentlen(data, 20)
    results = []

    for index, examples in enumerate(binned_data):
        pred, _, _, acc = evaluate_with_results(model, examples, batch_fn, prep_fn)
        results.append(acc)
    
    # Results is not exaclty correct, as eval expects only one result, but this indicates the results for each 'bin' atm
    return _, _, _, results
    # bin = list of entries, where each entry is mapped to corresponding sent
    # length

# %%
exp2 = next(do_experiment(7, models))
# FOR TREE
# exp.eval(eval_fn=sent_len_evaluate, batch_fn=get_minibatch,
#                     prep_fn=prepare_treelstm_minibatch)

# For MINI_LSTM
exp2.eval(eval_fn=sent_len_evaluate, batch_fn=get_minibatch,
                    prep_fn=prepare_minibatch)