# -*- coding: UTF-8 -*-

_author_ = "vm2583@columbia.edu"

from preprocess.reactions import encode_dataset_reaction

## FORWARD ##
# fpath = "datasets/forward/test.txt"  # fwd prediction:
# kind = 'test'  # 'train', 'valid', 'test'
# forward = True
# verbose = True
# retro=False

## RETROSYNTHESIS ##
fpath = ("datasets/retro/test_targets.txt", "datasets/retro/test_sources.txt")  # reactants_file, products_file
kind = 'test'  # 'train', 'valid', 'test'
forward = False
verbose = True
retro = True

# Following varaibles are imported while training the models. Feel free to change but do not comment them out
maxlen_reactants_forward, maxlen_products_forward = 700, 300
maxlen_reactants_retro, maxlen_products_retro = 900, 300 + 1  # first token is for reaction class

if forward:
    maxlen_reactants, maxlen_products = maxlen_reactants_forward, maxlen_products_forward
else:
    maxlen_reactants, maxlen_products = maxlen_reactants_retro, maxlen_products_retro

if __name__ == "__main__":
    encode_dataset_reaction(fpath, kind, forward, maxlen_reactants, maxlen_products, verbose)
