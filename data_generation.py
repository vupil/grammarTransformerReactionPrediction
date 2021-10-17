# -*- coding: UTF-8 -*-

_author_ = "vm2583@columbia.edu"

from preprocess.reactions import encode_dataset_reaction

fpath = "datasets/forward/test.txt"
kind = 'test'  # 'train', 'valid', 'test'
forward = True
maxlen_reactants = 700
maxlen_products = 300
verbose = True

if __name__ == "__main__":
    encode_dataset_reaction(fpath, kind, forward, maxlen_reactants, maxlen_products, verbose)
