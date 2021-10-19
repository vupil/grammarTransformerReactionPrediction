# -*- coding: utf-8 -*-

_author_ = "vm2583@columbia.edu"

from training.transformer_forward import main_train_forward
from training.transformer_retro import main_train_retro
from data_generation import maxlen_reactants_retro, maxlen_products_retro, maxlen_products_forward, maxlen_reactants_forward
from preprocess.grammar import D

import logging as logging

LOGGER = logging.getLogger(__name__)


# ====================================================================================
#  Forward prediction model
# ====================================================================================
forward = True
datasets_filepath = "datasets/forward/test/"  # model training: datasets/forward/train/, evaluation: datasets/forward/test/
checkpoint_path = "pretrained_models/forward"  # otherwise, None
batch_size = 3000
epochs = 100
FRAC_LB_UB = 0.2  # for GPU: controls the number of files loaded in memory

TEST_FRAC_ID = 1  # only for evaluation
TEST_FRAC = 1.0  # only for evaluation
BEAM_SIZE = 2  # only for evaluation
EVAL_DIR = 'results/forward'  # only for evaluation

num_layers = 4  # number of layers per encoder/decoder
d_model = 256  # embedding dimensions
dff = 512  # dimensions of each feed-forward network dense layer
num_heads = 8  # num of heads in the multi-head attention
dropout_rate = 0.1
pe_inpt, pe_targ = maxlen_reactants_forward, maxlen_products_forward
input_vocab_size = D+1  # 81
target_vocab_size = D+1 # 81

hyperparams = (
    datasets_filepath, checkpoint_path, batch_size, epochs, FRAC_LB_UB, TEST_FRAC_ID, TEST_FRAC, BEAM_SIZE,
    EVAL_DIR, num_layers, d_model, dff, num_heads, dropout_rate, pe_inpt, pe_targ, input_vocab_size, target_vocab_size)

# # ====================================================================================
# #  Retrosynthesis prediction model
# # ====================================================================================
# datasets_filepath = "datasets/retro/test/"  # model training: datasets/forward/train/, evaluation: datasets/forward/test/
# forward = False
# NOCLASS = True
# if NOCLASS:
#     checkpoint_path = "pretrained_models/retro/noclass"  # otherwise, None
# else:
#     checkpoint_path = "pretrained_models/retro/withclass"  # otherwise, None
#
# batch_size = 3000
# epochs = 100
# FRAC_LB_UB = 0.2  # for GPU: controls the number of files loaded in memory
#
# TEST_FRAC_ID = 1  # only for evaluation
# TEST_FRAC = 1.0  # only for evaluation
# BEAM_SIZE = 2  # only for evaluation
# EVAL_DIR = 'results/retro'  # only for evaluation
#
# num_layers = 4  # number of layers per encoder/decoder
# d_model = 256  # embedding dimensions
# dff = 512  # dimensions of each feed-forward network dense layer
# num_heads = 8  # num of heads in the multi-head attention
# dropout_rate = 0.1
# pe_inpt, pe_targ = maxlen_products_retro, maxlen_reactants_retro
#
# if NOCLASS:
#     input_vocab_size = D+1
# else:
#     input_vocab_size = D+1+10  # 10 reaction classes
# target_vocab_size = D+1+2
#
# hyperparams = (
#     datasets_filepath, checkpoint_path, batch_size, epochs, FRAC_LB_UB, TEST_FRAC_ID, TEST_FRAC, BEAM_SIZE,
#     EVAL_DIR, num_layers, d_model, dff, num_heads, dropout_rate, pe_inpt, pe_targ, NOCLASS, input_vocab_size, target_vocab_size)
#


if __name__ == "__main__":

    if forward:
        LOGGER.warning('Loading/training the forward model')
        main_train_forward(hyperparams)

    else:
        LOGGER.warning('Loading/training the forward model')
        main_train_retro(hyperparams)

