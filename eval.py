# -*- coding: utf-8 -*-

_author_ = "vm2583@columbia.edu"

from training.transfomer_forward_eval import main_eval_forward
from data_generation import maxlen_reactants, maxlen_products
import logging as logging

LOGGER = logging.getLogger(__name__)


# ====================================================================================
#  Pretrained models' paths
# ====================================================================================
checkpoint_path_retro_noclass = "pretrained_models/retro/noclass"
checkpoint_path_retro_class = "pretrained_models/retro/withclass"

# ====================================================================================
#  Specify model type: 'forward' or 'retro'
# ====================================================================================
MODEL = 'forward'

# ====================================================================================
#  Forward prediction model
# ====================================================================================
datasets_filepath = "datasets/forward/test/"  # model training: datasets/forward/train/, evaluation: datasets/forward/test/
checkpoint_path_forward = "pretrained_models/forward"  # otherwise, None
batch_size = 3000
epochs = 100
FRAC_LB_UB=0.2
TEST_FRAC_ID = 1  # only for evaluation
TEST_FRAC = 1.0  # only for evaluation
BEAM_SIZE = 1 # only for evaluation

num_layers = 4  # number of layers per encoder/decoder
d_model = 256  # embedding dimensions
dff = 512  # dimensions of each feed-forward network dense layer
num_heads = 8  # num of heads in the multi-head attention
dropout_rate = 0.1
pe_inpt, pe_targ = maxlen_reactants, maxlen_products  # 700,300 PositionalEmbeddings input and target | should be the same reactants and products limits

hyperparams_forward = (
datasets_filepath, checkpoint_path_forward, batch_size, epochs, FRAC_LB_UB, TEST_FRAC_ID, TEST_FRAC, BEAM_SIZE, num_layers, d_model, dff, num_heads, dropout_rate,
pe_inpt, pe_targ)




if __name__ == "__main__":

    if MODEL == 'forward':
        LOGGER.warning('Loading/training the forward model')
        main_eval_forward(hyperparams_forward)
