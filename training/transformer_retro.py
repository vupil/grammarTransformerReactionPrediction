# -*- coding: UTF-8 -*-

_author_ = "vm2583@columbia.edu"

import os
from os.path import isfile, join
import tensorflow as tf
import time
import numpy as np
import re
import logging as logging

LOGGER = logging.getLogger(__name__)

from training.transformer_forward import *
from preprocess.grammar import D

input_vocab_size = D + 1 + 10  # 10 reaction classes
target_vocab_size = D + 1 + 2  # additional tokens for indicating separation and end of reactants input

WARMUP_STEPS = 8000


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        # self.cycle = 100
        # self.factor = 20.0
        # self.warmup_steps = 16000

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.cycles_interval = 20.0 * WARMUP_STEPS
        self.multip = tf.Variable(1.0)

    def __call__(self, step):
        # def f1(): return self.warmup_steps + tf.math.mod(step, self.cycle)
        # def f2(): return step
        # u_step = tf.cond(tf.greater_equal(step, self.cycle), f1, f2)
        #
        #
        # arg1 = tf.math.minimum(1.0, u_step/self.warmup_steps)
        # arg2 = tf.math.maximum(u_step, self.warmup_steps)
        #
        # lr = tf.multiply(self.factor, arg1/arg2)

        multip = self.multip

        step_new = tf.math.mod(step, self.cycles_interval) + 1.0

        arg1 = tf.math.rsqrt(step_new)
        arg2 = step_new * (WARMUP_STEPS ** -1.5)

        lr = tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

        def f3(): return multip * 0.5

        def f4(): return multip

        multip = tf.cond(tf.equal(step_new, 1), f3, f4)

        self.multip.assign = multip

        return lr * multip


def main_train_retro(hyperparams_retro):
    (filepath, checkpoint_path_forward, batch_size, epochs, FRAC_LB_UB, TEST_FRAC_ID, TEST_FRAC, BEAM_SIZE, EVAL_DIR,
     num_layers,
     d_model,
     dff, num_heads, dropout_rate, pe_inpt, pe_targ, NOCLASS, input_vocab_size, target_vocab_size) = hyperparams_retro

    rktnt_filenames = [filepath + r'/rctnts/' + f for f in os.listdir(filepath + 'rctnts') if
                       isfile(join(filepath + 'rctnts', f))]
    prdct_filenames = [filepath + r'/prdcts/' + f for f in os.listdir(filepath + 'prdcts') if
                       isfile(join(filepath + 'prdcts', f))]

    # retain only .npz file format
    rktnt_filenames = [f for f in rktnt_filenames if f.endswith(".npz")]
    prdct_filenames = [f for f in prdct_filenames if f.endswith(".npz")]

    # sort based on the filenames
    rktnt_filenames_sorted = natural_sort(rktnt_filenames)
    prdct_filenames_sorted = natural_sort(prdct_filenames)

    # shuffle the files now
    np.random.seed(42)
    np.random.shuffle(rktnt_filenames_sorted)
    np.random.seed(42)
    np.random.shuffle(prdct_filenames_sorted)

    train_rktnt_filenames = rktnt_filenames_sorted
    train_prdct_filenames = prdct_filenames_sorted

    my_training_batch_generator = MyCustomGenerator(train_rktnt_filenames, train_prdct_filenames, batch_size=batch_size,
                                                    frac_lb_ub=FRAC_LB_UB)

    # Train the model: load checkpoint if it exists
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    transformer = Transformer(num_layers, d_model, num_heads, dff,
                              input_vocab_size, target_vocab_size,
                              pe_input=pe_inpt,
                              pe_target=pe_targ,
                              rate=dropout_rate)

    # The @tf.function trace-compiles train_step into a TF graph for faster
    # execution. The function specializes to the precise shape of the argument
    # tensors. To avoid re-tracing due to the variable sequence lengths or variable
    # batch sizes (the last batch is smaller), use input_signature to specify
    # more generic shapes.

    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ]

    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)

    @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = transformer(inp, tar_inp,
                                         True,
                                         enc_padding_mask,
                                         combined_mask,
                                         dec_padding_mask)

            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(tar_real, predictions)

    # ====================================================================================
    #  Checkpoints and Training!
    # ====================================================================================

    LOGGER.warning('Starting...')

    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path_forward, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        LOGGER.warning('Latest checkpoint restored!!')
    else:
        LOGGER.warning('No checkpoint found. Training model from scratch!')

    for epoch in range(epochs):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        for batch, res in enumerate(my_training_batch_generator):
            for r in res:
                inp, tar = r[1], r[0]  # order reversed for retrosynthesis

                if NOCLASS:
                    inp = inp[:, 1:]
                train_step(inp, tar)

                if batch % 5 == 0:  # print metrics every 5 batches
                    LOGGER.warning('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                        epoch + 1, batch, train_loss.result(), train_accuracy.result()))

        if (epoch + 1) % 1 == 0:  # save checkpoints every 5 epochs
            ckpt_save_path = ckpt_manager.save()
            LOGGER.warning('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                         ckpt_save_path))

        LOGGER.warning('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                                     train_loss.result(),
                                                                     train_accuracy.result()))
        LOGGER.warning('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
