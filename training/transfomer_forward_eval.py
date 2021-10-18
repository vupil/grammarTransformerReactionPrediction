# -*- coding: utf-8 -*-

_author_ = "vm2583@columbia.edu"

import tensorflow as tf
import logging as logging
import numpy as np
import os
from difflib import SequenceMatcher
import preprocess.parse_trees as parse_trees
import nltk
import pandas as pd
from os.path import isfile, join
from training.transformer_forward import create_masks, MyCustomGenerator, natural_sort, Transformer, CustomSchedule, \
    loss_function

LOGGER = logging.getLogger(__name__)

# ====================================================================================
#  Evaluation
# ====================================================================================

def evaluate_beam(inp_sentence, transformer, pe_targ, beamSize=5):
    encoder_input = tf.expand_dims(inp_sentence, 0)
    decoder_input = [1]  # start token for the output: 1
    output = tf.expand_dims(decoder_input, 0)

    all_outputs = [(output, 0.0, True)]

    for i in range(pe_targ):  # for each step in the sequence, perform a beam search
        curr_output_and_scores = []
        for output_and_scores in all_outputs:  # generate a beam for each output sequence

            output, curr_score, is_decode = output_and_scores[0], output_and_scores[1], output_and_scores[2]

            if not is_decode:
                try:
                    a = (output, curr_score, is_decode) in curr_output_and_scores
                    if a == True:  # sequence decoded AND is present in curr_output_and_scores
                        continue

                    else:  # sequence decoded but NOT present in curr_output_and_scores; add to list and continue
                        curr_output_and_scores.append((output, curr_score, is_decode))
                        continue

                except ValueError:
                    # throws an error when the above tuple is NOT in curr_output_and_scores
                    # does the same thing as the 'else' statement above!:
                    # sequence decoded but NOT present in curr_output_and_scores; add to list and continue

                    curr_output_and_scores.append((output, curr_score, is_decode))
                    continue

            beamSizeAdjusted = beamSize

            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)

            ## TODO: ISSUE HERE WHEN USING WITH RETRO TRANSFORMER!!!!!
            predictions, attention_weights = transformer(encoder_input,
                                                         output,
                                                         False,
                                                         enc_padding_mask,
                                                         combined_mask,
                                                         dec_padding_mask)
            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            # COMPUTE AND STORE SCORES HERE INSTEAD OF ARGSORT(): args can be extracted later to assemble sequences
            predicted_ids_top_B = tf.cast(tf.argsort(predictions, axis=-1, direction='DESCENDING'), tf.int32)[0, 0,
                                  :beamSizeAdjusted]

            # The scores computation needs to be corrected, Maybe!
            predicted_scores_top_B = tf.math.log(tf.nn.softmax(tf.sort(predictions, axis=-1, direction='DESCENDING')))[
                                     0, 0, :beamSizeAdjusted]

            for (predicted_id, pred_score) in zip(np.array(predicted_ids_top_B), np.array(predicted_scores_top_B)):
                # return the result if the predicted_id is equal to the end token
                if np.array(predicted_id) == (target_vocab_size - 1):
                    # predicted_id = target_vocab_size - 1
                    # pred_score = 0.0
                    currIsDecode = False
                    # this sequence should not be decoded further

                else:
                    currIsDecode = True

                prdidTensor = tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(predicted_id), axis=-1),
                                             axis=-1)  # expand dims twice
                outputNext = tf.concat([output, prdidTensor], axis=-1)

                curr_output_and_scores.append((outputNext, curr_score + pred_score, currIsDecode))

        # retain only the top-B sequences from curr_output_and_scores

        all_outputs = sorted(curr_output_and_scores, key=lambda x: -x[1])  # sort based on the scores from highest to
        # lowest, that's why the -x[1]
        all_outputs = all_outputs[:beamSize]  # select only B sequences from the list of candidates

        # break out of the for-loop if all the sequences have been decoded i.e. isDecode=False for all
        if np.all([x[2] == False for x in all_outputs]):
            break

    return all_outputs


def translate_beam(sentence, transformer, pe_targ, beamSize=3):
    result = evaluate_beam(sentence, transformer, pe_targ, beamSize=beamSize)
    return result


def main_eval_forward(hyperparams_forward):
    (filepath, checkpoint_path_forward, batch_size, epochs, FRAC_LB_UB, TEST_FRAC_ID, TEST_FRAC, BEAM_SIZE, EVAL_DIR, num_layers, d_model, dff, num_heads, dropout_rate, pe_inpt, pe_targ, input_vocab_size, target_vocab_size) = hyperparams_forward


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

    lobound, upbound = int(TEST_FRAC * (TEST_FRAC_ID - 1) * len(train_rktnt_filenames)), int(
        TEST_FRAC * TEST_FRAC_ID * len(train_rktnt_filenames))
    train_rktnt_filenames = train_rktnt_filenames[lobound:upbound]
    train_prdct_filenames = train_prdct_filenames[lobound:upbound]

    LOGGER.warning('validating on {} test files'.format(len(train_rktnt_filenames)))

    # batch_size not used for evaluation
    my_evaluation_batch_generator = MyCustomGenerator(train_rktnt_filenames, train_prdct_filenames,
                                                      batch_size=batch_size,
                                                      frac_lb_ub=FRAC_LB_UB)

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
        LOGGER.warning('No checkpoint found. Exiting...')


    # create evaluation directory if does not exist
    if not os.path.exists(EVAL_DIR):
        os.makedirs(EVAL_DIR)

    validate_model(my_evaluation_batch_generator, transformer, pe_inpt, pe_targ, BEAM_SIZE, TEST_FRAC_ID, EVAL_DIR)


def validate_model(my_val_batch_generator, transformer, pe_inpt, pe_targ, BEAM_SIZE, TEST_FRAC_ID, EVAL_DIR):
    val_list, acc_list, sim_list, BLEU_SCORE = [], [], [], []

    resdf = pd.DataFrame(columns=['reactants', 'product', 'predicted', 'valid', 'similarity', 'acc', 'bleu'])
    rkncount = 0

    for batch, res in enumerate(my_val_batch_generator):
        for r in res:
            inp, tar = r[0], r[1]

            # =====================================================================================
            # Evaluate a sentence
            # =====================================================================================
            for idx in range(0, inp.shape[0]):
                trns_inp, trns_tar = inp[idx], tar[idx]
                res = np.array(translate_beam(trns_inp, transformer, pe_targ, beamSize=BEAM_SIZE))

                # Print parse tree from one-hot encoded results
                # subtract 1 from all rules, then append 79 to the end
                nwreslist = []
                for s in range(len(res)):
                    # predicted reactants (target) for the given row num
                    # ============================================================
                    # ignore 81: remove last element in array
                    # split s at 81's (80's after subtracting 1)

                    # subtract 1 required so that index starts from 0-- required for smiles reconstruction from parse_trees code
                    nwres = np.array(res[s][0] - 1).flatten()
                    nwreslist.append(nwres)

                trns_tar_same = list(trns_tar[:np.argmax(trns_tar)] - 1)
                trns_tar_same.append(target_vocab_size - 2)  # 81 vocab size, 80 index, 79 for parse_trees

                # actual product
                one_hot_a = np.zeros((pe_targ, target_vocab_size - 1))
                for i in range(len(trns_tar_same)):
                    one_hot_a[i, int(trns_tar_same[i])] = 1

                one_hot_a[np.all(one_hot_a == 0, axis=1), target_vocab_size - 2] = 1

                one_hot_a = one_hot_a.reshape((-1, one_hot_a.shape[0], one_hot_a.shape[1]))
                act = parse_trees.ZincGrammarModel().decode(one_hot_a, return_smiles=True)

                prdlist = []
                # predicted product
                for nwres in nwreslist:
                    one_hot = np.zeros((pe_targ, target_vocab_size - 1))
                    for i in range(len(nwres)):
                        one_hot[i, int(nwres[i])] = 1

                    one_hot[np.all(one_hot == 0, axis=1), target_vocab_size - 2] = 1

                    one_hot = one_hot.reshape((-1, one_hot.shape[0], one_hot.shape[1]))
                    prd = parse_trees.ZincGrammarModel().decode(one_hot, return_smiles=True)
                    prdlist.append(prd[1])

                # Reactants
                rktnts = []
                brk_idx = np.where(trns_inp == (input_vocab_size - 1))[0]
                rktnt_iter = 0

                # reactants
                for i in range(len(brk_idx) - 1):
                    seq = trns_inp[rktnt_iter:brk_idx[i]]
                    seq = list(seq - 1)
                    seq.append(input_vocab_size - 2)

                    rktnt_iter = brk_idx[i] + 1

                    one_hot = np.zeros((pe_inpt, input_vocab_size - 1))

                    for i in range(len(seq)):
                        one_hot[i, int(seq[i])] = 1
                    one_hot = one_hot.reshape((-1, one_hot.shape[0], one_hot.shape[1]))

                    seq_parse = parse_trees.ZincGrammarModel().decode(one_hot, return_smiles=True)

                    rktnts.append(seq_parse[1])

                # =====================================================================================
                # Compute aggregated statistics
                # =====================================================================================

                if BEAM_SIZE == 1:
                    BLEU_SCORE.append(nltk.translate.bleu_score.sentence_bleu([[*act[1][0]]], [*prd[1][0]]))
                    sim_list.append(SequenceMatcher(None, prd[1][0], act[1][0]).ratio())
                else:
                    BLEU_SCORE.append(0)
                    sim_list.append(0)

                rkncount += 1
                LOGGER.warning('Reaction id: {}'.format(rkncount))
                try:
                    LOGGER.warning(
                        '{}---> true: {}, pred: {}'.format(rktnts, act[1], prdlist))

                    if [act[1][0]] in prdlist:
                        acc_list.append(1)
                    else:
                        acc_list.append(0)

                    valid_count = 0
                    for v in prdlist:
                        if v != ['']:
                            valid_count += 1

                    val_list.append(valid_count / len(prdlist))

                    LOGGER.warning('Valid SMILES fraction: {} %'.format(np.mean(val_list) * 100))
                    LOGGER.warning('Accuracy fraction: {} %'.format(np.mean(acc_list) * 100))
                    LOGGER.warning('Similarity fraction: {} %'.format(np.mean(sim_list) * 100))
                    LOGGER.warning('BLEU score: {} %'.format(np.mean(BLEU_SCORE) * 100))

                    LOGGER.warning('==' * 50)

                    # create a csv file : rktnts, agnts , products, predicted, sim score, acc (binary), bleu score
                    resdf.loc[resdf.shape[0], :] = [rktnts, act[1], prdlist,
                                                    val_list[-1], sim_list[-1], acc_list[-1], BLEU_SCORE[-1]]

                except:
                    val_list.append(0)

                    LOGGER.warning('Valid SMILES fraction: {} %'.format(np.mean(val_list) * 100))
                    LOGGER.warning('Accuracy fraction: {} %'.format(np.mean(acc_list) * 100))
                    LOGGER.warning('Similarity fraction: {} %'.format(np.mean(sim_list) * 100))
                    LOGGER.warning('BLEU score: {} %'.format(np.mean(BLEU_SCORE) * 100))
                    LOGGER.warning('SKIPPED')

                    LOGGER.warning('==' * 100)
                    LOGGER.warning('==' * 100)

                    # create a csv file : rktnts, agnts , products, predicted, sim score, acc (binary), bleu score
                    resdf.loc[resdf.shape[0], :] = [rktnts, act[1], prdlist,
                                                    val_list[-1], sim_list[-1], acc_list[-1], BLEU_SCORE[-1]]

                resdf.to_csv(EVAL_DIR+'/'+'testRes_{}.csv'.format(TEST_FRAC_ID))
