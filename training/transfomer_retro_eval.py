# -*- coding: utf-8 -*-

_author_ = "vm2583@columbia.edu"

from training.transfomer_forward_eval import *
from preprocess.grammar import D

LOGGER = logging.getLogger(__name__)


def validate_model_retro(my_val_batch_generator, transformer, pe_inpt, pe_targ, BEAM_SIZE, TEST_FRAC_ID, EVAL_DIR,
                         NOCLASS, input_vocab_size, target_vocab_size):
    resdf = pd.DataFrame(columns=['rknclass', 'source', 'true_reactants', 'predicted', 'acc', 'acc_frac', 'valid'])
    rkncount = 0

    for batch, res in enumerate(my_val_batch_generator):
        for r in res:
            inp, tar = r[1], r[0]  # reversed the direction of translation (chemical reaction)
            for j in range(0, inp.shape[0]):  # iterate over all the rows in the input

                # add D to reaction class
                inp[j][0] = inp[j][0] + D

                trns_inp, trns_tar = inp[j], tar[j]

                if NOCLASS:
                    res = np.array(
                        translate_beam(trns_inp[1:], transformer, pe_targ, target_vocab_size, beamSize=BEAM_SIZE))
                else:
                    res = np.array(
                        translate_beam(trns_inp, transformer, pe_targ, target_vocab_size, beamSize=BEAM_SIZE))

                prd_all, act_all, prd_smiles, act_smiles = [], [], [], []
                rkn_class = inp[j][0] - D

                for s in range(len(res)):
                    # ignore 81: remove last element in array
                    # split s at 81's (80's after subtracting 1)

                    nwres = np.array(res[s][0] - 1).flatten()
                    nwres = nwres[:-1]  # remove the last one: always 82

                    # nwresList = split at 80
                    prd_split_idx = np.where(nwres == D)[0]  # -3 because -2 and then another -1

                    if len(prd_split_idx) == 0:  # if single reactant predicted
                        nwresList = [nwres]
                    else:  # if multiple reactants predicted, split at 81 (80 after -1)
                        nwresList = np.split(nwres, prd_split_idx)

                    # iterate over elements in nwresList
                    ctr = 0

                    for nwres in nwresList:
                        if ctr >= 1:
                            nwres = nwres[1:]  # because 80 at the start of each non-first multiple predicted reactants
                        ctr = ctr + 1

                        # predicted product
                        one_hot = np.zeros((pe_targ, D))
                        for i in range(len(nwres)):
                            one_hot[i, int(nwres[i])] = 1

                        one_hot[np.all(one_hot == 0, axis=1), D - 1] = 1

                        one_hot = one_hot.reshape((-1, one_hot.shape[0], one_hot.shape[1]))
                        prd = parse_trees.ZincGrammarModel().decode(one_hot, return_smiles=True)

                        prd_all.append(prd)

                # actual reactants (target) for the given row num
                # ============================================================

                trns_tar_same = trns_tar[:np.argmax(trns_tar)] - 1

                tar_split_idx = np.where(trns_tar_same == D)[0]

                if len(tar_split_idx) == 0:  # if single reactant predicted
                    trns_tar_same_list = [trns_tar_same]
                else:  # if multiple reactants predicted, split at 81 (80 after -1)
                    trns_tar_same_list = np.split(trns_tar_same, tar_split_idx)

                # iterate over elements in trns_tar_same_list
                ctr_tar = 0

                for trns_tar_same in trns_tar_same_list:
                    if ctr_tar >= 1:
                        trns_tar_same = trns_tar_same[1:]

                    ctr_tar = ctr_tar + 1

                    # target (actual)
                    one_hot_a = np.zeros((pe_targ, D))
                    for i in range(len(trns_tar_same)):
                        one_hot_a[i, int(trns_tar_same[i])] = 1

                    one_hot_a[np.all(one_hot_a == 0, axis=1), D - 1] = 1

                    one_hot_a = one_hot_a.reshape((-1, one_hot_a.shape[0], one_hot_a.shape[1]))
                    act = parse_trees.ZincGrammarModel().decode(one_hot_a, return_smiles=True)

                    act_all.append(act)

                inp_source = list(
                    trns_inp[:np.argmax(trns_inp[1:])][1:] - 1)  # to remove the encoding for the reaction class

                inp_source.append(D - 1)

                one_hot = np.zeros((pe_inpt, D))  # to remove the encoding for the reaction class

                seq = inp_source
                for i in range(len(seq)):
                    one_hot[i, int(seq[i])] = 1
                one_hot = one_hot.reshape((-1, one_hot.shape[0], one_hot.shape[1]))

                seq_parse = parse_trees.ZincGrammarModel().decode(one_hot, return_smiles=True)

                rkncount += 1
                LOGGER.warning('Reaction id: {}'.format(rkncount))

                try:
                    LOGGER.warning('Source (major product) ---> {}'.format(seq_parse[1]))
                    LOGGER.warning('Actual target (true reactants): ')
                    for act in act_all:
                        LOGGER.warning('{}'.format(act[1]))
                        act_smiles.append(act[1][0])

                    LOGGER.warning('Predicted Reactants')
                    for prd in prd_all:
                        LOGGER.warning('{}'.format(prd[1]))
                        prd_smiles.append(prd[1][0])

                    LOGGER.warning('==' * 50)

                    acc_frac = len(set(act_smiles).intersection(set(prd_smiles))) / len(act_smiles)
                    acc = int(acc_frac == 1.0)

                    valid_count = 0
                    for v in prd_smiles:
                        if v != '':
                            valid_count += 1

                    valid_frac = valid_count / len(prd_smiles)

                    resdf.loc[resdf.shape[0], :] = [rkn_class, seq_parse[1][0], act_smiles, prd_smiles, acc, acc_frac,
                                                    valid_frac]

                except:

                    LOGGER.warning('==' * 100)
                    LOGGER.warning('SKIPPED')
                    LOGGER.warning('==' * 100)

                    resdf.loc[resdf.shape[0], :] = [seq_parse[1][0], act_smiles, prd_smiles, '', '', '']

                resdf.to_csv(EVAL_DIR + '/' + 'top_{}_test_frac_id_{}.csv'.format(BEAM_SIZE, TEST_FRAC_ID))

                # print the averagae stats every 5 rows
                if resdf.shape[0] % 5 == 0:
                    LOGGER.warning('AVERAGE STATS:')
                    LOGGER.warning('==' * 100)
                    LOGGER.warning(resdf.mean())


def main_eval_retro(hyperparams_retro):
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
        exit()

    # create evaluation directory if does not exist
    if not os.path.exists(EVAL_DIR):
        os.makedirs(EVAL_DIR)

    validate_model_retro(my_evaluation_batch_generator, transformer, pe_inpt, pe_targ, BEAM_SIZE, TEST_FRAC_ID,
                         EVAL_DIR, NOCLASS, input_vocab_size, target_vocab_size)
