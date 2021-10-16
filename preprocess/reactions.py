# -*- coding: UTF-8 -*-

from rdkit import Chem
from rdkit import RDLogger
import numpy
import parse_trees
import logging as logging
import os

LOGGER = logging.getLogger(__name__)
RDLogger.DisableLog('rdApp.*')

_author_ = "vm2583@columbia.edu"


def mols_from_smiles_list(all_smiles):
    '''
    Create list of rdkit mols from a given list of smiles strings
    '''
    mols = []
    for smiles in all_smiles:
        if not smiles: continue
        mols.append(Chem.MolFromSmiles(smiles))
    return mols


# REACTION_DB = open(r"../datasets/data_jin/train.txt", "rb")
# REACTION_DB = open(r"datasets/data_jin/train.txt", "rb")
# REACTION_DB = open(r"datasets/human/test/human.txt", "rb")

def get_reactions(fpath, verbose=False):
    REACTION_DB = open(fpath, "rb")
    counter = 0
    reactants, agents, products, major_products = [], [], [], []
    for example_doc in REACTION_DB:
        try:
            reaction_smiles = str(example_doc, 'utf-8')
            reaction_smiles = reaction_smiles.strip("\r\n ").split()[0]

            rctnts, agnts, prdcts = [mols_from_smiles_list(x) for x in
                                     [mols.split('.') for mols in reaction_smiles.split('>')]]

            # Sanitize the molecules: not necessary but recommended
            [Chem.SanitizeMol(mol) for mol in rctnts + prdcts]

            # Remove atom mappings from SMILES
            for rctnt in rctnts:
                [x.ClearProp('molAtomMapNumber') for x in rctnt.GetAtoms() if x.HasProp('molAtomMapNumber')]
            for prdct in prdcts:
                [x.ClearProp('molAtomMapNumber') for x in prdct.GetAtoms() if x.HasProp('molAtomMapNumber')]

        except Exception as e:
            LOGGER.warning(e)
            LOGGER.warning('Could not load or sanitize SMILES')
            continue

        reactants.append([Chem.MolToSmiles(rctnt) for rctnt in rctnts])
        products.append([Chem.MolToSmiles(prdct) for prdct in prdcts])
        counter += 1

        # for debuggin
        if counter == 300:
            return reactants, products

        if verbose:
            if counter % 500 == 0:
                LOGGER.warning('Processed {} reactions'.format(counter))
    return reactants, products


# get_reactions('../datasets/forward/valid.txt', True)

def encode_reaction(reaction, gram_mod, max_r, max_p):
    rktnts, prdcts = reaction[0], reaction[1]
    return gram_mod.encode((rktnts, prdcts), max_r, max_p)  # ignoring agnts completely


def order_reactants(reactants):
    # order based on SMILES string length
    lensmiles = [len(x) for x in reactants]
    sorted_mols = [x for (y, x) in sorted(zip(lensmiles, reactants), reverse=True, key=lambda pair: pair[0])]
    return sorted_mols


def encode_dataset_reaction(fpath, kind='train', forward=True, maxlen_reactants=700, maxlen_products=300, verbose=True):
    rktnts_all, prdcts_all = get_reactions(fpath, verbose=verbose)

    # create directories
    curr_dir = '../datasets/'
    if forward:
        curr_dir = curr_dir + 'forward'
    else:
        curr_dir = curr_dir + 'retro'

    rktnts_path, prdcts_path = curr_dir + '/' + kind + '/' + 'rctnts/', curr_dir + '/' + kind + '/' + 'prdcts/'
    if not os.path.exists(rktnts_path):
        os.makedirs(rktnts_path)

    if not os.path.exists(prdcts_path):
        os.makedirs(prdcts_path)

    prev_perc, i = None, 0
    numrkns_total = len(prdcts_all)

    rktnts_array, prdcts_arr = None, None

    # Iterate over the reactions consisting: (reactants, product)
    rkn_cnt = 0
    for rktnts, prdcts, i in zip(rktnts_all, prdcts_all, range(0, numrkns_total)):

        rktnts = order_reactants(rktnts)  # order reactants
        curr_reaction = (rktnts, prdcts)
        res = encode_reaction(curr_reaction, parse_trees.ZincGrammarModel(), maxlen_reactants, maxlen_products)

        if not res:
            LOGGER.warning('skipped reaction: {}'.format(i + 1))
            continue

        rseq, pseq = res

        # pad zeros
        react_padlen, prdct_padlen = max(0, maxlen_reactants - rseq.shape[0]), max(0, maxlen_products - pseq.shape[0])
        rseq, pseq = numpy.pad(rseq, (0, react_padlen)), numpy.pad(pseq, (0, prdct_padlen))

        if rktnts_array is None:
            rktnts_array, prdcts_arr = rseq, pseq
        else:
            rktnts_array, prdcts_arr = numpy.vstack([rktnts_array, rseq]), numpy.vstack([prdcts_arr, pseq])

        # print the status of the current iteration
        perc = int((i + 1) / numrkns_total * 100)
        if perc % 10 == 0:
            if not prev_perc == perc:
                print('#' * perc, '||', '{}%'.format(perc))

        # Store the arrays and flush the memory after every 100 reactions
        if (i + 1) % 100 == 0:
            LOGGER.warning('Writing the results upto i = : {}'.format(i))

            numpy.savez_compressed(rktnts_path + 'rktnts_array_{}.npz'.format(i), rktnts_array)
            numpy.savez_compressed(prdcts_path + 'prdcts_arr_{}.npz'.format(i), prdcts_arr)

            LOGGER.warning("Finished writing the .npz files! Flushing the memory...")

            rkn_cnt += prdcts_arr.shape[0]
            LOGGER.warning('Num reactions: {}'.format(rkn_cnt))

            rktnts_array, prdcts_arr = None, None

        prev_perc = perc

    LOGGER.warning('Writing the results upto i = : {}'.format(i))

    numpy.savez_compressed(rktnts_path + 'rktnts_array_{}.npz'.format(i), rktnts_array)
    numpy.savez_compressed(prdcts_path + 'prdcts_arr_{}.npz'.format(i), prdcts_arr)

    LOGGER.warning("Finished writing the .npz files! Flushing the memory...")

    return


encode_dataset_reaction('../datasets/forward/valid.txt', kind='valid')
