# -*- coding: UTF-8 -*-

_author_ = "vm2583@columbia.edu"

import nltk
import numpy as np
import preprocess.grammar as gram
import logging as logging
LOGGER = logging.getLogger(__name__)


def get_tokenizer(cfg):
    long_tokens = filter(lambda a: len(a) > 1, cfg._lexical_index.keys())
    long_tokens = list(long_tokens)
    replacements = ['$', '%', '^']  # ,'&']

    assert len(list(long_tokens)) == len(replacements)
    for token in replacements:
        assert not (token in cfg._lexical_index)

    def tokenize(smiles):
        for i, token in enumerate(long_tokens):
            smiles = smiles.replace(token, replacements[i])
        tokens = []
        for token in smiles:
            try:
                ix = replacements.index(token)
                tokens.append(long_tokens[ix])
            except:
                tokens.append(token)
        return tokens

    return tokenize

def pop_or_nothing(S):
    try:
        return S.pop()
    except:
        return 'Nothing'

def prods_to_eq(prods):
    seq = [prods[0].lhs()]
    for prod in prods:
        if str(prod.lhs()) == 'Nothing':
            break
        for ix, s in enumerate(seq):
            if s == prod.lhs():
                seq = seq[:ix] + list(prod.rhs()) + seq[ix + 1:]
                break
    try:
        return ''.join(seq)
    except:
        return ''


class ZincGrammarModel(object):
    def __init__(self):
        self._grammar = gram
        self._productions = self._grammar.GCFG.productions()
        self._prod_map = {}
        for ix, prod in enumerate(self._productions):
            self._prod_map[prod] = ix
        self._parser = nltk.ChartParser(self._grammar.GCFG)
        self._tokenize = get_tokenizer(self._grammar.GCFG)
        self._n_chars = len(self._productions)
        self._lhs_map = {}
        for ix, lhs in enumerate(self._grammar.lhs_list):
            self._lhs_map[lhs] = ix

    def encode(self, reaction, MAX_LEN_r, MAX_LEN_p):
        self.MAX_LEN_r = MAX_LEN_r
        self.MAX_LEN_p = MAX_LEN_p

        p, r = reaction[1], reaction[0]
        assert type(p) == list
        assert type(r) == list

        tokens_p = map(self._tokenize, p)
        tokens_p = list(tokens_p)

        tokens_r = map(self._tokenize, r)
        tokens_r = list(tokens_r)

        remov_idx_p, remov_idx_r = [], []

        # encode PRODUCTS
        for t, idx in zip(tokens_p, range(len(tokens_p))):
            try:
                parse_trees = [self._parser.parse(t).__next__() for t in [t]]
                productions_seq = [tree.productions() for tree in
                                   parse_trees]  # extracts the productions sequences for the parse trees
                indices = [np.array([self._prod_map[prod] for prod in entry], dtype=int) for entry in productions_seq]

                if len(indices[0]) >= self.MAX_LEN_p:
                    remov_idx_p.append(idx)
            except:
                remov_idx_p.append(idx)

        for index in sorted(remov_idx_p, reverse=True):
            print('removed: {}'.format(p[index]))
            del tokens_p[index]

        if len(tokens_p) == 0:
            return False

        parse_trees = [self._parser.parse(t).__next__() for t in tokens_p]  # generates an NLTK tree
        productions_seq = [tree.productions() for tree in parse_trees]  # extracts the production sequence


        indices = [np.array([self._prod_map[prod] for prod in entry], dtype=int) for entry in productions_seq]

        # add '80' in the end to indicate end of representation
        # add 1 to indices so that the first rule starts from 1 (and not 0)
        indices = [np.append(ind,gram.D-1) for ind in indices]
        ind_prod = [ind+1 for ind in indices]

        # encode REACTANTS
        for t, idx in zip(tokens_r, range(len(tokens_r))):
            try:
                parse_trees = [self._parser.parse(t).__next__() for t in [t]]
                productions_seq = [tree.productions() for tree in
                                   parse_trees]  # extracts the productions sequences for the parse trees
                indices = [np.array([self._prod_map[prod] for prod in entry], dtype=int) for entry in productions_seq]

                # check if the length of rules exceed the max len
                if len(indices[0]) >= self.MAX_LEN_p:
                    remov_idx_r.append(idx)
            except:
                remov_idx_r.append(idx)

        for index in sorted(remov_idx_r,
                            reverse=True):  # deleting in reverse order so as not to throw incorrect indices
            print('removed: {}'.format(r[index]))
            del tokens_r[index]

        if len(tokens_r) == 0:
            return False

        parse_trees = [self._parser.parse(t).__next__() for t in
                       tokens_r]  # generates an NLTK tree: Nice Graphic as well!

        productions_seq = [tree.productions() for tree in
                           parse_trees]  # extracts the productions sequences for the parse trees
        indices = [np.array([self._prod_map[prod] for prod in entry], dtype=int) for entry in productions_seq]

        # add '80' in the end to indicate end of representation
        # add 1 to indices so that the first rule starts from 1 (and not 0)
        indices = [np.append(ind,gram.D-1) for ind in indices]
        ind_react = [ind+1 for ind in indices]

        # concatenate the reactants together
        ind_react_concat = np.concatenate(ind_react)

        # check if len exceeds a given threshold, skip the reaction
        if ind_react_concat.shape[0]>self.MAX_LEN_r:
            LOGGER.warning('Reaction length exceeded. Skipping...')
            return False

        # check if len exceeds a given threshold, skip the reaction
        if ind_prod[0].shape[0]>self.MAX_LEN_p:
            LOGGER.warning('Reaction length exceeded. Skipping...')
            return False

        return ind_react_concat, ind_prod[0]