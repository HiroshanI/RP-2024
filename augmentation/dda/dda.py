import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')
import random
import gensim
import spacy_udpipe
import re

import requests
from tqdm import tqdm
stop_words = list(set(nltk.corpus.stopwords.words('english')))

class DDA:
    def __init__(self, wv_path, glove=False):
        if glove:
            self.wv_from_text = gensim.models.KeyedVectors.load_word2vec_format(
                wv_path, 
                binary=False, 
                no_header=True
            )
        else:
            self.wv_from_text = gensim.models.KeyedVectors.load_word2vec_format(
                wv_path, 
                binary=False
            )

    # RANDOM SWAP ---------------------------------------------------
    def swap_word(new_words):
        random_idx_1 = random.randint(0, len(new_words) - 1)
        random_idx_2 = random_idx_1
        counter = 0
        while random_idx_2 == random_idx_1:
            random_idx_2 = random.randint(0, len(new_words) - 1)
            counter += 1
            if counter > 3:
                return new_words
        new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
        return new_words
    
    def random_swap(self, words, n):
        new_words = words.copy()
        for _ in range(n):
            new_words = swap_word(new_words)
        return new_words

    # RANDOM INSERTION ---------------------------------------------------
    def add_word(new_words):
        synonyms = []
        counter = 0

        while len(synonyms) <1:
            random_word = new_words[random.randint(0, len(new_words)-1)]
            #synonyms = self.synonyms_cadidates(random_word, self.df)
            synonyms = list(get_synonyms_vec(random_word))
            counter += 1
            if counter > 10:
                return

        random_synonym = synonyms[0]
        random_idx = random.randint(0, len(new_words)-1)
        new_words.insert(random_idx, random_synonym)

    def random_insertion(self, words, n):
        new_words = words.copy()
        for _ in range(n):
            add_word(new_words)
        return new_words

    # RANDOM DELETION ---------------------------------------------------
    def random_deletion(self, words, p):
        """
        Randomly delete words from a sentence with probability p
        :param words:
        :param p:
        :return:
        """
        if len(words) == 1:
            return words
        new_words = []
        for word in words:
            r = random.uniform(0, 1) # random number between 0.0 and 1.0
            if r > p: #kinda elegant when you think about it
                new_words.append(word)
        #if you end up deleting all words, just return a random word
        if len(new_words) == 0:
            rand_int = random.randint(0, len(words)-1)
            return [words[rand_int]]

        return new_words

    # VECTOR-BASED SYNONYM REPLACEMENT ----------------------------------
    def get_synonyms_vec(self, word):
        synonyms = set()
        flag = False
        vec = None
        try:
            vec = self.wv_from_text.similar_by_word(word.lower())
        except KeyError:
            flag = True
            pass
        if flag is False:
            synonyms.add(vec[0][0])
        if word in synonyms:
            synonyms.remove(word)
        return synonyms
    
    def synonym_replacement_vec(self, words, n, wv_path):
        new_words = words.copy()
        random_word_list = list(set([word for word in words if word not in stop_words]))
        random.shuffle(random_word_list)
        num_replaced = 0
        for random_word in random_word_list:
            synonyms = get_synonyms_vec(random_word, wv_path)
            if len(synonyms) >= 1:
                synonym = random.choice(list(synonyms))
                new_words = [synonym if word.lower() == random_word else word for word in new_words]
                # print("replaced", random_word, "with", synonym)
                num_replaced += 1
            if num_replaced >= n:  # only replace up to n words
                break
        # this is stupid but we need it, trust me
        sentence = ' '.join(new_words)
        new_words = sentence.split(' ')
        return new_words

    def augmentation(self, sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, alpha_rd=0.1, num_aug=9):
        """
        @param sentence
        @param alpha_sr synonym replacement rate, percentage of the total sentence
        @param alpha_ri random insertion rate, percentage of the total sentence
        @param alpha_rs random swap rate, percentage of the total sentence
        @param alpha_rd random deletion rate, percentage of the total sentence
        @param num_aug how many augmented sentences to create

        @return list of augmented sentences
        """
        words_list = sentence.split(' ')  # list of words in the sentence
        words = [word for word in words_list if word != '']  # remove empty words
        num_words = len(words_list)  # number of words in the sentence

        augmented_sentences = []
        num_new_per_technique = int(num_aug / 4) + 1 # number of augmented sentences per technique
    
        #synonmym replacement
        if (alpha_sr > 0):
            n_sr = max(1, int(alpha_sr * num_words)) # number of words to be replaced per technique
            #print("Number of words to be replaced per technique: ", n_sr)
            for _ in range(num_new_per_technique):
                a_words = synonym_replacement_vec(words, n_sr)
                augmented_sentences.append(' '.join(a_words))
        #random insertion
        if (alpha_ri > 0):
            n_ri = max(1,int(alpha_ri * num_words))
            for _ in range(num_new_per_technique):
                a_words = random_insertion(words, n_ri)
                augmented_sentences.append(' '.join(a_words))
        #Random Deletion
        if (alpha_rd > 0):
            for _ in range(num_new_per_technique):
                a_words = random_deletion(words, alpha_rd)
                augmented_sentences.append(' '.join(a_words))
        #Random Swap
        if (alpha_rs > 0):
            n_rs = max(1, int(alpha_rs * num_words))
            for _ in range(num_new_per_technique):
                a_words = random_swap(words, n_rs)
                augmented_sentences.append(' '.join(a_words))

        return augmented_sentences
