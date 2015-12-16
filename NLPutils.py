"""
NLP-Deep Text Processing Utility Functions

Inspired from Michael Fire's notebook
'dato.com/learn/gallery/notebooks/deep_text_learning.html'
"""

from numpy import average
import os
import gensim
import re
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import numpy as np
import graphlab as gl


def load_json_from_file(filename):
    """
    Load JSON from a file.
    INPUT:  filename  Name of the file to be read.
    RETURN: Output SFrame
    """
    # Read the entire file into a SFrame with one row
    sf = gl.SFrame.read_csv(filename, delimiter='\n', header=False)
    
    # The dictionary can be unpacked to generate the individual columns.
    sf = sf.unpack('X1', column_name_prefix='')
    return sf


class TrainSentences(object):
    """
    Iterator class that returns Sentences from texts files in a input directory
    """
    RE_WIHTE_SPACES = re.compile("\s+")
    STOP_WORDS = set(stopwords.words("english"))
    
    def __init__(self, filename):
        """
        Initialize a TrainSentences object with a input filename that contains text files for training
        :param filename: file name which contains the text       
        """
        self.filename = filename

    def __iter__(self):
        """
        Sentences iterator that return sentences parsed from files in the input directory.
        Each sentences is returned as list of words
        """
        # read line from file (Without reading the entire file)
        for line in file(self.filename, "r"):
            # split the read line into sentences using NLTK
            for sent in txt2sentences(line):
                # split the sentence into words using regex
                w =txt2words(sent,
                             lower=True,
                             remove_stop_words=False,
                             remove_none_english_chars=True)
                
                #skip short sentences with less than 3 words
                if len(w) < 3:
                    continue
                yield w
                
def txt2sentences(txt, remove_none_english_chars=True):
    """
    Split the English text into sentences using NLTK
    :param txt: input text.
    :param remove_none_english_chars: if True then remove non-english chars from text
    :return: string in which each line consists of single sentence from the original input text.
    :rtype: str
    """
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    # split text into sentences using nltk packages
    for s in tokenizer.tokenize(txt):
        if remove_none_english_chars:
            #remove none English chars
            s = re.sub("[^a-zA-Z]", " ", s)
        yield s

        
def txt2words(txt, lower=True, remove_none_english_chars=True, remove_stop_words=True):
    """
    Split text into words list
    :param txt: the input text
    :param lower: if to make the  text to lowercase or not.
    :param remove_none_english_chars: if True then remove non-english chars from text
    :param remove_stop_words: if True then remove stop words from text
    :return: words list create from the input text according to the input parameters.
    :rtype: list
    """
    if lower:
        txt = txt.lower()
    if remove_none_english_chars:
        txt = re.sub("[^a-zA-Z]", " ", txt)

    words = TrainSentences.RE_WIHTE_SPACES.split(txt.strip().lower())
    
    if remove_stop_words:
        #remove stop words from text
        words = [w for w in words if w not in TrainSentences.STOP_WORDS]
    return words


class DeepTextAnalyzer(object):
    def __init__(self, word2vec_model):
        """
        Construct a DeepTextAnalyzer using the input Word2Vec model
        :param word2vec_model: a trained Word2Vec model
        """
        self._model = word2vec_model

    def txt2vectors(self,txt):
        """
        Convert input text into an iterator that returns the corresponding vector representation of each
        word in the text, if it exists in the Word2Vec model
        :param txt: input text
        :return: iterator of vectors created from the words in the text using the Word2Vec model.
        """
        words = txt2words(txt, lower=True, remove_none_english_chars=True, remove_stop_words=True)
        words = [w for w in words if w in self._model]
        if len(words) != 0:
            for w in words:
                yield self._model[w]


    def txt2avg_vector(self, txt):
        """
        Calculate the average vector representation of the input text
        :param txt: input text
        :return the average vector of the vector representations of the words in the text  
        """
        vectors = self.txt2vectors(txt)
        vectors_sum = next(vectors, None)
        if vectors_sum is None:
            return None
        count =1.0
        for v in vectors:
            count += 1
            vectors_sum = np.add(vectors_sum,v)
        
        #calculate the average vector and replace +infy and -inf with numeric values 
        avg_vector = np.nan_to_num(vectors_sum/count)
        return avg_vector
    

def print_statistics(result):
    print "*" * 30
    print "Accuracy        : ", result["accuracy"]
    print "Precision       : ", result['precision']
    print "Recall          : ", result['recall']
    print "AUC             : ", result['auc']
    print "Confusion Matrix: \n", result["confusion_matrix"]