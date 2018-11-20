import re
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import numpy as np

def main():
    corpus, y = prepare_data('trainreviews.txt')
    run_with_word_embedding(corpus, y)
    #run_with_bag_of_words(corpus, y)


# Purpose: runs classifiers with bag of words
#
#
def run_with_bag_of_words(corpus, y):
    print('Running with bag of words...\n\n')
    X = get_bag_of_words(corpus)
    X_train, X_validate, y_train, y_validate = \
        train_test_split(X, y, test_size = .33, shuffle = True, stratify = y)
    print(X_train, y_train, X_validate, y_validate)

# Purpose: runs classifiers with word embeddings
#
#
def run_with_word_embedding(corpus, y):
    print('Running with word embeddings...\n\n')
    X = get_word_embedding(corpus)
    X_train, X_validate, y_train, y_validate = \
        train_test_split(X, y, test_size = .33, shuffle = True, stratify = y)
    print(X_train, y_train, X_validate, y_validate)

# Purpose: Runs SVM, Logistic Regression, and bagging
# Pre-Condition: y is a np.array
# 
def get_bag_of_words(corpus):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus).toarray()
    return X
    # check that these are correct later

#
#
#
def get_word_embedding(corpus):
    X = []
    word_dict = gloVe_to_dict()
    for line in corpus:
        word_vector = np.zeros(50)
        for word in line.split():
            print(word)
            word_vector = np.add(word_vector, word_dict[word])
        print(word_vector)
        X.append(word_vector)
    return np.array(X)

def run_naive_bayes():
    pass

def run_log_reg():
    pass

def run_SVM():
    pass

def run_decision_tree():
    pass

# Purpose: Takes a text file with \t separated class labels from data 
#          instances
# Pre-condition: filename is the name of a file with a sentences
#                separated by a \t from a class label 
# Returns:
#          corpus:
#                type: string list
#                purpose: holds strings for NLP training
#                         parallel to class_labels
#          class_labels:
#                type: int list
#                purpose: holds class labels for corpus.
#                         Parallel to corpus
def prepare_data(filename):
    corpus = []                    # holds instance words
    class_labels = []              # holds instance class_labels
    with open(filename, 'r') as f:
        for line in f:
            instance = line.split('\t')
            corpus.append(instance[0])     
            class_labels.append(int(instance[1].replace('\n', '')))
    return corpus, np.array(class_labels)

# Purpose: Turns gloVe.6B.50d.txt into a dictionary for use in training
# Pre-Condition: gloVe.6B.50d.txt is in the working directory and 
# Returns:
#          word_embeddings:
#                type: dictionary
#                Purpose: holds word word-vector pairs for
#                         use in processing training data
#                keys: words values: 50d array of floats

#
#
#
def get_strat_folds(X, y):
    train = []
    validate = []
    skf = StratifiedKFold(n_splits = 5, shuffle = True)
    
    for train_indices, validate_indices in skf.split(X, y):
        train.append([X[train_indices], y[train_indices]])
        validate.append([X[validate_indices], y[validate_indices]])
    
    return train, validate

#
#
#
def gloVe_to_dict():
    print('Loading gloVe....')
    word_embeddings = dict()
    with open('glove.6B.50d.txt') as f:
        for line in f:
            embedding = line.split()
            word = embedding[0]
            vector = embedding[1:]
            word_embeddings[word] = vector
    return word_embeddings

if __name__ == '__main__':
    main()
