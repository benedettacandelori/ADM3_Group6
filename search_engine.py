from bs4 import BeautifulSoup
import requests
import os.path
import re
import pandas as pd
import numpy as np
import csv
import json
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
from collections import Counter
from IPython.core.display import HTML
import time
import nltk
import heapq

# custom libraries
from data_collection import *
from tsv_management import *

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer




########################################################################################################################################################
#                                                                      search_engine.py                                                                #
#                                                                                                                                                      #
#       library of function useful to process every document for the search, initialize the search engine, and perform the actual search               #
#                                                                                                                                                      #
########################################################################################################################################################


##########################################################################
#                                                                        #
#     functions to preprocess the text to have it ready for a search     #
#                                                                        #
##########################################################################

def preprocessing(text):
    '''
    this function preprocesses a string to prepare it either for the inverted
    index creation or for the search of the query
    in details, here we:
        - tokenize the string (with a regex tokenizer)
        - convert the words to lowercase
        - remove the english stopwords
        - stem the words using Porter stemmer
    
    input: string to preprocess
    output: preprocessed list of words
    '''
    
    # initialize tokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    
    # initialize the stemmer
    porter = PorterStemmer()
    
    # tokenize the text
    word_list = tokenizer.tokenize(text)
    
    processed_text = []
    for word in word_list:
        if word.lower() not in stopwords.words('english'):
            stemmed_word = porter.stem(word)
            processed_text.append(stemmed_word)
    
    return(processed_text)


def list_preprocessing( list_of_strings ):
    '''
    this function preprocesses every string in a list of strings
    input: list of strings
    output: None
    non-explicit output: every string in the list is preprocessed and becomes a list of words
    '''
    
    for i in tqdm(range(len(list_of_strings)), desc="Preprocessing documents"):
        if list_of_strings[i] is not None:
            list_of_strings[i] = preprocessing(list_of_strings[i])
        else:
            list_of_strings[i] = []
    
    return


##########################################################################
#                                                                        #
#     functions to create and manage a vocabulary that maps each word    #
#      of our anime descriptions to an integer number                    #
#                                                                        #
##########################################################################

def vocabulary_generation():
    '''
    NOTE: there are two different tqdm progress bar called in this function
    
    this function generates a vocabulary using all the anime descriptions
    and saves everything in a json file
    '''

    # retrieving the descriptions of all the anime
    description = column_retrieval('animeDescription')
    
    # preprocessing every description
    list_preprocessing( description )
    
    # generating a vocabulary of words that associates every word to an integer
    vocabulary = vocabulary_creation(description)
    
    # saving the vocabulary to the disk
    with open("content/vocabulary.json", 'w') as f:
        json.dump(vocabulary, f)
    
    return


def vocabulary_creation(list_of_lists):
    '''
    here we create a vocabulary of all words from a list of lists of words
    input: a list that contains lists of words
    output: dictionary that associates words to integers starting from 0
    '''
    
    # initializing the set of all the words
    set_of_words = set()
    
    for words_list in list_of_lists:
        # adding the words to the set of all the words
        set_of_words.update(words_list)
    
    
    # initializing the vocabulary
    vocabulary = {}
    
    for i in range(len(set_of_words)):
        # assigning to a random word the value i
        vocabulary[ set_of_words.pop() ] = i
    
    return( vocabulary )


def vocabulary_retrieval():
    '''
    this function reads the vocabulary from the disk
    and returns it as a dictionary
    
    input: None
    output: vocabulary dictionary
    '''
    
    term_dict = json_loading('./content/vocabulary.json')
    
    return(term_dict)


def json_loading(path):
    '''
    this function parses a json file in path and returns it
    input: json file path
    output: data retrieved from the json file
    '''
    
    with open(path) as json_file:
        data = json.load(json_file)
        
    return(data)


def vocabulary_inversion(vocabulary):
    '''
    reverses the input vocabulary
    
    input: dictionary {key1:value1, ...}
    output: dictionary {value1:key, ...}
    '''
    
    inverted_vocabulary = {value : key for (key, value) in vocabulary.items()}
    
    return(inverted_vocabulary)


def vocabulary_conversion(words_list, vocabulary):
    '''
    this function converts a list of words according to a certain vocabulary
    
    input: (list of words to convert, vocabulary)
    output: list of word ids according to the vocabulary
    '''
    
    ids = []
    
    for word in words_list:
        if word in vocabulary.keys():
            ids.append(vocabulary[word])
        else:
            ids = []
            break
        
    return(ids)


#################################################################################################################################################################################
#                                                                                                   #############################################################################
#       the following set of functions are used in the simple unranked search engine                #############################################################################
#       which only performs a conjunctive search on the words of the query                          #############################################################################
#                                                                                                   #############################################################################
#################################################################################################################################################################################


##########################################################################
#                                                                        #
#           functions to create and manage the inverted index            #
#                                                                        #
##########################################################################

def unranked_inverted_index_creation(list_of_documents, vocabulary):
    '''
    this function builds an inverted index using a list of documents and a vocabulary
    
    NOTE: for simplicity of search, every word in our inverted index will 
          belong to a dummy 0 document that contains every word
          (our anime documents are indexed starting from 1)
          
    NOTE: because we only consider document_ids in increasing order,
          our inverted index is automatically sorted
    
    input: (list of the (preprocessed) documents, vocabulary of terms)
    output: list containing the inverted index
    
    NOTE: inverted_index[i] will be the inverted list
          associated with the word vocabulary[i]
    '''
    
    number_of_words = len(vocabulary)
    
    # initializing the inverted index list with lists that contain 0
    inverted_index = []
    for i in range(number_of_words):
        inverted_index.append(list([0]))
    
    for i in tqdm(range(len(list_of_documents)), desc="Building the inverted index"):
        # our documents start from 1
        document_id = i+1
        
        document = list_of_documents[i]
        
        for word in document:
            # converting the word to its id according to the vocabulary
            word_id = vocabulary[word]
            
            if document_id not in inverted_index[word_id]:    # if the document id isn't already associated to the current word id
                inverted_index[word_id].append(document_id)   # then we add it to the corresponding list

                    
    return ( inverted_index )


def unranked_inverted_index_generation():
    '''
    NOTE: there are three different tqdm progress bar called in this function
    
    this function generates an inverted index using all the anime descriptions
    and saves everything in a json file
    '''

    # retrieving the descriptions of all the anime
    description = column_retrieval('animeDescription')
    
    # processing every every description
    list_preprocessing( description )
    
    # retrieving the vocabulary from the disk
    vocabulary = vocabulary_retrieval()
    
    # generating an inverted index list
    inverted_index = unranked_inverted_index_creation(description, vocabulary)
    
    # saving the inverted index to the disk
    with open("content/unranked_inverted_index.json", 'w') as f:
        json.dump(inverted_index, f)
    
    return


def unranked_inverted_index_retrieval():
    '''
    this function reads the unranked inverted index from the disk
    and returns it as a list
    
    input: None
    output: inverted index list
    '''

    inverted_index = json_loading('./content/unranked_inverted_index.json')
    
    return(inverted_index)


###############################################################################################
#                                                                                             #
#           functions used to intersect two or more elements of the inverted index            #
#                                                                                             #
###############################################################################################

def intersection_pointers(inverted_words):
    '''
    NOTE: this function assumes that exists a 'universal' document indexed by 0
          so that the intersection will never be empty
          and we won't have to do several check on the list lengths
    
    computes the intersection on the elements of the inverted index
    input: list of ordered lists of integers
    output: a list containing the intersection among the elements of the input lists
    
    NOTE: this algorithm compares the last element of every list instead of the first
          so that the last element (which is the first of every list) will always be a match
    '''
    
    number_of_words = len(inverted_words)
    
    # an array of indices that points to the last element of every list in inverted_words
    pointers = list( map(lambda x: len(x) - 1, inverted_words) )
    
    # creating output set
    intersection = []
    
    # j will the index used to navigate the elments of inverted_words
    
    while( pointers[0] >= 0):   # the algorithm stops when the first list has been scanned completely
        
        current_element = inverted_words[0][pointers[0]] # we always start comparing the pointed element of the first list
        j = 1                                            # with the pointed element of the second list
        
        
        while( j < number_of_words):  # this cycle only ends when a common element is found
                                      # thus the need for the common 0 element
            
            if current_element > inverted_words[j][ pointers[j] ]:                        # if the pointed element of this list is smaller than the current element
                current_element = decrement_some_pointers(inverted_words, j, pointers)    # then I decrement all the previous lists' pointers to match this one
                j = 1                                                                     # and I restart the cycle from the second list
                
            elif current_element < inverted_words[j][ pointers[j] ]:                           # if the pointed element of this list is bigger than the current element
                j += decrement_one_pointers(inverted_words[j], current_element, pointers, j)   # then I need to decrement the pointer of the current list, and if after decrementing
                                                                                               # the pointed element coincides with the current element I go on with the cycle
                                                                                               # otherwise I repeat this cycle and I fall back in the previous if case
                        
            else:       # if the pointed element of this list is equal to the current element
                j+=1    # I go on with the cycle
        
        
                                              # I arrive here only if current_element is in every list
        intersection.append(current_element)  # so I add it to the solution
        decrement_all_pointers(pointers)      # and I remove it from the lists
    
    return (intersection)


def decrement_one_pointers(inverted_word, document, pointers, i):
    '''
    input: (ordered list of integer, integer, list of integer, integer)
    output: 1 if the integer document is in the list, 0 otherwise
    non-explicit output: the function will decrement the value of pointers[i] until
                         inverted_words[i][ pointers[i] ] <= document
    '''
    
    # removes all the elements bigger than document
    while(inverted_word[ pointers[i] ] > document ):
        pointers[i] -= 1
    
    # checks if document is in inverted_word
    if(inverted_word[ pointers[i] ] == document):
        return(1)
    else:
        return(0)
    
    
def decrement_some_pointers(inverted_words, word, pointers):
    '''
    input: (list of lists of ordered integers, integer, list of integers)
    output: the greatest element of inverted_words[0] that
            is smaller or equal than inverted_words[word][ pointers[word] ]
    non-explicit output: takes the first 1 to word lists of inverted_words
                         and decrements their pointer until every pointed element
                         is smaller or equal than inverted_words[word][ pointers[word] ]
    '''
    
    document = inverted_words[word][ pointers[word] ]
    
    for i in range(word):
        decrement_one_pointers(inverted_words[i], document, pointers, i)   # decrements pointers[i] until
                                                                           # inverted_words[i][ pointers[i] ] <= document
    
    return(inverted_words[0][ pointers[0] ])

    
def decrement_all_pointers(pointers):
    '''
    input: list of lists of integers
    output: None
    non-explicit output: decrements every element in the list by 1
    '''
    
    for i in range(len(pointers)):
        pointers[i] -= 1   # decrement by one the i^th pointer
    
    return


###############################################################################################
#                                                                                             #
#    functions used to initialize the unranked search engine and perform the actual search    #
#                                                                                             #
###############################################################################################


def unranked_search_engine_initialization():
    '''
    initialize the unranked search engine by retrieving the inverted index,
    the vocabulary and the anime informations from the disk
    
    input: None
    output: (vocabulary, inverted index)
    '''
    
    # retrieve vocabulary from disk
    vocabulary = vocabulary_retrieval()
    
    # retrieve inverted index from disk
    inverted_index = unranked_inverted_index_retrieval()
    
    return(vocabulary, inverted_index)


def unranked_search(vocabulary, inverted_index):
    '''
    this is the actual search engine:
    given a query, it will print some brief information about the matching anime
    
    input: vocabulary of words, inverted index
    output: None
    non-explicit input: the function will ask for a search query
    non-explicit output: the function will print some brief information about the anime that matches the query
    '''
    
    # retrieving the search query
    query = input('Input a query:')
    print('')
    
    # preprocessing the query
    query = preprocessing(query)
    
    # converting the words in the query into id
    query_ids = vocabulary_conversion(query, vocabulary)
    
    # retrieving query inverted lists
    inverted_words = []
    for i in query_ids:
        inverted_words.append(inverted_index[i])
    
    if len(inverted_words) > 0:
        # searching query match
        search_results = intersection_pointers(inverted_words)
        print(f'{len(search_results)-1} search results found!\n')
    else:
        search_results = [0]
    
    # printing results
    unranked_search_result_printing(search_results, 'animeTitle', 'animeDescription', 'Url')
    
    return
    
    
def unranked_search_result_printing(anime_idx, *columns):
    '''
    this function displays the columns information about the anime in anime_idx
    it prints a 'Nothing Found' message if the only element in anime_idx is 0
    
    input: list of integer indices, column informations to display
    output: None
    non-explicit output: prints the search results
    '''
    
    # removing the dummy 0 anime document
    anime_idx.remove(0)
    
    # taking care of an empty search result
    if len(anime_idx) == 0:
        print("Couldn't find the query in the document. Try changing the terms or using less words.")
        return
    
    # retrieving information about the matching anime
    information_df = pd.DataFrame(columns = columns)
    for i in anime_idx:
        information_df.loc[i] = anime_information_retrieval(i, columns)
    
    # print the informations on screen
    display(HTML(information_df.to_html(index = False)))
    
    return


#################################################################################################################################################################################
#                                                                                                   #############################################################################
#       the following set of functions are used in the ranked search engine which performs          #############################################################################
#       a conjunctive search and ranks the results based on the similarity with the query           #############################################################################
#                                                                                                   #############################################################################
#################################################################################################################################################################################


################################################################################################
#                                                                                              #
#           functions to create and manage the inverted index and the tfidf vectors            #
#                                                                                              #
################################################################################################

def ranked_inverted_index_creation(corpus, vocabulary, unranked_inverted_index):
    '''
    this function creates the inverted index with the associated tfidf
    and the vector of tfidf for all the documents
    
    NOTE: for simplicity of search, every word in our inverted index will 
          belong to a dummy 0 document with a zero tfidf
          
    NOTE: because we only consider document_ids in an increasing fashion,
          our inverted index is automatically sorted
    
    input: list of anime description, vocabulary, unranked_inverted_index
    output: (inverted index dictionary of the form {word : [(doc, tfidf), (doc, tfidf), ...], ...},
             tfidf vector dictionary of the form {doc_id: {word_id : tfidf, word_id : tfidf, ...}, ...})
    '''
    
    # total number of documents
    N = len(corpus)
    
    # inverted index dictionary
    # we insert the tuple (0,0) in every word
    # for convenience in searching
    inverted_tf_idf = defaultdict(lambda: [(0,0)])
    
    # tfidf vectors dictionary
    tf_idf = defaultdict(dict) # {doc_id:{word_id:tfidf}}
    
    for i in tqdm(range(N), desc = 'Ranked inverted index generation'):
        document = corpus[i]
        doc_id = i+1
        
        # total number of words in the document
        words_count = len(document)
        
        # dictionary of occurrences of every word in the document
        counter = Counter(document)
        
        for word in np.unique(document):
            word_id = vocabulary[word]
            
            # tf = # occurrences of word / total number of words in the document
            tf = counter[word]/words_count
            
            # df = occurrences of (word in document) across all documents
            # is equal to the length of the corresponding inverted list (removing the 0 document)
            df = len(unranked_inverted_index[word_id])-1
            
            # idf = log10(#number of document / occurrences of (word in document) across all documents)
            idf = np.log10(N/df)
            
            tf_idf[doc_id][word_id] =  tf*idf
            inverted_tf_idf[word_id].append((doc_id,tf*idf))
            
    
    return(inverted_tf_idf, tf_idf)


def ranked_inverted_index_generation():
    '''
    NOTE: there are three different tqdm progress bar called in this function
    
    this function generates the tfidf vectors of all the documents and the ranked inverted index
    using all the anime descriptions, and saves everything in two json files
    '''
    
    # retrieving the descriptions of all the anime
    description = column_retrieval('animeDescription')
    
    # processing every every description
    list_preprocessing( description )
    
    # retrieving the vocabulary from the disk
    vocabulary = vocabulary_retrieval()
    
    # generating the ranked inverted index
    # and the tfidf vectors
    inverted_index, tfidf = ranked_inverted_index_creation(description, vocabulary, unranked_inverted_index_retrieval())
    
    # saving the tfidf vectors to the disk
    with open("content/tfidf_vectors.json", 'w') as f:
        json.dump(tfidf, f)
    
    # saving the inverted index to the disk
    with open("content/ranked_inverted_index.json", 'w') as f:
        json.dump(inverted_index, f)
    
    return


def ranked_inverted_index_retrieval():
    '''
    this function reads the inverted index from the disk
    and returns it as a dictionary
    
    input: None
    output: inverted index dictionary
    '''

    inverted_index = json_loading('./content/ranked_inverted_index.json')
    
    # because the json loading parses our integer keys as strings
    # we need to cast them
    for key in list(inverted_index.keys()):
        inverted_index[int(key)] = inverted_index.pop(key)
    
    return(inverted_index)


def tfidf_vector_retrieval():
    '''
    this function reads the tfidf vectors from the disk
    and returns it as a dictionary
    
    input: None
    output: tfidf vectors dictionary
    '''

    tfidf = json_loading('./content/tfidf_vectors.json')
    
    # because the json loading parses our integer keys as strings
    # we need to cast them
    for key in list(tfidf.keys()):
        
        for inner_key in list(tfidf[key].keys()):
            tfidf[key][int(inner_key)] = tfidf[key].pop(inner_key)
            
        tfidf[int(key)] = tfidf.pop(key)
    
    return(tfidf)


def vectorize_query(query_ids, inverted_words, total_documents):
    '''
    this function generates the vector of tfidf relative to the query
    
    input: (list of word id in the query,
            slice of inverted index needed to compute the tfidf,
            total number of documents)
            
    output: vector of tfidf relative to the query, it will be a dictionary
            of the form {word1 : tfidf1query, word2 : tfidf2query, ...}
    '''
    
    # total number of words in the query
    words_count = len(query_ids)
    
    # initializing output vector
    tfidf_vector = {}
    
    # counting the occurrences of every word in the query
    occurrences_count = Counter(query_ids)
    
    for word in occurrences_count.keys():
        # calculating the tfidf as   -----------------------idf_word-------------------- * -----------tf_word,query-----------
        tfidf_vector[word] =         np.log10(total_documents/len(inverted_words[word])) * occurrences_count[word]/words_count
    
    return(tfidf_vector)


###############################################################################################
#                                                                                             #
#           functions used to intersect two or more elements of the inverted index            #
#                                                                                             #
###############################################################################################

def ranked_intersection_pointers(inverted_words):
    '''
    NOTE: this function assumes that exists a 'universal' document indexed by 0
          so that the intersection will never be empty
          and we won't have to do several check on the list lengths
    
    computes the intersection on the elements of the inverted index
    input: dictionary of lists of (document_id, tfidf) ordered by the first element
    output: the tfidf partial vector of the documents in the input dictionary
            it will be a dictionary of the form
            {document1 : {word1 : tfidf11, word2 : tfidf21, ...}, document2 : {word1 : tfidf12, word2 : tfidf22, ...}, ...}
            where word is a word that is both in the query and in the document

    NOTE: this algorithm compares the last element of every list instead of the first
          so that the last element (which is the first of every list) will always be a match
    '''
    
    # retrieving word list
    words = list(inverted_words.keys())
    
    # setting total number of lists to intersect
    number_of_words = len(words)
    
    # defining the pointers dictionary
    pointers = {}
    
    # setting the pointers to the last element of every list
    for word in words:
        pointers[word] = len(inverted_words[word]) - 1
    
    # initializing the output vectors as a dictionary with
    # an empty dictionary as default value
    document_vectors = defaultdict(dict)
    
    
    # the cycle will stop when when the first list has been scanned completely
    # because the lists have in common the 0 document, the pointers will
    # become negative all at the same time
    while( pointers[words[0]] >= 0):
        
        current_element = inverted_words[words[0]][ pointers[words[0]] ][0] # we always start comparing the pointed element of the first list
        j = 1                                                               # with the pointed element of the second list
        
        
        while( j < number_of_words):  # this cycle only ends when a common element is found
                                      # thus the need for the common 0 element
            
            if current_element > inverted_words[words[j]][ pointers[words[j]] ][0]:                     # if the pointed element of this list is smaller than the current element
                current_element = ranked_decrement_some_pointers(inverted_words, words, j, pointers)    # then I decrement all the previous lists' pointers to match this one
                j = 1                                                                                   # and I restart the cycle from the second list
                
            elif current_element < inverted_words[words[j]][ pointers[words[j]] ][0]:                              # if the pointed element of this list is bigger than the current element
                j += ranked_decrement_one_pointers(inverted_words[words[j]], current_element, pointers, words[j])  # then I need to decrement the pointer of the current list, and if after decrementing
                                                                                                                   # the pointed element coincides with the current element I go on with the cycle
                                                                                                                   # otherwise I repeat this cycle and I fall back in the previous if case
                
            else:       # if the pointed element of this list is equal to the current element
                j+=1    # I go on with the cycle
        
        # populating the output dictionary with the tfidf
        for word in words:                                                                            # I arrive here only if current_element is in every list
            document_vectors[ current_element ][word] = inverted_words[word][ pointers[word] ][1]     # so I add it to the solution
            
        ranked_decrement_all_pointers(pointers, words)                                                # and I remove it from the lists
    
    # removing the dummy 0 anime document
    document_vectors.pop(0)
    
    return (document_vectors)


def ranked_decrement_one_pointers(inverted_word, document, pointers, idx):
    '''
    input: (list element of the inverted index, integer, dictionary of pointers, key of the list)
    output: 1 if the integer document is in the list, 0 otherwise
    non-explicit output: the function will decrement the value of pointers[idx] until
                         inverted_word[ pointers[idx] ] <= document
    '''
    
    # removes all the elements bigger than document
    while(inverted_word[ pointers[idx] ][0] > document ):
        pointers[idx] -= 1
    
    # checks if document is in inverted_word
    if(inverted_word[ pointers[idx] ][0] == document):
        return(1)
    else:
        return(0)
    
    
def ranked_decrement_some_pointers(inverted_words, words, j, pointers):
    '''
    input: (subdictionary of the inverted index, keys of the inverted index, integer, dictionary of pointers)
    output: the greatest doc_id of inverted_words[words[0]] that
            is smaller or equal than inverted_words[words[j]][ pointers[words[j]] ]
    non-explicit output: takes the first 1 to j lists of inverted_words
                         and decrements their pointer until every pointed element
                         is smaller or equal than inverted_words[words[j]][ pointers[words[j]] ]
    '''
    
    document = inverted_words[words[j]][ pointers[words[j]] ][0]
    
    for i in range(j):
        ranked_decrement_one_pointers(inverted_words[ words[i] ], document, pointers, words[i])   # decrements pointers[words[i]] until
                                                                                                  # inverted_words[words[i]][ pointers[words[i]] ] <= document
    
    return( inverted_words[ words[0] ][ pointers[words[0]] ][0])

    
def ranked_decrement_all_pointers(pointers, words):
    '''
    input: list of lists of integers
    output: None
    non-explicit output: decrements every element in the list by 1
    '''
    
    for i in words:
        pointers[i] -= 1   # decrement by one the i^th pointer
    
    return


###############################################################################################
#                                                                                             #
#     functions used to initialize the ranked search engine and perform the actual search     #
#                                                                                             #
###############################################################################################

def ranked_search_engine_initialization():
    '''
    initialize the ranked search engine by retrieving the inverted index,
    the vocabulary and the anime informations from the disk
    
    input: None
    output: (vocabulary, inverted index, tfidf vector, total number of document)
    '''
    
    # retrieve vocabulary from disk
    vocabulary = vocabulary_retrieval()
    
    # retrieve inverted index from disk
    inverted_index = ranked_inverted_index_retrieval()
    
    # retrieve tfidf vectors
    tfidf_vectors = tfidf_vector_retrieval()
    
    # retrieve total number of anime
    total_document = anime_count()
    
    return(vocabulary, inverted_index, tfidf_vectors, total_document)


def ranked_search(vocabulary, inverted_index, tfidf_dict, total_documents, k = 5):
    '''
    this is the actual search engine:
    given a query, it will print some brief information
    about the matching anime in order of similarity with the query
    
    input: (vocabulary of words, inverted index, tfidf vectors,
            total number of documents, number of search match to display)
    output: None
    non-explicit input: the function will ask for a search query
    non-explicit output: the function will print some brief information
                         about the anime that matches the query
    '''
    
    # retrieving the search query
    query = input('Input a query:')
    print('')
    
    # preprocessing the query
    query = preprocessing(query)
    
    # converting the words in the query into id
    query_ids = vocabulary_conversion(query, vocabulary)
    
    # retrieving query inverted lists
    inverted_words = {}
    for i in query_ids:
        inverted_words[i] = inverted_index[i]
    
    
    if len(inverted_words) > 0:
        
        # vectorizing the query
        vectorized_query = vectorize_query(query_ids, inverted_words, total_documents)
        
        # vectorizing all the documents that have
        # some words in common with the query
        vectorized_documents = ranked_intersection_pointers(inverted_words)
        
    else:
        # if len(inverted_words) == 0 none of the words
        # in the query are present in our vocabulary
        print("Couldn't find the query in the document. Try changing the terms or using less words.")
        return
    
    # finding the top_k list of documents
    top_k = find_top_k(tfidf_dict, vectorized_documents, vectorized_query, k)
    
    # printing search results
    ranked_search_result_printing(top_k, 'animeTitle', 'animeDescription', 'Url')
    
    return


def find_top_k(tfidf_dict, vec_documents, vec_query, k):
    '''
    this function finds the top k document in 
    vec_documents based on the cosine similarity with the query
    using an minheap (from the heapq library) to keep the top k
    
    input: (dictionary of tfidf document vectors, partial tfidf document vectors (they only contain the word that matches the query)
            tfidf vector of the query, k)
    output: list of tuples of the form (document_id, cosine_similarity)
    '''
    
    # initializing the document list
    heap = []
    
    # retrieving the ids of the document to compare
    documents_ids = vec_documents.keys()
    
    for document in documents_ids:
        
        # computing similarity between current document and query
        similarity = cosine_similarity(tfidf_dict[document], vec_query, vec_documents[document])
        
        # inserting the tuple (similarity, document) in the heap
        heapq.heappush(heap, (similarity, document))
        
        if len(heap) > k:
            # if the heap contains more than k documents
            # we remove the smallest one (according to similarity)
            heapq.heappop(heap)
    
    return(heap)

def cosine_similarity(complete_vec_doc, vec_query, vec_doc):
    '''
    NOTE: order of input IS important, because
          we ignore the norm of the query in this calculation
          
    NOTE: because of the fact that we ignore the norm of the query
          in this calculation, the result will not be the actual cosine
          similarity, but it will maintain the ordering relationship
          between the other similarities
        
    input: (total document vector, query vector, intersection document vector)
    output: cosine similarity between the two input
            (modulo the constant factor of the query norm)
    '''
    
    # computing the norm of the input document
    doc_norm = np.linalg.norm(list(complete_vec_doc.values()))
    
    dot_product = 0
    for word in vec_query.keys():
        dot_product += vec_query[word]*vec_doc[word]
    
    return(dot_product/doc_norm)


def ranked_search_result_printing(top_k, *columns):
    '''
    this function displays the columns information about the anime in top_k
    
    input: (list of tuples of the form (document_id, cosine_similarity),
            column informations to display)
    output: None
    non-explicit output: prints the search results in order of similarity
    '''
    
    # initializing the dataframe that will contain the anime informations
    information_df = pd.DataFrame(columns = columns+('Similarity',))
    
    for couples in top_k:
        anime_id = couples[1]
        
        # retrieving information about the matching anime
        information_df.loc[anime_id] = anime_information_retrieval(anime_id, columns) + [couples[0]]
    
    # sorting the search result by similarity
    information_df = information_df.sort_values(by='Similarity', ascending=False)
    
    # print the informations on screen
    display(HTML(information_df.to_html(index = False)))
    
    return