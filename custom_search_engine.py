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

# custom libraries
from data_collection import *
from tsv_management import *
from search_engine import *

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer 

#nltk.download('wordnet')    #da scaricare una sola volta
#nltk.download('stopwords')  
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('universal_tagset')




########################################################################################################################################################
#                                                              custom_search_engine.py                                                                 #
#                                                                                                                                                      #
#       library of function useful to process every document for the search, initialize the custom search engine, and perform the actual search        #
#                                                                                                                                                      #
########################################################################################################################################################


##########################################################################
#                                                                        #
#     functions to preprocess the text to have it ready for a search     #
#                                                                        #
##########################################################################

def lemmatization_preprocessing(text):
    '''
    this function preprocesses a string to prepare it either for the inverted
    index creation or for the search of the query
    in details, here we:
        - tokenize the string (with a regex tokenizer)
        - convert the words to lowercase
        - tag the words
        - remove the english stopwords
        - lemmatize the words using Word Net lemmatizer
    
    input: string to preprocess
    output: preprocessed list of words
    '''
    
    # initialize tokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    
    # initialize the lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # tokenize the text
    word_list = tokenizer.tokenize(text)
    
    # tagging the text
    word_list = word_tag_retrieval(word_list)
    
    processed_text = []
    for couples in word_list:
        word = couples[0].lower()
        tag = couples[1]
        if word not in stopwords.words('english'):
            lemmatized_word = lemmatizer.lemmatize(word, tag)
            processed_text.append(lemmatized_word)
    
    return(processed_text)


def word_tag_retrieval(word_list):
    '''
    this function parses the tag of the word to give to the lemmatizer
    
    if the tag is not one among adjective, adverb, verb, noun then the function returns the noun tag
    
    input: list of words
    output: list of tuples (word, tag)
    '''
    
    # tag the words accoring to nltk tag names
    words_and_tag_list = nltk.pos_tag(word_list, tagset = 'universal')
    
    # define a dictionary to convert the nltk tags into WordNet tags
    tag_conversion_dictionary = defaultdict(lambda : wordnet.NOUN)
    tag_conversion_dictionary.update({'ADJ': wordnet.ADJ, 'NOUN': wordnet.NOUN, 'VERB': wordnet.VERB, 'ADV': wordnet.ADV})
    
    # tag the words
    for i in range(len(words_and_tag_list)):
         words_and_tag_list[i] =  (words_and_tag_list[i][0], tag_conversion_dictionary[words_and_tag_list[i][1]])
        
    return(words_and_tag_list)


def lemmatization_list_preprocessing( list_of_strings ):
    '''
    this function preprocesses every string in a list of strings
    input: list of strings
    output: None
    non-explicit output: every string in the list is preprocessed and becomes a list of words
    '''
    
    for i in tqdm(range(len(list_of_strings)), desc="Preprocessing documents"):
        if list_of_strings[i] is not None:
            list_of_strings[i] = lemmatization_preprocessing(list_of_strings[i])
        else:
            list_of_strings[i] = []
    
    return


##########################################################################
#                                                                        #
#     functions to create and manage a vocabulary that maps each word    #
#      of our anime descriptions to an integer number                    #
#                                                                        #
##########################################################################

def titles_descriptions_vocabulary_generation():
    '''
    NOTE: there are two different tqdm progress bar called in this function
    
    this function generates a vocabulary using all the anime descriptions,titles,characters,voices and staff
    and saves everything in a json file
    '''

    # retrieving the descriptions of all the anime
    description = column_retrieval('animeDescription')
    
    # retrieving the titles of all the anime
    titles = column_retrieval('animeTitle')
    
    # retrieving the characters of all the anime
    characters = column_retrieval('animeCharacters')
    for i in range(len(characters)):
        characters[i] = ' '.join(characters[i])
    
    # retrieving the voices of all the anime
    voices = column_retrieval('animeVoices')
    for i in range(len(voices)):
        voices[i] = ' '.join(voices[i])
    
    # retrieving the staff of all the anime
    staff = column_retrieval('animeStaff')
    
    # extracting only the staff names
    for i in range(len(staff)):
        anime = staff[i]
        staff_names = []
        
        for person in anime:
            staff_names.append(person[0])
        
        staff[i] = ' '.join(staff_names)
    
    
    # processing every description
    lemmatization_list_preprocessing( description )
    
    # processing every title
    lemmatization_list_preprocessing( titles )
    
    # processing every character name
    lemmatization_list_preprocessing( characters )
    
    # processing every every voices name
    lemmatization_list_preprocessing( voices )
    
    # processing every staff name
    lemmatization_list_preprocessing( staff )
    
    
    # merging the four datasets
    merged = []
    for i in range(len(titles)):
        merged.append( description[i] + titles[i] + characters[i] + voices[i] + staff[i] )
    
    
    # generating a vocabulary of words that associates every word to an integer
    vocabulary = vocabulary_creation(merged)
    
    # saving the vocabulary to the disk
    with open("content/titles_descriptions_vocabulary.json", 'w') as f:
        json.dump(vocabulary, f)
    
    return
    
    
def titles_descriptions_vocabulary_retrieval():
    '''
    this function reads the vocabulary from the disk
    and returns it as a dictionary
    
    input: None
    output: vocabulary dictionary
    '''
    
    term_dict = json_loading('./content/titles_descriptions_vocabulary.json')
    
    return(term_dict)


###############################################################################################
#                                                                                             #
#           functions to create and manage the inverted index and the tfidf vectors           #
#                                                                                             #
###############################################################################################

def titles_descriptions_inverted_index_generation():
    '''
    NOTE: there are three different tqdm progress bar called in this function
    
    this function generates the tfidf vectors 
    and an inverted index using all the anime descriptions
    and saves everything in three json files
    '''
    
    # retrieving the descriptions of all the anime
    description = column_retrieval('animeDescription')
    
    # retrieving the titles of all the anime
    titles = column_retrieval('animeTitle')
    
    # retrieving the characters of all the anime
    characters = column_retrieval('animeCharacters')
    for i in range(len(characters)):
        characters[i] = ' '.join(characters[i])
    
    # retrieving the voices of all the anime
    voices = column_retrieval('animeVoices')
    for i in range(len(voices)):
        voices[i] = ' '.join(voices[i])
    
    # retrieving the staff of all the anime
    staff = column_retrieval('animeStaff')
    
    # extracting only the staff names
    for i in range(len(staff)):
        anime = staff[i]
        staff_names = []
        
        for person in anime:
            staff_names.append(person[0])
        
        staff[i] = ' '.join(staff_names)
    
    # processing every description
    lemmatization_list_preprocessing( description )
    
    # processing every title
    lemmatization_list_preprocessing( titles )
    
    # processing every character
    lemmatization_list_preprocessing( characters )
    
    # processing every every voices
    lemmatization_list_preprocessing( voices )
    
    # processing every staff name
    lemmatization_list_preprocessing( staff )
    
    
    # merging the four datasets while removing duplicates
    merged = []
    for i in range(len(titles)):
        merged.append( list(set(titles[i] + characters[i] + voices[i] + staff[i] )))
    
    # retrieving the vocabulary from the disk
    vocabulary = titles_descriptions_vocabulary_retrieval()
    
    # generating the inverted index
    # and the tfidf vectors
    inverted_index, description_tf_idf, partial_tf_idf = titles_descriptions_inverted_index_creation(description, merged, vocabulary)
    
    # saving the tfidf vectors to the disk
    with open("content/description_tfidf_vectors.json", 'w') as f:
        json.dump(description_tf_idf, f)
    with open("content/partial_tfidf_vectors.json", 'w') as f:
        json.dump(partial_tf_idf, f)
        
    # saving the inverted index to the disk
    with open("content/titles_descriptions_inverted_index.json", 'w') as f:
        json.dump(inverted_index, f)
    
    return


def titles_descriptions_inverted_index_creation(description_corpus, partial_corpus, vocabulary):
    '''
    this function creates an unranked inverted index based on the datasets of anime descriptions, titles, characters, voices, staff
    and the (distinct) vector of tfidf for all the document descriptions and the titles,characters,voices,staff
    
    NOTE: for simplicity of search, every word in our inverted index will 
          belong to a dummy 0 document that contains every word
          (our anime id starts from 1)
    
    
    input: list of anime descriptions, (list of merged titles,characters,voices,staff) , vocabulary
    output: (inverted index list of the form [[doc, doc, ...], [doc]...],
             tfidf vector dictionary of the form {doc_id: {word_id : tfidf, word_id : tfidf, ...}, ...} relative to description,
             tfidf vector dictionary of the form {doc_id: {word_id : tfidf, word_id : tfidf, ...}, ...} relative to titles,characters,voices,staff)
    '''
    
    # total number of documents
    N = len(description_corpus)
    
    # creating the total corpus
    total_corpus = []
    for i in range(N):
        total_corpus.append(description_corpus[i]+partial_corpus[i])
    
    # creating the inverted index list
    # (we associate the document 0 to every word for convenience in searching)
    inverted_index = unranked_inverted_index_creation(total_corpus, vocabulary)
    
    # initializing the tfidf vectors dictionary
    description_tf_idf = {} # {doc_id:{word_id:tfidf}}
    partial_tf_idf = {} # {doc_id:{word_id:tfidf}}
    
    for i in tqdm(range(N), desc = 'TfIdf vectors generation'):
        description_document = description_corpus[i]
        partial_document = partial_corpus[i]
        doc_id = i+1
        
        # initializing word dictionary
        partial_tf_idf[doc_id] = {}
        description_tf_idf[doc_id] = {}
        
        # total number of words in the documents
        description_words_count = len(description_document)
        partial_words_count = len(partial_document)
        
        # dictionary of occurrences of every word in the document
        description_counter = Counter(description_document)
        partial_counter = Counter(partial_document)
        
        # removing duplicated words
        unique_partial_document = list(np.unique(partial_document))
        unique_description_document = list(np.unique(description_document))
        
        # populating the tfidf vector for the titles(etc..)
        for word in unique_partial_document:
            word_id = vocabulary[word]
            
            # tf = # occurrences of word / total number of words in the document
            tf = partial_counter[word]/partial_words_count
            
            # df = occurrences of (word in document) across all documents
            # is equal to the length of the corresponding inverted list (removing the 0 document)
            df = len(inverted_index[word_id])-1
            
            # idf = log10(#number of document / occurrences of (word in document) across all documents)
            idf = np.log10(N/df)
            
            partial_tf_idf[doc_id][word_id] =  tf*idf
        
        # populating the tfidf vector for the descriptions
        for word in unique_description_document:
            word_id = vocabulary[word]
            
            # tf = # occurrences of word / total number of words in the document
            tf = description_counter[word]/description_words_count

            # df = occurrences of (word in document) across all documents
            # is equal to the length of the corresponding inverted list (removing the 0 document)
            df = len(inverted_index[word_id])-1

            # idf = log10(#number of document / occurrences of (word in document) across all documents)
            idf = np.log10(N/df)

            description_tf_idf[doc_id][word_id] =  tf*idf

    
    return(inverted_index, description_tf_idf, partial_tf_idf)


def titles_descriptions_inverted_index_retrieval():
    '''
    this function reads the inverted index from the disk
    and returns it as a list
    
    input: None
    output: inverted index list
    '''

    inverted_index = json_loading('./content/titles_descriptions_inverted_index.json')
    
    return(inverted_index)


def description_titles_tfidf_vector_retrieval():
    '''
    this function reads the tfidf vectors from the disk
    and returns them as a dictionary
    
    input: None
    output: tfidf vectors dictionary
    '''

    description_tf_idf = json_loading('./content/description_tfidf_vectors.json')
    partial_tf_idf = json_loading('./content/partial_tfidf_vectors.json')
    
    # because the json loading parses our integer keys as strings
    # we need to cast them
    for key in list(description_tf_idf.keys()):
        
        for inner_key in list(description_tf_idf[key].keys()):
            description_tf_idf[key][int(inner_key)] = description_tf_idf[key].pop(inner_key)
            
        description_tf_idf[int(key)] = description_tf_idf.pop(key)
    for key in list(partial_tf_idf.keys()):
        
        for inner_key in list(partial_tf_idf[key].keys()):
            partial_tf_idf[key][int(inner_key)] = partial_tf_idf[key].pop(inner_key)
            
        partial_tf_idf[int(key)] = partial_tf_idf.pop(key)
        
    
    return(description_tf_idf, partial_tf_idf)


##########################################################################################################
#                                                                                                        #
#           functions used to perform the union of two or more elements of the inverted index            #
#                                                                                                        #
##########################################################################################################

def union_pointers(inverted_words):
    '''
    NOTE: this function assumes that exists a 'universal' document indexed by 0
          so that we won't have to do several check on the list lengths
    
    computes the unions on the elements of the inverted index
    
    input: dictionary of inverted words
    output: dictionary containing the union of the document id in the lists as keys
            and the corresponding words as values (in a list)
    
    NOTE: this algorithm compares the last element of every list instead of the first
    '''  
    words = inverted_words.keys()
    
    # an dictionary of indices that points to the last element of every list in inverted_words
    pointers = {}
    for word in words:
        pointers[word] = len(inverted_words[word]) - 1
    
    # creating output dictionary
    union = defaultdict(list)
    
    # choosing a starting pointer to start the cycle
    representative_pointer = 0

    # the cycle will stop when one of the pointers goes under zero
    # because the lists have in common the 0 document, the pointers will
    # become negative all at the same time
    while(representative_pointer >= 0):
        
        # retrieving the words that have the greatest pointed element
        current_word = union_argmax( inverted_words, pointers )
        
        for word in current_word:
            # retrieving the pointed document id associated to word
            document_id = inverted_words[word][ pointers[word] ]
            
            # populating the output dictionary
            union[document_id].append(word)
        
        # decrementing the pointers relative to the greatest element by one
        decrement_these_pointers(pointers, current_word)
        
        # choosing a new starting pointer to continue the cycle
        representative_pointer = pointers[current_word[0]]
    
    return (union)


def union_argmax( inverted_words, pointers ):
    '''
    this function computes the indices corresponding to
    the greatest pointed value of the dictionary of inverted words
    
    input: dictionary of inverted words, pointers
    output: indices corresponding to the greatest value in the input lists
    '''
    
    # initializing the list that will store the keys and the greatest value
    max_couple = [[],-1]
    
    for current_word in inverted_words.keys():
        
        if inverted_words[current_word][pointers[current_word]] > max_couple[1]:
            # if I find a greater element I replace the values in max_couple
            max_couple[0] = [current_word]
            max_couple[1] = inverted_words[current_word][pointers[current_word]]
            
        elif inverted_words[current_word][pointers[current_word]] == max_couple[1]:
            # if I find another (equal) maximum, I store the index
            max_couple[0].append(current_word)
    
    return(max_couple[0])


def decrement_these_pointers(pointers, indices_list):
    '''
    decrement by one all the values in the pointers dictionary relative to the keys in indices_list
    
    input: (dictionary of integer values, integer, sublist of keys of the dictionary)
    output: None
    non-explicit output: decrement by one all the values in the pointers
                         dictionary relative to the keys in indices_list
    '''
    
    for idx in indices_list:
        pointers[idx] -= 1
            
    return


###############################################################################################
#                                                                                             #
#             functions used to compute a custom score associated to every anime              #
#                                                                                             #
###############################################################################################

def anime_scoring_info_creation():
    '''
    here we create a file in which we store some data about every anime
    including two custom scores based on rank and popularity
    '''
    
    # retrieving Score, Popularity and Type of every anime
    score = column_retrieval('animeScore')
    popularity = column_retrieval('animePopularity')
    animeType = column_retrieval('animeType')
    
    # initializing output dictionary
    anime_scoring = defaultdict(dict)
    
    for i in range(len(score)):
        
        # computing a custom ranking score
        if score[i] == None:
            anime_scoring[i+1]['animeScore'] = bump_function(2, 10)
        else:
            anime_scoring[i+1]['animeScore'] = bump_function(score[i], 10)
        
        # computing a custom popularity score
        temp = 19129 - popularity[i]
        anime_scoring[i+1]['animePopularity'] = bump_function(temp, 19128)
        
        # storing the anime type
        anime_scoring[i+1]['animeType'] = animeType[i]
    
    # saving everything to the disk
    with open("content/anime_scoring_info.json", 'w') as f:
        json.dump(anime_scoring, f)
    
    return

def bump_function(x, max = 1):
    '''
    this is a custom bump function that we use to squeeze the scores in (0,1)
    '''
    
    def g(y):
        try:
            res = np.exp(-1/y)
        except ZeroDivisionError:
            res = 0
        return(res)
    
    result = g(x/max)/(g(x/max)+g(1-x/max))
    
    return (result)


def anime_scoring_info_retrieval():
    '''
    this function reads the anime scoring info from the disk
    and returns it as a list
    
    input: None
    output: anime scoring info list
    '''

    anime_scoring_info = json_loading('./content/anime_scoring_info.json')
    
    # because the json loading parses our integer keys as strings
    # we need to cast them
    for key in list(anime_scoring_info.keys()):
        anime_scoring_info[int(key)] = anime_scoring_info.pop(key)
    
    return(anime_scoring_info)


###############################################################################################
#                                                                                             #
#                   functions used to perform the ranking during the search                   #
#                                                                                             #
###############################################################################################

# global variable useful in ranking
type_modifier = 1.6

def scoring_function(score, anime_scoring_info, Type, popularity_relevance):
    '''
    this function computes a final custom score based on the inputs of the user
    '''

    global type_modifier
    global absolute_popularity_modifier
    
    popularity_index = (anime_scoring_info['animePopularity'] + anime_scoring_info['animeScore'])/2
    score = score*(1.5-popularity_relevance) + popularity_index * popularity_relevance
    
    if Type == 'all':
        return(score)
    else:
        if anime_scoring_info['animeType'] == None:
            return(score)
        elif anime_scoring_info['animeType'].lower() == Type:
            return(score * type_modifier)
        else:
            return(score)
    
    
def custom_find_top_k(description_tf_idf, partial_tf_idf, union_documents, vec_query, anime_scoring_info, parameters_list, k):
    
    '''
    this function finds the top k document in 
    union_documents based on the cosine similarity with the query and on a custom scoring function
    using an minheap (from the heapq library) to keep the top k
    
    input: (dictionary of description document vectors,
            dictionary of titles/voices/characters/staff document vectors,
            list of documents to rank,
            vector query, list of custom parameters, k)
    output: list of tuples of the form (custom_score, document_id)
    '''
    
    isSpecific = parameters_list[0]
    
    # initializing the document list
    heap = []
    
    # retrieving the ids of the document to compare
    documents_ids = union_documents.keys()
    
    for document in documents_ids:
        
        # computing the custom score between current document and query
        custom_score = description_titles_cosine_similarity(description_tf_idf[document], partial_tf_idf[document], vec_query, union_documents[document], isSpecific)
        
        custom_score = scoring_function(custom_score, anime_scoring_info[document], *(parameters_list[1:]))
        
        # inserting the tuple (document, custom_score) in the heap
        heapq.heappush(heap, (custom_score, document))#(document, custom_score))
        
        if len(heap) > k:
            # if the heap contains more than k documents
            # we remove the smallest one (according to custom_score)
            heapq.heappop(heap)
    
    return(heap)


def description_titles_cosine_similarity(description_tf_idf, partial_tf_idf, vec_query, relevant_words, isSpecific):
    '''
    input: (description vector, titles vector, query vector, words in common between query and document, weight constant)
    output: cosine similarity between the documents input
    '''
    
    def ellipse (x):
        '''
        custom function used to renormalize the similarity
        '''
        if x>69:
            return 1
        else:
            f = -20 * np.sqrt(1-0.000209*(x-70)**2)+21
            return(f)
    
    
    # generating the total document vector
    complete_vec_doc = defaultdict(lambda : 0)
    for word in partial_tf_idf.keys():
        complete_vec_doc[word] += isSpecific * partial_tf_idf[word]
    for word in description_tf_idf.keys():
        complete_vec_doc[word] += (1 - isSpecific) * description_tf_idf[word]
    
    # computing the norm of the input document and the query
    doc_norm = np.linalg.norm(list(complete_vec_doc.values()))
    query_norm = np.linalg.norm(list(vec_query.values()))
    
    # computing the dot product
    dot_product = 0
    for word in relevant_words:
        dot_product += vec_query[word]*complete_vec_doc[word]
    
    # computing the cosine similarity
    similarity = dot_product/(doc_norm*query_norm)
    
    # normalizing the cosine similarity accordin
    # to the number of words in the reviews
    similarity /= ellipse(len(complete_vec_doc))
    
    return(similarity)


#####################################################################################################
#                                                                                                   #
#    functions used to initialize the search engine, customize it, and perform the actual search    #
#                                                                                                   #
#####################################################################################################

def custom_search_engine_initialization():
    '''
    initialize the custom search engine by retrieving the inverted index,
    the vocabulary and the anime informations from the disk
    
    input: None
    output: (vocabulary, inverted index, description tfidf vector, titles tfidf vector, anime_scoring_info, default parameters list, total number of documents)
    '''
    
    # retrieve vocabulary from disk
    vocabulary = titles_descriptions_vocabulary_retrieval()
    
    # retrieve inverted index from disk
    inverted_index = titles_descriptions_inverted_index_retrieval()
    
    # retrieve tfidf vectors
    description_tf_idf, partial_tf_idf = description_titles_tfidf_vector_retrieval()
    
    # retrieve anime scoring info
    anime_scoring_info = anime_scoring_info_retrieval()
    
    # retrieve total number of anime
    total_document = anime_count()
    
    # setting default search engine parameters
    default_parameters = [0.5, 'all' , 0.5]
    
    return(vocabulary, inverted_index, description_tf_idf, partial_tf_idf, anime_scoring_info, default_parameters, total_document)


def custom_search(vocabulary, inverted_index, description_tf_idf, partial_tf_idf, total_documents, anime_scoring_info, parameters_list, k = 5):
    '''
    this is the actual search engine:
    given a query, it will print some brief information
    about the matching anime in order of score with the query
    
    input: (vocabulary of words, inverted index,
            description tfidf vector, titles tfidf vector,
            total number of documents, anime_scoring_info, custom search parameters,
            number of search match to display)
    output: None
    non-explicit input: the function will ask for a search query
    non-explicit output: the function will print some brief information
                         about the anime that matches the query
    '''
    
    # retrieving the search query
    query = input('Input a query:')
    print('')
    
    # preprocessing the query
    query = lemmatization_preprocessing(query)
    
    # converting the words in the query into id
    query_ids = vocabulary_conversion(query, vocabulary)
    
    # retrieving query inverted lists
    inverted_words = {}
    for i in query_ids:
        inverted_words[i] = inverted_index[i]
    
    
    if len(inverted_words) > 0:
        
        # vectorizing the query
        vectorized_query = vectorize_query(query_ids, inverted_words, total_documents)
        
        # retrieving all the documents that have
        # some words in common with the query
        union_documents = union_pointers(inverted_words)
        
    else:
        # if len(inverted_words) == 0 none of the words
        # in the query are present in our vocabulary
        print("Couldn't find the query in the document. Try changing the terms.")
        return
    
    # removing the dummy 0 document from the search results
    union_documents.pop(0)
    
    # finding the top_k list of documents
    top_k = custom_find_top_k( description_tf_idf, partial_tf_idf, union_documents, vectorized_query, anime_scoring_info, parameters_list, k)
    
    # printing search results
    custom_search_result_printing(top_k, 'animeTitle', 'animeDescription', 'animeType', 'animeScore', 'animePopularity', 'Url')
    
    return


def custom_search_result_printing(top_k, *columns):
    '''
    this function displays the columns information about the anime in top_k
    
    input: (list of tuples of the form (custom score, document_id),
            column informations to display)
    output: None
    non-explicit output: prints the search results in order of custom score
    '''
    
    # initializing the dataframe that will contain the anime informations
    information_df = pd.DataFrame(columns = columns+('Custom score',))
    
    for couples in top_k:
        anime_id = couples[1]
        
        # retrieving information about the matching anime
        information_df.loc[anime_id] = anime_information_retrieval(anime_id, columns) + [couples[0]]
    
    # sorting the search result by similarity
    information_df = information_df.sort_values(by='Custom score', ascending=False)
    
    # print the informations on screen
    display(HTML(information_df.to_html(index = False)))
    
    return


def search_engine_customization():
    '''
    this function asks the user to set the desired parameters for the custom search engine
    '''
    
    print('Hi! Here you can tune some search engine parameters to fit your search needings.\n')
    
    correct_answer = False
    answer_dict = {'y':True, 'n':False, 'yes':True, 'no':False}
    
    while not correct_answer:
        detailed = input('Do you want to have a detailed explanation of the role of every parameter? (Y/n)')
        if detailed.lower() in answer_dict:
            correct_answer = True
            detailed = answer_dict[detailed]
    
    
    correct_answer = False
    
    while not correct_answer:
        
        if detailed: print('To compute the final tfidf vector associated to the document we computed two separate tfidf vectors: one for the description and one for the titles, characters, voices and staff; and we combined the tfidf for each word according to the following formula:\n\
                            final_tfidf = alpha * title_tfidf + (1 - alpha) * description_tfidf\n\
                            where title_tfidf is the tfidf of the word in the titles, characters, voices and staff texts,\
                            and description_tfidf is the tfidf of the word in the description text.\
                            The parameter we ask you to choose is the alpha in the above formula, and it represents the weight of the titles texts in the final computation\
                            for the cosine similarity.\n\
                            alpha = 1 will give max priority to words in the query that match a word in titles, characters, voices and staff texts\n\
                            alpha = 0 will give max priority to words in the query that match a word in the description\n\
                            Thus by varying this parameter you will be able to decide if you search for a specific anime (alpha near 1), or if you\
                            want to do a more generic search (alpha near 0).\n\
                            NOTE: We will automatically renormalize your score in (0.1, 0.9) as choosing too extreme values can lead to unexpected behaviours.\n')
        
        isSpecific = input('Are you searching for a specific anime or you want something generic that fits your query?\n Input a number in [0,1] where 0 is generic and 1 is specific: ')
        
        try:
            isSpecific = float(isSpecific)
            if isSpecific <= 1 and isSpecific >= 0:
                correct_answer = True
        except ValueError:
            pass
    isSpecific = isSpecific * 0.8 + 0.1
            
            
    correct_answer = False
    
    while not correct_answer:
        
        if detailed: print('After computing the cosine similarity based on the parameter above, we convolute the similarity with a specific score that is based on the popularity of the anime according to the following:\n\
                            final_score = similarity * ( 1.5 - alpha ) + popularity_ranking_score * alpha\n\
                            where the popularity_ranking_score is computed starting from the animePopularity and animeScore in the data in the following way:\n\
                            rank_score = f(animeScore, 10)\n\
                            popularity_score = f(19129 - animePopularity, 19128)\n\
                            where f(x,m) is a certain smooth function that squeezes all the numbers in (0,m) in the interval (0,1) in a nonlinear way near the borders.\n\
                            See the implementation in the library custom_search_engine.py, in anime_scoring_info_creation() for more details about the function.\n\
                            After obtaining the rank_score and the popularity_score, we compute the final rank as a simple average:\n\
                            popularity_ranking_score = (popularity_score + rank_score)/2\n\
                            The parameter we ask you to choose is the alpha in the above formula, and it represents the weight of the popularity and the rank of each anime in the classification.\n\
                            alpha = 1 will give max priority to popularity and rank, sending up in the final classification the most popular anime\n\
                            alpha = 0 will give no priority to popularity and rank, letting the cosine similarity choose the classification\n\
                            NOTE: We will automatically renormalize your score in (0.1, 0.9) as choosing too extreme values can lead to unexpected behaviours.\n\
                            NOTE: Whatever you choose as the parameter, the final classification WILL be influenced by the rank and popularity of the anime. This is an intended choice and it\n\
                            reflects the assumption that anime which are not popular nor ranked high are probably of low quality in general and would not be searched much.\n\
                            Thus by varying this parameter you will be able to decide if you prefer to have popular anime (alpha near 1) or high ranked anime (alpha near 0) in your search results.')
        
        popularity_relevance = input('Would you like to rank the anime based on their popularity or based on their score?\n Input a number in [0,1] where 0 is rank by popularity and 1 is rank by score: ')
        
        try:
            popularity_relevance = float(isSpecific)
            if popularity_relevance <= 1 and popularity_relevance >= 0:
                correct_answer = True
        except ValueError:
            pass
    popularity_relevance = popularity_relevance * 0.8 + 0.1
            

    if detailed: print('Here we ask you to specify if you prefer a specific type of anime among the following:\n\
                        Movie, Music, ONA, OVA, Special, TV\n\
                        This is an important choice since most of the anime that are not TV are special or side stories of a TV one.\n\
                        This will allow you to filter out some of the secondary story anime or to search specifically for them.\n\
                        The ranking boost works in this way: given a final similarity score, if the anime type coincides with the desired one\n\
                        the score will be increased by a fixed percentage (see the global variable type_modifier in the library custom_search_engine.py).\n\
                        NOTE: if you choose All or if you type an incorrect anime type, we will not take into account at all the type of the anime in the final score.')
        
    Type = input('Are you searching for a specific anime Type? Choose among the following:\nMovie, Music, ONA, OVA, Special, TV, All\n')
        
    Type = Type.lower()
    if Type not in ['movie', 'music', 'ona', 'ova', 'special', 'tv', 'all']:
        Type = 'all'
    
    
    if detailed:
        print('The parameter you just chose are passed to the search engine as a list:\n\
               [isSpecific, Type, popularity_relevance]\n\
               With this knowledge you can edit these parameters directly if you wish, and you can also bypass the restriction of the numerical parameters being in (0.1, 0.9).\n\
               This last thing can bring unexpected results, experiment freely with this in mind.\n')
    
    return ([isSpecific, Type, popularity_relevance])