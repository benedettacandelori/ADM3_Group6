from bs4 import BeautifulSoup
import requests
import os.path
import re
import pandas as pd
import csv
from datetime import datetime
from tqdm import tqdm
import time
from data_collection import *



####################################################################################################################################
#                                                        tsv_management.py                                                         #
#                                                                                                                                  #
#           library of function useful to retrieve information from the tsv and txt files that we created from the data            #
#                                                                                                                                  #
####################################################################################################################################



#################################################################################
#                                                                               #
#    functions to retrieve the anime info from the tsv in the correct format    #
#                                                                               #
#################################################################################

def tsv_retrieval(anime_num):
    
    '''
    reads the tsv file relative to the input number
    and puts it into a dictionary
    '''
    
    page = anime_num//50
    tsvname = f'content/alltsv/{page}page/anime_{anime_num}.tsv'
    
    with open(tsvname, 'r') as f:
        read_tsv = csv.reader(f, delimiter="\t")
        
        # the file contains only two row
        # the header and the data
        lines = []
        for row in read_tsv:
            lines.append(row)
        
        anime = to_dict(lines[1], lines[0])
    
    
    return(anime)


def column_retrieval(column):
    '''
    return a list that contains the value of 'column' for each anime
    '''
    
    # retrieving anime links
    filename = 'linksfile.txt'
    with open(filename) as f:
        urls = f.readlines()
    
    if column.lower() == 'url':
        return(urls)
    
    total_anime = len(urls)
    
    common_anime_column = []
    for i in tqdm(range(1, total_anime+1), desc="Retrieving tsv"):
        anime = tsv_retrieval(i)
        common_anime_column.append(anime[column])
    
    return(common_anime_column)


def load_all_tsv():
    '''
    return a list that contains the dictionary associated to every anime
    '''
    
    total_anime = anime_count()
    
    all_the_anime = []
    for i in tqdm(range(1, total_anime+1), desc="Retrieving tsv"):
        anime = tsv_retrieval(i)
        all_the_anime.append(anime)
    
    return(all_the_anime)


def url_retrieval(anime_num, filename = 'linksfile.txt'):
    '''
    returns the url relative to the anime indexed by anime_num
    input: integer starting from 1
    output: string
    '''
    
    with open(filename) as fp:
        for i, line in enumerate(fp):
            if i == anime_num-1:
                url = line.strip()
                break
    
    return(url)


def anime_information_retrieval(anime_num, columns):
    '''
    returns a list of columns information about the anime associated with anime_num
    '''

    anime = tsv_retrieval(anime_num)
    anime['Url'] = url_retrieval(anime_num)

    
    anime_info = []
    for column_name in columns:
        anime_info.append( anime[column_name] )
    
    return(anime_info)


def to_dict(anime, col):
    '''
    convert the output of the tsv reading into a dictionary
    '''
    
    # insert data into dictionary
    anime_dict = {}
    for i in range(len(anime)):
        anime_dict[col[i]] = anime[i]
        
        
    ## converting data to correct format ##
    
    # convert Title
    if anime_dict['animeTitle'] == '':
        anime_dict['animeTitle'] = None
    
    # convert Type
    if anime_dict['animeType'] == '':
        anime_dict['animeType'] = None
        
    # convert NumEpisode
    if anime_dict['animeNumEpisode'] == '':
        anime_dict['animeNumEpisode'] = None
    else:
        anime_dict['animeNumEpisode'] = int(anime_dict['animeNumEpisode'])
    
    # convert releaseDate
    if anime_dict['releaseDate'] == '':
        anime_dict['releaseDate'] = None
    else:
        anime_dict['releaseDate'] = datetime.strptime(anime_dict['releaseDate'], '%Y-%m-%d %H:%M:%S')
        
    # convert endDate
    if anime_dict['endDate'] == '':
        anime_dict['endDate'] = None
    else:
        anime_dict['endDate'] = datetime.strptime(anime_dict['endDate'], '%Y-%m-%d %H:%M:%S')
        
    # convert NumMembers
    if anime_dict['animeNumMembers'] == '':
        anime_dict['animeNumMembers'] = None
    else:
        anime_dict['animeNumMembers'] = int(anime_dict['animeNumMembers'])
        
    # convert Score
    if anime_dict['animeScore'] == '':
        anime_dict['animeScore'] = None
    else:
        anime_dict['animeScore'] = float(anime_dict['animeScore'])
        
    # convert Users
    if anime_dict['animeUsers'] == '':
        anime_dict['animeUsers'] = None
    else:
        anime_dict['animeUsers'] = int(anime_dict['animeUsers'])
        
    # convert Rank
    if anime_dict['animeRank'] == '':
        anime_dict['animeRank'] = None
    else:
        anime_dict['animeRank'] = int(anime_dict['animeRank'])    
        
    # convert Popularity
    if anime_dict['animePopularity'] == '':
        anime_dict['animePopularity'] = None
    else:
        anime_dict['animePopularity'] = int(anime_dict['animePopularity'])   
        
    # convert Description
    if anime_dict['animeDescription'] == '':
        anime_dict['animeDescription'] = None
        
    # convert Related
    anime_dict['animeRelated'] = string_to_list(anime_dict['animeRelated'])
    
    # convert Characters
    anime_dict['animeCharacters'] = string_to_list(anime_dict['animeCharacters'])
    anime_dict['animeCharacters'] = remove_commas(anime_dict['animeCharacters'])
    
    # convert Voices
    anime_dict['animeVoices'] = string_to_list(anime_dict['animeVoices'])
    anime_dict['animeVoices'] = remove_commas(anime_dict['animeVoices'])
    
    # convert Staff
    anime_dict['animeStaff'] = string_to_nested_list(anime_dict['animeStaff'])
        
    return(anime_dict)


def string_to_list(str):
    '''
    convert the string "['Elric, Edward', 'Elric, Alphonse', 'Armstrong, Olivier Mira']"
    to the list ['Elric, Edward', 'Elric, Alphonse', 'Armstrong, Olivier Mira']
    '''
    
    # convert the null string
    if str == '':
        return([])
    
    # convert to string "'Elric, Edward', 'Elric, Alphonse', 'Armstrong, Olivier Mira'"
    str = str[1:-1]
    
    # convert to list ["'element1'", "'element2'", "'element3'"]
    str = str.split("', '")
    
    # convert each element from "'element1'" to "element1"
    for i in range(len(str)):
        str[i] = str[i].replace("'", '')
    
    return( str )


def remove_commas(list):
    '''
    removes the commas from all the element of a list of strings
    '''
    
    for i in range(len(list)):
        list[i] = list[i].replace(',', '')
        
    return(list)


def string_to_nested_list(str):
    '''
    convert the string "[['Cook Justin', ['Producer']], ['Yonai Noritomo', ['Producer']], ['Irie Yasuhiro', ['Director', 'Episode Director', 'Storyboard']], ['Mima Masafumi', ['Sound Director']]]"
    to the list [['Cook Justin', ['Producer']], ['Yonai Noritomo', ['Producer']], ['Irie Yasuhiro', ['Director', 'Episode Director', 'Storyboard']], ['Mima Masafumi', ['Sound Director']]]
    '''
    
    # convert the null string
    if str == '':
        return([])
    
    # replace " with ' inside the string
    str = str.replace('"', "'")
    
    # convert to string "['Cook Justin', ['Producer']], ['Yonai Noritomo', ['Producer']], ['Irie Yasuhiro', ['Director', 'Episode Director', 'Storyboard']], ['Mima Masafumi', ['Sound Director']]"
    str = str[1:-1]
    
    # convert to list ["['Cook Justin', ['Producer']", "['Yonai Noritomo', ['Producer']", "['Irie Yasuhiro', ['Director', 'Episode Director', 'Storyboard']", "['Mima Masafumi', ['Sound Director']]"]
    str = str.split('], ')
    
    # convert each element
    for i in range(len(str)):
        
        # convert string "['Cook Justin', ['Producer']" to string "Cook Justin', ['Producer']"
        str[i] = str[i][2:]
        
        # convert string "Irie Yasuhiro', ['Director', 'Episode Director', 'Storyboard']"
        # to list ['Irie Yasuhiro', "'Director', 'Episode Director', 'Storyboard']"]
        str[i] = str[i].split("', [")
        
        # convert string "'Director', 'Episode Director', 'Storyboard']"
        # to string "'Director', 'Episode Director', 'Storyboard'"
        str[i][1] = str[i][1].replace(']', '')
        
        # convert string "'Director', 'Episode Director', 'Storyboard'"
        # to list ["'Director'", "'Episode Director'", "'Storyboard'"]
        str[i][1] = str[i][1].split(', ')
        
        # convert "'Director'" to "Director"
        for j in range(len(str[i][1])):
            str[i][1][j] = str[i][1][j].replace("'", '')
            
    return(str)


def txt_retrieval(anime_num):
    
    '''
    reads the txt file relative to the input number
    and puts it into list of strings
    '''
    
    page = anime_num//50
    txtname = f'content/alltxt/{page}page/review_{anime_num}.txt'
    
    with open(txtname, 'r') as file1:
        lines = file1.readlines()
    
    # stripping the reviews of the last \n character
    for i in range(len(lines)):
        lines[i] = lines[i].strip()
      
    return(lines)


def load_all_txt():
    '''
    return a list that contains the list of reviews associated to every anime
    '''
    
    total_anime = anime_count()
    
    all_the_reviews = []
    for i in tqdm(range(1, total_anime+1), desc="Retrieving txt"):
        anime = txt_retrieval(i)
        all_the_reviews.append(anime)
    
    return(all_the_reviews)


def merge_txt(tsvname = 'content/all_reviews.tsv'):
    '''
    here we merge all the txt files in one file for better file management
    (as it is hard to move multiple smaller files around)
    '''
    
    all_reviews = load_all_txt()
    
    with open(tsvname, 'w') as f:
        tsv_writer = csv.writer(f, delimiter='\t')
        for anime in all_reviews:
            tsv_writer.writerow(anime)
    
    return


def load_all_reviews(tsvname = 'content/all_reviews.tsv'):
    '''
    return a list that contains the list of reviews associated to every anime
    '''
    
    with open(tsvname, 'r') as f:
        read_tsv = csv.reader(f, delimiter="\t")
        
        all_the_reviews = []
        for row in read_tsv:
            all_the_reviews.append(row)
    
    return(all_the_reviews)
