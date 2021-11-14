from bs4 import BeautifulSoup
import requests
import os.path
import re
import csv
from datetime import datetime
from tqdm import tqdm
import time

def link_retrieval(str = 'linksfile.txt'):
    '''
    this function write all the links to the page of the anime to analyze in a txt file
    
    input: name of the file that will contain the links
    output: total number number of pages (starting from 0)
    '''
    
    
    with open(str, 'w') as f:
        
        for i in tqdm(range(400)):
            
            # because the animes are organized in pages, we generate the link for every page
            url = 'https://myanimelist.net/topanime.php?limit=' + str(50*i)
            
            # retrieving the url
            page = requests.get(url)
            
            soup = BeautifulSoup(page.content, "html.parser")
            
            # check if the pages are finished
            if str(soup.title) == "<title>404 Not Found - MyAnimeList.net\n</title>":
                number_of_pages = i
                break

            links = []  # anime pages links in each page
            
            # we inferred from the source code of the site that the following tags and attribute contain the links we want
            for anime in soup.find_all('td', class_="title al va-t word-break"):
                links.append(anime.a['href'])
            
            # appending the links in the file
            for link in links:
                f.write(link)
                f.write('\n')
    
    return(number_of_pages)


def folder_generation(num_pages, name):
    '''
    this function creates the folder structure to save each page in
    
    input: total number of pages
    output: None
    '''    
    
    # retrieve the current working directory
    cwd = os.getcwd()
    
    # we will create a folder for each page
    # and we will put these folders inside ~/content/allpagess
    
    
    parent = 'content'
    if not os.path.isdir(parent):
        os.mkdir(parent)
    
    os.chdir(parent)
    
    parent = name
    if not os.path.isdir(parent):
        os.mkdir(parent)
    
    os.chdir(parent)
    
    
    for i in range(num_pages):
        path = str(i) +'page'
        if not os.path.isdir(path):
            os.mkdir(path)
    
    os.chdir(cwd)  #reset the current working directory
    
    return


def article_downloading(str = 'linksfile.txt', start_line = 1):
    '''
    this function downloads the page of every anime and saves it in a file named article_i.html
    it stops the crawling if it detects that the site imposed a captcha to be able to access the page
    
    input: name of the link file, number of line anime from which start the crawling
    output: number of the last crawled anime and a boolean which indicates whether the anime was the last one
    '''
    
    # retrieve the current working directory
    cwd = os.getcwd()
    
    
    with open(str, 'r') as f:
        
        # change directory to ~/content/allpagess/
        os.chdir('content/allpagess/')
        
        line_num = 0
        for line in tqdm(f):
            line_num += 1
            
            if (line_num>=start_line):
                
                # we wait 0.1 seconds between each crawl to prevent the site from overloading
                time.sleep(0.1)
                
                # download and parse the html of the page
                page = requests.get(line)
                soup = BeautifulSoup(page.content, "html.parser")
                
                # this checks whether the page we downloaded is the one which contains the captcha
                # and stops the crawling in case it is
                if (soup.find('div', class_="display-submit") is not None):
                    
                    os.chdir(cwd)
                    print(f'Problem in line {line_num}\nStopping the crawl...')
                    return((line_num, False))
                
                # change directory to the folder page
                folder_name = str(line_num//50) +'page'
                os.chdir(folder_name)
                
                # saves the html of the anime page
                name = 'article_' + str(line_num)+".html"
                with open(name, 'w') as anime:
                    anime.write(str(soup))
                
                # resets the directory to ~/content/allpagess/
                os.chdir('..')
    
    os.chdir(cwd)  #reset the current working directory
    
    total_lines = line_num
    return((total_lines,True))


def crawl(str = 'linksfile.txt', start_line = 1):
    '''
    this function restarts the crawling in case it fails because of the site overloading
    
    input: name of the link file, number of line anime from which start the crawling
    output: total number of downloaded anime pages
    '''
    
    finished = False

    while not finished:
        
        #restart the crawl from the last (incorrectly) crawled anime
        start_line, finished = article_downloading(str, start_line)

        # we wait for 60 seconds hoping that the site will let us in then
        if not finished:
            print('Waiting a minute...', end = '')
            time.sleep(60)
            print('done!')
    
    return(total_anime)


def data_crawling(str = 'linksfile.txt'):
    '''
    this function retrieves every anime page link, and then crawls every anime page
    
    input: name of the file in which we save the links
    output: total number of anime
    '''
    # we count the number of pages to generate the folder structure
    number_of_pages = anime_count(str)//50 + 1
    
    # we generate the folder structure
    folder_generation(number_of_pages, 'allpagess')
    
    # we crawl every anime page and save it into an html file
    total_anime = crawl()
    
    return(total_anime)


def datify(a: list):
    """
    Here I turn a list of strings in datetime format.
    
    :param a: list of strings, an example is ['Mar', '18,', '2019']
    returns date in datetime format
    """
    
    date = ' '.join(a)
    
    # remove commas
    date = date.replace(',', '')
    
    # parse dates
    if len(a) == 3:
        date = datetime.strptime(date, '%b %d %Y')
    elif len(a) == 2:
        date = datetime.strptime(date, '%b %Y')
    else:
        date = datetime.strptime(date, '%Y')
        
    return date


def find_aired(aired):
    
    aired = aired.split()[1:]
    
    if aired == ['Not', 'available']:
        return([None, None])
    
    try:
        to = aired.index('to')
    except ValueError:
        to = -1
    
    # parse release date
    if to != -1:
        releaseDate = datify(aired[:to])
    else:
        releaseDate = datify(aired)
        endDate = None
        return([releaseDate, endDate])
    
    # parse end date
    aired = aired[to+1:]
    if aired[0] == '?':
        endDate = None
    else:
        endDate = datify(aired)
        
    return([releaseDate, endDate])


def createanimerelated(a, retrieve_links = False):
    """
    Here I create a list of unique related anime.
    If retrieve_links = True the function saves the link of the related anime
    If retrieve_links = False the function saves the names of the related anime
    """
    
    if a is None:
        return (a)
    
    related = []
    for i in a.find_all('a', href=True):
        if (retrieve_links):
            url = 'https://myanimelist.net' + i['href']    #Since in the original html only the latter branches of the url are added, I add the fist part of the link
            if url not in related:
                related.append(url)
        else:
            name = i.text
            if name not in related:
                related.append(name)
            
    return related


def createanimecharacters(soup):
    """
    this function is used to retrieve every character, voice actor, and staff name
    """
    characters = []
    
    for char in soup:
        characters.append(char.a.text)

    return(characters)


def createanimestaff(soup):
    """
    this function parses the staff names and roles
    """
    
    # this should return the portion of text which corresponds
    # to the staff table in the page
    stafftable = soup.find('h2', text = 'Staff').parent
    
    # taking care of holes in the database
    if((stafftable.text).find('No staff for this anime have been added to this title. Help improve our database by adding staff for this anime here.') != -1):
        return (None)
    
    
    stafftable = stafftable.next_sibling.next_sibling
    
    
    # the elements of the list staff will be lists which contain:
    # in the  first entry: the name of the staff member
    # in the second entry: a list which contains the role assigned to the staff member
    staff = []
    
    for elements in stafftable.find_all('td', class_ = "borderClass", width = False):
        name = elements.a.text
        name = name.replace(',','')   #name of the member
        
        role = elements.div.small.text
        role = role.split(', ')       #roles of the member
        
        staff.append([name,role])
        
    return(staff)


def article_parsing(anime_num):
    
    '''
    this function reads the anime_num^th html page and retrieves the relevant information
    and returns them in a dictionary
    '''
    
    # setting file path
    page = anime_num//50
    path = f'content/allpagess/{page}page/article_{anime_num}.html'
    
    with open(path, 'r') as f:

        soup = BeautifulSoup(f, "html.parser")

        # parsing the title
        animeTitle = soup.find(class_="title-name h1_bold_none")
        animeTitle = animeTitle.strong.text

        # parsing the type
        animeType = soup.find(text = re.compile('Type:'), class_="dark_text").parent.text.split()[1]
        if animeType == 'Unknown':
            animeType = None

        # parsing the number of episodes
        animeNumEpisode = soup.find(text = re.compile('Episodes:'), class_="dark_text").parent.text.split()[1]
        if animeNumEpisode == 'Unknown':
            animeNumEpisode = None

        # parsing the release and end dates
        aired = soup.find(text = re.compile('Aired:'), class_="dark_text").parent.text
        releaseDate, endDate = find_aired(aired)

        # parsing the number of members
        animeNumMembers = soup.find(class_="numbers members").text.split()[1]
        animeNumMembers = animeNumMembers.replace(',', '')

        # parsing the score
        animeScore = soup.find( attrs={'data-title' : re.compile('score')} ).div.text
        if animeScore == 'N/A':
            animeScore = None

        # parsing the users
        animeUsers = soup.find( attrs={'data-title' : re.compile('score')} )['data-user']
        if animeUsers in ['- users', '- user']:
            animeUsers = None
        else:
            animeUsers = animeUsers.split()[0].replace(',', '')

        # parsing the rank
        animeRank = soup.find(class_="numbers ranked").text.split()[1]
        if animeRank == 'N/A':
            animeRank = None
        else:
            animeRank = animeRank.replace('#', '')

        # parsing the popularity
        animePopularity = soup.find(class_="numbers popularity").text.split()[1]
        animePopularity = animePopularity.replace('#', '')

        # parsing the description
        null_description = ['No synopsis has been added for this series yet.  Click here to update this information.', ' ', 'No synopsis information has been added to this title. Help improve our database by adding a synopsis here.']
        animeDescription = soup.find(itemprop="description").text.replace('\n', ' ')
        if animeDescription in null_description:
            animeDescription = None

        # parsing the related anime
        animeRelated = soup.find(class_="anime_detail_related_anime")
        animeRelated = createanimerelated(animeRelated)

        # parsing the characters
        animeCharacters = soup.find_all(class_="h3_characters_voice_actors")
        animeCharacters = createanimecharacters(animeCharacters)

        # parsing the voices
        animeVoices = soup.find_all(class_="va-t ar pl4 pr4")
        animeVoices = createanimecharacters(animeVoices)

        # parsing the staff
        animeStaff = createanimestaff(soup)
    
    
    # here we pack the gathered information in a dictionary
    anime = {}
    anime['animeTitle'] = animeTitle
    anime['animeType'] = animeType
    anime['animeNumEpisode'] = animeNumEpisode
    anime['releaseDate'] = releaseDate
    anime['endDate'] = endDate
    anime['animeNumMembers'] = animeNumMembers
    anime['animeScore'] = animeScore
    anime['animeUsers'] = animeUsers
    anime['animeRank'] = animeRank
    anime['animePopularity'] = animePopularity
    anime['animeDescription'] = animeDescription
    anime['animeRelated'] = animeRelated
    anime['animeCharacters'] = animeCharacters
    anime['animeVoices'] = animeVoices
    anime['animeStaff'] = animeStaff
    
    return(anime)


def search_tab_newline(str):
    '''
    return True if the input string contains a \t or \n character
    '''
    if str.find('\t') != -1: return True
    if str.find('\n') != -1: return True
    return False


def anime_check(anime, tab_removal = False, verbose = True):
    
    '''
    here we search for tab and newline character in the parsed data
    if verbose == True we print them on screen
    if tab_removal == True we remove the \t characters
    
    input anime: dictionary that contains the data we want to store in the tsv
    '''
    
    
    if search_tab_newline(anime['animeTitle']):
        if tab_removal:
            anime['animeTitle'].replace('\t', '')
        if verbose:
            print(f'Found a tab or newline in the Title, anime', anime['animeTitle'], 'rank', anime['animeRank'])
    
    if anime['animeType'] is not None:
        if search_tab_newline(anime['animeType']):
            if tab_removal:
                anime['animeType'].replace('\t', '')
            if verbose:
                print(f'Found a tab or newline in the Type, anime', anime['animeTitle'], 'rank', anime['animeRank'])

    if anime['animeNumEpisode'] is not None:
        if search_tab_newline(anime['animeNumEpisode']):
            if tab_removal:
                anime['animeNumEpisode'].replace('\t', '')
            if verbose:
                print(f'Found a tab or newline in the number of Episodes, anime', anime['animeTitle'], 'rank', anime['animeRank'])
                  
    if search_tab_newline(anime['animeNumMembers']):
        if tab_removal:
            anime['animeNumMembers'].replace('\t', '')
        if verbose:
            print(f'Found a tab or newline in the number of Members, anime', anime['animeTitle'], 'rank', anime['animeRank'])
                  
    if anime['animeScore'] is not None:
        if search_tab_newline(anime['animeScore']):
            if tab_removal:
                anime['animeScore'].replace('\t', '')
            if verbose:
                print(f'Found a tab or newline in Score, anime', anime['animeTitle'], 'rank', anime['animeRank'])
                  
    if anime['animeUsers'] is not None:              
        if search_tab_newline(anime['animeUsers']):
            if tab_removal:
                anime['animeUsers'].replace('\t', '')
            if verbose:
                print(f'Found a tab or newline in the Users, anime', anime['animeTitle'], 'rank', anime['animeRank'])
                
    if anime['animeRank'] is not None:
        if search_tab_newline(anime['animeRank']):
            if tab_removal:
                anime['animeRank'].replace('\t', '')
            if verbose:
                print(f'Found a tab or newline in the Rank, anime', anime['animeTitle'], 'rank', anime['animeRank'])
                  
    if search_tab_newline(anime['animePopularity']):
        if tab_removal:
            anime['animePopularity'].replace('\t', '')
        if verbose:
            print(f'Found a tab or newline in the Popularity, anime', anime['animeTitle'], 'rank', anime['animeRank'])
    
    if anime['animeDescription'] is not None:
        if search_tab_newline(anime['animeDescription']):
            if tab_removal:
                anime['animeDescription'].replace('\t', ' ')
            if verbose:
                print(f'Found a tab or newline in the Description, anime', anime['animeTitle'], 'rank', anime['animeRank'])
    
    if anime['animeRelated'] is not None:
        for i in range(len(anime['animeRelated'])):
            if search_tab_newline(anime['animeRelated'][i]):
                if tab_removal:
                    anime['animeRelated'][i].replace('\t', '')
                if verbose:
                    print(f'Found a tab or newline in the anime related names, anime', anime['animeTitle'], 'rank', anime['animeRank'])
                
    for i in range(len(anime['animeCharacters'])):
        if search_tab_newline(anime['animeCharacters'][i]):
            if tab_removal:
                anime['animeCharacters'][i].replace('\t', '')
            if verbose:
                print(f'Found a tab or newline in the anime Characters, anime', anime['animeTitle'], 'rank', anime['animeRank'])
            
    for i in range(len(anime['animeVoices'])):
        if search_tab_newline(anime['animeVoices'][i]):
            if tab_removal:
                anime['animeVoices'][i].replace('\t', '')
            if verbose:
                print(f'Found a tab or newline in the anime Voices, anime', anime['animeTitle'], 'rank', anime['animeRank'])
    
    if anime['animeStaff'] is not None:
        for i in range(len(anime['animeStaff'])):
            if search_tab_newline(anime['animeStaff'][i][0]):
                if tab_removal:
                    anime['animeStaff'][i][0].replace('\t', '')
                if verbose:
                    print(f'Found a tab or newline in the staff Names, anime', anime['animeTitle'], 'rank', anime['animeRank'])
            for j in range(len(anime['animeStaff'][i][1])):
                if search_tab_newline(anime['animeStaff'][i][1][j]):
                    if tab_removal:
                        anime['animeStaff'][i][1][j].replace('\t', '')
                    if verbose:
                        print(f'Found a tab or newline in the staff Roles, anime', anime['animeTitle'], 'rank', anime['animeRank'])
                    
    return

    
def anime_count(filename = 'linksfile.txt'):
    '''
    input: file name
    output: number of line in the file
    '''
    with open(filename) as f:
        data = f.readlines()
    return(len(data))


def tsv_gen(filename = 'linksfile.txt', verbose = False):
    '''
    this function create the folder structure and the tsv file
    if verbose == True the files are checked for tab and newline characters and the results are written on screen
    '''
    
    # define tsv columns
    col = ['animeTitle', 'animeType', 'animeNumEpisode', 'releaseDate', 'endDate', 'animeNumMembers', 'animeScore', \
           'animeUsers', 'animeRank', 'animePopularity', 'animeDescription', 'animeRelated', 'animeCharacters', \
           'animeVoices', 'animeStaff']
    
    # count all the anime and pages
    total_anime = anime_count(filename)
    total_pages = total_anime//50 + 1
    
    #creating the folder structure to save the tsv
    folder_generation(total_pages, 'alltsv')
    
    for i in tqdm(range(1,total_anime+1)):
        
        # parse the article i in a dictionary
        anime = article_parsing(i)
        
        # check and correct problems in the file (removes \t in the data)
        anime_check( anime, tab_removal = True, verbose = verbose )
        
        # set tsv file name
        page = i//50
        tsvname = f'content/alltsv/{page}page/anime_{i}.tsv'
        
        # write to tsv        
        with open(tsvname, 'w') as f:
            tsv_writer = csv.writer(f, delimiter='\t')
            tsv_writer.writerow(col)
            tsv_writer.writerow(list(anime.values()))
        
    return

