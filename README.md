# Homework 3 - Group 6

In this repository, you can find all you need to implement a search engine over *Top Anime Series* from the list of [MyAnimeList](https://myanimelist.net).

Here a list of the files/folders that are there and their function.

* **data_collection.py**
> It is a python script which contains the functions that are needed to collect our data for the search engine. 
> 
> Specifically we used them to get all the links from the list of Anime and then to scrape the collected web pages which have been stored in files called `article.html`. Moreover, there is a function that creates one file with extention `tsv` for each *article.htlm* which contanins some imformations about the anime and its popularity, such as the title, the type, dates of release and end, characters and more over.
>
> Due the number of the Anime that we have to analyze, these files (`anime.tsv`) have been organized in different [pages](https://www.dropbox.com/sh/yj17csp9f630rf8/AACfnQne-eRctns0bXaXE7q6a?dl=0).
 

- **linksfile.txt**
> It contains the links of all the Anime in the list.

-  **tsv_management.py**
> This script contains some useful functions which we used to manipulate easily our data from `anime.tsv` files.

-  **search_engine.py**
> In this script there are some functions that we used to preprocess the description of the anime, to crete some useful dictionary and to implement three different search engine, given a query.
> 1. It is the basic one. Its purpose is to select the Anime that have all the words of the query in their description.
> 2. The second search engine is based on the output of the first one and in addition it sorts the output Anime on a similarity score between the Anime and the query (cosine similarity).
> 3. Finally, this search engine is a bit more complex then the other ones. It searches among descriptions and title and the selected Anime are not sorted based only on tf-idf score but on a new score that considers the popularity rank, the score and the Anime Type. 

- **Dictionary**
> A folder which includes some helpful dictionaries for search engine. There are:
>
> 1. vocabulary.json
> 
> This is a dictionary that maps each *'processed'* word to a integer.
>
> 2. unranked_inverted_index.json
> 
> This is a dictionary that contanins for each word of the volabulary a list of Anime that contains that word in the description. We have used it in the fisrt search      engine.
>
> 3.  ranked_inverted_index.json
> 
> This is a dictionary that is useful in the second search engine. It contains not only the Anime that contains each word in the description, as the previous one, but also the tf-idf score. 
>
> 4.  tfidf_vectors.json
> 
> It is a doctionary that links to each Anime description a dictionary that contains the tf-idf score for all the words in it. 
> So that, we have vectors of Anime with tf-idf scores (greater than zero) that we have used for compute cosine similarity.


 - **Notebook**
 > It is a python notebook that contains 3/4 ???? main sections, one for the data collection, one for search engines.




