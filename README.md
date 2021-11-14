# Homework 3 - Group 6

In this repository, you can find all you need to implement a search engine over *Top Anime Series* from the list of [MyAnimeList](https://myanimelist.net).

Here a list of the files/folders that are there and their function.

* **data_collection.py**
> It is a python library which contains the functions that are needed to collect our data for the search engine. 
> 
> Specifically we used them to get all the links from the list of Anime and then to scrape the collected web pages which have been stored in files called `article.html`. Moreover, there is a function that creates one file with extension `tsv` for each *article.htlm* which contanins some imformations about the anime and its popularity, such as the title, the type, dates of release and end, characters and more over. It also contains some analogous functions used to download the first five reviews of each anime.
>
> Due the number of the Anime that we have to analyze, these files (`anime.tsv`) have been organized in different [pages](https://www.dropbox.com/sh/lfy85uhcojfawee/AAB9s7NzE6FU12ZMs44vyY8Fa?dl=0).
 

- **linksfile.txt**
> It contains the links of all the Anime in the list.

-  **tsv_management.py**
> This library contains some useful functions that we used to manipulate easily our data from `anime.tsv` files.

-  **search_engine.py**
> In this library there are some functions that we used to preprocess the description of the anime, to create some useful dictionaries and to implement two different search engines:
> 1. It is the basic one. Its purpose is to select the Anime that have all the words of the query in their description.
> 2. The second search engine is based on the output of the first one and in addition it sorts the output Anime on a similarity score between the Anime and the query (cosine similarity).

-  **custom_search_engine.py**
> In this library there are some functions that we used to preprocess the description, titles, characters voices and staff names, for each anime, and to create some useful dictionaries used to implement a custom search engine:
> * Given a query, this third search engine selects all the anime that have at least one word in common with the query (searching in the anime description, title, characters, voices, staff); these results are then ranked based on a custom score that the user can customize.

- **content**
> A folder which includes some helpful dictionaries for search engine and the reviews dataset. There are:
>
> 1. vocabulary.json
> 
> This is a dictionary that maps each *'processed'* word to a integer.
>
> 2. unranked_inverted_index.json
> 
> This is a list that contains for each word of the vocabulary (the id of the word corresponds to the index in the list) a list of Anime that contains that word in the description. We have used it in the first search engine.
>
> 3.  ranked_inverted_index.json
> 
> This is a dictionary that is useful in the second search engine. It contains not only the Anime that contains each word in the description, as the previous one, but also the tf-idf score for each word.
>
> 4.  tfidf_vectors.json
> 
> It is a dictionary that links to each Anime description a dictionary that contains the tf-idf score for all the words in it. 
> So that, we have vectors of Anime with tf-idf scores (greater than zero) that we have used to compute cosine similarity.
> 
> 5. titles_descriptions_vocabulary.json
> 
> This is a dictionary that maps each *'processed'* word to a integer and is built on the set of all words contained in the anime descriptions, titles, characters, voices and staff.
>
> 6.  anime_scoring_info.json
> 
> It is a dictionary that links to each Anime its type and its popularity and rank score (defined customly starting from the animeScore and animePopularity columns in the dataset)
> 
> 7.  description_tfidf_vectors.json
> 
> It is a dictionary that links to each Anime description a dictionary that contains the tf-idf (in the context in which all the other documents contains also the titles, characters, voices, staff) score for all the words in it.
> 
> 8.  partial_tfidf_vectors.json
> 
> It is a dictionary that links to each Anime partial document (i.e. a document that contains the title, characters, voices and staff) a dictionary that contains the tf-idf (in the context in which all the other documents contains also the description) score for all the words in it.
> 
> 9.  titles_descriptions_inverted_index.json
> 
> This is a list that contains for each word of the titles_descriptions_vocabulary (the id of the word corresponds to the index in the list) a list of Anime that contains that word in the description (or title, or characters, or voices or staff). We have used it in the first custom search engine.
> 
> 10.  all_reviews.tsv
> 
> It is a dataset that contains the first five reviews for each anime (actually about the first 500 characters of each review). Each line is associated to an anime and the reviews are tab-separated.

 - **ADM-HW3.ipynb**
 > It is a python notebook that contains 5 main sections, one for the data collection, one for search engines.


