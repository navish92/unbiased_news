import streamlit as st
import pandas as pd
import numpy as np
import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import copy
import random
import requests
from bs4 import BeautifulSoup
import random
import time
from newspaper import Article
import re
import unicodedata

# Open Entry field to enter a left leaning source --- DONE
# Open entry field to enter a right leaning source --- DONE
# function to scrape the news from each of these sources --- DONE
# function to explode into sentences & clean up sentences a bit --- DONE
# function to encode into embeddings & store
# function to create comparisons & printout

def general_scraper(url):
    """
    Scrapes the provided url link using python's requests module and returns a BS object containing returned information text
    Scraping wrapped around try-except blocks along with conditional check if status code is 200.

    Args:
        url ([str]): website to be scrapped.

    Returns:
        soup [Beautiful Soup object]: Returns scraped text in a Beautiful Soup object type.
    """

    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html5lib')
            return soup
        else:
            print(f"Did not get status code 200 for url: \n{url}\n.Instead got status code {response.status_code}")
            return None
    except Exception as err_msge:
        print(f"Error while scraping: {err_msge}")
        return None

def scrape_news(story_links):

    if type(story_links) == str:
        story_links = [story_links]

    stories_columns = ['news_title', 'news_source', 'global_bias', 'News_link', 'text']
    stories_df = pd.DataFrame(columns=stories_columns)

    for link in story_links:

        story_dict = {} 
        story_dict['news_link'] = link

        if link.find("nytimes.com") >= 0:

            try:
                soup = general_scraper(link)
                text = "\n\n".join([para.text for para in soup.find_all('p', class_="css-158dogj evys1bk0")])
                title = soup.find('h1', class_='css-ymxi58 e1h9rw200').text
                story_dict['text'] = text
                story_dict['news_title'] = title
                story_dict['news_source'] = 'New York Times (News)'
                story_dict['global_bias'] = 'From the Left'
            except:
                print(f"Error retrieving article from {link}")

        elif link.find("washingtontimes.com") >= 0:

            try:
                soup = general_scraper(link)
                title = soup.find('h1', class_='page-headline').text
                text = "\n\n".join([para.text for para in soup.find('div', class_='storyareawrapper').find_all('p')[:-5]])
                story_dict['text'] = text
                story_dict['news_title'] = title
                story_dict['news_source'] = 'Washington Times'
                story_dict['global_bias'] = 'From the Right'
            except:
                print(f"Error retrieving article from {link}")

        else:

            try:
                article = Article(link)
                article.download()
                article.parse()
                
                authors = article.authors
                publish_date = article.publish_date
                text = article.text
                news_title = article.title
                
                story_dict['text'] = text
                story_dict['news_title'] = news_title

                if link.find("washingtonpost.com") >= 0:
                    story_dict['news_source'] = 'Washington Post'
                    story_dict['global_bias'] = 'From the Left'
                
                if link.find("huffpost.com") >= 0:
                    story_dict['news_source'] = 'HuffPost'
                    story_dict['global_bias'] = 'From the Left'  

                if link.find("foxnews.com") >= 0:
                    story_dict['news_source'] = 'Fox News (Online News)'
                    story_dict['global_bias'] = 'From the Right'
            except:
                print(f"Error retrieving article from {link}")

        stories_df = stories_df.append(story_dict, ignore_index=True)

    return stories_df

def sent_split(article):
    sent_list2 = []
    sent_list = nltk.sent_tokenize(article)
    for sent in sent_list:
        sent_list2.extend(sent.split('\n\n'))
    return sent_list2
    
def simple_cleaning(text_sent):
    text_sent = text_sent.lower()

    if ((text_sent == 'ad') or (text_sent.find('click here') >= 0 ) or (text_sent.find('sign up here') >= 0 ) or
        (text_sent.find('sign up for daily') >= 0 ) or (text_sent.find('sign up for the') >= 0 ) or
        (text_sent.find('contributed to this') >= 0 ) or (text_sent.find('all rights reserved') > 0 ) or
        (text_sent.find('reported from') >= 0 ) or (text_sent.find('contributed reporting') >= 0 ) or
        (text_sent.find('want fox news') >= 0) or (text_sent == '') or
        (text_sent.find('the washington times, llc') >= 0) or (text_sent.find('sign up for our') >= 0) or
        (text_sent.find('daily to your inbox') >= 0)
       ): 
        return False
    elif len((re.sub('[^a-z\s]', '', text_sent)).split()) <= 5:
        return False
    else:
        return True

def explode_and_clean(df):
    
    df['text_ascii'] = df.text.map(lambda x: unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('ascii'))

    df_sentences = df[['news_title', 'global_bias','news_link','news_source','text_ascii']].copy(deep=True)

    # # Splitting each para into a list of paras
    df_sentences['text_sent_list'] = df_sentences.text_ascii.map(sent_split)

    # # Exploding the paragraphs into a dataframe, where each row has a paragraph
    df_sentences_col = pd.DataFrame(df_sentences.text_sent_list.explode())
    df_sentences_col.rename(columns={'text_sent_list':'text_sent'}, inplace=True)

    df_sentences_col = df_sentences_col[df_sentences_col.text_sent.map(simple_cleaning)]
    df_sentences_col = df_sentences_col[~(df_sentences_col.text_sent.isna())]

    # # Joining the exploded dataframe back, so that other metadata can be associated with it
    df_sentences = df_sentences.join(df_sentences_col, how='left').reset_index()
    df_sentences.rename(columns={'index':'article'}, inplace=True)
    df_sentences.drop(columns='text_sent_list', inplace=True)

    # getting paragraph numbering
    df_sentences['text_count'] = df_sentences.groupby('article').cumcount()

    del df_sentences_col
    
    return df_sentences

def unbias_gen(df):
    
    article_left = df[df.global_bias == 'From the Left']
    article_right = df[df.global_bias == 'From the Right']
    
    embeddings_left = model.encode(article_left.text_sent.tolist(), num_workers = 8)
    embeddings_right = model.encode(article_right.text_sent.tolist(), num_workers = 8)
    
    left_article_len = len(article_left)
    right_article_len = len(article_right)

    if left_article_len >= right_article_len:
        smaller_article = article_right.copy(deep=True)
        smaller_embedding = copy.deepcopy(embeddings_right)
        bigger_article = article_left.copy(deep=True)
        bigger_embedding = copy.deepcopy(embeddings_left)
    else:
        smaller_article = article_left.copy(deep=True)
        smaller_embedding = copy.deepcopy(embeddings_left)
        bigger_article = article_right.copy(deep=True)
        bigger_embedding = copy.deepcopy(embeddings_right)  

    smaller_article_html_text = smaller_article.text_ascii.iloc[0]
    bigger_article_html_text = bigger_article.text_ascii.iloc[0]

    cosine_scores = np.array(util.pytorch_cos_sim(smaller_embedding, bigger_embedding))

    cos_scores_df = pd.DataFrame(cosine_scores)
    pairs = []

    for row in range(len(smaller_article)):

        i = cos_scores_df.max(axis='columns').idxmax()
        j = cos_scores_df.loc[i].idxmax()

        pairs.append({'index': [i, j], 'score': cos_scores_df.loc[i,j]})

        cos_scores_df.drop(index = i, inplace=True)
        cos_scores_df.drop(columns = j, inplace=True)


    filtered_pairs = []
    counter = 0
    sent_limit = min(10,round(len(smaller_article)/2))
    summary_article = []

    for pair in sorted(pairs, key=lambda x: x['index'][0]):

        score = pair['score']

        if score >= 0.55 and counter < min(10,sent_limit):
            counter += 1
            filtered_pairs.append(pair)


    for pair in filtered_pairs:

        smaller_art_sent_len = len(smaller_article.iloc[pair['index'][0]].loc['text_sent'])
        bigger_art_sent_len = len(bigger_article.iloc[pair['index'][1]].loc['text_sent'])

        if smaller_art_sent_len >= bigger_art_sent_len:
            summary_article.append(smaller_article.iloc[pair['index'][0]].loc['text_sent'])
        else:
            summary_article.append(bigger_article.iloc[pair['index'][1]].loc['text_sent'])

        start_pos_smaller = smaller_article_html_text.find(smaller_article.iloc[pair['index'][0]].loc['text_sent'])
        end_pos_smaller = start_pos_smaller + smaller_art_sent_len
        
        if (start_pos_smaller >= 0) and (end_pos_smaller >= 10):
            smaller_article_html_text = smaller_article_html_text[:start_pos_smaller] + '<span class="highlight-green">' + \
                                        smaller_article_html_text[start_pos_smaller:end_pos_smaller] + '</span>' + \
                                        smaller_article_html_text[end_pos_smaller:]
        
        start_pos_bigger = bigger_article_html_text.find(bigger_article.iloc[pair['index'][1]].loc['text_sent'])
        end_pos_bigger = start_pos_bigger + bigger_art_sent_len
        
        if (start_pos_bigger >= 0) and (end_pos_bigger >= 10):
            bigger_article_html_text = bigger_article_html_text[:start_pos_bigger] + '<span class="highlight-green">' + \
                                    bigger_article_html_text[start_pos_bigger:end_pos_bigger] + '</span>' + \
                                    bigger_article_html_text[end_pos_bigger:]


    sum_art_to_publish = '<br>'.join(summary_article)

    st.write("<br><b><u>Summarized Unbiased Article</u></b>", unsafe_allow_html=True)
    st.markdown(sum_art_to_publish, unsafe_allow_html=True)
    st.markdown('''---''')
    st.write('<br><b><u>Source Articles Analysis</u></b><br>', unsafe_allow_html=True)
    st.write('<b>Note:</b> Sentences that were determined to be common in both articles are highlighted in green below.' , unsafe_allow_html=True)

    st.write(f"<u>Article {smaller_article.iloc[0].loc['global_bias']} source is published by {smaller_article.iloc[0].loc['news_source']}</u>", unsafe_allow_html=True)
    st.write(f"Title: <b>{smaller_article.iloc[0].loc['news_title']} </b>", unsafe_allow_html = True)

    st.markdown(smaller_article_html_text, unsafe_allow_html = True)

    st.write()

    st.write(f"<u>Article {bigger_article.iloc[0].loc['global_bias']} source is published by {bigger_article.iloc[0].loc['news_source']}</u>", unsafe_allow_html=True)
    st.write(f"Title: <b>{bigger_article.iloc[0].loc['news_title']} </b>", unsafe_allow_html = True)
    st.markdown(bigger_article_html_text, unsafe_allow_html = True)

    return summary_article

# def make_clickable_link (url, text):
#     '''
#     Will make any link a hyperlink of the given text.
#     '''

#     return f'<a target="_blank" href="{url}">{text}</a>'

if __name__ == "__main__":
    # model = SentenceTransformer("E:\\roberta_large_sentence_transformer")
    # a = model.encode(['I hope this works. I am hungry.','Sentence two for example lets make it slightly longer and leaner.'], num_workers = 4)
    # st.write('Encoding Finished')
    # st.write(a)
    sentence_green_highlight = """
    <style>
        .highlight-green {
            background-color: #00cc00;
        }
    </style>
    """
    st.markdown(sentence_green_highlight,  unsafe_allow_html=True)
    st.markdown('''
    ## Welcome
    To the first of its kind - An automated Unbiased News Aggregator!
    ''')
    st.write()

    st.markdown('''
    Please choose the link to a left & right leaning news article from any the specified news outlets:  
    ''')

    st.markdown('''
    ### Left Leaning News Outlets:
    - HuffPost 
    - New York Times (News)
    - Washington Post 
    ''')
    left_url = st.text_input('Enter link for an article that leans left')
    st.write()
    st.markdown('''
    ### Right Leaning News Outlets:
    - Washington Times 
    - Fox News (Online News)
    ''')
    right_url = st.text_input('Enter link for an article that leans right')

    button_summarize = st.button("Summarize for me!")
    
    model = SentenceTransformer("E:\\roberta_large_sentence_transformer")

    if button_summarize:
        stories_df = scrape_news([left_url, right_url])
        df_sentences = explode_and_clean(stories_df)
        summary = unbias_gen(df_sentences)





