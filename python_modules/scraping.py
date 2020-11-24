import pandas as pd
import requests
from bs4 import BeautifulSoup
import random
import time
from newspaper import Article
import re 

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

def allsides_sitemap_story_parser(urls):
    """
    Scrapes & Parses allsides.com sitemap to return all links that have '/story/' in them

    Args:
        urls [str or list]: Sitemap url. Can pass in a list of url strings or a single url as a list or string)

    Returns:
        story_links_all [list]: List of all 'story' URL links found in the sitemap url.
    """

    story_links_all = []
    
    # Check if a single link was passed as string. 
    # Convert to a list of 1 element to ensure downstream compatibility
    if type(urls) == str:
        urls = [urls]
    
    for url in urls:
        soup = general_scraper(url)
        if soup:
            story_links = [link.text for link in soup.find_all('loc') if link.text.find('/story/') > 0]
            story_links_all.extend(story_links)
        else:
            print(f"No results from {url}")
    
    return story_links_all

def allsides_story_parser(story_links, filename = "", verbose = 0):
    """
    Takes in '/story/' links for allsides.com and returns a dictionary containing information parsed from the site.

    Args:
        story_links ([list] or [str]): A list (or single string) of links to what allsides.com calls stories.
                                       With a summary & three stories being linked.
        filename (str, optional): if the output should be saved, pass in filename with .csv extension. Defaults to "".
        verbose (Int, optional): If greater than 0, prints a snippet of the scraping output every nth iteration, as specified here.
                                So, if 20 is passed for verbose, output printed every 20 iterations.
    Returns:
        stories_df [Pandas Dataframe]: Returns a pandas dataframe containing the parsed info. 
    """

    stories_columns = ['title', 'date', 'summary', 'news_sources']
    stories_df = pd.DataFrame(columns=stories_columns)

    # If a single string is passed instead of list, changing it a list of one element
    if type(story_links) == str:
        story_links = [story_links]

    counter = 0

    for link in story_links:

        counter += 1
        story_dict = {}
        news_sources = [] 
        soup = general_scraper(link)
        
        story_dict['link'] = link

        try:
            story_dict['title'] = soup.find('h1').text.strip()
        except:
            story_dict['title'] = None
    
        try:
            story_dict['date'] = soup.find('span', class_='date-display-single').text.strip()
        except:
            story_dict['date'] = None

        try:
            story_dict['summary'] = [para.text.strip() for para in soup.find('div', class_='story-id-page-description').find_all('p')]
        except:
            story_dict['summary'] = []

        try:
            external_links = soup.find('div', class_='region-help').find_all('div', class_='quicktabs-views-group')
        except:
            external_links = []
        
        for ext_link in external_links:
            ext_news_dict = {}
            
            try:
                ext_news_dict['news_title'] = ext_link.find('div', class_='news-title').text.strip()
            except:
                ext_news_dict['news_title'] = None
            
            try:
                ext_news_dict['news_title'] = ext_link.find('div', class_='news-title').text.strip()
            except:
                ext_news_dict['news_title'] = None
                
            try:
                ext_news_dict['news_source'] = ext_link.find('div', class_='news-source').text.strip()
            except:
                ext_news_dict['news_source'] = None
            
            try:
                ext_news_dict['news_link'] = ext_link.find('div', class_='news-title').find('a').get('href')
            except:
                ext_news_dict['news_link'] = None
            
            try:
                ext_news_dict['global_bias'] = ext_link.find('div', class_='global-bias').text.strip()
            except:
                ext_news_dict['global_bias'] = None
            
            try:
                ext_news_dict['bias'] = ext_link.find('div', class_='bias-image').find('img').get('alt').split(':')[-1].strip()
            except:
                ext_news_dict['bias'] = None
                
            try:
                ext_news_dict['paras'] = ext_link.find('div', class_='news-body').text.strip()
            except:
                ext_news_dict['paras'] = None

            news_sources.append(ext_news_dict)
        
        story_dict['news_sources'] = news_sources
        
        stories_df = stories_df.append(story_dict, ignore_index=True)

        if filename:
            stories_df.to_csv(filename, index=False)
        
        if (verbose) and (counter % verbose == 0):
            print(counter, story_dict['link'],story_dict['title'],story_dict['date'])
 
        time.sleep(.5+.5*random.random())

    return stories_df

def newspaper3k_articles(stories_flat_df):

    for row_idx, row in stories_flat_df.iterrows():
        url = row['news_link']
        authors = []
        publish_date = None
        text = None
        
        try:
            article = Article(url)
            article.download()
            article.parse()
            
            authors = article.authors
            publish_date = article.publish_date
            text = article.text
            
        except:
            print(f"Error retrieving article from {url}")
        
        stories_flat_df.loc[row_idx, 'authors'] = authors
        stories_flat_df.loc[row_idx, 'publish_date'] = publish_date
        stories_flat_df.loc[row_idx, 'text'] = text

    return stories_flat_df

def fox_news_url_cleaner(url):
    """
    

    Args:
        url ([type]): [description]

    Returns:
        [type]: [description]
    """

    if url.find("www.foxnews.com") >= 0:
        url = re.sub(r'/\d*/\d*/\d*', '', url)
        url = re.sub(r'.html\b', '', url)
        url = url.rstrip('/')
        return url
    else:
        return url