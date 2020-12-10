import streamlit as st
import pandas as pd
import numpy as np
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import copy
import random

# @st.cache(allow_output_mutation=True)
# def load_data():
#     df_small = pd.read_csv("Data/smaller_sent_df.csv")
#     df_embeds = pd.read_csv("Data/smaller_sent_embeddings.csv", header = None)
    
#     return df_small, df_embeds
#     # return df_small, df_embeds, model

sentence_green_highlight = """
<style>
    .highlight-green {
        background-color: #00cc00;
    }
</style>
"""
st.markdown(sentence_green_highlight,  unsafe_allow_html=True)

def unbias_gen(number):
    
    article_left = df_small[(df_small.number == number) & (df_small.global_bias == 'From the Left')]
    article_right = df_small[(df_small.number == number) & (df_small.global_bias == 'From the Right')]
    
    embeddings_left = df_embeds.loc[article_left.index,:].values
    embeddings_right = df_embeds.loc[article_right.index,:].values
    
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

        if score >= 0.55 and counter < sent_limit:
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
        
        print(start_pos_bigger, end_pos_bigger)
        if (start_pos_bigger >= 0) and (end_pos_bigger >= 10):
            
            bigger_article_html_text = bigger_article_html_text[:start_pos_bigger] + '<span class="highlight-green">' + \
                                    bigger_article_html_text[start_pos_bigger:end_pos_bigger] + '</span>' + \
                                    bigger_article_html_text[end_pos_bigger:]


    sum_art_to_publish = ' <br> '.join(summary_article)

    # summary_size = len(sum_art_to_publish)

    # sum_art_to_publish = sum_art_to_publish[:summary_size//4] + '<span class="highlight-green">' + \
    #                         sum_art_to_publish[summary_size//4:summary_size//2] + '</span>' + sum_art_to_publish[summary_size//2:]
    st.markdown(sum_art_to_publish, unsafe_allow_html=True)

    st.write()

    st.write(f"Article {smaller_article.iloc[0].loc['global_bias']} source is published by {smaller_article.iloc[0].loc['news_source']}")
    st.write(f" Titled: <b>{smaller_article.iloc[0].loc['news_title']} </b>", unsafe_allow_html = True)

    st.markdown(smaller_article_html_text, unsafe_allow_html = True)
    st.write()
    st.write(f"Article {bigger_article.iloc[0].loc['global_bias']} source is published by {bigger_article.iloc[0].loc['news_source']}")
    st.write(f" Title: <b>{bigger_article.iloc[0].loc['news_title']} </b>", unsafe_allow_html = True)

    st.markdown(bigger_article_html_text, unsafe_allow_html = True)

# df_small, df_embeds, model = load_data()
# df_small, df_embeds = load_data()

# model = SentenceTransformer("E:\\roberta_large_sentence_transformer")

# number_list_size = list(set(df_small.number.tolist()))
# button = st.button("Generate Article Randomly!")

# st.write("reached here")
# a = model.encode(['I hope this works. I am hungry.','Sentence two for example lets make it slightly longer and leaner.'], num_workers = 4)
# st.write(a)

# if button:
    # random_number = random.randint(0,len(number_list_size)-1)
    # unbias_gen(number_list_size[random_number])


if __name__ == "__main__":
    model = SentenceTransformer("E:\\roberta_large_sentence_transformer")
    a = model.encode(['I hope this works. I am hungry.','Sentence two for example lets make it slightly longer and leaner.'], num_workers = 4)
    st.write('Encoding Finished')
    st.write(a)





