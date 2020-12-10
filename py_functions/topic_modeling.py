import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation, NMF
from sklearn.metrics.pairwise import cosine_similarity

def lda_topic_modeling(word_matrix, vocab, n = 5):

    lda = LatentDirichletAllocation(n_components=n, random_state=0, max_iter = 100, n_jobs = -1, verbose = 1)
    lda.fit(word_matrix)
    topic_matrix = pd.DataFrame(lda.transform(word_matrix)).add_prefix("topic_")
    word_matrix = pd.DataFrame(lda.components_, \
        columns = vocab).T.add_prefix('topic_')

    return lda, lda.bound_, topic_matrix, word_matrix

def nmf_topic_modeling (word_matrix, vocab, n = 5):

    nmf = NMF(n_components = n, max_iter = 1000)
    nmf.fit(word_matrix)

    topic_matrix = pd.DataFrame(nmf.transform(word_matrix)).add_prefix("topic_")
    word_matrix = pd.DataFrame(nmf.components_, \
        columns = vocab).T.add_prefix('topic_')

    return nmf, nmf.reconstruction_err_, topic_matrix, word_matrix

def top_reviews(topic_matrix_df, topic, n_reviews):
    return (topic_matrix_df
            .sort_values(by=f'topic_{topic}', ascending=False)
            .head(n_reviews)['raw_review']
            .values)

def top_words(word_topic_matrix_df, topic, n_words):
    return (word_topic_matrix_df
            .sort_values(by=f'topic_{topic}', ascending=False)
            .head(n_words))[f'topic_{topic}']

def top_words_for_all_topics(word_matrix, n_topics, n_words = 15):
    for i in range(n_topics):
        # topic = "topic_" + str(i)
        topic = i
        print(f'Topic {i}')
        words = top_words(word_matrix, topic, n_words).index
        for word in words:
            print(word, end = ', ')
        print('\n')
    
    return None

    
def lsa_topic_modeling(word_matrix, vocab, n = 5):

    lsa = TruncatedSVD(n)
    lsa.fit(word_matrix)
    topic_matrix = pd.DataFrame(lsa.transform(word_matrix)).add_prefix('topic_')
    # lsa.explained_variance_ratio_

    word_matrix = pd.DataFrame(lsa.components_, columns = vocab).T.add_prefix('topic_')

    return lsa, lsa.explained_variance_, topic_matrix, word_matrix
