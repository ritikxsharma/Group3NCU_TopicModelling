import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import re
import string

import spacy

import gensim
from gensim import corpora
from nltk.stem import PorterStemmer

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords


"""CLEANING"""

def clean_text(t): 
    delete_dict = {sp_character: '' for sp_character in string.punctuation}
    delete_dict[' '] = ' ' 
    txt = ' '.join(map(str,t))
    table = str.maketrans(delete_dict)
    text1 = txt.translate(table)
    textArr= text1.split()
    text2 = ' '.join([w for w in textArr if ( not w.isdigit() and  ( not w.isdigit() and len(w)>3))]) 
    return list(text2.lower().split('|'))



"""REMOVING STOPWORDS"""

def remove_stopwords(t):
    en = spacy.load('en_core_web_sm')
    stpwrds = en.Defaults.stop_words
    all_stopwords = stpwrds.union(stopwords.words('english'))
    finalList = []
    for i in t:
        textArr = i.split(' ')    
        rem_text = ' '.join([i for i in textArr if i not in all_stopwords])
        finalList.append(rem_text)
    return finalList


    """LEMMATIZATION"""

def lemmatize(t):
    nlp = spacy.load('en_core_web_sm', disable=['parser','ner'])
    #text_list = text.tolist()
    splem = []
    wordlist = []
    for i in t:
        wordlist.append(i)
        lst = word_tokenize(i)
        #print(lst)
        #wordlist.append(lst)
        doc = nlp(' '.join(map(str,lst)))
        lem = " ".join([token.lemma_ for token in doc])
        #print(lem)
        splem.append(lem)
    #return(' '.join(map(str,splem)))
    return splem


    """Making the doc_term_matrix"""
def mtrx(t):
    tokens = [d.split() for d in t]
    dictionary = corpora.Dictionary(tokens)
    doc_term_matrix = [dictionary.doc2bow(rev) for rev in tokens]
    return doc_term_matrix


"""MODEL BUILDING"""

def lda_model_build(t,doc_term_matrix):
    tokens = [d.split() for d in t]
    dictionary = corpora.Dictionary(tokens)
    #doc_term_matrix = [dictionary.doc2bow(rev) for rev in tokens]
    LDA = gensim.models.ldamodel.LdaModel
    lda_model = LDA(corpus=doc_term_matrix, id2word=dictionary, num_topics=10, random_state=100,
                chunksize=1000, passes=50,iterations=100)
    return lda_model
    


"""SHOW THE DOMINANT TOPIC WITH THE KEYWORDS"""

def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
            # if(j<=1):
            #     wp = ldamodel.show_topic(topic_num)
            #     topic_keywords = ", ".join([word for word, prop in wp])
            #     sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            # if(j==0):
            #     continue
            # else:
            #     break

    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    df_dominant_topic = sent_topics_df.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    return df_dominant_topic


# df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=doc_term_matrix, texts=for_end)

# # Format
# df_dominant_topic = df_topic_sents_keywords.reset_index()
# df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# # Show
# df_dominant_topic.head(10)


def chnge_to_list(text):
    sample_list = list(text.split('|'))
    #for_end = list(text)
    a = clean_text(sample_list)
    b = remove_stopwords(a)
    c = lemmatize(b)
    d = remove_stopwords(c)
    e = mtrx(d)
    f = lda_model_build(d,e)
    #g = print_topics(f)
    g = format_topics_sentences(ldamodel=f, corpus=e, texts=sample_list)
    #print(g)
    #print(type(g))
    g_list = g.values.tolist()
    return g_list

