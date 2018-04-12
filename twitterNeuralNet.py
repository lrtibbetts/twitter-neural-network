import tensorflow as tf
import pandas as pd
import numpy as np
import scipy
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
from textstat.textstat import*

# https://towardsdatascience.com/another-twitter-sentiment-analysis-bb5b01ebad90

stemmer = nltk.PorterStemmer()
sentiment_analyzer = VS()

# FUNCTIONS

# remove handles, extra spaces, urls, hashtags, and html
def preprocess(text_string):
    text_without_handles = re.sub(r'@[\w!:]+', '', text_string)
    text_without_spaces = re.sub(r'\s+', ' ', text_without_handles)
    text_without_urls = re.sub(r'https?://[A-Za-z0-9./]+', '', text_without_spaces)
    text_without_hashtags = re.sub(r'#[\w]+', '', text_without_urls)
    parsed_text = re.sub(r'&[\w;]+', '', text_without_hashtags)
    return parsed_text

# remove punctuation, set to lowercase, stem words
def tokenize(tweet):
    tweet = re.sub(r'[^\w\s]', '', tweet.lower())
    tokens = [stemmer.stem(t) for t in tweet.split()]
    return tokens

def basic_tokenize(tweet):
    tweet = re.sub(r'[^\w\s]', '', tweet.lower())
    return tweet.split()

def get_pos_tags(tweets):
    tweet_tags = []
    for t in tweets:
        tokens = basic_tokenize(preprocess(t))
        tags = nltk.pos_tag(tokens)
        tag_list = [x[1] for x in tags]  # isolate only the tag, removing the words themselves
        tag_str = " ".join(tag_list)  # convert to string
        tweet_tags.append(tag_str)
    return tweet_tags

def other_features(tweet):
    sentiment = sentiment_analyzer.polarity_scores(tweet)
    words = preprocess(tweet)
    syllables = textstat.syllable_count(words) # count syllables in words
    num_chars = sum(len(w) for w in words) # num chars in words
    num_chars_total = len(tweet)
    num_terms = len(tweet.split())
    num_words = len(words.split())
    avg_syl = round(float((syllables+0.001))/float(num_words+0.001),4)
    num_unique_terms = len(set(words.split()))

    # Modified FK grade, where avg words per sentence is just num words/1
    FKRA = round(float(0.39 * float(num_words)/1.0) + float(11.8 * avg_syl) - 15.59,1)
    # Modified FRE score, where sentence fixed to 1
    FRE = round(206.835 - 1.015*(float(num_words)/1.0) - (84.6*float(avg_syl)),2)

    features = [FKRA, FRE, syllables, num_chars, num_chars_total, num_terms, num_words,
                num_unique_terms, sentiment['compound']]
    return features

def get_features(tweets):
    feats=[]
    for t in tweets:
        feats.append(other_features(t))
    return np.array(feats)

def main():
    # use nltk stopwords (unimportant words)
    stop_words = stopwords.words('english')
    other_exclusions = ["#ff", "ff", "rt"]
    stop_words.extend(other_exclusions)

    # read in data
    df = pd.read_csv("twitterData.csv", header=0)
    tweets = df.tweet # isolate text of tweets
    tweets = [x for x in tweets if type(x) == str] # check that all tweets are type string

    # test the fxns preprocess and tokenize, print results for first 30 lines
    '''for x in range(len(tweets)) :
        tweets[x] = preprocess(tweets[x])
        tweets[x] = tokenize(tweets[x])

    for x in range(30) :
        print(tweets[x])
    '''
    vectorizer = TfidfVectorizer(tokenizer=tokenize, preprocessor=preprocess, ngram_range=(1, 3),
        stop_words=stop_words, use_idf=True, smooth_idf=False, norm=None, decode_error='replace',
        max_features=10000, min_df=5, max_df=0.75)

    pos_vectorizer = TfidfVectorizer(lowercase=False, preprocessor=None, ngram_range=(1, 3),
        stop_words=None, use_idf=False, smooth_idf=False, norm=None, decode_error='replace',
        max_features=5000, min_df=5, max_df=0.75)
    
    # generate tfidf array using sklearn.feature_extraction
    tfidf = vectorizer.fit_transform(tweets).toarray()
    vocab = {v:i for i, v in enumerate(vectorizer.get_feature_names())}
    idf_vals = vectorizer.idf_
    idf_dict = {i:idf_vals[i] for i in vocab.values()} #keys are indices; values are IDF scores

    tweet_tags = get_pos_tags(tweets)
    pos = pos_vectorizer.fit_transform(pd.Series(tweet_tags)).toarray()
    pos_vocab = {v:i for i, v in enumerate(pos_vectorizer.get_feature_names())}

    feats = get_features(tweets)
    M = np.concatenate([tfidf, pos, feats], axis=1)
    print(M.shape)

main()