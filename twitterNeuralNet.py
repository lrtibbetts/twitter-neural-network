import tensorflow as tf
import pandas as pd
import numpy as np
import scipy
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import string

stop_words = stopwords.words('english')
other_exclusions = ["#ff", "ff", "rt"]
stop_words.extend(other_exclusions)

stemmer = nltk.PorterStemmer()

df = pd.read_csv("twitterData.csv", header=0)
tweets = df.tweet
tweets = [x for x in tweets if type(x) == str]

# fix tweets
fixed_tweets = []
for i, t_orig in enumerate(tweets):
    s = t_orig
    try:
        s = s.encode("latin1")
    except:
        try:
            s = s.encode("utf-8")
        except:
            pass
    if type(s) != np.unicode:
            fixed_tweets.append(np.unicode(s, errors="ignore"))
    else:
        fixed_tweets.append(s)
assert len(tweets) == len(fixed_tweets), "shouldn't remove any tweets"
tweets = fixed_tweets
print(len(tweets), " tweets to classify")

# remove handles, extra spaces, urls, hashtags, and html
def preprocess(text_string):
    text_without_handles = re.sub(r'@[\w!:]+', '', text_string)
    text_without_spaces = re.sub(r'\s+', ' ', text_without_handles)
    text_without_urls = re.sub(r'https?://[A-Za-z0-9./]+', '', text_without_spaces)
    text_without_hashtags = re.sub(r'#[\w]+', '', text_without_urls)
    parsed_text = re.sub(r'&[\w;]+', '', text_without_hashtags)
    return parsed_text

'''
for x in range(len(tweets)) :
    tweets[x] = preprocess(tweets[x])

for x in range(30) :
    print(tweets[x])'''

# remove punctuation, set to lowercase
def tokenize(tweet):
    tweet = re.sub(r'[^\w\s]', '', tweet.lower())
    tokens = [stemmer.stem(t) for t in tweet.split()]
    print(tokens)
    return tokens

vectorizer = TfidfVectorizer(
    tokenizer=tokenize,
    preprocessor=preprocess,
    ngram_range=(1, 3),
    stop_words=stop_words,
    use_idf=True,
    smooth_idf=False,
    norm=None,
    decode_error='replace',
    max_features=10000,
    min_df=5,
    max_df=0.75
    )

tfidf = vectorizer.fit_transform(tweets).toarray()
# print(tfidf)
