import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from textstat.textstat import*
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn
import keras

# resource utilized: https://github.com/t-davidson/hate-speech-and-offensive-language/blob/master/classifier/classifier.py

# NLTK stemmer
stemmer = nltk.PorterStemmer()

# Function definitions

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

def main():
    # use nltk stopwords (unimportant words)
    stop_words = stopwords.words('english')
    other_exclusions = ["#ff", "ff", "rt"]
    stop_words.extend(other_exclusions)

    # read in data
    df = pd.read_csv("twitterData.csv", header=0)
    tweets = df.tweet # isolate text of tweets
    tweets = [x for x in tweets if type(x) == str] # check that all tweets are type string

    # tfidf vectorizer
    vectorizer = TfidfVectorizer(tokenizer=tokenize, preprocessor=preprocess, ngram_range=(1, 3),
        stop_words=stop_words, use_idf=True, smooth_idf=False, norm=None, decode_error='replace',
        max_features=10000, min_df=5, max_df=0.75)

    # generate tfidf array using sklearn.feature_extraction and convert to array
    tfidf = vectorizer.fit_transform(tweets).toarray()

    # prepare data
    X = pd.DataFrame(tfidf)
    y = df['class'].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.15)
    y_train_labels = keras.utils.to_categorical(y_train, num_classes=3)
    y_test_labels = keras.utils.to_categorical(y_test, num_classes=3)

    # set up neural network
    model = Sequential()
    model.add(Dense(256, activation='relu', input_dim=7234))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train_labels, batch_size=16, epochs=5, verbose=1,
              validation_data=(x_test, y_test_labels))
    y_preds = model.predict_classes(x_test)

    # confusion matrix
    from sklearn.metrics import confusion_matrix
    confusion_matrix = confusion_matrix(y_test, y_preds)
    print(confusion_matrix)
    matrix_proportions = np.zeros((3, 3))
    for i in range(0, 3):
        matrix_proportions[i, :] = confusion_matrix[i, :] / float(confusion_matrix[i, :].sum())
    names = ['Hate', 'Offensive', 'Neither']
    confusion_df = pd.DataFrame(matrix_proportions, index=names, columns=names)
    plt.figure(figsize=(5, 5))
    seaborn.heatmap(confusion_df, annot=True, annot_kws={"size": 12}, cmap='gist_gray_r', cbar=False, square=True,
                    fmt='.2f')
    plt.ylabel(r'True categories', fontsize=14)
    plt.xlabel(r'Predicted categories', fontsize=14)
    plt.tick_params(labelsize=12)
    plt.show()

main()