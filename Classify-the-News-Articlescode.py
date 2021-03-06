# --------------
# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score ,confusion_matrix


# Code starts here

# load data
news = pd.read_csv(path)

# subset data
news = news[["TITLE", "CATEGORY"]]

# distribution of classes
dist = news["CATEGORY"].value_counts()

# display class distribution
print(dist)

# display data
print(news.head())

# Code ends here


# --------------
stop = set(stopwords.words('english'))
news["TITLE"] = news["TITLE"].apply(lambda x: re.sub("[^a-zA-Z]", " ",x))
news["TITLE"] = news["TITLE"].apply(lambda x: x.lower().split())
news["TITLE"] = news["TITLE"].apply(lambda x: [i for i in x if i not in stop])
news["TITLE"] = news["TITLE"].apply(lambda x: " ".join(x))
X_train, X_test, Y_train, Y_test = train_test_split(news["TITLE"], news["CATEGORY"], test_size=0.2, random_state=3)


# --------------
count_vectorizer = CountVectorizer()
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,3))
X_train_count = count_vectorizer.fit_transform(X_train)
X_test_count = count_vectorizer.transform(X_test)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


# --------------
nb_1 = MultinomialNB()
nb_2 = MultinomialNB()
nb_1.fit(X_train_count, Y_train)
nb_2.fit(X_train_tfidf, Y_train)
acc_count_nb = accuracy_score(nb_1.predict(X_test_count), Y_test)
acc_tfidf_nb = accuracy_score(nb_2.predict(X_test_tfidf), Y_test)
print(acc_count_nb, acc_tfidf_nb)


# --------------
import warnings
warnings.filterwarnings('ignore')

# initialize logistic regression
logreg_1 = OneVsRestClassifier(LogisticRegression(random_state=10))
logreg_2 = OneVsRestClassifier(LogisticRegression(random_state=10))
logreg_1.fit(X_train_count, Y_train)
logreg_2.fit(X_train_tfidf, Y_train)
acc_count_logreg = accuracy_score(logreg_1.predict(X_test_count), Y_test)
acc_tfidf_logreg = accuracy_score(logreg_2.predict(X_test_tfidf), Y_test)
print(acc_count_logreg, acc_tfidf_logreg)


