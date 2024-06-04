
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def process(path,X_test):
	df = pd.read_csv(path)
	print(df.head())
	X_train=df["text"]
	y = df.label
	
	X_test=[X_test]

	#X_test=["Iran reportedly makes new push for uranium concessions in nuclear talks"]

	print(X_train)
	print(X_test)
	tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)    # This removes words which appear in more than 70% of the articles
	tfidf_train = tfidf_vectorizer.fit_transform(X_train) 
	tfidf_test = tfidf_vectorizer.transform(X_test)


	clf = MultinomialNB(alpha=.01, fit_prior=True)
	clf.fit(tfidf_train, y)
	pred1 = clf.predict(tfidf_test)
	print(pred1)


	X_train=df["text"]
	y = df.Sentiment

	tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)    # This removes words which appear in more than 70% of the articles
	tfidf_train = tfidf_vectorizer.fit_transform(X_train) 
	tfidf_test = tfidf_vectorizer.transform(X_test)


	clf = MultinomialNB(alpha=.01, fit_prior=True)
	clf.fit(tfidf_train, y)
	pred2 = clf.predict(tfidf_test)
	print(pred2)

	return pred1[0],pred2[0]


