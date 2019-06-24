import numpy as np   
import pandas as pd  

import matplotlib
import matplotlib.pyplot as plt

from Pre_Processing import Pre_Process_tweet

#read dataset from comma separated file
dataset = pd.read_csv('Tweets.csv');

#extract and pre-process tweets
all_tweets = []  
len = len(dataset)
for i in range(0, len):
    tweet = dataset['text'][i]
    tweet = Pre_Process_tweet(tweet)
    all_tweets.append(tweet)
    

#feature extraction
from sklearn.feature_extraction.text import CountVectorizer  
#use ngrams with range 2
ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
ngram_vectorizer.fit(all_tweets)
X = ngram_vectorizer.transform(all_tweets)

# y contains each tweet airline_sentiment (labels)
Y = dataset.iloc[:, 1].values      

#split into training and testing sets
from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, shuffle=True)



from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 

from sklearn.linear_model import LogisticRegression
"""
   loop was used to determine the best C value
   for c in [0.01, 0.05, 0.25, 0.5, 1]:  
        lr = LogisticRegression(C=c, solver = 'saga', multi_class='auto', max_iter=250)
       lr.fit(X_train, Y_train)
       Y_pred = lr.predict(X_test) 
       print(classification_report(Y_test, Y_pred))
"""  
#build and fit model
model = LogisticRegression(C=0.5, solver = 'saga', multi_class='auto', max_iter=250)
model.fit(X_train, Y_train)

#predict using the trained model
Y_pred = model.predict(X_test) 

#evaluation metrics
print(classification_report(Y_test, Y_pred))  
print(confusion_matrix(Y_test, Y_pred))

#create pie representation for the true labels
unique, counts = np.unique(Y_test, return_counts=True)
sizes = [counts[np.where(unique == 'negative')], counts[np.where(unique == 'neutral')], counts[np.where(unique == 'positive')]]
explode = (0, 0, 0.1)
labels = 'Negative', 'Neutral', 'Positive'
fig = plt.figure(figsize=(5,5))
fig.suptitle('Airline Sentiment [True]')
fig.canvas.set_window_title('True Labels Pie Representation')
plt.pie(np.array(sizes).flatten(), explode=explode, colors="bwr", labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90, wedgeprops={'alpha':0.8})
plt.axis('equal') 

#create pie representation for the predicted labels
unique, counts = np.unique(Y_pred, return_counts=True)
sizes = [counts[np.where(unique == 'negative')], counts[np.where(unique == 'neutral')], counts[np.where(unique == 'positive')]]
explode = (0, 0, 0.1)
labels = 'Negative', 'Neutral', 'Positive'
fig = plt.figure(figsize=(5,5))
fig.suptitle('Using Logistic Regression [Predicted]')
fig.canvas.set_window_title('Predicted Labels Pie Representation')
plt.pie(np.array(sizes).flatten(), explode=explode, colors="bwr", labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90, wedgeprops={'alpha':0.8})
plt.axis('equal') 
plt.show()

#save trained model to file to load later
import pickle
with open('Saved_Model', 'wb') as f:
    pickle.dump(model, f)

# load trained model if re-use of classification model was needed                                                                                                                                                                                                     
with open('Saved_Model', 'rb') as f:
    NewModel = pickle.load(f)
Y_pred = NewModel.predict(X_test) 
print(classification_report(Y_test, Y_pred)) 



