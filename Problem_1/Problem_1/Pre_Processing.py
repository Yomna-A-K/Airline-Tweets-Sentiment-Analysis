# library to clean data 
import re
from nltk.tokenize import word_tokenize  
from nltk.stem.porter import PorterStemmer 

def Pre_Process_tweet(tweet):
    tweet = ' '.join(re.sub('(@[A-Za-z0-9]+)', ' ', tweet).split()) 
    #removing symbols
    tweet = re.sub(r'\W', ' ', tweet)
    #replacing multiple spaces with single space
    tweet = re.sub(r'\s+', ' ', tweet, flags=re.I)
    tweet = tweet.lower()
    tweet = word_tokenize(tweet)
    ps = PorterStemmer()
    tweet = [ps.stem(word) for word in tweet]               
    tweet = ' '.join(tweet)
    return tweet
