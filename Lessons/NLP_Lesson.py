
# coding: utf-8

# # Natural Language Processing (NLP)

# ## Introduction
# 
# *Adapted from [NLP Crash Course](http://files.meetup.com/7616132/DC-NLP-2013-09%20Charlie%20Greenbacker.pdf) by Charlie Greenbacker and [Introduction to NLP](http://spark-public.s3.amazonaws.com/nlp/slides/intro.pdf) by Dan Jurafsky*

# ### What is NLP?
# 
# - Using computers to process (analyze, understand, generate) natural human languages
# - Most knowledge created by humans is unstructured text, and we need a way to make sense of it
# - Build probabilistic model using data about a language
# - Also referred to as machine learning with text.

# ### What are some of the higher level task areas?
# 
# - **Information retrieval**: Find relevant results and similar results
#     - [Google](https://www.google.com/)
# - **Information extraction**: Structured information from unstructured documents
#     - [Events from Gmail](https://support.google.com/calendar/answer/6084018?hl=en)
# - **Machine translation**: One language to another
#     - [Google Translate](https://translate.google.com/)
# - **Text simplification**: Preserve the meaning of text, but simplify the grammar and vocabulary
#     - [Rewordify](https://rewordify.com/)
#     - [Simple English Wikipedia](https://simple.wikipedia.org/wiki/Main_Page)
# - **Predictive text input**: Faster or easier typing
#     - [A friend's application](https://justmarkham.shinyapps.io/textprediction/)
#     - [A much better application](https://farsite.shinyapps.io/swiftkey-cap/)
# - **Sentiment analysis**: Attitude of speaker
#     - [Hater News](http://haternews.herokuapp.com/)
# - **Automatic summarization**: Extractive or abstractive summarization
#     - [autotldr](https://www.reddit.com/r/technology/comments/35brc8/21_million_people_still_use_aol_dialup/cr2zzj0)
# - **Natural Language Generation**: Generate text from data
#     - [How a computer describes a sports match](http://www.bbc.com/news/technology-34204052)
#     - [Publishers withdraw more than 120 gibberish papers](http://www.nature.com/news/publishers-withdraw-more-than-120-gibberish-papers-1.14763)
# - **Speech recognition and generation**: Speech-to-text, text-to-speech
#     - [Google's Web Speech API demo](https://www.google.com/intl/en/chrome/demos/speech.html)
#     - [Vocalware Text-to-Speech demo](https://www.vocalware.com/index/demo)
# - **Question answering**: Determine the intent of the question, match query with knowledge base, evaluate hypotheses
#     - [How did supercomputer Watson beat Jeopardy champion Ken Jennings?](http://blog.ted.com/how-did-supercomputer-watson-beat-jeopardy-champion-ken-jennings-experts-discuss/)
#     - [IBM's Watson Trivia Challenge](http://www.nytimes.com/interactive/2010/06/16/magazine/watson-trivia-game.html)
#     - [The AI Behind Watson](http://www.aaai.org/Magazine/Watson/watson.php)

# ### What are some of the lower level components?
# 
# - **Tokenization**: breaking text into tokens (words, sentences, n-grams)
# - **Stopword removal**: a/an/the
# - **Stemming and lemmatization**: root word
# - **TF-IDF**: word importance
# - **Part-of-speech tagging**: noun/verb/adjective
# - **Named entity recognition**: person/organization/location
# - **Spelling correction**: "New Yrok City"
# - **Word sense disambiguation**: "buy a mouse"
# - **Segmentation**: "New York City subway"
# - **Language detection**: "translate this page"
# - **Machine learning**

# ### Why is NLP hard?
# 
# - **Ambiguity**:
#     - Hospitals are Sued by 7 Foot Doctors
#     - Juvenile Court to Try Shooting Defendant
#     - Local High School Dropouts Cut in Half
# - **Non-standard English**: text messages
# - **Idioms**: "throw in the towel"
# - **Newly coined words**: "retweet"
# - **Tricky entity names**: "Where is A Bug's Life playing?"
# - **World knowledge**: "Mary and Sue are sisters", "Mary and Sue are mothers"
# - **Texts with the same words and phrases can having different meanings **: 
# State farm commercial where two different people say "Is this my car? What? This is ridiculous! This can't be happening! Shut up! Ahhhh!!!"
# 
# 
# NLP requires an understanding of the **language** and the **world**.

# ## Part 1: Reading in the Yelp Reviews

# - "corpus" = collection of documents
# - "corpora" = plural form of corpus

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import seaborn as sb
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import metrics
from textblob import TextBlob, Word
from nltk.stem.snowball import SnowballStemmer
get_ipython().magic(u'matplotlib inline')


# In[2]:

# read yelp.csv into a DataFrame
url = 'yelp.csv'
yelp = pd.read_csv(url, encoding='unicode-escape')


# In[3]:

# Create a new DataFrame called yelp_best_worst that only contains the 5-star and 1-star reviews
yelp_best_worst = yelp[(yelp.stars==5) | (yelp.stars==1)]


# In[4]:

yelp_best_worst.head()


# In[5]:

# define X and y
X = yelp_best_worst.text
y = yelp_best_worst.stars
print y.value_counts(normalize=True)

# split the new DataFrame into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


# In[6]:

X_train[0]


# ## Part 2: Tokenization

# - **What:** Separate text into units such as sentences or words
# - **Why:** Gives structure to previously unstructured text
# - **Notes:** Relatively easy with English language text, not easy with some languages

# From the [scikit-learn documentation](http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction):
# 
# > Text Analysis is a major application field for machine learning algorithms. However the raw data, a sequence of symbols cannot be fed directly to the algorithms themselves as most of them expect **numerical feature vectors with a fixed size** rather than the **raw text documents with variable length**.
# 
# We will use [CountVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) to "convert text into a matrix of token counts":

# In[7]:

# example documents
simple_train = ['call you tonight', 'Call me a cab', 'please call me... PLEASE!']


# In[8]:

# Term Frequency
vect = CountVectorizer()
dtm = vect.fit_transform(simple_train)
tf = pd.DataFrame(dtm.toarray(), columns=vect.get_feature_names())
tf


# In[9]:

# transforming a new sentence, what do you notice?
new_sentence = ['please call yourself a cab']
pd.DataFrame(vect.transform(new_sentence).toarray(), columns=vect.get_feature_names())


# In[10]:

# use CountVectorizer to create document-term matrices from X_train and X_test
vect = CountVectorizer()
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)


# In[11]:

# rows are documents, columns are terms (phrases) (aka "tokens" or "features")
print X_train_dtm.shape
print X_test_dtm.shape
# Why do they have the same number of features


# In[12]:

# first 50 features
print vect.get_feature_names()[:50]


# In[13]:

# last 50 features
print vect.get_feature_names()[-50:]


# In[14]:

# show vectorizer options
vect


# [CountVectorizer documentation](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)

# - **lowercase:** boolean, True by default
# - Convert all characters to lowercase before tokenizing.

# In[15]:

#Create a count vectorizer that doesn't lowercase the words
vect = CountVectorizer(lowercase=False)
X_train_dtm = vect.fit_transform(X_train)
X_train_dtm.shape # has more features


# - **ngram_range:** tuple (min_n, max_n)
# - The lower and upper boundary of the range of n-values for different n-grams to be extracted. All values of n such that min_n <= n <= max_n will be used.

# In[16]:

# include 1-grams and 2-grams
vect = CountVectorizer(ngram_range=(1, 2))
X_train_dtm = vect.fit_transform(X_train)
X_train_dtm.shape


# In[17]:

# last 50 features
print vect.get_feature_names()[-50:]


# **Predicting the star rating with Naive Bayes**

# We will use [multinomial Naive Bayes](https://www.analyticsvidhya.com/blog/2015/09/naive-bayes-explained/):
# 
# "It is a classification technique based on Bayes’ Theorem with an assumption of independence among predictors. In simple terms, a Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature. For example, a fruit may be considered to be an apple if it is red, round, and about 3 inches in diameter. Even if these features depend on each other or upon the existence of the other features, all of these properties independently contribute to the probability that this fruit is an apple and that is why it is known as ‘Naive’."
# 
# For more explanation on NB click on link.
# 

# ### <b>Pros</b>: 
# #### - Very fast. Adept at handling tens of thousands of features which is why it's used for text classification
# #### - Works well with a small number of observations
# #### - Isn't negatively affected by "noise"
# 
# ### <b>Cons</b>:
# #### - Useless for probabilities. Most of the time assigns probabilites that are close to zero or one
# #### - It is literally "naive". Nearly impossible to have a set of features that are independent.

# In[18]:

from sklearn.naive_bayes import MultinomialNB


# In[19]:

#test model on the whole data then do a cross valdiation
vect = CountVectorizer()
Xdtm = vect.fit_transform(X)
nb = MultinomialNB()
nb.fit(Xdtm, y)
nb.score(Xdtm, y)


# In[20]:

# make a countvectorizer for a train test split
vect = CountVectorizer()
# create document-term matrices
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)

# use multinomial naive bayes with document feature matrix, NOT the text column
nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)
y_pred_class = nb.predict(X_test_dtm)
# calculate accuracy
print metrics.accuracy_score(y_test, y_pred_class)


# In[21]:

# calculate null accuracy, which is the accuracy of our null model (just guessing the most common thing)
y_test_binary = np.where(y_test==5, 1, 0)
max(y_test_binary.mean(), 1 - y_test_binary.mean())


# In[22]:

# Predict on new text
new_text = ["I had a decent time at this restaurant. The food was delicious but the service was poor. I recommend the salad but do not eat the french fries."]
new_text_transform = vect.transform(new_text)


# In[23]:

nb.predict(new_text_transform)


# In[24]:

# EXERCISE define a function, tokenize_test,  that does five things:
def tokenize_test(vect):
    nb = MultinomialNB()
    X_dtm = vect.fit_transform(X)
    print 'Features: ', X_dtm.shape[1]
    print 'Accuracy: ', cross_val_score(nb, X_dtm, y, cv=5, scoring='accuracy').mean()


# In[26]:

# include 1-grams and 2-grams
vect = CountVectorizer(ngram_range=(1, 2))
tokenize_test(vect)
#2-grams add too much noise; not as good as just 1-grams


# ## Part 3: Stopword Removal

# - **What:** Remove common words that will likely appear in any text
# - **Why:** They don't tell you much about your text

# In[ ]:

# show vectorizer options
vect


# - **stop_words:** string {'english'}, list, or None (default)
# - If 'english', a built-in stop word list for English is used.
# - If a list, that list is assumed to contain stop words, all of which will be removed from the resulting tokens.
# - If a word is equally like to show up in a rap lyric as medical paper then its most likely a stop word.
# - Corpus-specific stopwords, that words that aren't regular stopwords but become stopwords depending on the context.
# - If None, no stop words will be used. max_df can be set to a value in the range [0.7, 1.0) to automatically detect and filter stop words based on intra corpus document frequency of terms.

# In[27]:

# remove English stop words
vect = CountVectorizer(stop_words='english', ngram_range=(1, 2))
tokenize_test(vect)


# In[28]:

# set of stop words
print vect.get_stop_words()


# ## Part 4: Other CountVectorizer Options

# - **max_features:** int or None, default=None
# - If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus.

# In[29]:

# remove English stop words and only keep 100 features, MUCH FASTER
vect = CountVectorizer(stop_words='english', max_features=100)
tokenize_test(vect)


# In[30]:

# all 100 features
print vect.get_feature_names()


# In[32]:

# include 1-grams and 2-grams, and limit the number of features - inreases accuracy!
vect = CountVectorizer(ngram_range=(1, 2), max_features=10000)
tokenize_test(vect)


# - **min_df:** float in range [0.0, 1.0] or int, default=1
# - When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold. This value is also called cut-off in the literature. If float, the parameter represents a proportion of documents, integer absolute counts.

# In[34]:

# include 1-grams and 2-grams, and only include terms that appear at least 3 times - increases accuracy more
vect = CountVectorizer(ngram_range=(1, 2), min_df=3)
tokenize_test(vect)


# ## Part 5: Introduction to TextBlob

# TextBlob: "Simplified Text Processing"

# In[35]:

# print the first review
print yelp_best_worst.text[0]


# In[36]:

# save it as a TextBlob object
review = TextBlob(yelp_best_worst.text[0])


# In[51]:

dir(review)


# In[37]:

# list the words
review.words[:50]


# In[38]:

# list the sentences
review.sentences[:5]


# In[39]:

# some string methods are available
review.lower()


# In[40]:

# Parts-of-speech tagging. Identifies nouns, verbs, adverbs, etc...
review.tags


# POS Tags guide: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

# ## Part 6: Stemming and Lemmatization

# **Stemming:**
# 
# - **What:** Reduce a word to its base/stem/root form
# - **Why:** Often makes sense to treat related words the same way
# - **Notes:**
#     - Uses a "simple" and fast rule-based approach
#     - Stemmed words are usually not shown to users (used for analysis/indexing)
#     - Some search engines treat words with the same stem as synonyms

# In[41]:

# initialize stemmer
stemmer = SnowballStemmer('english')


# Compare and contrast the words with their stems.

# In[42]:

review.words[:100]


# In[43]:

# stem each word
print [stemmer.stem(word) for word in review.words[:100]]


# **Lemmatization**
# 
# - **What:** Derive the canonical form ('lemma') of a word
# - **Why:** Can be better than stemming
# - **Notes:** Uses a dictionary-based approach (slower than stemming)

# In[44]:

from nltk.stem.wordnet import WordNetLemmatizer


# In[45]:

word = Word('indices')
stemmer.stem(word)


# In[46]:

lem = WordNetLemmatizer()


# In[47]:

#Try it with words that look very different when pluralized like indices and octopi
lem.lemmatize("indices")


# Compare and contrast the originals words with their "lemons"

# In[48]:

print [word for word in review.words[:100]]


# In[49]:

# assume every word is a noun
print [word.lemmatize() for word in review.words[:100]]


# In[50]:

# assume every word is a verb
print [word.lemmatize(pos='v') for word in review.words]


# In[52]:

# define a function that accepts text and returns a list of lemmas
def word_tokenize_stem(text):
    words = TextBlob(text).words
    return [stemmer.stem(word) for word in words]
def word_tokenize_lemma(text):
    words = TextBlob(text).words
    return [word.lemmatize() for word in words]


# In[53]:

# use word_tokenize LEMMA as the feature extraction function (WARNING: SLOW!)
# this will lemmatize each word
vect = CountVectorizer(analyzer=word_tokenize_stem)
tokenize_test(vect)


# In[54]:

# use word_tokenize STEM as the feature extraction function (WARNING: SLOW!)
# this will lemmatize each word
vect = CountVectorizer(analyzer=word_tokenize_lemma)
tokenize_test(vect)


# ## Part 7: Term Frequency-Inverse Document Frequency (TF-IDF)

# - **What:** Computes "relative frequency" that a word appears in a document compared to its frequency across all documents
# - **Why:** More useful than "term frequency" for identifying "important" words in each document (high frequency in that document, low frequency in other documents). Court, ball, shooting, passing will show up frequently in a basketball corpus, but essentially add no meaning.
# - **Notes:** Used for search engine scoring, text summarization, document clustering

# In[55]:

# example documents
simple_train = ['call you tonight', 'Call me a cab', 'please call me... PLEASE!']


# In[56]:

# Term Frequency
vect = CountVectorizer()
tf = pd.DataFrame(vect.fit_transform(simple_train).toarray(), columns=vect.get_feature_names())
tf


# In[57]:

# Document Frequency
vect = CountVectorizer(binary=True)
df = vect.fit_transform(simple_train).toarray().sum(axis=0)
pd.DataFrame(df.reshape(1, 6), columns=vect.get_feature_names())


# In[58]:

# Term Frequency-Inverse Document Frequency (simple version)
tf/df


# In[59]:

# TfidfVectorizer. Why does please have the highest score?
vect = TfidfVectorizer()
pd.DataFrame(vect.fit_transform(simple_train).toarray(), columns=vect.get_feature_names())


# **More details:** [TF-IDF is about what matters](http://planspace.org/20150524-tfidf_is_about_what_matters/)

# In[60]:

# create a document-term matrix using TF-IDF
vect = TfidfVectorizer(stop_words='english')
dtm = vect.fit_transform(yelp.text)
features = vect.get_feature_names()
dtm.shape


# In[62]:

vect = TfidfVectorizer(stop_words='english')
tokenize_test(vect)
#doesn't improve accuracy in this case


# ## Part 8: Sentiment Analysis

# In[63]:

print review


# In[64]:

review.sentiment


# In[65]:

#Apply polarity and sentiment over yelp reviews df
yelp["polarity"] = yelp.text.apply(lambda x:TextBlob(x).polarity)
yelp["subjectivity"] = yelp.text.apply(lambda x:TextBlob(x).subjectivity)


# In[66]:

yelp["review_length"] = yelp.text.str.len()


# In[67]:

pd.set_option('max_colwidth', 500)


# In[68]:

yelp[yelp.polarity == 1].text.head()


# In[69]:

yelp[yelp.polarity == -1].text.head()


# In[70]:

yelp[(yelp.stars == 5) & (yelp.polarity < -0.3)]["text"].head(2)


# In[71]:

yelp[(yelp.stars == 1) & (yelp.polarity > 0.5)]["text"].head(2)


# In[72]:

yelp.polarity.plot(kind="hist", bins=20);


# In[73]:

yelp.subjectivity.plot(kind="hist", bins=20)


# In[74]:

#Plot scatter plot of polarity vs subjectivity scores
plt.scatter(yelp.polarity, yelp.subjectivity)
plt.xlabel("Polarity Scores")
plt.ylabel("Subjectivity Scores")


# In[75]:

#Plot boxplots of the polarity by yelp stars
yelp.boxplot(column='polarity', by='stars')


# ## Part 9: Calculating "spaminess" of a token

# In[76]:

#Load in ham or spam text dataset
df = pd.read_table("sms.tsv",encoding="utf-8", names= ["label", "message"])
df.head()


# In[77]:

#Look at null accuracy
df.label.value_counts(normalize=True)


# In[78]:

X = df.message
y = df.label
vect =CountVectorizer()
Xdtm = vect.fit_transform(X)
nb = MultinomialNB()
nb.fit(Xdtm,y)
nb.score(Xdtm,y)


# In[79]:

tokens = vect.get_feature_names()
len(tokens)


# In[80]:

#Print first 50 features
print vect.get_feature_names()[:50]


# In[81]:

#Print random slice of features
print vect.get_feature_names()[3200:3250]


# In[82]:

#How many times does a word appear in each class
nb.feature_count_


# In[83]:

nb.feature_count_.shape


# In[84]:

ham_token_count = nb.feature_count_[0,:]
ham_token_count


# In[85]:

spam_token_count = nb.feature_count_[1, :]
spam_token_count


# In[86]:

# create a DataFrame of tokens with their separate ham and spam counts
df_tokens = pd.DataFrame({'token':tokens, 'ham':ham_token_count, 'spam':spam_token_count}).set_index('token')
df_tokens.sample(10, random_state=3)


# In[87]:

# add 1 to ham and spam counts to avoid dividing by 0
df_tokens['ham'] = df_tokens.ham + 1
df_tokens['spam'] = df_tokens.spam + 1
df_tokens.sample(10, random_state=3)


# In[88]:

# Naive Bayes counts the number of observations in each class
nb.class_count_


# In[89]:

# convert the ham and spam counts into frequencies
df_tokens['ham'] = df_tokens.ham / nb.class_count_[0]
df_tokens['spam'] = df_tokens.spam / nb.class_count_[1]
df_tokens.sample(10, random_state=3)


# In[90]:

# calculate the ratio of spam-to-ham for each token
df_tokens['spam_ratio'] = df_tokens.spam / df_tokens.ham
df_tokens.sample(10, random_state=3)


# In[91]:

# examine the DataFrame sorted by spam_ratio
df_tokens.sort_values('spam_ratio', ascending=False).head(10)


# In[92]:

#Try looking up scores of different words
word = "win"
df_tokens.loc[word, 'spam_ratio']


# ## Conclusion
# 
# - NLP is a gigantic field
# - Understanding the basics broadens the types of data you can work with
# - Simple techniques go a long way
# - Use scikit-learn for NLP whenever possible

# In[93]:

ls


# # Lab time
# - There are three other datasets pitchfork album reviews, fake/real news, and political lean.
# - Pick one of those three datasets and try to build a model that differentiate between good/bad review, real/fake news, or liberal/conservative leaning. Make sure to examine the false positives and the false negatives texts. Use the "spamminess" technique on the corpus as well. 
# - Use both count and tfidf vectorizers. Use textblob to determine sentiment and polarity.
# - I've included some bonus material if you want to explore. 
#     
#     -How to summarize a text
#     
#     -How to use gridsearch to find the optimal parameters for countvectorizer.
#     

# ## Bonus: Using TF-IDF to Summarize a Yelp Review
# 
# Reddit's autotldr uses the [SMMRY](http://smmry.com/about) algorithm, which is based on TF-IDF!

# In[136]:

tfidf=TfidfVectorizer()
dtm=tfidf.fit_transform(yelp.text)
features=tfidf.get_feature_names()


# In[95]:

def summarize():
    # choose a random review that is at least 300 characters
    review_length = 0
    while review_length < 300:
        review_id = np.random.randint(0, len(yelp))
        review_text = yelp.text[review_id]
        review_length = len(review_text)
    # create a dictionary of words and their TF-IDF scores
    word_scores = {}
    for word in TextBlob(review_text).words:
        word = word.lower()
        if word in features:
            word_scores[word] = dtm[review_id, features.index(word)]
    
    # print words with the top 5 TF-IDF scores
    print 'TOP SCORING WORDS:'
    top_scores = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    for word, score in top_scores:
        print word
    
    # print 5 random words
    print '\n' + 'RANDOM WORDS:'
    random_words = np.random.choice(word_scores.keys(), size=5, replace=False)
    for word in random_words:
        print word
    
    # print the review
    print '\n' + review_text


# In[96]:

summarize()


# 

# In[ ]:




# ## Gridsearch/pipelining and vectorization

# In[97]:

from sklearn.grid_search import GridSearchCV


# In[98]:

#make a pipeline 
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(CountVectorizer(), MultinomialNB())


# In[99]:

#pipe steps
pipe.steps


# In[100]:

#Set range of parameters
param_grid = {}
param_grid["countvectorizer__max_features"] = [1000,5000,10000]
param_grid["countvectorizer__ngram_range"] = [(1,1), (1,2), (2,2)]
param_grid["countvectorizer__lowercase"] = [True, False]
param_grid["countvectorizer__analyzer"] = ["word", word_tokenize_lemma]


# In[101]:

from sklearn.grid_search import GridSearchCV
grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')


# In[102]:

#This will take a while
grid.fit(X,y)


# In[103]:

#Look at the best parameters and the best scores
print(grid.best_params_)
print(grid.best_score_)


# In[104]:

#Helpful for understanding how to create your param grid.
grid.get_params().keys()


# In[ ]:



