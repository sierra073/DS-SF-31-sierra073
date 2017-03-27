
# coding: utf-8

# ## Data Cleaning

# In[210]:

import pandas as pd


# In[211]:

#load the page level data and document level data for each document class (stored in different folders)
rel_page=pd.read_csv('Data/betadata-master/medical_records_release_form/pdf_text_by_page.csv',error_bad_lines=False,encoding='utf-8')
rel_page['class']='1'
rel_doc=pd.read_csv('Data/betadata-master/medical_records_release_form/pdf_text_by_file.csv',error_bad_lines=False,encoding='utf-8')
rel_doc['class']='1'
cons_page=pd.read_csv('Data/betadata-master/informed consent form/pdf_text_by_page.csv',error_bad_lines=False,encoding='utf-8')
cons_page['class']='2'
cons_doc=pd.read_csv('Data/betadata-master/informed consent form/pdf_text_by_file.csv',error_bad_lines=False,encoding='utf-8')
cons_doc['class']='2'
int_page=pd.read_csv('Data/betadata-master/patient intake form/pdf_text_by_page.csv',error_bad_lines=False,encoding='utf-8')
int_page['class']='3'
int_doc=pd.read_csv('Data/betadata-master/patient intake form/pdf_text_by_file.csv',error_bad_lines=False,encoding='utf-8')
int_doc['class']='3'
med_page=pd.read_csv('Data/betadata-master/medical form/pdf_text_by_page.csv',error_bad_lines=False,encoding='utf-8')
med_page['class']='4'
med_doc=pd.read_csv('Data/betadata-master/medical form/pdf_text_by_file.csv',error_bad_lines=False,encoding='utf-8')
med_doc['class']='4'
#print the dimensions
dataframes=[rel_page,rel_doc,cons_page,cons_doc,int_page,int_doc,med_page,med_doc]
for d in dataframes:
    print d.shape


# In[212]:

#combine into one page-level dataframe and one document-level dataframe
doc_level=[rel_doc,cons_doc,int_doc,med_doc]
page_level=[rel_page,cons_page,int_page,med_page]
data_doc=pd.concat(doc_level)
data_page=pd.concat(page_level)
print data_doc.shape 
print data_page.shape 


# In[213]:

#shuffle rows
data_doc = data_doc.sample(frac=1).reset_index(drop=True)
data_page= data_page.sample(frac=1).reset_index(drop=True)
print data_doc['class'].value_counts(normalize=True)
print data_page['class'].value_counts(normalize=True)


# In[214]:

#remove duplicates and NAs based on text content - expected to be a lot of these
data_doc=data_doc.drop_duplicates(subset=['text']).dropna(subset=['text'])
data_page=data_page.drop_duplicates(subset=['text']).dropna(subset=['text'])
#add an extra id column
data_doc['id'] = range(1, len(data_doc) + 1)
data_page['id'] = range(1, len(data_page) + 1)
print data_doc.shape
print data_page.shape


# In[215]:

#plot class distributions
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(1,2,figsize=(9,4))
i=0
for d in [data_doc,data_page]:
    plot=sns.countplot(x="class", data=d, order=['1','2','3','4'],ax=ax[i])
    if d.shape[1]==8:
        plot.set_title('Document-Level Data Class Distribution')
        i=i+1
    else:
        plot.set_title('Page-Level Data Class Distribution')
        i=i+1


# In[216]:

# define X and y for each dataframe
Xd = data_doc.text
yd = data_doc['class']
Xp = data_page.text
yp = data_page['class']


# ## Tokenization

# In[217]:

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from textblob import TextBlob, Word
from nltk.stem.snowball import SnowballStemmer

vect = CountVectorizer(stop_words='english')
Xd_dtm=vect.fit_transform(Xd)
Xp_dtm=vect.fit_transform(Xp)
print Xd_dtm.shape
print Xp_dtm.shape


# In[218]:

#check for strange words
print vect.get_feature_names()[5900:12000]


# In[219]:

#need to replace ____ with whitespace
data_doc['text']=data_doc['text'].apply(lambda x: x.replace("_"," "))
data_page['text']=data_page['text'].apply(lambda x: x.replace("_"," "))
Xd=Xd.apply(lambda x: x.replace("_"," "))
Xp=Xp.apply(lambda x: x.replace("_"," "))


# In[220]:

#run CountVectorizer again
Xd_dtm=vect.fit_transform(Xd)
Xp_dtm=vect.fit_transform(Xp)
print Xd_dtm.shape
print Xp_dtm.shape
print vect.get_feature_names()[5900:12000]


# In[221]:

#stemming
stemmer = SnowballStemmer('english')
def stem_function(x):
    words = TextBlob(x).words
    list_of_stems = [stemmer.stem(i) for i in words]
    return (" ").join(list_of_stems)

data_doc['stems'] = data_doc.text.apply(stem_function)
data_page['stems'] = data_page.text.apply(stem_function)

print 'done'


# In[222]:

#implement CountVectorizer on stems
vect = CountVectorizer(stop_words='english')
Xd_dtm2=vect.fit_transform(data_doc['stems'])
Xp_dtm2=vect.fit_transform(data_page['stems'])
print Xd_dtm2.shape
print Xp_dtm2.shape


# In[223]:

#ngrams
vect = CountVectorizer(stop_words='english',ngram_range=(1, 2))
Xd_dtm3=vect.fit_transform(Xd)
Xp_dtm3=vect.fit_transform(Xp)
print Xd_dtm3.shape
print Xp_dtm3.shape
#only include terms that occur 3+ times
vect = CountVectorizer(stop_words='english',ngram_range=(1, 2),min_df=3)
Xd_dtm4=vect.fit_transform(Xd)
Xp_dtm4=vect.fit_transform(Xp)
print Xd_dtm4.shape
print Xp_dtm4.shape


# In[224]:

#histogram of 10 most common words in each class
vect = CountVectorizer(stop_words='english', max_features=1000)

def create_freq_vector(data,col,classval):
    data=data[data[col]==classval]
    dtm = vect.fit_transform(data['text'])
    freqs = [(word, dtm.getcol(idx).sum()) for word, idx in vect.vocabulary_.items()]
    #sort from largest to smallest
    freq_vector=pd.DataFrame(sorted(freqs, key = lambda x: -x[1]))
    freq_vector.columns=['word', 'frequency']
    freq_vector[col]=classval
    return freq_vector[0:10]

#document data plot
fig, ax = plt.subplots(1,4, sharey=True,figsize=(15,9))
i=0
for classval in ['1','2','3','4']:
    v=create_freq_vector(data_doc,'class',classval)
    b=sns.barplot('word', 'frequency', data=v, ax=ax[i]) 
    b.set_xticklabels(v['word'], rotation=80);
    b.set_ylabel('frequency')
    b.set_title('Document class '+classval)
    i=i+1



# In[225]:

#correlations between 22 most common words
vect = CountVectorizer(stop_words='english', max_features=22)
dd=pd.DataFrame(vect.fit_transform(Xd).toarray(), columns=vect.get_feature_names())
dp=pd.DataFrame(vect.fit_transform(Xp).toarray(), columns=vect.get_feature_names())
dd.head()


# In[226]:

def corrplots(data):
    fig, ax = plt.subplots(1,2, figsize=(18,6.5))
    i=0
    for d in data:
        plot=sns.heatmap(d.corr(),ax=ax[i])
        if i==0:
            plot.set_title('Document-Level Data')
            i=i+1
        else:
            plot.set_title('Page-Level Data')
            i=i+1
corrplots([dd,dp])
#Interesting - not as many high correlations with page-level data


# In[227]:

#TF-IDF - correlations between 22 most common words
vect = TfidfVectorizer(stop_words='english', max_features=22)
dd_2=pd.DataFrame(vect.fit_transform(Xd).toarray(), columns=vect.get_feature_names())
dp_2=pd.DataFrame(vect.fit_transform(Xp).toarray(), columns=vect.get_feature_names())
dd_2.head()


# In[228]:

corrplots([dd_2,dp_2])
#page-level data looks more similar to document-level data


# In[229]:

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
scaler = StandardScaler(with_mean=False)
vect = TfidfVectorizer(stop_words='english',max_features=10000)

#Cluster to see if there is inherent structure in the document-level data - try for 2, 3, 4 clusters. See is the clusters are similar to the classes
def fit_plot_clusters(k_range,data):
    def fitcluster(k,data):
        TfIdfm=vect.fit_transform(data)
        X=scaler.fit_transform(TfIdfm)  
        km = KMeans(n_clusters=k, random_state=1)
        kfit=km.fit(X)
        #Silhouette Coefficient for each k
        score=metrics.silhouette_score(X, km.labels_)
        return kfit, score

    kfits=[]
    scores = []

    for k in k_range:
        kfit,score=fitcluster(k,data)
        kfits.append(kfit)
        scores.append(score)

    plt.plot(k_range, scores)
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Coefficient')
    plt.grid(True)

#not good and gets worse as k increases
fit_plot_clusters([2,3,4],data_doc['text'])
#try with 3 classes
fit_plot_clusters([2,3,4],data_doc['text'][data_doc['class'] !='4'])


# In[235]:

#most common words in each cluster (just showing for 2 clusters)
kfits=[]
scores = []
for k in [2,3,4]:
    kfit,score=fitcluster(k,data_doc['text'])
    kfits.append(kfit)
    scores.append(score)
    col=str(k)+'_cluster'
    data_doc[col]=kfit.labels_

fig, ax = plt.subplots(1,2, figsize=(10,3))
i=0
for k in [0,1]:
    v=create_freq_vector(data_doc,'2_cluster',k)
    b=sns.barplot('word', 'frequency', data=v, ax=ax[i]) 
    b.set_xticklabels(v['word'], rotation=80);
    b.set_ylabel('frequency')
    b.set_title('Cluser' +str(k))
    i=i+1


# In[237]:

#class distribution by cluster - algotithm strongly prefers 1 (or 2) clusters, and doesn't seem to systematically detect the class except for perhaps 3 (intake forms)
for k in [2,3,4]:
    col=str(k)+'_cluster'
    print data_doc.groupby([col,'class']).size()


# In[246]:

#Look at document correlations using tf-idf matrix - see which classes of documents are correlated, and if the same classes are the most correlated
vect = TfidfVectorizer(stop_words='english',max_features=1000)
d_tfidf_matrix=pd.DataFrame(vect.fit_transform(Xd).toarray(), columns=vect.get_feature_names())
print d_tfidf_matrix.shape
d_corr_matrix=d_tfidf_matrix.dot(d_tfidf_matrix.transpose())
print d_corr_matrix.shape
d_corr_matrix.head()


# In[249]:

#To get pairs with highest absolute correlation - adapted from stackoverflow:
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations")
print(get_top_abs_correlations(d_corr_matrix, 10))


# In[261]:

corr_pairs=pd.DataFrame(get_top_abs_correlations(d_corr_matrix, 500))
corr_pairs.reset_index(level=[0, 1],inplace=True)
#get the classes
corr_pairs.columns=['id1','id2','corr']
corr_pairs=pd.merge(left=corr_pairs,right=data_doc[['id','class']], left_on='id1', right_on='id')
corr_pairs=pd.merge(left=corr_pairs,right=data_doc[['id','class']], left_on='id2', right_on='id')
corr_pairs.head()
#All document classes correlate with class 1 (release form) the majority of the time, followed by class 3 (intake form)
corr_pairs.groupby(['class_x','class_y']).size()


# In[238]:

#try a basic model
from sklearn.naive_bayes import MultinomialNB
vect = TfidfVectorizer(stop_words='english',max_features=10000)
Xd_dtm = vect.fit_transform(data_doc['stems'])
Xp_dtm = vect.fit_transform(data_page['stems'])
nb = MultinomialNB()
fitd=nb.fit(Xd_dtm, yd)
fitp=nb.fit(Xp_dtm, yp)
print 'Document level data accuracy: '
print fitd.score(Xd_dtm, yd)
print 'Page level data accuracy: ' 
print fitp.score(Xp_dtm, yp)


# In[239]:

#They don't do well with the 4th class there, seems like it's too similar to the other 3. Will try taking it out, or experimenting with other methods of classifying the 4th class
from sklearn.metrics import confusion_matrix
print confusion_matrix(yd,fitd.predict(Xd_dtm))
print confusion_matrix(yp,fitp.predict(Xp_dtm))


# In[ ]:



