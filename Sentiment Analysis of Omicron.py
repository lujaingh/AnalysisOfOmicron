#!/usr/bin/env python
# coding: utf-8

# # This Research done by Lujain Ghazalat and Rand Agha

# In[2]:


'''
import libraries
'''
from deep_translator import GoogleTranslator
import tweepy
import pandas as pd
import nltk
from xgboost import XGBClassifier
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
from nltk import pos_tag #post tag
import nltk.classify.util
from nltk.tokenize import word_tokenize
from nltk.classify import NaiveBayesClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from nltk.corpus import names #to classify gender based on names
import numpy as np  
import sys
import os
import re
from textblob import TextBlob
import numpy as np
from datetime import datetime, timedelta
from IPython.display import clear_output
import matplotlib.pyplot as plt
import csv
import time
import matplotlib.pyplot as plt
import string
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,accuracy_score, classification_report
from datetime import date
from googletrans import Translator, constants
from pprint import pprint
from nltk.stem.isri import ISRIStemmer
from textblob import TextBlob
import re
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import unicodedata as ud
#import camel_tools as ct
from nltk.stem.isri import ISRIStemmer
#from ar_wordcloud import ArabicWordCloud
import time
import matplotlib.pyplot as plt
import numpy as np
#model
from textblob import TextBlob
#from camel_tools.sentiment import SentimentAnalyzer
from sklearn.metrics import classification_report, accuracy_score
import argparse
from nltk.stem.isri import ISRIStemmer
import sys


# In[3]:


'''
Read data, almost preprocessed before 
'''
df_1 = pd.read_excel('annotation1.xlsx')
df_2 = pd.read_csv('annotation2.csv')
df_8 = pd.read_csv('annotation3.csv')
df_7 = pd.read_csv('annotation4.csv')

df_1['Sentiment'] = df_1['Sentiment'].map({'positive':1, 'negative':0, 'neutral':1, 'Neutral':1, 'Positive':1, 'Negative':0})
df_1['Sentiment'].fillna(0.0 , inplace = True)
df_1['Sentiment'] = df_1['Sentiment'].astype('int64')

df_2['Sentiment'] = df_2['Sentiment'].map({'positive':1, 'negative':0, 'neutral':1, 'Neutral':1, 'Positive':1, 'Negative':0})
df_2['Sentiment'].fillna(0.0 , inplace = True)
df_2['Sentiment'] = df_2['Sentiment'].astype('int64')

df_8['Sentiment'] = df_8['Sentiment'].map({'positive':1, 'negative':0, 'neutral':1, 'Neutral':1, 'Positive':1, 'Negative':0})
df_8['Sentiment'].fillna(0.0 , inplace = True)
df_8['Sentiment'] = df_8['Sentiment'].astype('int64')

df_7['Sentiment'] = df_7['Sentiment'].replace({2:1})
df_7['Sentiment'].fillna(0.0 , inplace = True)
df_7['Sentiment'] = df_7['Sentiment'].astype('int64')

df_all = pd.concat([df_1, df_2,df_7,df_8]) ######
dfcheck = df_all.copy()


# In[117]:


'''
To draw word cloud
'''
dll = df_all.copy()
def rem(text):
    return ' '.join(re.sub('[a-zA-Z_]|#|http\S+', " ", text,  flags=re.UNICODE).split()) 
def cleanup_string(text):
    '''
    Remove prepositions
    '''
    try:      
        str1 = text.replace("في"," ").replace("الى"," ").replace("عن"," ").replace("على"," ").replace("من"," ")
    except Exception:
        print('' , end = '')
        #print('<----',str_in,'---->')
    
    return str1 # .ljust(5) padding

dll['text'] = dll['text'].apply(rem)
dll['text'] = dll['text'].apply(cleanup_string)

dfcheck['text'] = dfcheck['text'].apply(rem)
dfcheck['text'] = dfcheck['text'].apply(cleanup_string)


# In[62]:


'''
Word cloud
'''
from bidi.algorithm import get_display
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import arabic_reshaper # this was missing in your code

l = list(dll.text.values)
str1 = " ".join(l)

data = arabic_reshaper.reshape(str1)
data = get_display(data) # add this line
WordCloud = WordCloud(font_path = 'arial', background_color = 'white',
                  mode='RGB', width = 2000, height = 1000 , colormap = 'plasma').generate(data)
plt.title("wordcloud")
plt.imshow(WordCloud)
plt.axis("off")
plt.show()


# In[63]:


'''
Data before Balancing
'''
import seaborn as sns
plt.figure(figsize = (4,4))
ax = sns.countplot( x = 'Sentiment', order = df_all['Sentiment'].value_counts().index[0:2] , data = df_all, palette = ["#8d2865" , "#fe5862" , "#eb717c" , "#fe5862" , "#a54780" , "#8d2865" ,  "#6b1e47" , "#3d033b" , "#1e0034"] ) # , "" , "#6b1e47"  , "#7b174a" ,
plt.xticks( [0,1] , ['Positive','Negative']  )
i = 0


# In[66]:


'''
Map classes to numeric ones'
'''
df_all.Sentiment.replace({'positive' : 1 , 'Negative' : 0 , 'Neutral' : 1} , inplace = True)
df_all = df_all.loc[(df_all.Sentiment != 'irrelevant')]


# In[ ]:


'''
Draw a pie chart before balancing
'''
plt.pie(  df_all.Sentiment.value_counts()  )##palette = ["#8d2865" , "#fe5862" , "#eb717c" , "#fe5862" , "#a54780" , "#8d2865" ,  "#6b1e47" , "#3d033b" , "#1e0034"] )
plt.legend(['Positive','Negative'])
plt.show() 


# In[27]:


'''
Preprocessing
'''
def preprocessing(text):
    
    COMMA = u'\u060C'
    SEMICOLON = u'\u061B'
    QUESTION = u'\u061F'
    HAMZA = u'\u0621'
    ALEF_MADDA = u'\u0622'
    ALEF_HAMZA_ABOVE = u'\u0623'
    WAW_HAMZA = u'\u0624'
    ALEF_HAMZA_BELOW = u'\u0625'
    YEH_HAMZA = u'\u0626'
    ALEF = u'\u0627'
    BEH = u'\u0628'
    TEH_MARBUTA = u'\u0629'
    TEH = u'\u062a'
    THEH = u'\u062b'
    JEEM = u'\u062c'
    HAH = u'\u062d'
    KHAH = u'\u062e'
    DAL = u'\u062f'
    THAL = u'\u0630'
    REH = u'\u0631'
    ZAIN = u'\u0632'
    SEEN = u'\u0633'
    SHEEN = u'\u0634'
    SAD = u'\u0635'
    DAD = u'\u0636'
    TAH = u'\u0637'
    ZAH = u'\u0638'
    AIN = u'\u0639'
    GHAIN = u'\u063a'
    TATWEEL = u'\u0640'
    FEH = u'\u0641'
    QAF = u'\u0642'
    KAF = u'\u0643'
    LAM = u'\u0644'
    MEEM = u'\u0645'
    NOON = u'\u0646'
    HEH = u'\u0647'
    WAW = u'\u0648'
    ALEF_MAKSURA = u'\u0649'
    YEH = u'\u064a'
    MADDA_ABOVE = u'\u0653'
    HAMZA_ABOVE = u'\u0654'
    HAMZA_BELOW = u'\u0655'
    ZERO = u'\u0660'
    ONE = u'\u0661'
    TWO = u'\u0662'
    THREE = u'\u0663'
    FOUR = u'\u0664'
    FIVE = u'\u0665'
    SIX = u'\u0666'
    SEVEN = u'\u0667'
    EIGHT = u'\u0668'
    NINE = u'\u0669'
    PERCENT = u'\u066a'
    DECIMAL = u'\u066b'
    THOUSANDS = u'\u066c'
    STAR = u'\u066d'
    MINI_ALEF = u'\u0670'
    ALEF_WASLA = u'\u0671'
    FULL_STOP = u'\u06d4'
    BYTE_ORDER_MARK = u'\ufeff'

    # Diacritics
    FATHATAN = u'\u064b'
    DAMMATAN = u'\u064c'
    KASRATAN = u'\u064d'
    FATHA = u'\u064e'
    DAMMA = u'\u064f'
    KASRA = u'\u0650'
    SHADDA = u'\u0651'
    SUKUN = u'\u0652'

    #Ligatures
    LAM_ALEF = u'\ufefb'
    LAM_ALEF_HAMZA_ABOVE = u'\ufef7'
    LAM_ALEF_HAMZA_BELOW = u'\ufef9'
    LAM_ALEF_MADDA_ABOVE = u'\ufef5'
    SIMPLE_LAM_ALEF = u'\u0644\u0627'
    SIMPLE_LAM_ALEF_HAMZA_ABOVE = u'\u0644\u0623'
    SIMPLE_LAM_ALEF_HAMZA_BELOW = u'\u0644\u0625'
    SIMPLE_LAM_ALEF_MADDA_ABOVE = u'\u0644\u0622'
    HARAKAT_PAT = re.compile(u"["+u"".join([FATHATAN, DAMMATAN, KASRATAN,
                                            FATHA, DAMMA, KASRA, SUKUN,
                                            SHADDA])+u"]")
    HAMZAT_PAT = re.compile(u"["+u"".join([WAW_HAMZA, YEH_HAMZA])+u"]")
    ALEFAT_PAT = re.compile(u"["+u"".join([ALEF_MADDA, ALEF_HAMZA_ABOVE,
                                           ALEF_HAMZA_BELOW, HAMZA_ABOVE,
                                           HAMZA_BELOW])+u"]")
    LAMALEFAT_PAT = re.compile(u"["+u"".join([LAM_ALEF,
                                              LAM_ALEF_HAMZA_ABOVE,
                                              LAM_ALEF_HAMZA_BELOW,
    LAM_ALEF_MADDA_ABOVE])+u"]")
    """ https://github.com/cltk/cltk/blob/master/cltk/corpus/arabic/alphabet.py """
    WESTERN_ARABIC_NUMERALS = ['0','1','2','3','4','5','6','7','8','9']
    EASTERN_ARABIC_NUMERALS = [u'۰', u'۱', u'۲', u'۳', u'٤', u'۵', u'٦', u'۷', u'۸', u'۹']
    eastern_to_western_numerals = {}
    for i in range(len(EASTERN_ARABIC_NUMERALS)):
        eastern_to_western_numerals[EASTERN_ARABIC_NUMERALS[i]] = WESTERN_ARABIC_NUMERALS[i]
    # Punctuation marks
    COMMA = u'\u060C'
    SEMICOLON = u'\u061B'
    QUESTION = u'\u061F'
    # Other symbols
    PERCENT = u'\u066a'
    DECIMAL = u'\u066b'
    THOUSANDS = u'\u066c'
    STAR = u'\u066d'
    FULL_STOP = u'\u06d4'
    MULITIPLICATION_SIGN = u'\u00D7'
    DIVISION_SIGN = u'\u00F7'
    arabic_punctuations = COMMA + SEMICOLON + QUESTION + PERCENT + DECIMAL + THOUSANDS + STAR + FULL_STOP + MULITIPLICATION_SIGN + DIVISION_SIGN
    all_punctuations = string.punctuation + arabic_punctuations + '()[]{}'
    all_punctuations = ''.join(list(set(all_punctuations)))
    def remove_all_punctuations(text):
        for punctuation in all_punctuations:
            text = text.replace(punctuation, ' ')
        return text
    def remove_stop_words(text):
        stop_words = stopwords.words()
        text = ' '.join(word for word in text.split() if word not in stop_words)
        return text
    def remove_tashkeel(text):
        arabic_diacritics = re.compile("""
                                     ّ    | # Shadda
                                     َ    | # Fatha
                                     ً    | # Tanwin Fath
                                     ُ    | # Damma
                                     ٌ    | # Tanwin Damm
                                     ِ    | # Kasra
                                     ٍ    | # Tanwin Kasr
                                     ْ    | # Sukun
                                     ـ     # Tatwil/Kashida
                                 """, re.VERBOSE)
        text = re.sub(arabic_diacritics, '', text)
        return text
    def remove_tatweel(text):
        text = re.sub("[إأآا]", "ا", text)
        text = re.sub("ى", "ي", text)
        text = re.sub("ؤ", "ء", text)
        text = re.sub("ئ", "ء", text)
        text = re.sub("ة", "ه", text)
        text = re.sub("گ", "ك", text)
        return text
    def remove_non_arabic(text):
        return ' '.join(re.sub('[a-zA-Z_]|#|http\S+', " ", text,  flags=re.UNICODE).split()) 
    def normalize_hamza(text):
        text = ALEFAT_PAT.sub(ALEF, text)
        return HAMZAT_PAT.sub(HAMZA, text)
    def normalize_lamalef(text):
        return LAMALEFAT_PAT.sub(u'%s%s'%(LAM, ALEF), text)
    def normalize_spellerrors(text):
        text = re.sub(u'[%s]' % TEH_MARBUTA, HEH, text)
        return re.sub(u'[%s]' % ALEF_MAKSURA, YEH, text)
    def remove_retweet_tag(text):
        return re.compile('\#').sub('', re.compile('rt @[a-zA-Z0-9_]+:|@[a-zA-Z0-9_]+').sub('', text).strip())
    def remove_extra_spaces(text):
        return ' '.join(text.split())
    def convert_eastern_to_western_numerals(text): 
        for num in EASTERN_ARABIC_NUMERALS:
            text = text.replace(num, eastern_to_western_numerals[num])
        return text
    def replace_urls(text):
        return re.sub(r"http\S+|www.\S+", "", text)
    def replace_emails(text):
        emails = re.findall(r'[\w\.-]+@[\w\.-]+', text)
        for email in emails:
            text = text.replace(email,' يوجدايميل ')
        return text
    def remove_underscore(text):
        return ' '.join(text.split('_')) 
    def replace_phone_numbers(text):
        return re.sub(r'\d{10}', ' ', text) 
    def cleanup_string(text):
        '''
        Remove prepositions
        '''
        try:      
            str1 = text.replace("في"," ").replace("الى"," ").replace("عن"," ").replace("على"," ").replace("من"," ")
        except Exception:
            print('' , end = '')
            #print('<----',str_in,'---->')
        return str1 # .ljust(5) padding
    text = cleanup_string(text)
    text = replace_urls(text)
    text=remove_all_punctuations(text)
    text=remove_tashkeel(text)
    text=remove_tatweel(text)
    text=remove_stop_words(text)
    #text=remove_non_arabic(text)
    text=normalize_hamza(text)
    text = normalize_lamalef(text)
    text = normalize_spellerrors(text)
    text = remove_retweet_tag(text)
    text = remove_extra_spaces(text)
    text = convert_eastern_to_western_numerals(text)
    text= replace_emails(text)
    text = remove_underscore(text)
    text = replace_phone_numbers(text)
    text=remove_non_arabic(text)
    return text
def light_stem(text):
    words = text.split()
    result = list()
    stemmer = ISRIStemmer()
    for word in words:
        word = stemmer.norm(word, num=1)      # remove diacritics which representing Arabic short vowels
        if not word in stemmer.stop_words:    # exclude stop words from being processed
            word = stemmer.pre32(word)        # remove length three and length two prefixes in this order
            word = stemmer.suf32(word)        # remove length three and length two suffixes in this order
            word = stemmer.waw(word)          # remove connective ‘و’ if it precedes a word beginning with ‘و’
            word = stemmer.norm(word, num=2)  # normalize initial hamza to bare alif
        result.append(word)
    return ' '.join(result)
def lemitizing(text):
    return ISRIStemmer().suf32(text)
df_all['text'] = df_all['text'].apply(lambda x: preprocessing(x))

# duplicates already removed 
df_all["text"]  = df_all["text"] .apply(light_stem)
df_all["text"]  = df_all["text"] .apply(lemitizing)
dfcheck['text'] = dfcheck['text'].apply(lambda x: preprocessing(x))
dfcheck["text"] = dfcheck["text"] .apply(light_stem)
dfcheck["text"] = dfcheck["text"] .apply(lemitizing)


# In[37]:


'''
Down sampling
'''
d1 = df_all.loc[((df_all['Sentiment'] == 1))].sample(n = 807 , random_state = 1)
d2 = df_all.loc[((df_all['Sentiment'] == 0))].sample(n = 807 , random_state = 1)
df_all = pd.concat( [d1,d2] , sort = False )
dfcheck = pd.concat( [d1,d2] , sort = False )


# In[46]:


'''
Data after balancing
'''
import seaborn as sns
plt.figure(figsize = (4,4))
ax = sns.countplot( x = 'Sentiment', order = df_all['Sentiment'].value_counts().index[0:2] , data = df_all, palette = ["#8d2865" , "#fe5862" , "#eb717c" , "#fe5862" , "#a54780" , "#8d2865" ,  "#6b1e47" , "#3d033b" , "#1e0034"] ) # , "" , "#6b1e47"  , "#7b174a" ,
plt.xticks( [0,1] , ['Positive','Negative']  )
i = 0


# In[38]:


'''
LogisticRegression Two classes
'''
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer

feature = dfcheck.text
target = dfcheck.Sentiment.astype('int')
# splitting into train and tests
X_train, X_test, Y_train, Y_test = train_test_split(feature, target, test_size = .2, random_state = 100)
# make pipeline
pipe = make_pipeline(TfidfVectorizer(),
                    LogisticRegression())
# make param grid
param_grid = {'logisticregression__C': [0.01, 0.1, 0.5, 1, 5, 30, 10, 100]}

# create and fit the model
model = GridSearchCV(pipe, param_grid, cv = 20)
model.fit(X_train,Y_train)

# make prediction and print accuracy
prediction = model.predict(X_test)
print(f"Accuracy score is {accuracy_score(Y_test, prediction):.2f}")
print(classification_report(Y_test, prediction))


# In[55]:


'''
RandomForestClassifier Two classes
'''
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer

feature = dfcheck.text
target = dfcheck.Sentiment.astype('int')
# splitting into train and tests
X_train, X_test, Y_train, Y_test = train_test_split(feature, target, test_size = .2, random_state = 100)
# make pipeline

pipe = make_pipeline(TfidfVectorizer(),
                    RandomForestClassifier())

param_grid = {'randomforestclassifier__n_estimators':[10, 100, 250, 1000],
             'randomforestclassifier__max_features':['sqrt', 'log2']}

rf_model = GridSearchCV(pipe, param_grid, cv = 10)
rf_model.fit(X_train,Y_train)

prediction = rf_model.predict(X_test)
print(f"Accuracy score is {accuracy_score(Y_test, prediction):.2f}")
print(classification_report(Y_test, prediction))


# In[54]:


'''
MultinomialNB Two classes
'''
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer

feature = dfcheck.text
target = dfcheck.Sentiment.astype('int')
# splitting into train and tests
X_train, X_test, Y_train, Y_test = train_test_split(feature, target, test_size = .2, random_state=100)
# make pipeline

pipe = make_pipeline(TfidfVectorizer(),
                    MultinomialNB())
pipe.fit(X_train,Y_train)
prediction = pipe.predict(X_test)
print(f"Accuracy score is {accuracy_score(Y_test, prediction):.2f}")
print(classification_report(Y_test, prediction))


# In[50]:


'''
SVC Two classes
'''
pipe = make_pipeline(TfidfVectorizer(),
                     SVC())
param_grid = {'svc__kernel': ['rbf', 'linear', 'poly'],
             'svc__gamma': [0.1, 1, 10, 100],
             'svc__C': [0.1, 1, 10, 100]}

svc_model = GridSearchCV(pipe, param_grid, cv = 4)
svc_model.fit(X_train, Y_train)

prediction = svc_model.predict(X_test)
print(f"Accuracy score is {accuracy_score(Y_test, prediction):.2f}")
print(classification_report(Y_test, prediction))


# In[49]:


'''
AdaBoostClassifier NB Two classes
'''
pipe = make_pipeline(TfidfVectorizer(),
                    AdaBoostClassifier(
     MultinomialNB(), n_estimators = 250,
    algorithm = "SAMME", learning_rate = 0.1, random_state = 42))
pipe.fit(X_train,Y_train)
prediction = pipe.predict(X_test)
print(f"Accuracy score is {accuracy_score(Y_test, prediction):.2f}")
print(classification_report(Y_test, prediction))


# In[43]:


'''
XGBClassifier Two classes
'''
pipe = make_pipeline(TfidfVectorizer(),
                    XGBClassifier())
pipe.fit(X_train,Y_train)
prediction = pipe.predict(X_test)
print(f"Accuracy score is {accuracy_score(Y_test, prediction):.2f}")
print(classification_report(Y_test, prediction))


# In[42]:


'''
LGBMClassifier Two classes
'''
from lightgbm import LGBMClassifier
pipe = make_pipeline(TfidfVectorizer(),
                    LGBMClassifier())
pipe.fit(X_train,Y_train)
prediction = pipe.predict(X_test)
print(f"Accuracy score is {accuracy_score(Y_test, prediction):.2f}")
print(classification_report(Y_test, prediction))


# In[41]:


'''
AdaBoostClassifier RF Two classes
'''
pipe = make_pipeline(TfidfVectorizer(),
                    AdaBoostClassifier(
    RandomForestClassifier(n_estimators  = 250), n_estimators = 250,
    algorithm="SAMME", learning_rate = 0.02, random_state = 42))
pipe.fit(X_train,Y_train)
prediction = pipe.predict(X_test)
print(f"Accuracy score is {accuracy_score(Y_test, prediction):.2f}")
print(classification_report(Y_test, prediction))


# In[72]:


'''
AdaBoostClassifier DT Two classes
'''
pipe = make_pipeline(TfidfVectorizer(),
                    AdaBoostClassifier(
    DecisionTreeClassifier(max_depth = 1), n_estimators = 350,
    algorithm="SAMME.R", learning_rate = 0.6, random_state = 42))
pipe.fit(X_train,Y_train)
prediction = pipe.predict(X_test)
print(f"Accuracy score is {accuracy_score(Y_test, prediction):.2f}")
print(classification_report(Y_test, prediction))


# In[68]:


'''
AdaBoostClassifier LR Two classes
'''
pipe = make_pipeline(TfidfVectorizer(),
                    AdaBoostClassifier(
     LogisticRegression(), n_estimators = 250,
    algorithm = "SAMME", learning_rate = .5, random_state = 42))
pipe.fit(X_train,Y_train)
prediction = pipe.predict(X_test)
print(f"Accuracy score is {accuracy_score(Y_test, prediction):.2f}")
print(classification_report(Y_test, prediction))


# ## Question2 Analysis

# In[12]:


'''
Read data about covid19
'''
eng = pd.read_csv('training1.csv')
eng.drop(['Labels' , 'ID'] , axis = 1 , inplace  = True)
eng[0:1560]


# In[13]:


"""
TextBlob for classify posts  as positive , neutral or negative polarity
"""
def sentiment_analysis(df):
    def getSubjectivity(text):
           return TextBlob(text).sentiment.subjectivity
    #Create a function to get the polarity
    def getPolarity(text):
           return TextBlob(text).sentiment.polarity

    #Create two new columns ‘Subjectivity’ & ‘Polarity’
    eng['TextBlob_Subjectivity'] = eng['Tweet'].apply(getSubjectivity)
    eng['TextBlob_Polarity'] = eng['Tweet'].apply(getPolarity)
    def getAnalysis(score):
        if score < 0:
            return 0
        elif score == 0:
            return 1
        else:
            return 1
    df ['TextBlob_Analysis'] = eng['TextBlob_Polarity'].apply(getAnalysis)
    return df

sentiment_analysis(eng)


# In[14]:


"""
Remove Symbols from English texts
"""
def preproc_eng(str_in):
    def cleanup_string(str_in):
        '''
        Remove any symbols
        '''
        try:      
            str1 = str_in.replace("’"," ").replace("."," ").replace("!"," ").replace(")"," ").replace("("," ").replace("‘"," ").replace("“"," ").replace("”"," ").replace("–"," ").replace("\n"," ").replace("\r"," ").replace("§"," ")
            #cleanups
            for char in st.punctuation:
                str1 = str1.replace(char, ' ')        


        except Exception:
            print('' , end = '')
        return str1


    """
    Remove emojis from posts and comments
    """
    def cleanup_emojis(str_in):
        '''
        Remove any emoji
        '''
        try:      
            emoji_pattern = re.compile("["
                    u"\U0001F600-\U0001F64F"  # emotions
                    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                    u"\U0001F680-\U0001F6FF"  # transport & map symbols
                    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                    u"\U00002500-\U00002BEF"  # chinese char
                    u"\U00002702-\U000027B0"
                    u"\U00002702-\U000027B0"
                    u"\U000024C2-\U0001F251"
                    u"\U0001f926-\U0001f937"
                    u"\U00010000-\U0010ffff"
                    u"\u2640-\u2642" 
                    u"\u2600-\u2B55"
                    u"\u200d"
                    u"\u23cf"
                    u"\u23e9"
                    u"\u231a"
                    u"\ufe0f"  # dingbats
                    u"\u3030"
                    "]+", flags=re.UNICODE)
            str1 = emoji_pattern.sub(r' ', str_in) # no emoji    
        except Exception:
            print('', end = '')
            #print('<----',str_in,'---->')
        return str1
    str1 =  cleanup_emojis(str_in)
    str1 =  cleanup_string(str_in)
    return str1

eng['Tweet'] = eng['Tweet'].apply(lambda x: preproc_eng(x))


# In[17]:


'''
Translate from english to arabic
'''
from googletrans import Translator, constants
from deep_translator import GoogleTranslator
from pprint import pprint

translator = Translator()

# 500  
sentences = list(eng['Tweet'].iloc[:500])
arabicList = []
for i in sentences:
    english_Text = GoogleTranslator(source = 'auto', target = 'ar').translate(i)
    arabicList.append(english_Text)
arabicList    


# In[18]:


# 1000  
sentences = list(eng['Tweet'].iloc[500:1000])
for i in sentences:
    english_Text = GoogleTranslator(source = 'auto', target = 'ar').translate(i)
    arabicList.append(english_Text)    


# In[69]:


'''
read data after translating
'''
eng2 = pd.read_csv('covidData.csv')


# In[70]:


'''
preprocess covid data
'''
eng2['text'] = eng2['text'].apply(preprocessing)
eng2['text']  = eng2['text'].apply(light_stem)
eng2['text']  = eng2['text'].apply(lemitizing)


# In[83]:


'''
Analysis of Covid19 and Omicron
'''
plotdata = pd.DataFrame({
    "Covid19":[ len(eng2.text.loc[(eng2.TextBlob_Analysis == 1)]) , len(eng2.text.loc[(eng2.TextBlob_Analysis == 0)]) ],
    "Omicron":[ len(df_all.text.loc[(df_all.Sentiment == 1)]) , len(df_all.text.loc[(df_all.Sentiment == 0)]) ]},
    index = ["Positive", "Negative"])

plotdata.plot(kind = "bar",figsize = (15, 8) , color = [ "#8d2865" , "#fe5862" , "#fe5862" , "#a54780" , "#eb717c" ,  "#8d2865" ,  "#6b1e47" , "#3d033b" , "#1e0034"])
plt.title("Comparsion")
plt.xlabel("Class")
plt.xticks(rotation = 360)
plt.ylabel("Count")
plt.savefig('pic1.png')


# In[ ]:


'''
Measure perecents of omicron and covid19 data
'''
covpos = len(eng2.loc[(eng2.TextBlob_Analysis == 1)]) / ( len(eng2.loc[(eng2.TextBlob_Analysis == 1)]) + len(eng2.loc[(eng2.TextBlob_Analysis == 0)]) )
ompos = len(df_all.loc[(df_all.Sentiment == 1)]) / ( len(df_all.loc[(df_all.Sentiment == 1)]) + len(df_all.loc[(df_all.Sentiment == 0)]) )
ompos

