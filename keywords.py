# -*- coding: utf-8 -*-
"""
@author: patricklam
Copyright Gary King, Patrick Lam, and Molly Roberts 2015. Not to be distributed without permission.
"""

##### Note: Need to install the following depending on types of data #####


import os
import sys
import string
import re
import time
import random
import numpy as np
import scipy.stats
import pandas as pd


from collections import OrderedDict
from collections import defaultdict
from collections import namedtuple
from nltk.stem.snowball import SnowballStemmer
from pandas import DataFrame
from pandas import Series
from math import lgamma
from sklearn import svm
from sklearn import lda
from sklearn import tree
from sklearn import naive_bayes
from sklearn import linear_model
from sklearn import ensemble
from sklearn import neighbors
from sklearn.feature_extraction.text import CountVectorizer


##### Read in corpus from a directory #####
def ReadDirectory(directory, remove_startswithdot=True, sample=None, seed=12345, deduplicate=True):
    """Read a corpus of text from a directory of .txt files."""
    file_names = os.listdir(directory)

    if remove_startswithdot:
        file_names = [file for file in file_names if os.path.isfile(directory+'/'+file) and not file.startswith('.')]
    else:
        file_names = [file for file in file_names if os.path.isfile(directory+'/'+file)]

    if sample != None:
        random.seed(seed)
        file_names = random.sample(file_names, sample)

    text_list = []
    id_list = []
    for fn in file_names:
        f = open(directory+'/'+fn, 'r', encoding='utf-8')
        text_list.append(' '.join(f.readlines()))
        f.close()
        id_list.append(fn)
    data_dict = {}
    data_dict['id'] = id_list
    data_dict['text'] = text_list
    data = DataFrame.from_dict(data_dict).sort('id')
    if deduplicate:
        data = data[data.duplicated('text') == False]

    return data



##### Read in corpus from a CSV file #####
def ReadSheet(filename, sample=None, seed=12345, deduplicate=True, delimiter=',', escapechar=None, text_colname=None, id_colname=None, date_colname=None, date_start=None, date_end=None):
    """Read a corpus of text from a .csv, .xls, or .xlsx spreadsheet."""
    if text_colname == None:
        print('Please specify a column name identifying the text.')
        return

    if filename.endswith('.csv'):
        data = pd.read_csv(open(filename, 'rU'), sep=delimiter, escapechar=escapechar, encoding='utf-8', dtype={id_colname: str}, error_bad_lines=False)
    elif filename.endswith('.xls') or filename.endswith('.xlsx'):
        data = pd.read_excel(filename)

    if date_colname != None and (date_start != None or date_end != None):
        if date_start != None:
            data = data[data[date_colname] >= date_start]
        if date_end != None:
            data = data[data[date_colname] <= date_end]

    if deduplicate:
        data = data[data.duplicated(text_colname) == False]

    if sample != None:
        random.seed(seed)
        samp_rows = random.sample(list(range(len(data))), sample)
        data = data.iloc[samp_rows]

    if text_colname != 'text' and 'text' in list(data.columns):
        data.rename(columns={'text':'old_text_column_renamed'}, inplace=True)
    if id_colname != 'id' and 'id' in list(data.columns):
        data.rename(columns={'id':'old_id_column_renamed'}, inplace=True)
    if id_colname == None:
        data['id'] = list(range(len(data)))
        id_colname = 'id'
    data.rename(columns={text_colname:'text', id_colname:'id', 'text_processed':'old_text_processed_column_renamed'}, inplace=True)
    data.sort_values('id', inplace=True)
    data.text = [t if type(t) == str else '' for t in data.text]
    return data



##### Process Text #####
def ProcessText(corpus, min_wordlength=1, stem=True, remove_numbers=True, remove_punct=True, remove_stopwords=True, remove_wordlist=None, keep_twitter_symbols=True, keep_urls=True, language='english'):
    """Take a list of text entries and processes them."""
    if stem and language in SnowballStemmer.languages:
        stemmer = SnowballStemmer(language).stem
    #elif stem and language not in Stemmer.algorithms():
    #    print("No support for stemming in %s.  Stem argument set to False." % language)
    #    stem = False

    stoplist = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
                'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
                'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
                'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
                'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
                's', 't', 'can', 'will', 'just', 'don', 'should', 'now']
    remove_stoplist = '|'.join(stoplist)
    pattern = re.compile('\w')
    text_list = []
    stem_map = {}

    if type(remove_wordlist) == str:
        remove_wordlist = remove_wordlist.split()

    if stem and remove_wordlist != None:
        remove_wordlist = list(set(remove_wordlist + list(set([stemmer(w) for w in remove_wordlist]))))

    for text in corpus:

        text = text.replace('http://', ' http://')
        text = text.replace('https://', ' https://')
        text = text.replace('\u201c', '"')
        text = text.replace('\u201d', '"')
        text = text.replace('\u2019', "'")

        keep = []
        if keep_urls:
            urls = [w for w in text.split() if w.lower().startswith('http://') or w.lower().startswith('https://') or w.lower().startswith('www.')]
            keep = keep + urls
            text = ' '.join([w for w in text.split() if w not in urls])

        text = text.lower()

        if keep_twitter_symbols:
            keep = keep + re.findall(r'\B#\w+\b', text) + re.findall(r'\b#\w+\b', text) + re.findall(r'\B@\w+\b', text)
            regex = re.compile(r'\B#\w+\b|\b#\w+\b|\B@\w+\b')
            text = regex.sub(' ', text)

        if remove_wordlist != None:
            keep = [w for w in keep if w not in remove_wordlist]

        if remove_numbers:
            text = re.sub('[0-9]', ' ', text)
            #text = text.translate(string.maketrans(string.digits, ' '*len(string.digits)))

        if remove_punct:
            text = re.sub(r'[!"#$%&()*+,\-./:;<=>?@[\\\]^_`{|}~\']', ' ', text)
            #text = re.sub("'", " ", text)
            #text = text.translate(string.maketrans(punct, " "*len(punct)), "'")

        if stem:
            unstemmed = text.split()
            stemmed = [stemmer(w) for w in unstemmed]
            changed = [(i,j) for i,j in zip(stemmed, unstemmed) if i != j]
            for w in changed:
                if w[0] in stem_map:
                    stem_map[w[0]].update([w[1]])
                else:
                    stem_map[w[0]] = set([w[1]])

            text = ' '.join(stemmed)

        if remove_stopwords:
            regex = re.compile(r'\b('+remove_stoplist+r')\b', flags=re.IGNORECASE)
            text = regex.sub(' ', text)

        if remove_wordlist != None:
            for w in remove_wordlist:
                if pattern.match(w) == None:
                    regex = re.compile(' ' + w + r'\b|^' + w + r'\b', flags=re.IGNORECASE)
                    text = regex.sub(' ', text)
                else:
                    regex = re.compile(r'\b'+w+r'\b', flags=re.IGNORECASE)
                    text = regex.sub(' ', text)

        if min_wordlength > 1:
            text = ' '.join([w for w in text.split() if len(w) >= min_wordlength])

        text = ' '.join(text.split())
        if len(keep) > 0:
            text = text + ' ' + ' '.join(keep)

        text_list.append(text)

    if stem:
        for k,v in stem_map.items():
            stem_map[k] = ' '.join(list(v))

    processText_obj = namedtuple('processText_object', 'text stem_map')
    res = processText_obj(text_list, stem_map)

    return res



##### Get document-term matrix from corpus dictionary #####

def WSTokenizer(text):
    return text.split()

def GetDTM(corpus_data, min_df=1, max_df=1.0, vocabulary=None, ngram_range=(1,1), tokenizer=WSTokenizer):
    """Output a document-term matrix from a corpus."""
    if vocabulary == None:
        vectorizer = CountVectorizer(min_df=min_df, max_df=max_df, decode_error='ignore', tokenizer=tokenizer, ngram_range=ngram_range)
    else:
        vectorizer = CountVectorizer(min_df=min_df, max_df=max_df, decode_error='ignore', tokenizer=tokenizer, vocabulary=vocabulary, ngram_range=ngram_range)
    dtm_object = namedtuple('dtm_object', 'dtm docs terms')
    dtm = vectorizer.fit_transform(corpus_data.text_processed)
    terms = vectorizer.get_feature_names()
    res = dtm_object(dtm=dtm, terms=terms, docs=list(corpus_data.id))
    return res


##### Classifiers #####

def Classifiers(y_train, X_train, X_test, label, algorithms=['nbayes', 'nearest', 'logit', 'SVM', 'LDA', 'tree', 'gboost', 'rf'], rf_trees=200, seed=12345):
    """Run classifiers and output test set predictions for each classifier."""
    ## Get probability of reference set from classifiers
    classify_dict = {}

    ## Naive Bayes
    if 'nbayes' in algorithms:
        ts = time.time()
        clf_nb = naive_bayes.MultinomialNB()
        clf_nb.fit(X_train, y_train)
        classify_dict['nbayes'] = clf_nb.predict(X_test).tolist()
        te = time.time()
        print("Time for Naive Bayes: {} seconds".format(round(te-ts, 2)))

    ## Nearest Neighbor
    if 'nearest' in algorithms:
        ts = time.time()
        clf_nn = neighbors.KNeighborsClassifier()
        clf_nn.fit(X_train, y_train)
        classify_dict['nearest'] = clf_nn.predict(X_test).tolist()
        te = time.time()
        print("Time for Nearest Neighbor: {} seconds".format(round(te-ts, 2)))

    ## Logit
    if 'logit' in algorithms:
        ts = time.time()
        clf_logit = linear_model.LogisticRegression()
        clf_logit.fit(X_train, y_train)
        classify_dict['logit'] = clf_logit.predict(X_test).tolist()
        te = time.time()
        print("Time for Logit: {} seconds".format(round(te-ts, 2)))

    ## Support vector machine
    if 'SVM' in algorithms:
        ts = time.time()
        clf_svm = svm.SVC(C=100, probability=True, random_state=seed)
        clf_svm.fit(X_train, y_train)
        classify_dict['svm'] = clf_svm.predict(X_test).tolist()
        te = time.time()
        print("Time for SVM: {} seconds".format(round(te-ts, 2)))

    ## Linear discriminant
    if 'LDA' in algorithms:
        ts = time.time()
        clf_lda = lda.LDA()
        clf_lda.fit(X_train.toarray(), y_train)
        classify_dict['lda'] = clf_lda.predict(X_test.toarray()).tolist()
        te = time.time()
        print("Time for LDA: {} seconds".format(round(te-ts, 2)))

    ## Tree
    if 'tree' in algorithms:
        ts = time.time()
        clf_tree = tree.DecisionTreeClassifier(random_state=seed)
        clf_tree.fit(X_train.toarray(), y_train)
        classify_dict['tree'] = clf_tree.predict(X_test.toarray()).tolist()
        te = time.time()
        print("Time for Tree: {} seconds".format(round(te-ts, 2)))

    ## Gradient boosting
    if 'gboost' in algorithms:
        ts = time.time()
        clf_gboost = ensemble.GradientBoostingClassifier(random_state=seed)
        clf_gboost.fit(X_train.toarray(), y_train)
        classify_dict['gboost'] = clf_gboost.predict(X_test.toarray()).tolist()
        te = time.time()
        print("Time for Gradient Boosting: {} seconds".format(round(te-ts, 2)))

    ## Random forest
    if 'rf' in algorithms:
        ts = time.time()
        clf_rf = ensemble.RandomForestClassifier(n_estimators=rf_trees, random_state=seed)
        clf_rf.fit(X_train.toarray(), y_train)
        classify_dict['rf'] = clf_rf.predict(X_test.toarray()).tolist()
        te = time.time()
        print("Time for Random Forest: {} seconds".format(round(te-ts, 2)))

    return classify_dict


##### Find number of documents with given word #####
def DocCounts(frequent_words, dtm_obj, doc_names=None):
    """Find the number of documents with a given list of words and document-term matrix."""
    counts = {}
    terms = dtm_obj.terms
    if doc_names == None:
        dtm = dtm_obj.dtm
    else:
        doc_index = [dtm_obj.docs.index(name) for name in dtm_obj.docs if name in doc_names]
        dtm = dtm_obj.dtm[doc_index,]
    for word in frequent_words:
        dtm_sub = dtm[:,terms.index(word)]
        counts[word] = len(dtm_sub.nonzero()[0])
    return counts


def BooleanSearch(text, all_words=None, any_words=None, none_words=None):
    """Take a list of text and a set of boolean keyword searches and return the index numbers of the text entries that match the boolean search."""
    if type(all_words) == str:
        all_words = [all_words]
    if type(any_words) == str:
        any_words = [any_words]
    if type(none_words) == str:
        none_words = [none_words]
    if all_words == None and any_words == None and none_words == None:
        print("Please specify at least one search option with all_words, any_words, or none_words.")
        return

    ## Start with index of entire text list
    text_index = list(range(len(text)))

    ## Narrow list down to only indices where text contains all of the 'all_words'
    if all_words != None:
        all_words_list = text_index
        for w in all_words:
            if re.match('\W', w[0]) == None:
                regex = re.compile(r'\b' + w + r'\b', flags=re.IGNORECASE)
            else:
                regex = re.compile(r'\B' + w + r'\b', flags=re.IGNORECASE)
            temp_list = [i for i in text_index if regex.search(text[i]) != None]
            all_words_list = sorted(list(set(all_words_list).intersection(set(temp_list))))
        text_index = all_words_list

    ## Narrow list down to only indices where text contains one or more of the 'any_words'
    if any_words != None:
        any_words_list = []
        for w in any_words:
            if re.match('\W', w[0]) == None:
                regex = re.compile(r'\b' + w + r'\b', flags=re.IGNORECASE)
            else:
                regex = re.compile(r'\B' + w + r'\b', flags=re.IGNORECASE)
            temp_list = [i for i in text_index if regex.search(text[i]) != None]
            any_words_list = any_words_list + temp_list
        text_index = sorted(list(set(any_words_list)))

    ## Narrow list down to only indices where text does not contain any of the 'none_words'
    if none_words != None:
        none_words_list = text_index
        for w in none_words:
            if re.match('\W', w[0]) == None:
                regex = re.compile(r'\b' + w + r'\b', flags=re.IGNORECASE)
            else:
                regex = re.compile(r'\B' + w + r'\b', flags=re.IGNORECASE)
            temp_list = [i for i in text_index if regex.search(text[i]) == None]
            none_words_list = sorted(list(set(none_words_list).intersection(set(temp_list))))
        text_index = none_words_list

    return text_index



class Keywords:

    def __init__(self):
        """A Keyword class"""
        print("Keyword object initialized.")

    def LoadDataset(self, data, sample=None, deduplicate=True, seed=12345, delimiter=',', escapechar=None, text_colname=None, id_colname=None, date_colname=None, date_start=None, date_end=None):
        """Load a corpus from a .csv or .xls file or a directory of .txt files."""
        ts = time.time()
        if data.endswith('.csv') or data.endswith('.xls') or data.endswith('.xlsx'):
            self.dataset = ReadSheet(data, sample, seed, deduplicate, delimiter, escapechar, text_colname, id_colname, date_colname, date_start, date_end)
        else:
            self.dataset = ReadDirectory(data, remove_startswithdot=True, sample=sample, seed=seed, deduplicate=deduplicate)
        te = time.time()
        print("Loaded corpus of size {} in {} seconds.".format(str(len(self.dataset)), round(te-ts, 2)))


    def ReferenceSet(self, data=None, sample=None, deduplicate=True, seed=12345, delimiter=',', escapechar=None, text_colname=None, id_colname=None,
                     date_colname=None, date_start=None, date_end=None, search_string=None, retweets=False, language='english', all_words=None, any_words=None, none_words=None):
        """Load a reference set from a .csv or .xls file or a directory of .txt files or subset reference set from full dataset by keyword."""
        ts = time.time()
        if data == None and hasattr(self, 'dataset') == False:
            print("Please specify data source.")
            return
        if data == None and hasattr(self, 'dataset'):
            docs = self.dataset
            if date_colname != None and date_start != None:
                docs = docs[docs[date_colname] >= date_start]
            if date_colname != None and date_end != None:
                docs = docs[docs[date_colname] <= date_end]
            docs_index = BooleanSearch(list(docs.text), all_words, any_words, none_words)
            self.reference_set = docs.iloc[docs_index]
        elif data == 'twitter':
            if sample == None or sample <= 0:
                print("For twitter data, please specify a sample size greater than 0.")
                return
            self.reference_set = GetTweets(search_string, sample, retweets, language)
        elif data.endswith('.csv') or data.endswith('.xls') or data.endswith('.xlsx'):
            self.reference_set = ReadSheet(data, sample, seed, deduplicate, delimiter, escapechar, text_colname, id_colname, date_colname, date_start, date_end)
        else:
            self.reference_set = ReadDirectory(data, remove_startswithdot=True, sample=sample, seed=seed, deduplicate=deduplicate)
        te = time.time()
        print("Loaded reference set of size {} in {} seconds.".format(str(len(self.reference_set)), round(te-ts, 2)))

    def AddToReferenceSet(self, data=None, sample=None, deduplicate=True, seed=12345, delimiter=',', escapechar=None, text_colname=None, id_colname=None,
                          date_colname=None, date_start=None, date_end=None, search_string=None, retweets=False, language='english', all_words=None, any_words=None, none_words=None):
        """Add to an existing reference set from a .csv or .xls file or a directory of .txt files or subset additional reference set from full dataset by keyword."""
        ts = time.time()
        if data == None and hasattr(self, 'dataset') == False:
            print("Please specify data source.")
            return
        if data == None and hasattr(self, 'dataset'):
            docs = self.dataset
            if date_colname != None and date_start != None:
                docs = docs[docs[date_colname] >= date_start]
            if date_colname != None and date_end != None:
                docs = docs[docs[date_colname] <= date_end]
            docs_index = BooleanSearch(list(docs.text), all_words, any_words, none_words)
            docs = docs.iloc[docs_index]

        elif data == 'twitter':
            if sample == None or sample <= 0:
                print("For twitter data, please specify a sample size greater than 0.")
                return
            docs = GetTweets(search_string, sample, retweets, language)
        elif data.endswith('.csv') or data.endswith('.xls') or data.endswith('.xlsx'):
            docs = ReadSheet(data, sample, seed, deduplicate, delimiter, escapechar, text_colname, id_colname, date_colname, date_start, date_end)
        else:
            docs = ReadDirectory(data, remove_startswithdot=True, sample=sample, seed=seed, deduplicate=deduplicate)

        ref = self.reference_set
        orig_n = len(ref)
        ref = ref.append(docs, ignore_index=True)
        ref = ref[ref.duplicated('text') == False]
        added = len(ref) - orig_n
        self.reference_set = ref
        te = time.time()
        print("Added {} reference set documents in {} seconds.".format(str(added), round(te-ts, 2)))


    def SearchSet(self, data=None, sample=None, deduplicate=True, seed=12345, delimiter=',', escapechar=None, text_colname=None, id_colname=None,
                  date_colname=None, date_start=None, date_end=None, search_string=None, retweets=False, language='english', all_words=None, any_words=None, none_words=None):
        """Load a search set from a .csv or .xls file or a directory of .txt files or subset search set from full dataset by keyword."""
        ts = time.time()
        if data == None and hasattr(self, 'dataset') == False:
            print("Please specify data source.")
            return
        if data == None and hasattr(self, 'dataset'):
            docs = self.dataset
            if date_colname != None and date_start != None:
                docs = docs[docs[date_colname] >= date_start]
            if date_colname != None and date_end != None:
                docs = docs[docs[date_colname] <= date_end]
            if all_words == None and any_words == None and none_words == None:
                docs_index = [i for i,ids in enumerate(docs.id) if ids not in list(self.reference_set.id)]
                docs = docs.iloc[docs_index]
            else:
                docs_index = BooleanSearch(list(docs.text), all_words, any_words, none_words)
                docs = docs.iloc[docs_index]
            self.search_set = docs

        elif data == 'twitter':
            if sample == None:
                print("For twitter data, please specify a sample size")
                return
            self.search_set = GetTweets(search_string, sample, retweets, language)
        elif data.endswith('.csv') or data.endswith('.xls') or data.endswith('.xlsx'):
            self.search_set = ReadSheet(data, sample, seed, deduplicate, delimiter, escapechar, text_colname, id_colname, date_colname, date_start, date_end)
        else:
            self.search_set = ReadDirectory(data, remove_startswithdot=True, sample=sample, seed=seed, deduplicate=deduplicate)
        te = time.time()
        print("Loaded search set of size {} in {} seconds.".format(str(len(self.search_set)), round(te-ts, 2)))


    def AddToSearchSet(self, data=None, sample=None, deduplicate=True, seed=12345, delimiter=',', escapechar=None, text_colname=None, id_colname=None,
                          date_colname=None, date_start=None, date_end=None, search_string=None, retweets=False, language='english', all_words=None, any_words=None, none_words=None):
        """Add to an existing search set from a .csv or .xls file or a directory of .txt files or subset additional search set from full dataset by keyword."""
        ts = time.time()
        if data == None and hasattr(self, 'dataset') == False:
            print("Please specify data source.")
            return
        if data == None and hasattr(self, 'dataset'):
            docs = self.dataset
            if date_colname != None and date_start != None:
                docs = docs[docs[date_colname] >= date_start]
            if date_colname != None and date_end != None:
                docs = docs[docs[date_colname] <= date_end]
            docs_index = BooleanSearch(list(docs.text), all_words, any_words, none_words)
            docs = docs.iloc[docs_index]

        elif data == 'twitter':
            if sample == None or sample <= 0:
                print("For twitter data, please specify a sample size greater than 0.")
                return
            docs = GetTweets(search_string, sample, retweets, language)
        elif data.endswith('.csv') or data.endswith('.xls') or data.endswith('.xlsx'):
            docs = ReadSheet(data, sample, seed, deduplicate, delimiter, escapechar, text_colname, id_colname, date_colname, date_start, date_end)
        else:
            docs = ReadDirectory(data, remove_startswithdot=True, sample=sample, seed=seed, deduplicate=deduplicate)

        search = self.search_set
        orig_n = len(search)
        search = search.append(docs, ignore_index=True)
        search = search[search.duplicated('text') == False]
        added = len(search) - orig_n
        self.search_set = search
        te = time.time()
        print("Added {} search set documents in {} seconds.".format(str(added), round(te-ts, 2)))



    def ProcessData(self, doc_set=None, min_wordlength=3, stem=True, remove_numbers=True, remove_punct=True, remove_stopwords=True, remove_wordlist=None, keep_twitter_symbols=True, keep_urls=True, language='english'):
        """Process text and return a processed dataset."""
        ts = time.time()
        pd.set_option('chained_assignment', None) #turn off incorrect warning
        if stem and language not in SnowballStemmer.languages:
            print("No support for stemming in {}.  Stem argument set to False.".format(language))
            stem = False
        if doc_set == None:
            if hasattr(self, 'reference_set'):
                ref_processtext = ProcessText(self.reference_set.text, min_wordlength, stem, remove_numbers, remove_punct, remove_stopwords, remove_wordlist, keep_twitter_symbols, keep_urls, language)
                self.reference_set['text_processed'] = ref_processtext.text
            if hasattr(self, 'search_set'):
                search_processtext = ProcessText(self.search_set.text, min_wordlength, stem, remove_numbers, remove_punct, remove_stopwords, remove_wordlist, keep_twitter_symbols, keep_urls, language)
                self.search_set['text_processed'] = search_processtext.text
        elif doc_set.lower() in ['reference', 'ref']:
            ref_processtext = ProcessText(self.reference_set.text, min_wordlength, stem, remove_numbers, remove_punct, remove_stopwords, remove_wordlist, keep_twitter_symbols, keep_urls, language)
            self.reference_set['text_processed'] = ref_processtext.text
        elif doc_set.lower() == 'search':
            search_processtext = ProcessText(self.search_set.text, min_wordlength, stem, remove_numbers, remove_punct, remove_stopwords, remove_wordlist, keep_twitter_symbols, keep_urls, language)
            self.search_set['text_processed'] = search_processtext.text

        if doc_set not in ['reference', 'ref'] and hasattr(self, 'search_set'):
            if len(search_processtext.stem_map) > 0:
                self.stem_map = search_processtext.stem_map
            else:
                self.stem_map = {}
        te = time.time()
        print("Time to process corpus: {} seconds".format(round(te-ts, 2)))



    def ReferenceKeywords(self, ngrams=1):
        """Find a list of the words ranked by appearance in number of documents in processed reference set."""
        if type(ngrams) == int:
            ngrams = (1, ngrams)
        if type(ngrams) == list:
            ngrams = tuple(ngrams)

        ref_dtm = GetDTM(self.reference_set, min_df=1, ngram_range=ngrams)
        ref_dc = DocCounts(ref_dtm.terms, ref_dtm, None)
        ref_stats = DataFrame.from_dict(ref_dc, orient='index')
        ref_stats.rename(columns={0:'counts'}, inplace=True)
        ref_stats.sort_values('counts', ascending=False, inplace=True)
        ref_stats['ranks'] = scipy.stats.rankdata(ref_stats.counts.max() - ref_stats.counts, method='min').astype('int')
        self.reference_stats = ref_stats
        self.reference_keywords = list(ref_stats.index)
        print("\n{} reference set keywords found.".format(str(len(self.reference_keywords))))


    def ClassifyDocs(self, seed=12345, min_df=1, max_df=1.0, ref_trainprop=.33, search_trainprop=.33, algorithms=['nbayes', 'nearest', 'logit', 'SVM', 'tree', 'rf']):
        """Classify search set (test set) documents as in reference or search set using classifiers and a sampled training set."""
        corpus = self.reference_set[['id', 'text_processed']].append(self.search_set[['id', 'text_processed']])
        ts = time.time()
        dtm = GetDTM(corpus, min_df=min_df, max_df=max_df)
        te = time.time()
        print("\nDocument Term Matrix: {} by {} with {} nonzero elements".format(dtm.dtm.shape[0], dtm.dtm.shape[1], dtm.dtm.nnz))
        print("\nTime to get document-term matrix: {} seconds".format(round(te-ts, 2)))

        ## Create training and test sets
        random.seed(seed)
        ref_sampind = random.sample(list(range(len(self.reference_set))), int(round(len(self.reference_set)*ref_trainprop)))
        search_sampind = random.sample(list(range(len(self.reference_set), len(corpus))), int(round(len(self.search_set)*search_trainprop)))
        train_ind = sorted(ref_sampind + search_sampind)
        test_ind = list(range(len(self.reference_set), len(corpus)))
        print("\nRef training size: {}; Search training size: {}; Training size: {}; Test size: {}\n".format(len(ref_sampind), len(search_sampind), len(train_ind), len(test_ind)))

        y = [0]*len(self.reference_set) + [1]*len(self.search_set)  # use 1 for search to avoid rare events problem
        y_train = [y[i] for i in train_ind]
        X_train = dtm.dtm[train_ind,:]
        X_test = dtm.dtm[test_ind,:]

        classify_dict = Classifiers(y_train, X_train, X_test, label=0, algorithms=algorithms, rf_trees=200, seed=seed)
        self.target_votematrix = 1-DataFrame(classify_dict, index=list(self.search_set.id)) # switch back to 1s for reference set classification (target set)
        self.target_votecount = self.target_votematrix.sum(1)
        self.target_tabvotes = self.target_votecount.value_counts().sort_index()



    def FindTargetSet(self, vote_min=1):
        """Identify an estimated target set within search set using classifier votes."""
        self.target_docnames = list(self.target_votecount[self.target_votecount >= vote_min].index)
        self.nontarget_docnames = list(self.target_votecount[self.target_votecount < vote_min].index)
        print("{} documents in target set".format(len(self.target_docnames)))
        print("{} documents in non-target set".format(len(self.nontarget_docnames)))


    def FindKeywords(self, support=10, ngrams=1):
        """Identify and rank keywords within target and non-target sets."""
        if type(ngrams) == int:
            ngrams = (1, ngrams)
        if type(ngrams) == list:
            ngrams = tuple(ngrams)

        frequent_words = GetDTM(self.search_set, min_df=support, ngram_range=ngrams).terms
        dtm = GetDTM(self.search_set, min_df=1, vocabulary=frequent_words)
        total_dc = DocCounts(frequent_words, dtm, None)
        target_dc = DocCounts(frequent_words, dtm, self.target_docnames)
        ntarget = len(self.target_docnames)
        nnontarget = len(self.nontarget_docnames)

        alpha1 = 1
        alpha0 = 1
        ranked_by = 'll'
        target_wordlist = []
        nontarget_wordlist = []
        target_stats = defaultdict(list)
        nontarget_stats = defaultdict(list)
        for word in frequent_words:
            n1 = target_dc[word]
            n0 = total_dc[word] - target_dc[word]
            p1 = (float(n1)/ntarget)*100
            p0 = (float(n0)/nnontarget)*100
            n1_not = ntarget - n1
            n0_not = nnontarget - n0
            ll = (lgamma(n1+alpha1) + lgamma(n0+alpha0) - lgamma(n1+alpha1+n0+alpha0)) + (lgamma(n1_not+alpha1) + lgamma(n0_not+alpha0) - lgamma(n1_not+alpha1+n0_not+alpha0))
            if hasattr(self, 'reference_keywords'):
                r_count = 0
                if word in self.reference_keywords:
                    r_count = self.reference_stats.loc[word, 'counts']
            else:
                r_count = None

            if p0 > p1:
                p1, p0 = p0, p1
                n1, n0 = n0, n1
                nontarget_wordlist.append(word)
                nontarget_stats['n1'].append(n1)
                nontarget_stats['n0'].append(n0)
                nontarget_stats['p1'].append(p1)
                nontarget_stats['p0'].append(p0)
                nontarget_stats['ll'].append(ll)
                nontarget_stats['T'].append(n0)
                nontarget_stats['S'].append(n0+n1)
                nontarget_stats['R'].append(r_count)
            else:
                target_wordlist.append(word)
                target_stats['n1'].append(n1)
                target_stats['n0'].append(n0)
                target_stats['p1'].append(p1)
                target_stats['p0'].append(p0)
                target_stats['ll'].append(ll)
                target_stats['T'].append(n1)
                target_stats['S'].append(n0+n1)
                target_stats['R'].append(r_count)

        target_stats = DataFrame(target_stats, index=target_wordlist)
        target_stats = target_stats.reindex_axis(['ll', 'n1', 'n0', 'p1', 'p0','T','S','R'], axis=1)
        target_stats.sort_values(ranked_by, ascending=False, inplace=True)
        nontarget_stats = DataFrame(nontarget_stats, index=nontarget_wordlist)
        nontarget_stats = nontarget_stats.reindex_axis(['ll', 'n1', 'n0', 'p1', 'p0','T','S','R'], axis=1)
        nontarget_stats.sort_values(ranked_by, ascending=False, inplace=True)

        if hasattr(self, 'reference_keywords'):
            ref_words = self.reference_keywords
            ref_dtm = GetDTM(self.search_set, min_df=1, vocabulary=ref_words)
            total_dc = DocCounts(ref_words, ref_dtm, None)
            target_dc = DocCounts(ref_words, ref_dtm, self.target_docnames)
            ref_T = []
            ref_S = []
            for word in ref_words:
                ref_T.append(target_dc[word])
                ref_S.append(total_dc[word])
            self.reference_stats['T'] = ref_T
            self.reference_stats['S'] = ref_S
            self.reference_stats['R'] = self.reference_stats['counts']

        self.target_stats = target_stats
        self.nontarget_stats = nontarget_stats
        self.target_keywords = list(target_stats.index)
        self.nontarget_keywords = list(nontarget_stats.index)
        print("{} target set keywords found".format(len(self.target_keywords)))
        print("{} non-target set keywords found".format(len(self.nontarget_keywords)))


    def PrintKeywords(self, n=100, doc_set='all', starts_with=None, complete_stem=False, stats=False, filename=None):
        """Display keyword (or stems) or print keywords to file."""
        if doc_set == 'target':
            kw = self.target_keywords
            kw_stats = self.target_stats
            docnames = self.target_docnames
        elif doc_set == 'nontarget':
            kw = self.nontarget_keywords
            kw_stats = self.nontarget_stats
            docnames = self.nontarget_docnames
        elif doc_set == 'reference':
            kw = self.reference_keywords
            kw_stats = self.reference_stats
            docnames = list(self.reference_set.id)
            if complete_stem:
                print("Note that stem completions come from search set, so some stems may not be completed")

        if doc_set == 'all':
            print('{:<30}{:<30}{:<}'.format('   Reference', 'Target', 'Non-target'))
            print('{:<30}{:<30}{:<}'.format('   ' + '-'*10,'-'*10,'-'*10))
            for i in range(n):
                if len(self.reference_keywords) <= i:
                    a = str(i+1) + '. ' + ''
                else:
                    a = str(i+1) + '. ' + self.reference_keywords[i]

                if len(self.target_keywords) <= i:
                    b = ''
                else:
                    b = self.target_keywords[i]

                if len(self.nontarget_keywords) <= i:
                    c = ''
                else:
                    c = self.nontarget_keywords[i]
                print('{:<30}{:<30}{:<}'.format(a,b,c))
            return


        if starts_with != None:
            starts_with = list(starts_with)
            kw_keep = []
            for word in kw:
                word_boolean = []
                for i in starts_with:
                    word_boolean.append(word.startswith(i))
                if True in word_boolean:
                    kw_keep.append(word)
            kw = kw_keep
            kw_stats = kw_stats.loc[kw,:]

        kw = kw[0:n]
        kw_stats = kw_stats[0:n]

        if complete_stem and len(self.stem_map) > 0:
            new_kw = []
            stem_map = self.stem_map
            for word in kw:
                if word in stem_map.keys():
                    comp_words = stem_map[word].split(' ')
                    endings = []
                    for cw in comp_words:
                        ending = []
                        for l in range(0,len(cw)):
                            if l <= (len(word)-1):
                                if cw[l] != word[l]:
                                    ending.append(cw[l])
                            else:
                                ending.append(cw[l])
                        endings.append('-' + ''.join(ending))
                    new_kw.append(word + '(' + ', '.join(endings) + ')')
                else:
                    new_kw.append(word)
        else:
            new_kw = kw

        if stats:
            if doc_set == 'reference':
                new_kw = [new_kw[i] + ' (' + str(list(kw_stats.counts)[i]) + ')' for i in range(len(new_kw))]
            else:
                n1 = kw_stats['n1']
                n0 = kw_stats['n0']
                p1 = kw_stats['p1']
                p0 = kw_stats['p0']
                ref_rank = []
                for w in kw:
                    if w in self.reference_stats.index:
                        ref_rank.append(str(self.reference_stats.loc[w, 'ranks']))
                    else:
                        ref_rank.append('not in ref')
                #new_kw = [new_kw[i] + "     (ref rank: %s -- n1/n0 -- %s/%s -- p1/p0 %s/%s)" % (ref_rank[i], n1[i], n0[i], round(p1[i], 2), round(p0[i], 2)) for i in range(len(new_kw))]
                new_kw = [new_kw[i] + '    Recall: {} -- Precision: {}'.format(round(p1[i],2), round((n1[i]/(n1[i]+n0[i]))*100, 2)) for i in range(len(new_kw))]

        new_kw = [str(i+1) + '. ' + word for i,word in enumerate(new_kw)]

        for word in new_kw:
            print(word)

        if filename != None:
            newfile = open(filename, 'w')
            if doc_set != 'reference':
                newfile.write("{} set size: {}; Entire search corpus size: {}\n\n".format(doc_set.capitalize(), len(docnames), len(self.search_set)))
            else:
                newfile.write("Reference set size: {}\n\n".format(len(docnames)))
            save_words = [word + '\n' for word in new_kw]
            newfile.write(''.join(save_words))
            newfile.close()
            print("\n{} file written".format(filename))


    def ViewDocs(self, doc_set='target', nprint=10, start=1, all_words=None, any_words=None, none_words=None, use_processed=True, csv_filename=None):
        """View samples from document sets by keywords."""
        if doc_set == 'target':
            sorted_searchnames = list(self.target_votecount.sort_values(ascending=False, inplace=False).index)
            docnames = [name for name in sorted_searchnames if name in self.target_docnames]
        elif doc_set == 'nontarget':
            sorted_searchnames = list(self.target_votecount.sort_values(ascending=True, inplace=False).index)
            docnames = [name for name in sorted_searchnames if name in self.nontarget_docnames]
        elif doc_set == 'reference':
            docnames = list(self.reference_set.id)
        elif doc_set == 'search':
            docnames = list(self.search_set.id)
        else:
            print("Invalid document set")
            return

        if doc_set == 'reference':
            docs = self.reference_set
        else:
            docs = self.search_set

        doc_index = [i for i,j in enumerate(docs.id) if j in docnames]
        docs = docs.iloc[doc_index]

        if all_words != None or any_words != None or none_words != None:
            if use_processed:
                doc_text = list(docs.text_processed)
            else:
                doc_text = list(docs.text)

            docs_index = BooleanSearch(doc_text, all_words, any_words, none_words)
            docs = docs.iloc[docs_index]

        if len(docs) == 0:
            print("No documents match criteria")
            self.viewed_docs = []
            return

        if csv_filename != None:
            docs.iloc[0:nprint].to_csv(csv_filename, index=False)
            print(csv_filename + ' written to file.')
            return docs

        if start > len(docs):
            print("Start index greater than the number of documents. Starting from document 1 instead.")
            start = 1

        if start == 0:
            start = 1

        if len(docs) < nprint:
            nprint = len(docs)

        index_range = list(range(start-1, min([start-1+nprint, len(docs)])))

        for i in index_range:
            print(str(list(docs.id)[i]) + ': ' + list(docs.text)[i])

        self.viewed_docs = list(docs.id)

        if all_words != None or any_words != None or none_words != None:
            print('\n' + str(len(docs)) + ' documents match keyword critera in the ' + doc_set + ' set.')
        else:
            print('\n' + str(len(docs)) + ' documents in the ' + doc_set + ' set.')
        #return  [list(docs.text)[i] for i in index_range]
