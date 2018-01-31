#TODO add important code from here https://markroxor.github.io/gensim/static/notebooks/lda_training_tips.html 

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import copy
import csv

tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

# compile sample documents into a list
doc_set = []

# list for tokenized documents in loop
texts = []

with open('postdata.csv') as csvLayerFile:
    csvReader = csv.reader(csvLayerFile)
    i = 0
    for row in csvReader:
        #if i==50:
            #break
        i = i+1
        doc_set.append(row[2])

# loop through document list
for i in doc_set:

    # clean and tokenize document string
    #print i
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    #print tokens

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]
    #print (stopped_tokens)
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

    # add tokens to list
    texts.append(stemmed_tokens)

    #print(stemmed_tokens)

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)

#print(dictionary.token2id)


# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]
#print(corpus[34])
print("Started Model creation...")
# generate LDA model
#num_topics=10, id2word = dictionary, passes=200 took 30 minutes
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=10, id2word = dictionary, passes=20)
print(ldamodel.print_topics(num_topics=10, num_words=8))
