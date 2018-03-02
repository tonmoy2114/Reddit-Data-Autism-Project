#TODO Read https://www.analyticsvidhya.com/blog/2017/01/ultimate-guide-to-understand-implement-natural-language-processing-codes-in-python/
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.corpora import Dictionary
from stop_words import get_stop_words
from gensim.models import Phrases
from pprint import pprint
from gensim.models import LdaModel
from gensim.models import LdaMulticore
from gensim.models import HdpModel
import gensim
import copy
import csv
from datetime import datetime
import pyLDAvis.gensim as gensimvis
import pyLDAvis
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

writeFileTokens = open('output/post_comment_tokens.csv', 'w+')
writeFileStopWordRemoved = open('output/post_comment_after_stopwords.csv', 'w+')

tokenizer = RegexpTokenizer(r'\w+')

docs = []

remove_word_list = ['http','https', 'www','ve','er','nt', 'can', 're', 'com']

# create English stop words list
en_stop = get_stop_words('en')

print("Program Started at: " + str(datetime.now()))

with open('postCommentCombined.csv') as csvLayerFile:
    csvReader = csv.reader(csvLayerFile)
    i = 0
    for row in csvReader:
        mStr = ""
        for item in row:
            mStr+=item +"\n"
        docs.append(mStr)

for idx in range(len(docs)):
    docs[idx] = docs[idx].lower()  # Convert to lowercase.
    docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.
    #remove word mentioned in remove_word_list
    docs[idx] = [x for x in docs[idx] if x not in remove_word_list]

    writeFileTokens.write(str(docs[idx])[1:-1])
    writeFileTokens.write("\n")
    #print(docs[idx])
    # remove stop words from tokens
    docs[idx] = [i for i in docs[idx] if not i in en_stop]
    writeFileStopWordRemoved.write(str(docs[idx])[1:-1])
    writeFileStopWordRemoved.write("\n")

writeFileTokens.close()
writeFileStopWordRemoved.close()

# Remove numbers, but not words that contain numbers.
docs = [[token for token in doc if not token.isnumeric()] for doc in docs]
# Remove words that are only one character.
docs = [[token for token in doc if len(token) > 1] for doc in docs]

mfile = open('output/post_after_numbers_1charremoved.csv', 'w+')
for doc in docs:
    mfile.write(str(doc)[1:-1])
    mfile.write("\n")



# Lemmatize the documents.


# Lemmatize all words in documents.
lemmatizer = WordNetLemmatizer()
docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]

mfile = open('output/post_comment_after_lemmatization.csv', 'w+')
for doc in docs:
    mfile.write(str(doc)[1:-1])
    mfile.write("\n")
# Compute bigrams.

# Add bigrams and trigrams to docs (only ones that appear 15 times or more).
bigram = Phrases(docs, min_count=15)
for idx in range(len(docs)):
    for token in bigram[docs[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            docs[idx].append(token)

# Remove rare and common tokens.
mfile = open('output/post_comment_after_bigram.csv', 'w+')
for doc in docs:
    mfile.write(str(doc)[1:-1])
    mfile.write("\n")


# Create a dictionary representation of the documents.
dictionary = Dictionary(docs)

# Filter out words that occur less than 20 documents, or more than 50% of the documents.
dictionary.filter_extremes(no_below=20, no_above=0.6)


# Vectorize data.

# Bag-of-words representation of the documents.
corpus = [dictionary.doc2bow(doc) for doc in docs]


print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(corpus))


# Train LDA model.

# Set training parameters.
num_topics = 15
chunksize = 20000
passes = 20
iterations = 400
eval_every = None  # Don't evaluate model perplexity, takes too much time.

# Make a index to word dictionary.
temp = dictionary[0]  # This is only to "load" the dictionary.
id2word = dictionary.id2token


model = LdaMulticore(corpus=corpus, id2word=id2word, chunksize=chunksize, \
                       alpha='auto', eta='auto', \
                       iterations=iterations, num_topics=num_topics, \
                       passes=passes, eval_every=eval_every, workers=3)


#model = LdaModel(corpus, num_topics=num_topics, id2word = id2word, passes=passes)
model.save('reddit_autism_pass20_topic15_iter400_lda.model')

#hdp = HdpModel(corpus, dictionary)
#hdp.save('reddit_autism_hdp.model')

print(model.print_topics(num_topics=num_topics, num_words=12))
print("Program Ended at: " + str(datetime.now()))
