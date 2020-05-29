import os
import csv
import re
import nltk
from nltk.corpus import wordnet, stopwords
from gensim import corpora
from nltk.stem.wordnet import WordNetLemmatizer


class Corpus:
    def __init__(self, name, texts=None):
        self.corpus_name = name
        self.num_docs = 0
        self.texts = texts if texts != None else []
        self.dictionary = None
        self.doc_term_mat = None
        self.num_elem = 0

    def setDocTermMatrix(self):
        print("Creating Document Term-Frequency Matrix....")
        self.dictionary = corpora.Dictionary(self.texts)
        self.doc_term_mat = [self.dictionary.doc2bow(text) for text in self.texts]
        print("Complete.")

    def updateTexts(self, text):
        self.texts.append(text)


def cleanData(doc):

    def get_wordnet_pos(treebank_tag):
        return tag_to_type.get(treebank_tag[:1], wordnet.NOUN)

    '''
    Perform stop word removal, lemmatization and other preprocessing steps.
    Return tokens.
    '''
    nonan = re.compile(r'[^-0-9a-zA-Z ]')
    stop = stopwords.words('english') + ['also', 'use', 'respective', 'respectively',
                                         'one', 'may', 'two', 'within', 'say', 'wherein', 'andor', 'thereof', 'eg', 'etc', 'along']
    tag_to_type = {'J': wordnet.ADJ, 'V': wordnet.VERB, 'R': wordnet.ADV}

    tokens = nltk.word_tokenize(nonan.sub('', doc.lower()))
    tokens = [token for token in tokens if not token in stop]
    lmtzr = WordNetLemmatizer()
    tags = nltk.pos_tag(tokens)

    finalTokens = []
    uniqueTokens = set()
    for word, tag in zip(tokens, tags):
        token = lmtzr.lemmatize(word, get_wordnet_pos(tag[1]))
        try:
            num = int(token)
            continue
        except:
            {}
        if len(token) < 2:
            continue
        if token in stop:
            continue
        if token.isspace():
            continue
        if token.strip("-").isspace():
            continue
        if len(token.strip("-")) < 2:
            continue
        finalTokens.append(token.strip('-'))
        uniqueTokens.add(token.strip('-'))

    # print "finalTokens:",finalTokens,len(finalTokens)
    # print "\nlength of uniqueTokens:",len(uniqueTokens)
    # raw_input()
    return finalTokens


def setVocab(c):
    print("setting vocab file....")
    word_id = c.dictionary.token2id
    word_id = sorted(word_id.items(), key=lambda x: x[1])
    path = 'hdp_data/preprocessed_vector/vocab'+c.corpus_name+'.txt'

    with open(path, 'w') as f:
        for pair in word_id:
            f.write(pair[0] + '\n')


def createLDA_C_File(c):
    print("Creating .dat file....")
    path = 'hdp_data/preprocessed_vector/' + c.corpus_name + '.dat'

    with open(path, 'w+') as f:
        for i in xrange(c.num_docs):
            f.write(str(len(c.doc_term_mat[i])) + ' ')
            for j in xrange(len(c.doc_term_mat[i])):
                f.write(str(c.doc_term_mat[i][j][0]) + ':' + str(c.doc_term_mat[i][j][1]) + ' ')
            f.write('\n')


def parseData(inputdir):
    for filename in os.listdir(inputdir):
        name = filename.split('.')[0]
        if ".csv" not in filename:
            continue

        print(filename)
        # raw_input()
        with open(inputdir + filename, 'r') as f:
            reader = csv.reader(f)
            rowCount = 0
            c = Corpus(name)

            for row in reader:
                rowCount += 1
                # print rowCount
                if (rowCount % 100 == 0):
                    print(rowCount)
                    # break

                doc = cleanData(row[5])
                # print doc
                c.updateTexts(doc)
                c.num_docs += 1

        c.setDocTermMatrix()
        setVocab(c)
        createLDA_C_File(c)


def main():
    parseData('hdp_data/cleaned_text/')


if __name__ == '__main__':
    main()
