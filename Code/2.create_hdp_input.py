import pickle
from six import iteritems
from collections import Counter
import pandas as pd
import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER, CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
import os
from gensim.utils import simple_preprocess
from gensim import corpora, models
import re
import nltk
from nltk.corpus import wordnet, stopwords
import pickle
import gensim
from nltk.stem.wordnet import WordNetLemmatizer
from collections import defaultdict
from tqdm import tqdm


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
        #self.dictionary.filter_extremes(no_below=.01, no_above=0.99)
        self.doc_term_mat = [self.dictionary.doc2bow(text) for text in self.texts]
        print("Complete.")

    def updateTexts(self, text):
        self.texts.append(text)


def setVocab(c):
    print("setting vocab file....")
    word_id = c.dictionary.token2id
    word_id = sorted(word_id.items(), key=lambda x: x[1])
    path = "hdp_data_app/1_preprocessed_vector/vocab"+c.corpus_name+".txt"

    with open(path, "w") as f:
        for pair in word_id:
            f.write(pair[0] + "\n")


def createLDA_C_File(c):
    print("Creating .dat file....")
    path = "hdp_data_app/1_preprocessed_vector/" + c.corpus_name + ".dat"

    with open(path, "w+") as f:
        for i in range(c.num_docs):
            f.write(str(len(c.doc_term_mat[i])) + " ")
            for j in range(len(c.doc_term_mat[i])):
                f.write(str(c.doc_term_mat[i][j][0]) + ":" + str(c.doc_term_mat[i][j][1]) + " ")
            f.write("\n")


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True, min_len=2))
# deacc: Remove letter accents from the given string
# deaccent("Šéf chomutovských komunistů dostal poštou bílý prášek")
# u"Sef chomutovskych komunistu dostal postou bily prasek"


stop = stopwords.words("english") + ["also", "use", "respective", "respectively",
                                     "one", "may", "two", "within", "say", "wherein", "andor", "thereof", "eg", "etc", "along"]


def remove_stopwords(texts):
    return [[word for word in doc if word not in stop] for doc in tqdm(texts)]


def clean_token(texts):
    newtext = [[word.strip("-") for word in doc if (word.isspace() == False) & (
        word.strip("-").isspace() == False) & (len(word.strip("-")) > 2)] for doc in tqdm(texts)]

    newtext = [[word for word in doc if not (word.isdigit())] for doc in tqdm(newtext)]
    return newtext


# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
def custom_tokenizer(nlp):
    infixes = (
        LIST_ELLIPSES
        + LIST_ICONS
        + [
            r"(?<=[0-9])[+\-\*^](?=[0-9-])",
            r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
                al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
            ),
            r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
            #r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
            r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
        ]
    )

    infix_re = compile_infix_regex(infixes)

    return Tokenizer(nlp.vocab, prefix_search=nlp.tokenizer.prefix_search,
                     suffix_search=nlp.tokenizer.suffix_search,
                     infix_finditer=infix_re.finditer,
                     token_match=nlp.tokenizer.token_match,
                     rules=nlp.Defaults.tokenizer_exceptions)


nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
nlp.tokenizer = custom_tokenizer(nlp)


def lemmatization(texts):
    texts_out = []
    for sent in tqdm(texts):
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc])
    return texts_out


def parseData(inputdir):
    allyear = pd.DataFrame()
    for filename in sorted(os.listdir(inputdir)):
        if ".csv" not in filename:
            continue
        print(filename)
        df = pd.read_csv(inputdir+filename)
        allyear = allyear.append(df)
        # raw_input()
    c = Corpus("all_years")

    allyear["text"] = allyear["text"].str.lower()
    data = allyear["text"].values.tolist()
    data = [re.sub(r"[^-0-9a-zA-Z ]", "", sent) for sent in data]
    print("tokenize")
    data_words = list(sent_to_words(data))
    # print(data_words[:1])
    # len(data)
    print("load bigrams")
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=10)  # higher threshold fewer phrases.
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    # print(bigram_mod[data_words[0]])

    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in tqdm(texts)]

    print("drop stopwords")
    data_words_nostops = remove_stopwords(data_words)
    print("create bigrams")
    data_words_bigrams = make_bigrams(data_words_nostops)
    print("lemmatize")
    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams)

    print("clean token")
    data_words_cleaned = clean_token(data_lemmatized)

    print("update c")
    for docu in tqdm(data_words_cleaned):
        c.updateTexts(docu)
        c.num_docs += 1
    #
    c.setDocTermMatrix()

    print("create doc_id")
    id = allyear["id"]
    id.to_csv("./hdp_data_app/1_paper_ids/"+c.corpus_name+"_docID.txt", index=False)

    print("set vocab")
    setVocab(c)
    createLDA_C_File(c)


def main():
    parseData("hdp_data_app/0_cleaned_text/")


if __name__ == "__main__":
    main()
