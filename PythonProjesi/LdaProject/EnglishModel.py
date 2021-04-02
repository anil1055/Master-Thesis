#!/usr/bin/python
import string
from collections import OrderedDict
import mIslemler
from stop_words import get_stop_words
import gensim
from gensim import corpora, models
from nltk.stem.porter import PorterStemmer

"""Coherence için topic oluşturma
doc_set = mIslemler.dosyaOkuma('uci-news5C.txt')
mIslemler.model_name = 'Uci5-LDA'
dictionary, corpus = mIslemler.englishSentence(doc_set)
for topic in range(10,55,5):
    mIslemler.modelOlusturmaTopic(topic, corpus, dictionary, False)
"""

topic_num = 40
doc_set = mIslemler.dosyaOkuma("uci-news5C.txt")
mIslemler.model_name = "Uci5-LDA"
dictionary, corpus = mIslemler.englishSentence(doc_set)
#mIslemler.modelOlusturmaTopic(topic_num, corpus, dictionary, False)
dictionary = gensim.corpora.Dictionary.load('Uci5-LDA.dict')
corpus = gensim.corpora.MmCorpus('Uci5-LDA.mm')
mIslemler.modelOlusturmaTopic(topic_num, corpus, dictionary, True, False, "Uci5-2LDA")
dictionary = gensim.corpora.Dictionary.load('Uci5-2LDA.dict')
corpus = gensim.corpora.MmCorpus('Uci5-2LDA.mm')
mIslemler.modelOlusturmaTopic(topic_num, corpus, dictionary, True, False, "Uci5-3LDA")

"""Coherence sıralama
trained_models = OrderedDict()
for topic_num in range(10,55,5):
    trained_models[topic_num] = gensim.models.LdaModel.load(str(topic_num) + '.model')
#Coherence işlemi
if __name__ == '__main__': 
    mIslemler.coherence(trained_models, dictionary)
"""

"""
mIslemler.modelOlusturmaTopic(32, corpus, dictionary, True)
dictionary = gensim.corpora.Dictionary.load('uci-news4C5Stage2.dict')
corpus = gensim.corpora.MmCorpus('uci-news4C5Stage2.mm')
mIslemler.modelOlusturmaTopic(12, corpus, dictionary)


dict = gensim.corpora.Dictionary.load('uci-news.dict')
corpus = gensim.corpora.MmCorpus('uci-news.mm')
lda = gensim.models.LdaModel.load('uci-news.model')

agirliklar = lda.show_topics(num_topics=16, num_words=600)
topics = str(agirliklar).split('\'), (')
topics[0] = str(topics[0])[2:]

words = []
weights = []
totalWords = []
for konular in topics:
    i = str(konular).find(',')
    konu = konular[:i]
    kelimeler = str(str(konular[i+3:]).strip()).split('+')
    toplam = 0
    for kelime in kelimeler:
        kelime = str(kelime).strip()
        i = str(kelime).find('*')
        agirlik = kelime[:i]
        k = str(kelime).rfind('"')
        kelime = kelime[i+2:k]
        if agirlik != "0.000":
            toplam += float(agirlik)
            weights.append(agirlik)
            words.append(kelime)
        else:
            break

    tWords = words
    ort_agirlik = toplam/(len(weights))
    wordCount = 0
    for weight in weights:
        if float(weight) >= ort_agirlik:
            totalWords.append(tWords[wordCount])
            wordCount += 1
        else:
            break
    
    words.clear()
    weights.clear()

totalWords = list(set(totalWords))
print("Kelimeleri indirgeme işlemi yapıldı...")

texts = mIslemler.texts
iterative_texts = []
for text in texts:
    tokens = [i for i in text if i in totalWords]
    iterative_texts.append(tokens)

print("Kelime haznesi oluşturuldu...")

mIslemler.model_name = "uci-newsIte"
dictionary = corpora.Dictionary(iterative_texts)
dictionary.save( mIslemler.model_name + '.dict')
corpus = [dictionary.doc2bow(text) for text in iterative_texts]
gensim.corpora.MmCorpus.serialize(mIslemler.model_name + '.mm', corpus)
print("Iterative Corpus-Dictionary oluşturuldu...")

ldamodel = mIslemler.modelOlusturmaTopic(16, corpus, dictionary)
"""