#!/usr/bin/python
from stop_words import get_stop_words
from gensim import corpora, models
import gensim
import snowballstemmer
import numpy
import pyLDAvis.gensim
import string
from gensim import utils, models
from collections import OrderedDict
from gensim.models import CoherenceModel
from gensim.models import LdaModel
#import plotly.plotly as py
#import plotly.graph_objs as go
import warnings
warnings.filterwarnings('ignore')

tr_stop = get_stop_words('turkish')
tr_stemmer = snowballstemmer.stemmer('turkish')

verb = open("verbsTR.txt")
haber = open("Haberler.txt")
tur = open("Turleri.txt")
haberler = haber.readlines()
turler = tur.readlines()
verbs = verb.readlines()
haber.close()
tur.close()
verb.close()
    
doc_set = []
for satir in haberler:
    doc_set.append(str(satir).rstrip("\n"))
    
tur_set = []
for emo in turler:
    tur_set.append(emo)

tr_verb = []
for fiil in verbs:
    tr_verb.append(str(fiil).rstrip("\n"))

texts = []
number = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
for i in doc_set:
    map = i.maketrans('', '', string.punctuation)
    out = i.translate(map)
    tokens = []
    not_verb = []
    #Türkçe karakterlerin düzeltilmesi
    for kelime in str(out).split(" "):
        if str(kelime).find("İ") != -1 or str(kelime).find("Ö") != -1 or str(kelime).find("Ü") != -1 or str(kelime).find("Ç") != -1 or str(kelime).find("Ş") != -1:
            kelime = str(kelime).replace("I","ı")
            kelime = str(kelime).replace("İ","i")
            kelime = str(kelime).replace("Ö","ö")
            kelime = str(kelime).replace("Ü","ü")
            kelime = str(kelime).replace("Ç","ç")
            kelime = str(kelime).replace("Ş","ş")
        kelime = str(kelime).lower().strip()
        tokens.append(kelime)
    stemmed_tokens_verb = [ tr_stemmer.stemWord(i) for i in tokens if not i in tr_stop]
    stemmed_tokens = [i for i in stemmed_tokens_verb if not i in tr_verb]
    delete = []
    #Köklerden sayıların ve tek harflilerin silinmesi
    for kok in stemmed_tokens:
        if len(kok) <= 2:
            delete.append(kok)
        else:
            for sayi in number:
                if kok[0].find(sayi) != -1:
                    delete.append(kok)
                    break
    for sil in delete:
        stemmed_tokens.remove(sil)

    stemmed_tokens = [item if len(item) > 7 else item[0:5] for item in stemmed_tokens]
    texts.append(stemmed_tokens)
    
dictionary = corpora.Dictionary(texts)
dictionary.save('haberlerDictionary.dict')
corpus = [dictionary.doc2bow(text) for text in texts]
gensim.corpora.MmCorpus.serialize('haberlerCorpus.mm', corpus)

"""Bound-Perplex hesaplama ve LDA modeli oluşturma 
bound_set = []
perplex_set = []
topic_set = []
trained_models = OrderedDict()
print("K - Perplex ---------- Bound")
for topic_num in range(5,55,5):
    ldamodel = LdaModel(corpus, num_topics=topic_num, id2word = dictionary, passes=10, iterations = 100, alpha = 'asymmetric')
    topic_set.append(topic_num)
    bound = ldamodel.bound(corpus)
    bound_set.append(bound)
    perplex = numpy.exp2(-bound / sum(cnt for document in corpus for _, cnt in document))
    perplex_set.append(perplex)
    print(str(topic_num) + "  " + str(perplex) + "   " + str(bound))
    trained_models[topic_num] = ldamodel
"""

print("Değerler Hesaplandı...")

"""Plotly grafik çizimi
py.sign_in('anilguven1055', 'pkfV8XPr8VQIoXDqCnnG')
trace1 = go.Scatter(
    x = topic_set,
    y = bound_set,
    mode = 'bound',
    name = 'bound'
)    

trace2 = go.Scatter(
    x = topic_set,
    y = perplex_set,
    mode = 'perplex',
    name = 'perplex'
)
data = [trace1, trace2]
py.iplot(data, filename='basic-line')
"""

"""Coherence hesabı
def print_coherence_rankings(coherences):
    avg_coherence = \
        [(num_topics, avg_coherence)
         for num_topics, (_, avg_coherence) in coherences.items()]
    ranked = sorted(avg_coherence, key=lambda tup: tup[1], reverse=True)
    print("Ranked by average '%s' coherence:\n" % cm.coherence)
    for item in ranked:
        print("num_topics=%d:\t%.4f" % item)
    print("\nBest: %d" % ranked[0][0])

if __name__ == '__main__':
    cm = CoherenceModel.for_models(trained_models.values(), texts = texts ,dictionary = dictionary, coherence='c_v')
    coherence_estimates = cm.compare_models(trained_models.values())
    coherences = dict(zip(trained_models.keys(), coherence_estimates))

    print_coherence_rankings(coherences)
    #c_v = []
    #cm = CoherenceModel(model=ldamodel, texts=texts, dictionary=dictionary, coherence='c_v')
    #c_v.append(cm.get_coherence())

"""

ldamodel = LdaModel(corpus, num_topics = 50, id2word = dictionary, iterations = 100, passes = 10, alpha = 'asymmetric')
ldamodel.save("haberlerTopicIlk5.model")

#pyLDAvis grafik çizimi
if __name__ == '__main__':
    data = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, mds='mmds')
    pyLDAvis.save_html(data, 'haberlerLdaIlk5Kok.html')

print("Konular beraber")
print(ldamodel.show_topics())
print('******************** \n')
    
print("Haberlere göre konu değerleri")

#Cümlelerin topiklerinin bulunması
all_topics = ldamodel.get_document_topics(corpus, per_word_topics=True)
i = 1
for doc_topics, word_topics, phi_values in all_topics:
    print('Haber topikleri:', doc_topics)        
    print('Word topics:', word_topics)
    print('Phi values:', phi_values)
    i += 1

print("Sonuca ulaşıldı...")


