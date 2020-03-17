from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
#import plotly.plotly as py
#import plotly.graph_objs as go
import numpy
from gensim import utils, models
from collections import OrderedDict
from gensim.models import CoherenceModel
from gensim.models import LdaModel
from gensim.corpora import TextCorpus, MmCorpus
import string
import pyLDAvis.gensim

class MyCorpus(TextCorpus):
    def getText(self):
        for doc in self.input.data:
            yield doc 

tokenizer = RegexpTokenizer('\w+')
en_stop = get_stop_words('en')
p_stemmer = PorterStemmer()

doc_set = []
with open("AffectiveEng250.txt", "r") as in_file:
    for line in in_file.readlines():
        doc_set.append(line)

duygu_set = []
with open("Affective5Duygu.txt", "r") as em_file:
    for line in em_file.readlines():
        duygu_set.append(line)
   
texts = []
for i in doc_set:
    raw = str(i).lower()
    map = raw.maketrans('', '', string.punctuation)
    out = raw.translate(map)

    tokens = tokenizer.tokenize(out)
    stemmed_tokens = [ p_stemmer.stem(i) for i in tokens if not i in en_stop]
    delete = []
    for kok in stemmed_tokens:
        if len(kok.strip()) == 1:
            delete.append(kok)
    for sil in delete:
        stemmed_tokens.remove(sil)
    texts.append(stemmed_tokens)

"""Kendi texts değişkenimizle corpus dictionary oluşturma
dictionary = corpora.Dictionary(texts)
lib = dictionary
corpus = [dictionary.doc2bow(text) for text in texts]
"""

#Türetilen class(TextCorpus) ile dictionary corpus oluşumu
mycorpus = MyCorpus('AffectiveEng250.txt')
texts = mycorpus.get_texts()
dictionary = mycorpus.dictionary
dictionary.save('Rada250Dictionary.dict')
corpus = [dictionary.doc2bow(text) for text in texts]
MmCorpus.serialize('Rada250.mm', corpus)
#c = gensim.corpora.MmCorpus('corpus.mm')

"""Bound-Perplex hesaplama ve LDA modeli oluşturma
bound_set = []
perplex_set = []
topic_set = []
trained_models = OrderedDict()
print("K - Perplex ---------- Bound")
for topic_num in range(5,110,10):
    ldamodel = LdaModel(corpus, num_topics=topic_num, id2word = dictionary, passes=10, iterations = 100
                        ,alpha='asymmetric', offset=64)
    topic_set.append(topic_num)
    bound = ldamodel.bound(corpus)
    bound_set.append(bound)
    perplex = numpy.exp2(-bound / sum(cnt for document in corpus for _, cnt in document))
    perplex_set.append(perplex)
    print(str(topic_num) + "  " + str(perplex) + "   " + str(bound))
    trained_models[topic_num] = ldamodel
"""

"""Plotly ile bound-perplex çizimi
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
py.iplot(data, filename='English dataset')
"""

print("Değerler Hesaplandı...")

"""
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
    cm = CoherenceModel.for_models(trained_models.values(), texts = mycorpus.get_texts() ,dictionary = dictionary, coherence='c_v')
    coherence_estimates = cm.compare_models(trained_models.values())
    coherences = dict(zip(trained_models.keys(), coherence_estimates))

    print_coherence_rankings(coherences)
    #c_v = []
    #cm = CoherenceModel(model=ldamodel, texts=texts, dictionary=dictionary, coherence='c_v')
    #c_v.append(cm.get_coherence())

"""
#Lda modelinin bulunması
ldamodel = LdaModel(corpus, num_topics=100, id2word = dictionary, iterations = 100)
ldamodel.save("Rada250Topic.model")

#pyLDAvis grafik çizimi
if __name__ == '__main__':
    data = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, mds='mmds')
    pyLDAvis.save_html(data, 'Rada250Lda.html')

print("Topicler beraber")
print(ldamodel.show_topics())
    
print("Cümleye göre topik değerleri")
i = 1
dogru_sayi = 0
#Cümleye göre topiclerin bulunması
all_topics = ldamodel.get_document_topics(corpus, per_word_topics=True)
for doc_topics, word_topics, phi_values in all_topics:
    print('Sentence topics:', doc_topics)

    i+=1

print("Sonuca ulaşıldı")

