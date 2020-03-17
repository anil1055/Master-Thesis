from stop_words import get_stop_words
from gensim import corpora, models
import gensim
import string
import snowballstemmer
from gensim import utils
from collections import OrderedDict
from gensim.models import CoherenceModel
from gensim.models import LdaModel
import pyLDAvis.gensim
import numpy

tr_stop = get_stop_words('turkish')
tr_stemmer = snowballstemmer.stemmer('turkish')

dosya = open("AffectiveCevirme.txt")
duygu = open("Affective5Duygu.txt")
satirlar = dosya.readlines()
duygular = duygu.readlines()
dosya.close()
duygu.close()
    
doc_set = []
for satir in satirlar:
    doc_set.append(satir)
    
duygu_set = []
for emo in duygular:
    duygu_set.append(emo)
    
texts = []
for i in doc_set:
    map = i.maketrans('', '', string.punctuation)
    out = i.translate(map)
    tokens = []
    for kelime in str(out).split(" "):
        if str(kelime).find("İ") != -1 or str(kelime).find("Ö") != -1 or str(kelime).find("Ü") != -1 or str(kelime).find("Ç") != -1 or str(kelime).find("Ş") != -1:
            kelime = str(kelime).replace("İ","i")
            kelime = str(kelime).replace("Ö","ö")
            kelime = str(kelime).replace("Ü","ü")
            kelime = str(kelime).replace("Ç","ç")
            kelime = str(kelime).replace("Ş","ş")
        kelime = str(kelime).lower().strip()
        tokens.append(kelime)
    stemmed_tokens = [tr_stemmer.stemWord(i) for i in tokens if not i in tr_stop]
    delete = []
    for kok in stemmed_tokens:
        if len(kok) == 1:
            delete.append(kok)
    for sil in delete:
        stemmed_tokens.remove(sil)
    texts.append(stemmed_tokens)

dictionary = corpora.Dictionary(texts)
dictionary.save('TurkceRadaDictionary.dict')
corpus = [dictionary.doc2bow(text) for text in texts]
gensim.corpora.MmCorpus.serialize('TurkceRadaCorpus.mm', corpus)

"""Bound-Perplex hesaplama ve LDA modeli oluşturma
bound_set = []
perplex_set = []
topic_set = []
trained_models = OrderedDict()
print("K - Perplex ---------- Bound")
for topic_num in range(5,115,10):
    ldamodel = LdaModel(corpus, num_topics=topic_num, id2word = dictionary)
    topic_set.append(topic_num)
    bound = ldamodel.bound(corpus)
    bound_set.append(bound)
    perplex = numpy.exp2(-bound / sum(cnt for document in corpus for _, cnt in document))
    perplex_set.append(perplex)
    print(str(topic_num) + "  " + str(perplex) + "   " + str(bound))
    trained_models[topic_num] = ldamodel
"""

"""Plotly bound-perplex grafik çizimi
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

ldamodel = LdaModel(corpus, id2word = dictionary, num_topics=5)
ldamodel.save("TurkceRadaLDA.model")

#pyLDAvis grafik çizimi
if __name__ == '__main__':
    data = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, mds='mmds')
    pyLDAvis.save_html(data, 'TurkceRada.html')
    
print("Konular beraber")
print(ldamodel.show_topics())
print('******************** \n')
    
print("Cümleye göre topik değerleri")
i = 1
dogru_sayi = 0
all_topics = ldamodel.get_document_topics(corpus, per_word_topics=True)

for doc_topics, word_topics, phi_values in all_topics:
    print('Document topics:', doc_topics)
    print('Word topics:', word_topics)
    print('Phi values:', phi_values)         
    
    """Ağırlığa göre konu sıralama
    max_value = doc_topics[0][1]
    max_index = doc_topics[0][0]
         
    for index in range(len(doc_topics)):
        if doc_topics[index][1] > max_value:
            max_value = doc_topics[index][1]
            max_index = doc_topics[index][0]
    
    if max_index == 0:
        print(str(i) + " kızgın")
        if "anger" in duygu_set[i-1]:
            dogru_sayi += 1
            print("Doğru")
        else:
            print("Yanlış")
    elif max_index == 1:
        print(str(i) + " korkunç")
        if "fear" in duygu_set[i-1]:
            dogru_sayi += 1
            print("Doğru")
        else:
            print("Yanlış")
    elif max_index == 2:
        print(str(i) + " eğlenceli")
        if "joy" in duygu_set[i-1]:
            dogru_sayi += 1
            print("Doğru")
        else:
            print("Yanlış")
    elif max_index == 3:
        print(str(i) + " üzgün")
        if "sadness" in duygu_set[i-1]:
            dogru_sayi += 1
            print("Doğru")
        else:
            print("Yanlış")
    elif max_index == 4:
        print(str(i) + " sürpriz")
        if "surprise" in duygu_set[i-1]:
            dogru_sayi += 1
            print("Doğru")
        else:
            print("Yanlış")
    i = i+1
    """
    print('-------------- \n')

#print("Toplam veri " + str(len(duygu_set)))
#print("Doğruluk oranı = " + str(dogru_sayi/len(duygu_set)*100))

print("Sonuca ulaşıldı...")
