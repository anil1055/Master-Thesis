from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
#import plotly.plotly as py
#import plotly.graph_objs as go
import numpy
import pyLDAvis.gensim
import warnings
import string
warnings.filterwarnings('ignore')

tokenizer = RegexpTokenizer('\w+')
en_stop = get_stop_words('en')
p_stemmer = PorterStemmer()

dosya = open("AffectiveEng250.txt")
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
    
dictionary = corpora.Dictionary(texts)
dictionary.save('dictionary.dict')
corpus = [dictionary.doc2bow(text) for text in texts]
"""
bound_set = []
perplex_set = []
topic_set = []
print("K - Perplex ---------- Bound")
for topic_num in range(10,205,10):
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=topic_num, id2word = dictionary)
    topic_set.append(topic_num)
    bound = ldamodel.bound(corpus)
    bound_set.append(bound)
    perplex = numpy.exp2(-bound / sum(cnt for document in corpus for _, cnt in document))
    perplex_set.append(perplex)
    print(str(topic_num) + "  " + str(perplex) + "   " + str(bound))

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

#gensim.corpora.MmCorpus.serialize('corpus.mm', corpus )
c = gensim.corpora.MmCorpus('corpus.mm')
d = gensim.corpora.Dictionary.load('dictionary.dict')
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=100, id2word = dictionary, iterations = 100)
ldamodel.save("topic.model")

if __name__ == '__main__':
    data = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, mds='mmds')
    pyLDAvis.save_html(data, 'lda.html')

print("Topicler beraber")
print(ldamodel.show_topics())
    
print("Cümleye göre topik değerleri")
i = 1
dogru_sayi = 0
all_topics = ldamodel.get_document_topics(corpus, per_word_topics=True)

for doc_topics, word_topics, phi_values in all_topics:
    print('Sentence topics:', doc_topics)
    print('Word topics:', word_topics)
    print('Phi values:', phi_values)
    i+=1

print("Sonuca ulaşıldı")

