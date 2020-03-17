from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim

tokenizer = RegexpTokenizer(r'\w+')
tr_stop = get_stop_words('tr')
p_stemmer = PorterStemmer()

dosya = open("deneme_verisetiDuygu.txt")
satirlar = dosya.readlines()
dosya.close()
doc_set = []
for satir in satirlar:
	doc_set.append(satir)

texts = []
for i in doc_set:
	raw = i.lower()
	tokens = tokenizer.tokenize(raw)
	stopped_tokens = [i for i in tokens if not i in tr_stop]
	stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
	texts.append(stemmed_tokens)

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=4, id2word = dictionary, passes=20)

print("Topicler beraber")
print(ldamodel.print_topics(num_topics = 4))
print("1. topic ilk 10 kelime")
print(ldamodel.print_topic(0, topn = 10))
print("2.topic ilk 10 kelime")
print(ldamodel.print_topic(1, topn = 10))

print("Cümleye göre topik değerleri")
i = 1
for d in doc_set:
	bow = dictionary.doc2bow(d.split())
	t = ldamodel.get_document_topics(bow)
	print(d)
	print(t)
	max_value = t[0][1]
	max_index = t[0][0]
	for index in range(len(t)):
		if t[index][1] > max_value:
			max_value = t[index][1]
			max_index = t[index][0]

	if max_index == 0:
		print(str(i) + ". karmaşık")
	elif max_index == 1:
		print(str(i) + ". neşeli")
	elif max_index == 2:
		print(str(i) + ". sinirli")
	else:
		print(str(i) + ". üzgün")
	i = i+1

print("Sonuca ulaşıldı...")
