from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
 
tokenizer = RegexpTokenizer(r'\w+')
tr_stop = get_stop_words('tr')
p_stemmer = PorterStemmer()
 
doc_a = "Çok güzel bir ortam ya bayıldım!"
doc_b = "Senden nefret ediyorum :("
doc_c = "Yaşasın, güzel bir üniversiteyi kazandım"
doc_d = "Moralim çok bozuk, tüm gün uyumak istiyorum."
doc_e = "Hadi ya, çok can sıkıcı bir durum."
doc_f = "İşte bu kadar başardım!"
doc_g = "Güzel bir hayat için neler vermezdim :("
doc_h = "Seni burada görmek ne hoş :)"
doc_set = [doc_a, doc_b, doc_c, doc_d, doc_e, doc_f, doc_g, doc_h]
texts = []
for i in doc_set:
	raw = i.lower()
	tokens = tokenizer.tokenize(raw)
	stopped_tokens = [i for i in tokens if not i in tr_stop]
	stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
	texts.append(stemmed_tokens)
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=2, id2word = dictionary, passes=20)
print(ldamodel.print_topics(num_topics=2))
