from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim

tokenizer = RegexpTokenizer(r'\w+')
tr_stop = get_stop_words('tr')
p_stemmer = PorterStemmer()
    
# create sample documents
doc_a = "Bu Turkcell ile  aramız her ne kadar iyi olmasa da böyle hediyeler verdikce gözüme giriyorlar."
doc_b = "Cem Yılmaz'ın oynadığı turkcell reklamı ne öyle ya gülme krizindeyim süper yapmışlar."
doc_c = "Bir tim calışanı bana kahve ısmarlıyor :)) tesekkurler."
doc_d = "Tüketici haklarına başvur tabi sözleşmen senden yanaysa..."
doc_e = "Sadece bir dakikam kaldı diye benimle rastgele gülerek dalga geçiyor. Sanki senin 1 dakikana kaldık. :)" 
doc_f = "Hayat paylaşınca güzel canım benim :))" 
doc_g = "Benim bulunduğum şehirde yok ee yani şansımı kayıp etmiş oluyorum:((" 
doc_h = "Lan bu Turkcell niye bu kadar kazık!" 
doc_i = "Dogru bilgiyi alabilmek icin ne yapmam lazim illa kavga illa siddet of turkcell of !!!!!" 

# compile sample documents into a list
doc_set = [doc_a, doc_b, doc_c, doc_d, doc_e, doc_f, doc_g, doc_h, doc_i]

# list for tokenized documents in loop
texts = []
for i in doc_set:
    
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in tr_stop]
    
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    
    # add tokens to list
    texts.append(stemmed_tokens)

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)
    
# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=2, id2word = dictionary, passes=20)

print(ldamodel.print_topics(num_topics=2))
