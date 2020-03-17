from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from gensim import corpora, models
import gensim
from TurkishStemmer import TurkishStemmer 

tokenizer = RegexpTokenizer('\w+')
tr_stop = get_stop_words('turkish')
tr_stemmer = TurkishStemmer()

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
    raw = i.lower()   
    tokens = tokenizer.tokenize(raw)
    stopped_tokens = [i for i in tokens if not i in tr_stop]
    stemmed_tokens = [tr_stemmer.stem(i) for i in stopped_tokens]
    texts.append(stemmed_tokens)
    
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
ldamodel = gensim.models.ldamodel.LdaModel(corpus, id2word = dictionary, num_topics=5)
    
print("Topicler beraber")
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
    print('-------------- \n')

print("Toplam veri " + str(len(duygu_set)))
print("Doğruluk oranı = " + str(dogru_sayi/len(duygu_set)*100))
print("Sonuca ulaşıldı...")

