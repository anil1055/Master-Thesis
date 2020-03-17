import gensim
from gensim import corpora, models
import mIslemler


dict = gensim.corpora.Dictionary.load('uci-news5CStage2.dict')
corpus = gensim.corpora.MmCorpus('uci-news5CStage2.mm')
lda = gensim.models.LdaModel.load('40.model')

mIslemler.model_name = 'UciTest'
print(lda.show_topics(num_topics=40, num_words=15))
test_doc = mIslemler.dosyaOkuma('uci-news-test.txt')
mIslemler.englishSentence(test_doc)
texts = mIslemler.texts

"""
test_doc = mIslemler.dosyaOkuma('affectivetext_test.txt')
emo_doc = mIslemler.dosyaOkuma('affectivetext_test_emo.txt')
mIslemler.model_name = 'EnglishRadaTest'
dictionary, corpus = mIslemler.englishSentence(test_doc)
texts = mIslemler.texts
"""

#topic konuları
bus = [5,8,13,16,22,29,35,36]
enter = [3,9,18,23,28,30,31,32]
medic = [4,7,11,12,14,21,24,26,33]
tech = [2,10,20,25,34,37,38]
sport = [0,1,6,15,17,19,27,39]

#2. ve 3. madde için test_doc -> texts olmalı
#doc.lower().split() -> doc olmalı
sayi = 1
topic = 0
dogruluk = 0
topic_true = []
for doc in texts:
    vec_bow = dict.doc2bow(doc)
    doc_lda = lda[vec_bow]

    max_value = doc_lda[0][1]
    max_index = doc_lda[0][0]
         
    for index in range(len(doc_lda)):
        if doc_lda[index][1] > max_value:
            max_value = doc_lda[index][1]
            max_index = doc_lda[index][0]

    print(max_index)

    if topic == 0:
        for konu in bus:
            if int(max_index) == konu:
                dogruluk += 1
                break

    if topic == 1:
        for konu in enter:
            if int(max_index) == konu:
                dogruluk += 1
                break

    if topic == 2:
        for konu in medic:
            if int(max_index) == konu:
                dogruluk += 1
                break

    if topic == 3:
        for konu in tech:
            if int(max_index) == konu:
                dogruluk += 1
                break
    
    if topic == 4:
        for konu in sport:
            if int(max_index) == konu:
                dogruluk += 1
                break

    if sayi == 200:
        sayi = 0
        topic += 1
        topic_true.append(dogruluk)

    sayi += 1

print("Konular bulundu...")
print("Doğruluk oranı: %" + str(dogruluk/1000*100))

print("Bus oranı: %" + str(topic_true[0]/200*100))
print("Enter oranı: %" + str((topic_true[1]-topic_true[0])/200*100))
print("Medic oranı: %" + str((topic_true[2]-topic_true[1])/200*100))
print("Tech oranı: %" + str((topic_true[3]-topic_true[2])/200*100))
print("Sport oranı: %" + str((topic_true[4]-topic_true[3])/200*100))





