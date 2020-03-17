import gensim
from gensim import corpora, models
import mIslemler

""" Önemli Bilgiler
    1-)Zemberekten elde edilen dosya bilgileri;
        Dosya ismi: HaberlerZemberek  
        Test dökümanı: HaberlerIsimFiilTest.txt
        10 Topic için: HaberlerZemberek10T

    2-)İlk 5 harfi kök olarak kabul etme dosya bilgileri;
        Dosya ismi: HaberlerIlk5Kok
        Test dökümanı: TestHaber.txt
        10 Topic için: HaberlerIlk5Kok10T

    3-)Snowball ve 5 Kök için dosya bilgisi;
        Dosya ismi: HaberlerSnowball_5Kok
        Test dökümanı: TestHaber.txt
        10 Topic için: HaberlerSnowball_5Kok10T

    *2. ve 3. modeller için test dosyası işlemleri;
        #HaberlerSnowball_5Kok için k = 1
        #HaberlerIlk5Kok k = 0
        #HaberlerZemberek k = 2

        mIslemler.model_name = "HaberlerSnowball_5KokTest"
        tr_verb = mIslemler.dosyaOkuma("verbsTR.txt")
        dictionary, corpus = mIslemler.cumleAyristirma(test_doc, tr_verb, k, True)
        texts = mIslemler.texts
"""

dict = gensim.corpora.Dictionary.load('Duygu5CYStage3.dict')
corpus = gensim.corpora.MmCorpus('Duygu5CYStage3.mm')
lda = gensim.models.LdaModel.load('Duygu5CYStage3.model')

print(lda.show_topics(num_topics=50, num_words=15))

mIslemler.model_name = "DuyguS2Test"
test_doc = mIslemler.dosyaOkuma('DuyguWeka.txt')

"""2. ve 3. madde için geçerli
mIslemler.model_name = "HaberlerSnowball_5KokTest"
tr_verb = mIslemler.dosyaOkuma("verbsTR.txt")
mIslemler.cumleAyristirma(test_doc, 1, True)
texts = mIslemler.texts
"""

#Haber konuları
ekonomi = [13,17,19,21,24]#anger
magazin = [10,12,14,20]#fear
siyaset = [2,15,18]#joy
spor = [7,22]#sadness
yasam = [0,1,3,4,5,6,8,9,11,16,23]#surprise
saglik = [6,11,17,18]
teknoloji = [12,13,16]


#2. ve 3. madde için test_doc -> texts olmalı
#doc.lower().split() -> doc olmalı
sayi = 1
topic = 0
dogruluk = 0
topic_true = []
for doc in test_doc:
    vec_bow = dict.doc2bow(doc.lower().split())
    doc_lda = lda[vec_bow]

    max_value = doc_lda[0][1]
    max_index = doc_lda[0][0]
         
    for index in range(len(doc_lda)):
        if doc_lda[index][1] > max_value:
            max_value = doc_lda[index][1]
            max_index = doc_lda[index][0]

    print(str(max_index))

    if topic == 0:
        for konu in ekonomi:
            if int(max_index) == konu:
                dogruluk += 1
                break
    
    if topic == 1:
        for konu in magazin:
            if int(max_index) == konu:
                dogruluk += 1                
                break

    if topic == 2:
        for konu in siyaset:
            if int(max_index) == konu:
                dogruluk += 1
                break
    
    if topic == 3:
        for konu in spor:
            if int(max_index) == konu:
                dogruluk += 1             
                break

    if topic == 4:
        for konu in yasam:
            if int(max_index) == konu:
                dogruluk += 1
                break
    
    if topic == 5:
        for konu in teknoloji:
            if int(max_index) == konu:
                dogruluk += 1
                break

    if topic == 6:
        for konu in saglik:
            if int(max_index) == konu:
                dogruluk += 1
                break
    

    if sayi == 120:
        sayi = 0
        topic += 1
        topic_true.append(dogruluk)

    sayi += 1

print("Konular bulundu...")
print("Doğruluk: %" + str(dogruluk/600*100))

print("Kızgın : %" + str(topic_true[0]/120*100))
print("Korkmuş : %" + str((topic_true[1]-topic_true[0])/120*100))
print("Mutlu : %" + str((topic_true[2]-topic_true[1])/120*100))
print("Üzgün : %" + str((topic_true[3]-topic_true[2])/120*100))
print("Sürpriz : %" + str((topic_true[4]-topic_true[3])/120*100))
#print("Teknoloji : %" + str((topic_true[5]-topic_true[4])/120*100))
#print("Sağlık : %" + str((topic_true[6]-topic_true[5])/120*100))

#mIslemler.grafikCiz(lda, 20)




