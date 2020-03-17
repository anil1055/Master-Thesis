import numpy
import nltk
import string
from stop_words import get_stop_words
import pandas as pd

#Dosyanın okunması
def dosyaOkuma(fileName):
    dosya = open(fileName)
    satirlar = dosya.readlines()
    dosya.close()

    dosya_ = []
    for satir in satirlar:
        dosya_.append(str(satir).rstrip("\n"))

    return dosya_

def testProcessing(dokuman, strDoc, strLast, strBool):
    news_df = pd.DataFrame({strDoc:dokuman})
    #Tweet için needlessTW , haber için verbsTR
    stop_verbs = dosyaOkuma("needlessTW.txt")
    # removing everything except alphabets`
    news_df[strLast] = news_df[strDoc].str.replace("[^a-zA-ZŞşÇçÖöİÜüı#]", " ")
    news_df[strLast] = news_df[strLast].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))
    news_df[strLast] = news_df[strLast].apply(lambda x: x.lower())

    from stop_words import get_stop_words
    stop_words = get_stop_words('turkish')
    # tokenization
    tokenized_doc = news_df[strLast].apply(lambda x: x.split())
    tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])
    tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_verbs])
    # de-tokenization
    detokenized_doc = []
    for i in range(len(news_df)):
        t = ' '.join(tokenized_doc[i])
        detokenized_doc.append(t)
    return detokenized_doc

def testMod():
    topic_list = []
    topics = dosyaOkuma("pLSAtopics5C_.txt")
    for topic in topics:
        topic = str(topic).strip().split(' ')
        topic_list.append(topic)
    #Tweet için DuyguT5C , haber için HaberlerLsa5CTest
    test_set = dosyaOkuma("DuyguT5C.txt")
    test_detoken = testProcessing(test_set, 'dokuman', 'docLast', False)        
    topics = []
    for cumle in test_detoken:
        cumle = str(cumle).strip().split(' ')
        weights = []
        for konular in topic_list:
            agirlik = 0.0
            for kelime in cumle:
                for info in konular:
                    bilgi = str(info).strip().split('-')
                    if kelime == str(bilgi[0]):
                        agirlik += float(bilgi[1])
                        break
            weights.append(agirlik)
        konu = weights.index(max(weights))
        topics.append(konu)
    print(topics)

    #Haber konuları
    ekonomi = [11,12]#anger
    magazin = [4,8,14,17]#fear
    siyaset = [1,2,5,9,10,13]#joy
    spor = [0,3,6,15]#sadness
    yasam = [7,16,18,19]#surprise
    tekno = [2,10,14,15]#sadness
    saglik = [1,4,8,12]#surprise
    sayi = 1
    dogruluk = 0
    topic = 0
    topic_true = []
    for doc in topics:
        if topic == 0:
            if (doc in ekonomi) == True:
                dogruluk += 1   
        if topic == 1:
            if (doc in magazin) == True:
                dogruluk += 1
        if topic == 2:
            if (doc in siyaset) == True:
                dogruluk += 1   
        if topic == 3:
            if (doc in spor) == True:
                dogruluk += 1
        if topic == 4:
            if (doc in yasam) == True:
                dogruluk += 1
        if topic == 5:
            if (doc in tekno) == True:
                dogruluk += 1
        if topic == 6:
            if (doc in saglik) == True:
                dogruluk += 1

        if sayi == 160:#Haber için 120
            sayi = 0
            topic += 1
            topic_true.append(dogruluk)
        sayi += 1

    print("Konular bulundu...")
    print("Doğruluk: %" + str(dogruluk/800*100))
    print("Ekonomi : %" + str(topic_true[0]/160*100))
    print("Magazin : %" + str((topic_true[1]-topic_true[0])/160*100))
    print("Siyaset : %" + str((topic_true[2]-topic_true[1])/160*100))
    print("Spor : %" + str((topic_true[3]-topic_true[2])/160*100))
    print("Yaşam : %" + str((topic_true[4]-topic_true[3])/160*100))    
    print("Teknoloji : %" + str((topic_true[5]-topic_true[4])/120*100))
    print("Sağlık : %" + str((topic_true[6]-topic_true[5])/120*100))

