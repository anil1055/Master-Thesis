import pandas as pd;
import numpy as np;
import scipy as sp;
import sklearn;
import sys;
from nltk.corpus import stopwords;
import nltk;
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer;
from sklearn.decomposition import NMF;
from sklearn.preprocessing import normalize;
import pickle;

token = []
#Dosyanın okunması
def dosyaOkuma(fileName):
    dosya = open(fileName)
    satirlar = dosya.readlines()
    dosya.close()

    dosya_ = []
    for satir in satirlar:
        dosya_.append(str(satir).rstrip("\n"))

    return dosya_

def preProcessing(dokuman, strDoc, strLast, strBool):
    global token
    news_df = pd.DataFrame({strDoc:dokuman})
    #Tweet için needlessTW , haber için verbsTR
    stop_verbs = dosyaOkuma("needlessTW.txt")
    # removing everything except alphabets`
    news_df[strLast] = news_df[strDoc].str.replace("[^a-zA-ZŞşÇçğÖöİÜüı#]", " ")
    # removing short words
    news_df[strLast] = news_df[strLast].apply(lambda x: ' '.join([w for w in x.split() if len(w)>1]))
    # make all text lowercase
    news_df[strLast] = news_df[strLast].apply(lambda x: x.lower())

    from stop_words import get_stop_words
    stop_words = get_stop_words('turkish')
    # tokenization
    tokenized_doc = news_df[strLast].apply(lambda x: x.split())
    # remove stop-words
    tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])
    tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_verbs])
    if strBool == True:
        tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item in stageWord_list])
    # de-tokenization
    token = tokenized_doc
    detokenized_doc = []
    for i in range(len(news_df)):
        t = ' '.join(tokenized_doc[i])
        detokenized_doc.append(t)

    return detokenized_doc

pd.set_option("display.max_colwidth", 200)
#Tweet için Duygu5C , haber için HaberlerLsa5C
doc_set = dosyaOkuma("Duygu5CAll.txt")
detoken = preProcessing(doc_set, 'document', 'docLast', False)


vectorizer = CountVectorizer(analyzer='word', max_features=10000);
x_counts = vectorizer.fit_transform(detoken);

# we set a TfIdf Transformer, and transform the counts with the model.
transformer = TfidfTransformer(smooth_idf=False);
x_tfidf = transformer.fit_transform(x_counts);

# And now we normalize the TfIdf values to unit length for each row.
xtfidf_norm = normalize(x_tfidf, norm='l1', axis=1)

# And finally, obtain a NMF model, and fit it with the sentences.
num_topics = 15
model = NMF(n_components=num_topics, init='nndsvd');

#fit the model
model.fit(xtfidf_norm)

def get_nmf_topics(model, n_top_words):
    
    #the word ids obtained need to be reverse-mapped to the words so we can print the topic names.
    feat_names = vectorizer.get_feature_names()
    
    word_dict = []
    word_weight = []
    topics = []
    for i in range(num_topics):       
        words_ids = model.components_[i].argsort()[:-n_top_words - 1:-1]
        words = [feat_names[key] for key in words_ids]
        word_dict.append(words)
        weight = [model.components_[i][key] for key in words_ids]
        word_weight.append(weight)
        topic = []
        for j in range(n_top_words):
            topic.append(word_dict[i][j] + ',' + str(word_weight[i][j]))
            #print('Topic' + str(i) + ': ' + word_dict[i][j] + ' -Weights: ' + str(word_weight[i][j]))
        topics.append(topic)

    return topics;


def testMod(test_detoken, topic_list):
    topics = []
    for cumle in test_detoken:
        cumle = str(cumle).strip().split(' ')
        weights = []
        for konular in topic_list:
            agirlik = 0.0
            for kelime in cumle:
                for info in konular:
                    bilgi = str(info).strip().split(',')
                    if kelime == str(bilgi[0]):
                        agirlik += float(bilgi[1])
                        break
            weights.append(agirlik)
        konu = weights.index(max(weights))
        topics.append(konu)
    print(topics)

    #Haber konuları
    ekonomi = [0,1,3,4,6,7,8,9]#anger
    magazin = [2]#fear
    siyaset = [5,10,11]#joy
    spor = [2]#sadness
    yasam = [6,13,16,17]#surprise
    tekno = [0,1,2,3,6,9,11,14,19]#surprise
    saglik = [10]#surprise

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

        if sayi == 120:
            sayi = 0
            topic += 1
            topic_true.append(dogruluk)
        sayi += 1

    print("Konular bulundu...")
    print("Doğruluk: %" + str(dogruluk/360*100))
    print("Ekonomi : %" + str(topic_true[0]/120*100))
    print("Magazin : %" + str((topic_true[1]-topic_true[0])/120*100))
    print("Siyaset : %" + str((topic_true[2]-topic_true[1])/120*100))
    print("Spor : %" + str((topic_true[3]-topic_true[2])/120*100))
    print("Yaşam : %" + str((topic_true[4]-topic_true[3])/120*100))


def txtFile(tokenFile, topic_list, topic_num):
    weights = []
    for words in tokenFile:
        toplam=[]
        for j in range(topic_num):
            toplam.append(0)
        for word in words: 
            i = 0
            for row in topic_list:                
                for kelime in row:
                    dizi = kelime = str(kelime).strip().split(',')
                    if dizi[0] == word:
                        toplam[i] += float(dizi[1])
                i += 1
        weights.append(toplam)
    
    haber = ["kizgin","korku","mutlu","uzgun","saskin"]
    sayi = 0
    toplam = 0
    thefile = open('TweetNMF5.txt', 'w')
    thefile.write("%s\n" % weights)
    for weight in weights:
        cumle = ""
        for deger in weight:
            cumle += str(deger) + ","
        cumle += haber[sayi]
        thefile.write("%s\n" % cumle)      
        toplam += 1
        if toplam == 800: 
            sayi += 1
            toplam = 0
   
    print("Txt işlemi tamamlandı...")


topic_list = get_nmf_topics(model, 50)
txtFile(token, topic_list, 20)
test_set = dosyaOkuma("DuyguT3C.txt")
test_detoken = preProcessing(test_set, 'test_document', 'test_docLast', False)
testMod(test_detoken, topic_list)
