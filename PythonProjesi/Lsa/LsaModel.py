import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    news_df = pd.DataFrame({strDoc:dokuman})
    #Tweet için needlessTW , haber için verbsTR
    stop_verbs = dosyaOkuma("needlessTW.txt")
    # removing everything except alphabets`
    news_df[strLast] = news_df[strDoc].str.replace("[^a-zA-ZŞşÇçÖöİÜüı#]", " ")
    # removing short words
    news_df[strLast] = news_df[strLast].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))
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
    detokenized_doc = []
    for i in range(len(news_df)):
        t = ' '.join(tokenized_doc[i])
        detokenized_doc.append(t)

    return detokenized_doc

def createLsa(detoken, topic_num):
    import time
    start = time. time()

    #Tweet için DuyguT5C , haber için HaberlerLsa7CTest
    test_set = dosyaOkuma("DuyguT3C.txt")
    test_detoken = preProcessing(test_set, 'test_document', 'test_docLast', False)

    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(# max_features= 10000,
                                max_df = 0.5, 
                                smooth_idf=True)

    X = vectorizer.fit_transform(detoken)
    X.shape # check shape of the document-term matrix

    from sklearn.decomposition import TruncatedSVD
    # SVD represent documents and terms in vectors 
    svd_model = TruncatedSVD(n_components=topic_num, algorithm='randomized', n_iter=100) #, random_state=122
    svd_model.fit(X)
    len(svd_model.components_)

    from sklearn import preprocessing
    topic_list = []
    agirlik_list = []
    terms = vectorizer.get_feature_names()
    for i, comp in enumerate(svd_model.components_):
        terms_comp = zip(terms, comp)
        sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:100]
        print("Topic "+str(i)+": ")
        topic = []
        agirliktop = 0.0
        for t in sorted_terms:
            if float(t[1]) >= 0.0:
                print("K:" + str(t[0]) + " ,A:" + str(t[1]))
                topic.append(str(t[0]) + "," + str(t[1]))
                agirliktop += float(t[1])
            else:
                break
        topic_list.append(topic)
        agirlik_list.append(str(agirliktop/len(topic)))
    
    end = time. time()
    print(end - start)
    testMod(test_detoken, topic_list)


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
    ekonomi = [2,7,8,10,11,12,13]#anger
    magazin = [0,4,9]#fear
    siyaset = [1,3,5,6,14]#joy
    spor = [6,13,16,18,19]#sadness
    yasam = [1,2,4,7,11]#surprise

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

        if sayi == 160:
            sayi = 0
            topic += 1
            topic_true.append(dogruluk)
        sayi += 1

    print("Konular bulundu...")
    print("Doğruluk: %" + str(dogruluk/480*100))
    print("Ekonomi : %" + str(topic_true[0]/160*100))
    print("Magazin : %" + str((topic_true[1]-topic_true[0])/160*100))
    print("Siyaset : %" + str((topic_true[2]-topic_true[1])/160*100))
    print("Spor : %" + str((topic_true[3]-topic_true[2])/160*100))
    print("Yaşam : %" + str((topic_true[4]-topic_true[3])/160*100))


pd.set_option("display.max_colwidth", 200)
#Tweet için Duygu5C , haber için HaberlerLsa5C
doc_set = dosyaOkuma("Duygu3C.txt")
detoken = preProcessing(doc_set, 'document', 'docLast', False)
createLsa(detoken,20)

#2-stage
indx = 0
stageWord_list = []
for konular in topic_list:
    for info in konular:
        bilgi = str(info).strip().split(',')
        if float(bilgi[1]) >= float(agirlik_list[indx]):
            stageWord_list.append(str(bilgi[0]))
    indx += 1
stageWord_list = list(set(stageWord_list))

detoken = preProcessing(doc_set, 'document2', 'docLast2', True)
createLsa(detoken,12)
"""
import umap
X_topics = svd_model.fit_transform(X)
embedding = umap.UMAP(n_neighbors=150, min_dist=0.5, random_state=12).fit_transform(X_topics)

plt.figure(figsize=(7,5))
plt.scatter(embedding[:, 0], embedding[:, 1], 
s = 20, # size
edgecolor='none'
)
plt.show()
"""


            
        



