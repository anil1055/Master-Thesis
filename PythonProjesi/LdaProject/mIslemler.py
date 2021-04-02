import numpy
import string
import gensim
from collections import OrderedDict
from gensim.models import LdaModel
from gensim.models import CoherenceModel
from gensim import corpora, models
import pyLDAvis.gensim
from stop_words import get_stop_words
import snowballstemmer

model_name = None
ldamodel = None
texts = []

#Dosyanın okunması
def dosyaOkuma(fileName):
    dosya = open(fileName)
    satirlar = dosya.readlines()
    dosya.close()

    dosya_ = []
    for satir in satirlar:
        dosya_.append(str(satir).rstrip("\n"))

    return dosya_

#Cümlenin ayrıştırılıp texts oluşturma
def cumleAyristirma(doc, secim, bool):
    #secim=0 ise ilk 5 kök alınır
    #secim = 1 ise snowball stemmer ile kok bulunup True False'a göre boyut azaltılır
    #secim= 2 ise zemberek için
    global model_name
    
    tr_stop = get_stop_words('turkish')
    tr_stemmer = snowballstemmer.stemmer('turkish')
    verb = dosyaOkuma("verbsTR.txt")
    if secim == 0:
        verb = [ item[0:5] for item in verb] 
        tr_stop = [ item[0:5] for item in tr_stop]

    global texts
    number = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    for i in doc:
        map = i.maketrans('', '', string.punctuation)
        out = i.translate(map)
        tokens = []
        not_verb = []
        #Türkçe karakterlerin düzeltilmesi
        for kelime in str(out).split(" "):
            if str(kelime).find("İ") != -1 or str(kelime).find("Ö") != -1 or str(kelime).find("Ü") != -1 or str(kelime).find("Ç") != -1 or str(kelime).find("Ş") != -1:
                kelime = str(kelime).replace("I","ı")
                kelime = str(kelime).replace("İ","i")
                kelime = str(kelime).replace("Ö","ö")
                kelime = str(kelime).replace("Ü","ü")
                kelime = str(kelime).replace("Ç","ç")
                kelime = str(kelime).replace("Ş","ş")
            kelime = str(kelime).lower().strip()
            tokens.append(kelime)
        if secim == 0 or secim == 2:
            stemmed_tokens_verb = [ i for i in tokens if not i in tr_stop]
            bool = True
        if secim == 1:            
            stemmed_tokens_verb = [ tr_stemmer.stemWord(i) for i in tokens if not i in tr_stop]       

        stemmed_tokens = [i for i in stemmed_tokens_verb if not i in verb]
        delete = []
        #Köklerden sayıların ve tek harflilerin silinmesi
        for kok in stemmed_tokens:
            if len(kok) <= 1:
                delete.append(kok)
            else:
                for sayi in number:
                    if kok[0].find(sayi) != -1:
                        delete.append(kok)
                        break
        for sil in delete:
            stemmed_tokens.remove(sil)
            
        if bool == True:
            if secim == 0:
                stemmed_tokens = [ item[0:5] for item in stemmed_tokens]                
            if secim == 1:
                stemmed_tokens = [item if len(item) > 7 else item[0:5] for item in stemmed_tokens]
            
            stemmed_tokens = [ i for i in stemmed_tokens if not i in tr_stop]
            stemmed_tokens = [ i for i in stemmed_tokens if not i in verb]
        
        texts.append(stemmed_tokens)

    print("Cümle ayrıştırılıp texts oluşturuldu...")
    dictionary = corpora.Dictionary(texts)
    dictionary.save( str(model_name) + '.dict')
    corpus = [dictionary.doc2bow(text) for text in texts]
    print("Kelime haznesi oluşturuldu...Kelime sayısı:" + str(dictionary.num_nnz))
    gensim.corpora.MmCorpus.serialize(str(model_name) + '.mm', corpus)
    print("Corpus-Dictionary oluşturuldu...")
    return dictionary, corpus


def englishSentence(doc):
    from nltk.stem.porter import PorterStemmer
    p_stemmer = PorterStemmer()
    en_stop = get_stop_words('english')

    global texts
    number = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    for i in doc:
        map = i.maketrans('', '', string.punctuation)
        out = i.translate(map)
        tokens = []
        not_verb = []
        #Türkçe karakterlerin düzeltilmesi
        for kelime in str(out).split(" "):
            kelime = str(kelime).lower().strip()
            tokens.append(kelime)
         
        stemmed = [ p_stemmer.stem(i) for i in tokens ]
        stemmed_tokens = [ i for i in stemmed if not i in en_stop]

        delete = []
        #Köklerden sayıların ve tek harflilerin silinmesi
        for kok in stemmed_tokens:
            if len(kok) <= 1:
                delete.append(kok)
            else:
                for sayi in number:
                    if kok[0].find(sayi) != -1:
                        delete.append(kok)
                        break
        for sil in delete:
            stemmed_tokens.remove(sil)
                    
        texts.append(stemmed_tokens)

    print("Cümle ayrıştırılıp texts oluşturuldu...")

    dictionary = corpora.Dictionary(texts)
    dictionary.save( str(model_name) + '.dict')
    corpus = [dictionary.doc2bow(text) for text in texts]
    gensim.corpora.MmCorpus.serialize(str(model_name) + '.mm', corpus)
    print("Corpus-Dictionary oluşturuldu...")
    print("Kelime haznesi oluşturuldu...Kelime sayısı:" + str(dictionary.num_nnz))

    return dictionary, corpus

#Bound perplex hesabı ile modelin oluşturulması
def boundPerplex(corpus, dictionary):
    bound_set = []
    perplex_set = []
    topic_set = []
    trained_models = OrderedDict()
    #print("K - Perplex ---------- Bound")

    for topic_num in range(10,55,5):
        ldamodel = LdaModel(corpus, num_topics=topic_num, id2word = dictionary, passes=10, iterations = 100, alpha = 'asymmetric')
        ldamodel.save(str(topic_num) + '.model')
        """
        topic_set.append(topic_num)
        bound = ldamodel.bound(corpus)
        bound_set.append(bound)
        perplex = numpy.exp2(-bound / sum(cnt for document in corpus for _, cnt in document))
        perplex_set.append(perplex)
        print(str(topic_num) + "  " + str(perplex) + "   " + str(bound))
        """
        trained_models[topic_num] = ldamodel

    print("Modeller oluşturuldu...")

    return trained_models

#Modeller için coherence değerlerinin hesaplanması
def coherence(models, dictionary):
    global texts
   
    cm = CoherenceModel.for_models(models.values(), texts = texts ,dictionary = dictionary, coherence='c_v')
    coherence_estimates = cm.compare_models(models.values())
    coherences = dict(zip(models.keys(), coherence_estimates))

    coherence_siralama(coherences, cm)

    print("Coherence sıralandı...")


#Coherence değerlerinin topic sayısına göre sıralanması
def coherence_siralama(coherences, cm):
    avg_coherence = \
        [(num_topics, avg_coherence)
         for num_topics, (_, avg_coherence) in coherences.items()]
    ranked = sorted(avg_coherence, key=lambda tup: tup[1], reverse=True)
    print("Ranked by average '%s' coherence:\n" % cm.coherence)
    for item in ranked:
        print("num_topics=%d:\t%.4f" % item)
    print("\nBest: %d" % ranked[0][0])

#Ldamodelin oluşumu ve döküman topici bulma
def modelOlusturmaTopic(topic_num, corpus, dictionary, iterative = False, bound = False, new_model = ''):
    global ldamodel
    global model_name
    global texts

    if iterative == False:
        import time
        start = time.time()
        ldamodel = LdaModel(corpus, num_topics = topic_num, id2word = dictionary, iterations = 100, passes = 10, alpha = 'asymmetric')
        ldamodel.save(str(model_name) + '.model')
        end = time.time()

        print('Topics')
        print(ldamodel.show_topics(num_topics=topic_num, num_words=25))
        print('****************')

        """ Cümlelerin topiklerinin bulunması
        all_topics = ldamodel.get_document_topics(corpus, per_word_topics=True)
        i = 1
        for doc_topics, word_topics, phi_values in all_topics:
            print(str(i) + '. Topic:', doc_topics)        
            #print('Word topics:', word_topics)
            #print('Phi values:', phi_values)
            i += 1
        """
        print('LDA modeli oluşturuldu...RunTime:' + str(end-start))
        arffOlustur(topic_num, model_name, "")
    else:
        #dict = gensim.corpora.Dictionary.load(str(model_name) + '.dict')
        #corpus = gensim.corpora.MmCorpus(str(model_name) + '.mm')
        lda = gensim.models.LdaModel.load(str(model_name) + '.model')

        agirliklar = lda.show_topics(num_topics=topic_num, num_words=500)
        topics = str(agirliklar).split('\'), (')
        topics[0] = str(topics[0])[2:]

        words = []
        weights = []
        totalWords = []
        for konular in topics:
            i = str(konular).find(',')
            konu = konular[:i]
            kelimeler = str(str(konular[i+3:]).strip()).split('+')
            toplam = 0
            for kelime in kelimeler:
                kelime = str(kelime).strip()
                i = str(kelime).find('*')
                agirlik = kelime[:i]
                k = str(kelime).rfind('"')
                kelime = kelime[i+2:k]
                if agirlik != "0.0001":
                    toplam += float(agirlik)
                    weights.append(agirlik)
                    words.append(kelime)
                else:
                    break

            tWords = words
            ort_agirlik = toplam/(len(weights))
            wordCount = 0
            for weight in weights:
                if float(weight) >= ort_agirlik:
                    totalWords.append(tWords[wordCount])
                    wordCount += 1
                else:
                    break
    
            words.clear()
            weights.clear()

        totalWords = list(set(totalWords))
        print("Kelimeleri indirgeme işlemi yapıldı...")

        sentences = texts
        iterative_texts = []
        for text in sentences:
            tokens = [i for i in text if i in totalWords]
            iterative_texts.append(tokens)
        
        texts = iterative_texts
        print("Kelime haznesi oluşturuldu...Kelime sayısı:" + str(len(totalWords)))

        model_name = new_model
        dictionary = corpora.Dictionary(iterative_texts)
        dictionary.save( model_name + '.dict')
        corpus = [dictionary.doc2bow(text) for text in iterative_texts]
        gensim.corpora.MmCorpus.serialize(model_name + '.mm', corpus)
        print("Corpus-Dictionary oluşturuldu...")
        ldamodel = modelOlusturmaTopic(topic_num, corpus, dictionary, False)
        if bound == True:
            boundPerplex(corpus, dictionary)     
        
    return ldamodel

def arffOlustur(topic_num, model_name, txtFile):
    global texts
    lda = gensim.models.LdaModel.load(str(model_name) + '.model')

    agirliklar = lda.show_topics(num_topics=topic_num, num_words=500)
    topics = str(agirliklar).split('\'), (')
    topics[0] = str(topics[0])[2:]

    list_topic = []
    total_words = []
    for konular in topics:
        i = str(konular).find(',')
        konu = konular[:i]
        kelimeler = str(str(konular[i+3:]).strip()).split('+')
        toplam = 0 
        rows = []
        for kelime in kelimeler:
            column = []

            kelime = str(kelime).strip()
            i = str(kelime).find('*')
            agirlik = kelime[:i]
            k = str(kelime).rfind('"')
            kelime = kelime[i+2:k]
            if str(agirlik) != "0.0001":
                column.append(kelime)
                column.append(agirlik)

                rows.append(column)
                total_words.append(kelime)
            else:
                break
        
        list_topic.append(rows)        
   
    print("Topic listesi oluşturuldu...")
    #doc_set = dosyaOkuma(txtFile)
    #model_name = "uci-news"
    #cumleAyristirma(doc_set, 2, True)
    #englishSentence(doc_set)
    
    total_words = list(set(total_words))
    sentences = texts
    iterative_texts = []
    for text in sentences:
        tokens = [i for i in text if i in total_words]
        iterative_texts.append(tokens)
     
    weights = []
    for words in iterative_texts:
        toplam=[]
        for j in range(topic_num):
            toplam.append(0)
        for word in words: 
            i = 0
            for row in list_topic:                
                for kelime, agirlik in row:
                    if kelime == word:
                        toplam[i] += float(agirlik)
                i += 1
        weights.append(toplam)
           
    duygu = ["kizgin","korku","mutlu","uzgun", "surpriz"]
    haber = ["ekonomi","magazin","siyaset","spor","yasam","teknoloji","saglik"]
    uci = ["business","entertain","medi","tech","sport"]
    sayi = 0
    toplam = 0
    thefile = open(str(model_name) + '.txt', 'w')
    #thefile.write("%s\n" % weights)
    for weight in weights:
        cumle = ""
        for deger in weight:
            cumle += str(deger) + ","
        cumle += uci[sayi]
        thefile.write("%s\n" % cumle)      
        toplam += 1
        if toplam == 1000: 
            sayi += 1
            toplam = 0
   
    print("İşlem tamamlandı...")

def csvFile(topic_num, model_name):
    global texts
    lda = gensim.models.LdaModel.load(str(model_name) + '.model')
    agirliklar = lda.show_topics(num_topics=topic_num, num_words=250)
    topics = str(agirliklar).split('\'), (')
    topics[0] = str(topics[0])[2:]

    list_topic = []
    total_words = []
    for konular in topics:
        i = str(konular).find(',')
        konu = konular[:i]
        kelimeler = str(str(konular[i+3:]).strip()).split('+')
        toplam = 0 
        rows = []
        for kelime in kelimeler:
            column = []

            kelime = str(kelime).strip()
            i = str(kelime).find('*')
            agirlik = kelime[:i]
            k = str(kelime).rfind('"')
            kelime = kelime[i+2:k]
            if agirlik != "0.000":
                column.append(kelime)
                column.append(agirlik)

                rows.append(column)
                total_words.append(kelime)
            else:
                break
        
        list_topic.append(rows)        
   
    print("Topic listesi oluşturuldu...")

    weights = []
    for words in texts:
        toplam=[]
        for j in range(topic_num):
            toplam.append(0)
        for word in words: 
            i = 0
            for row in list_topic:                
                for kelime in row:
                    dizi = str(kelime).strip().split(',')
                    kelime = dizi[0]
                    if str(kelime[2:-1]) == word:
                        agirlik = dizi[1]
                        toplam[i] += float(agirlik[2:-2])
                i += 1
        weights.append(toplam)
    
    haber = ["kizgin","korku","mutlu","uzgun","saskin"]
    sayi = 0
    toplam = 0
    thefile = open('TrHaberLDA.txt', 'w')
    thefile.write("%s\n" % weights)
    for weight in weights:
        cumle = ""
        for deger in weight:
            cumle += str(deger) + ","
        cumle += haber[sayi]
        thefile.write("%s\n" % cumle)      
        toplam += 1
        if toplam == 600: 
            sayi += 1
            toplam = 0
   
    print("Txt işlemi tamamlandı...")


def txtOlustur(dosya):
    global model_name
    
    tr_stop = get_stop_words('tr')
    global texts

    number = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    for i in dosya:
        map = i.maketrans('', '', string.punctuation)
        out = i.translate(map)
        tokens = []
        not_verb = []
        #Türkçe karakterlerin düzeltilmesi
        for kelime in str(out).split(" "):
            if str(kelime).find("İ") != -1 or str(kelime).find("Ö") != -1 or str(kelime).find("Ü") != -1 or str(kelime).find("Ç") != -1 or str(kelime).find("Ş") != -1:
                kelime = str(kelime).replace("I","ı")
                kelime = str(kelime).replace("İ","i")
                kelime = str(kelime).replace("Ö","ö")
                kelime = str(kelime).replace("Ü","ü")
                kelime = str(kelime).replace("Ç","ç")
                kelime = str(kelime).replace("Ş","ş")
            kelime = str(kelime).lower().strip()
            tokens.append(kelime)

        stemmed_tokens = [ i for i in tokens if not i in tr_stop]
        
        delete = []
        #Köklerden sayıların ve tek harflilerin silinmesi
        for kok in stemmed_tokens:
            if len(kok) <= 1:
                delete.append(kok)
            else:
                for sayi in number:
                    if kok[0].find(sayi) != -1:
                        delete.append(kok)
                        break
        for sil in delete:
            stemmed_tokens.remove(sil)
        
        texts.append(stemmed_tokens)

    print("Cümle ayrıştırıldı...")

    duygu = ["kizgin","korku","mutlu","surpriz","uzgun"]
    haber = ["ekonomi","magazin","saglik","siyaset","spor","teknoloji","yasam"]
    uci = ["business","entertainment","medicine","sport","technology"]
    sayi = 0
    toplam = 0
    thefile = open('EngHaber5Sinif.txt', 'w')
    for text in texts:
        cumle = ""
        for deger in text:
            cumle += str(deger) + " "
        cumle += "," + uci[sayi]
        thefile.write("%s\n" % cumle)      
        toplam += 1
        if toplam == 1000: 
            sayi += 1
            toplam = 0
   
    print("Txt oluşturuldu...")


def txtToArffKlasor(dosya):
    global model_name
    
    tr_stop = get_stop_words('tr')
    global texts

    for i in dosya:
        map = i.maketrans('', '', string.punctuation)
        out = i.translate(map)
        tokens = []
        not_verb = []
        #Türkçe karakterlerin düzeltilmesi
        for kelime in str(out).split(" "):
            if str(kelime).find("İ") != -1 or str(kelime).find("Ö") != -1 or str(kelime).find("Ü") != -1 or str(kelime).find("Ç") != -1 or str(kelime).find("Ş") != -1:
                kelime = str(kelime).replace("I","ı")
                kelime = str(kelime).replace("İ","i")
                kelime = str(kelime).replace("Ö","ö")
                kelime = str(kelime).replace("Ü","ü")
                kelime = str(kelime).replace("Ç","ç")
                kelime = str(kelime).replace("Ş","ş")
            kelime = str(kelime).lower().strip()
            tokens.append(kelime)
        
        texts.append(tokens)

    print("Cümle ayrıştırıldı...")

    sayi = 0
    toplam = 0
    
    for text in texts:
        dosyaIsmi = str(sayi) + ".txt"
        thefile = open(dosyaIsmi, 'w')
        cumle = ""
        for deger in text:
            cumle += str(deger) + " "
        thefile.write("%s\n" % cumle)      
        sayi += 1
   
    print("Txt oluşturuldu...")


def grafikCiz(ldamodel, topic_num):
    import numpy as np
    import matplotlib.pyplot as plt

    #py.sign_in('anilguven1055', 'hnzN2lEL5JnEcTZd50Q5')
    lda = gensim.models.LdaModel.load(str(ldamodel) + '.model')
    agirliklar = lda.show_topics(num_topics=topic_num, num_words=15)
    topics = str(agirliklar).split('\'), (')
    topics[0] = str(topics[0])[2:]

    words = []
    weights = []
    for konular in topics:
        i = str(konular).find(',')
        konu = konular[:i]
        kelimeler = str(str(konular[i+3:]).strip()).split('+')
        toplam = 0
        for kelime in kelimeler:
            kelime = str(kelime).strip()
            i = str(kelime).find('*')
            agirlik = kelime[:i]
            k = str(kelime).rfind('"')
            kelime = kelime[i+2:k]
            if agirlik != "0.000":
                toplam += float(agirlik)
                weights.append(agirlik)
                words.append(kelime)
            else:
                break

        objects = (words)
        incomes = (weights)
        y_post = np.arange(len(objects))

        plt.figure(figsize=(8,8))
        plt.bar(y_post, incomes,align='center',alpha=0.5)
        plt.xticks(y_post, objects, rotation = 60)
        plt.ylabel('Ağırlıklar')
        plt.title(konu + '-topic')
        plt.savefig(konu + '-topic.png')
    
        words.clear()
        weights.clear()

    print('Grafikler çizildi')