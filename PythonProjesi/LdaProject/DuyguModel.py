#!/usr/bin/python
import string
from collections import OrderedDict
#import plotly.plotly as py
#import plotly.graph_objs as go
import mIslemler
import gensim
from gensim import corpora, models

""" Önemli Bilgiler
    1-)Zemberekten elde edilen dosya bilgileri;
        Dosya ismi: DuyguZemberek  
        Test dökümanı: DuyguIsimFiilTest.txt
        10 Topic için: DuyguZemberek10T

    2-)İlk 5 harfi kök olarak kabul etme dosya bilgileri;
        Dosya ismi: DuyguIlk5Kok
        Test dökümanı: TestHaber.txt
        10 Topic için: DuyguIlk5Kok10T

    3-)Snowball ve 5 Kök için dosya bilgisi;
        Dosya ismi: DuyguSnowball_5Kok
        Test dökümanı: TestHaber.txt
        10 Topic için: DuyguSnowball_5Kok10T

    *Modeller için cumleAyristirma fonksiyonu secim(k) değerleri;
        #DuyguSnowball_5Kok için k = 1
        #DuyguIlk5Kok k = 0
        #DuyguZemberek k = 2
"""

#mIslemler.grafikCiz("uci-news5CStage2", 40)
mIslemler.model_name = "Tivit5S"
doc_set = mIslemler.dosyaOkuma("Duygu5CAll.txt")
#mIslemler.txtToArffKlasor(doc_set)

#mIslemler.arffOlustur(15, "uci-news3CStage2", "uci-news3C.txt")
mIslemler.model_name = "Duygu5CY"
dictionary, corpus = mIslemler.cumleAyristirma(doc_set, 2, True)
#dictionary = gensim.corpora.Dictionary.load('Duygu3CAll.dict')
#corpus = gensim.corpora.MmCorpus('Duygu3CY.mm')

mIslemler.csvFile(20, 'Duygu5CY')
#mIslemler.modelOlusturmaTopic(50, corpus, dictionary, True)
#mIslemler.boundPerplex(corpus, dictionary)
#mIslemler.modelOlusturmaTopic(6, corpus, dictionary)
#mIslemler.modelOlusturmaTopic(6, corpus, dictionary, True, True)

"""
lda = gensim.models.LdaModel.load('DuyguStage2C.model')

print(lda.show_topics(num_topics=15, num_words=15))
#mIslemler.modelOlusturmaTopic(20, corpus, dictionary)
dictionary = gensim.corpora.Dictionary.load('Duygu5K5CStage2.dict')
corpus = gensim.corpora.MmCorpus('Duygu5K5CStage2.mm')
mIslemler.model_name = "Duygu5K5CStage2"
mIslemler.modelOlusturmaTopic(50, corpus, dictionary)
"""
"""
mIslemler.model_name = "Duygu5CStage2"
dictionary = gensim.corpora.Dictionary.load('Duygu5CStage2.dict')
corpus = gensim.corpora.MmCorpus('Duygu5CStage2.mm')

mIslemler.modelOlusturmaTopic(45, corpus, dictionary)

mIslemler.modelOlusturmaTopic(6, corpus, dictionary, True)
dictionary = gensim.corpora.Dictionary.load('DuyguS5Stage2.dict')
corpus = gensim.corpora.MmCorpus('DuyguS5Stage2.mm')

dictionary = gensim.corpora.Dictionary.load('Duygu3CYStage2Stage3.dict')
corpus = gensim.corpora.MmCorpus('Duygu3CYStage2Stage3.mm')
trained_models = OrderedDict()
for topic_num in range(6,33,3):
    trained_models[topic_num] = gensim.models.LdaModel.load(str(topic_num) + '.model')

#Coherence işlemi
if __name__ == '__main__':
    mIslemler.coherence(trained_models, dictionary)
"""
"""
mIslemler.model_name = "DuyguZ3CStage2"
dictionary = gensim.corpora.Dictionary.load('DuyguZ3CStage2.dict')
corpus = gensim.corpora.MmCorpus('DuyguZ3CStage2.mm')
mIslemler.modelOlusturmaTopic(21, corpus, dictionary)
#mIslemler.modelOlusturmaTopic(6, corpus, dictionary, True, True)

"""
