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

    *Modeller için cumleAyristirma fonksiyonu secim(k) değerleri;
        #HaberlerSnowball_5Kok için k = 1
        #HaberlerIlk5Kok k = 0
        #HaberlerZemberek k = 2
"""
"""
#Bound Perplexity hesabı ve Model oluşumu
mIslemler.modelOlusturmaTopic(28, corpus, dictionary, True)
dictionary = gensim.corpora.Dictionary.load('HaberlerIF5CS2Stage3.dict')
corpus = gensim.corpora.MmCorpus('HaberlerIF5CS2Stage3.mm')
"""

"""Coherence için topic oluşturma
doc_set = mIslemler.dosyaOkuma('HaberlerWeka3C.txt')
mIslemler.model_name = 'TrHaber3-LDA'
dictionary, corpus = mIslemler.cumleAyristirma(doc_set, 2, True)
for topic in range(6,33,3):
    mIslemler.modelOlusturmaTopic(topic, corpus, dictionary, False)
"""

topic_num = 6
doc_set = mIslemler.dosyaOkuma("HaberlerWeka3C.txt")
mIslemler.model_name = "TrHaber3-LDA"
dictionary, corpus = mIslemler.cumleAyristirma(doc_set, 2, True)
mIslemler.modelOlusturmaTopic(topic_num, corpus, dictionary, False)
dictionary = gensim.corpora.Dictionary.load('TrHaber3-LDA.dict')
corpus = gensim.corpora.MmCorpus('TrHaber3-LDA.mm')
mIslemler.modelOlusturmaTopic(topic_num, corpus, dictionary, True, False, "TrHaber3-2LDA")
dictionary = gensim.corpora.Dictionary.load('TrHaber3-2LDA.dict')
corpus = gensim.corpora.MmCorpus('TrHaber3-2LDA.mm')
mIslemler.modelOlusturmaTopic(topic_num, corpus, dictionary, True, False, "TrHaber3-3LDA")

#trained_models = OrderedDict()
#for topic_num in range(10,55,5):
#    trained_models[topic_num] = gensim.models.LdaModel.load(str(topic_num) + '.model')
#Coherence işlemi
#if __name__ == '__main__':
#    mIslemler.coherence(trained_models, dictionary)
