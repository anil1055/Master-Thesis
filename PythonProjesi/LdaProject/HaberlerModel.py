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

doc_set = mIslemler.dosyaOkuma("YTU_deneme.txt")

mIslemler.model_name = "YTU_deneme"
dictionary, corpus = mIslemler.cumleAyristirma(doc_set, 0, True)
mIslemler.modelOlusturmaTopic(15, corpus, dictionary, False)
dictionary = gensim.corpora.Dictionary.load('HaberlerIF5CS2.dict')
corpus = gensim.corpora.MmCorpus('HaberlerIF5CS2.mm')
mIslemler.model_name = "HaberlerIF5CS2"
mIslemler.modelOlusturmaTopic(20, corpus, dictionary, True)
#mIslemler.modelOlusturmaTopic(28, corpus, dictionary)

#Bound Perplexity hesabı ve Model oluşumu
#mIslemler.modelOlusturmaTopic(28, corpus, dictionary, True)
dictionary = gensim.corpora.Dictionary.load('HaberlerIF5CS2Stage3.dict')
corpus = gensim.corpora.MmCorpus('HaberlerIF5CS2Stage3.mm')

trained_models = OrderedDict()
for topic_num in range(10,55,5):
    trained_models[topic_num] = gensim.models.LdaModel.load(str(topic_num) + '.model')

#Coherence işlemi
if __name__ == '__main__':
    mIslemler.coherence(trained_models, dictionary)

"""
mIslemler.modelOlusturmaTopic(21, corpus, dictionary, True)
dictionary = gensim.corpora.Dictionary.load('Haberler7CStage2.dict')
corpus = gensim.corpora.MmCorpus('Haberler7CStage2.mm')
mIslemler.modelOlusturmaTopic(21, corpus, dictionary)
"""