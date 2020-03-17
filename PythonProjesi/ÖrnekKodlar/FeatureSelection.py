import gensim
from gensim import corpora, models
import pandas
import numpy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from stop_words import get_stop_words
import snowballstemmer
import string
from gensim import utils, models
import warnings
warnings.filterwarnings('ignore')
import mIslemler

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()


doc_set = mIslemler.dosyaOkuma("Haberler.txt")
tr_verb = mIslemler.dosyaOkuma("verbsTR.txt")
turler = mIslemler.dosyaOkuma("Turleri.txt")

tr_stop = get_stop_words('turkish')
tr_stemmer = snowballstemmer.stemmer('turkish')

cumleler = []
number = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
for i in doc_set:
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
    stemmed_tokens_verb = [ tr_stemmer.stemWord(i) for i in tokens if not i in tr_stop]
    stemmed_tokens = [i for i in stemmed_tokens_verb if not i in tr_verb]
    delete = []
    #Köklerden sayıların ve tek harflilerin silinmesi
    for kok in stemmed_tokens:
        if len(kok) <= 2:
            delete.append(kok)
        else:
            for sayi in number:
                if kok[0].find(sayi) != -1:
                    delete.append(kok)
                    break
    for sil in delete:
        stemmed_tokens.remove(sil)

    stemmed_tokens = [item if len(item) > 7 else item[0:5] for item in stemmed_tokens]
    
    cumle = ''
    for kelime in stemmed_tokens:
        cumle += ' ' + kelime
    cumleler.append(cumle)


X = vectorizer.fit_transform(cumleler)
Y = turler
analyse = vectorizer.build_analyzer()
features = vectorizer.get_feature_names()

print (features)
test = SelectKBest(score_func=chi2)
fit = test.fit(X, Y)
# summarize scores
numpy.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
# summarize selected features
print(features[0:5, :])
