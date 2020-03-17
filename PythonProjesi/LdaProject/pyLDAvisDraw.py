import pyLDAvis.gensim
import gensim


d = gensim.corpora.Dictionary.load('Duygu5CY.dict')
c = gensim.corpora.MmCorpus('Duygu5CY.mm')
lda = gensim.models.LdaModel.load('Duygu5CY.model')

if __name__ == '__main__':
    data = pyLDAvis.gensim.prepare(lda, c, d, mds='mmds')
    pyLDAvis.save_html(data, 'Duygu5CY.html')
