import pyLDAvis.gensim
import gensim

d = gensim.corpora.Dictionary.load('haberlerIsim.dict')
c = gensim.corpora.MmCorpus('haberlerIsim.mm')
lda = gensim.models.LdaModel.load('haberlerIsim.model')

if __name__ == '__main__':
    data = pyLDAvis.gensim.prepare(lda, c, d, mds='mmds')
    pyLDAvis.save_html(data, 'haberlerIsim.html')
