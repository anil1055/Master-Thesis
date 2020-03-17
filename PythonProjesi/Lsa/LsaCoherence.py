from gensim import corpora
from gensim.models import LsiModel
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
import pandas as pd
import LsaModel

def preProcessing(dokuman, strDoc, strLast, strBool):
    news_df = pd.DataFrame({strDoc:dokuman})
    stop_verbs = LsaModel.dosyaOkuma("verbsTR.txt")
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

    return tokenized_doc

def prepare_corpus(doc_clean):
    """
    Input  : clean document
    Purpose: create term dictionary of our courpus and Converting list of documents (corpus) into Document Term Matrix
    Output : term dictionary and Document Term Matrix
    """
    # Creating the term dictionary of our courpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)
    dictionary = corpora.Dictionary(doc_clean)
    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
    # generate LDA model
    return dictionary,doc_term_matrix

def create_gensim_lsa_model(doc_clean,number_of_topics,words):
    """
    Input  : clean document, number of topics and number of words associated with each topic
    Purpose: create LSA model using gensim
    Output : return LSA model
    """
    dictionary,doc_term_matrix=prepare_corpus(doc_clean)
    # generate LSA model
    lsamodel = LsiModel(doc_term_matrix, num_topics=number_of_topics, id2word = dictionary)  # train model
    print(lsamodel.print_topics(num_topics=number_of_topics, num_words=words))
    return lsamodel

def computeLsa(dictionary, doc_term_matrix, stop, start=2, step=3):
    from collections import OrderedDict
    model_list = OrderedDict()
    for num_topics in range(start, stop, step):
        # generate LSA model
        model = LsiModel(doc_term_matrix, num_topics=num_topics, id2word = dictionary)  # train model
        model_list[num_topics] = model

    return model_list

def coherence(models, dictionary, doc_term_matrix, doc_clean):   
    cm = CoherenceModel.for_models(models.values(), texts = doc_clean ,dictionary = dictionary, coherence='c_v')
    coherence_estimates = cm.compare_models(models.values())
    coherences = dict(zip(models.keys(), coherence_estimates))
    coherenceOrder(coherences, cm)

#Coherence değerlerinin topic sayısına göre sıralanması
def coherenceOrder(coherences, cm):
    avg_coherence = \
        [(num_topics, avg_coherence)
         for num_topics, (_, avg_coherence) in coherences.items()]
    ranked = sorted(avg_coherence, key=lambda tup: tup[1], reverse=True)
    print("Ranked by average '%s' coherence:\n" % cm.coherence)
    for item in ranked:
        print("num_topics=%d:\t%.4f" % item)
    print("\nBest: %d" % ranked[0][0])

def plot_graph(doc_clean,start, stop, step):
    dictionary,doc_term_matrix=prepare_corpus(doc_clean)
    model_list, coherence_values = compute_coherence_values(dictionary, doc_term_matrix,doc_clean,
                                                            stop, start, step)
    # Show graph
    x = range(start, stop, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()

doc_set = LsaModel.dosyaOkuma("HaberlerLsa5C.txt")
detoken = preProcessing(doc_set, 'document', 'docLast', False)
dct,termmat = prepare_corpus(detoken)
model_list = computeLsa(dct,termmat, 30, 10, 5)
if __name__ == '__main__':
    coherence(model_list,dct,termmat,detoken)
