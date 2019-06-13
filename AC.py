from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import TweetTokenizer
from os import listdir
from os.path import isfile, join
import json
import nltk
from string import punctuation
import tqdm
import numpy as np
import pandas as pd
import re
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.utils.multiclass import unique_labels
from itertools import product

# Importing for graphics and plots.
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib

#setting style
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
rc={"lines.linewidth": 2.5})

from bs4 import BeautifulSoup

###FUNZIONI NECESSARIE PER ESTRAZIONE DATI DAGLI ARTICOLI PER POI CREARE LA TERM DOCUMENT MATRIX
# ------------------------------------------------------------------------------------------------- #
# prende in input un'articolo e esegue delle modifiche ai campi di interesse utilizzando funzioni e oggetti passati come parametri
def news_analyzer(news, tokenizer, stemmer, stop_words=[]):
    campi_interesse = ['testo','titolo_articolo','sottotitolo','tags']
    news['tags'] = " ".join(news['tags'])
    for campo in campi_interesse:
        news[campo] = BeautifulSoup(news[campo], "lxml").get_text()                                 # rimozione tag html
        news[campo] = news[campo].replace("'"," ")                                                  # rimossi apostrofi
        news[campo] = re.sub("http\S+", " link ", news[campo])                                      # rimossi link
        news[campo] = tokenizer.tokenize(news[campo])                                               # campi tokenizzati
        news[campo] = [token for token in news[campo] if not token.isdigit() and token not in stop_words]# tokenizzazione e rimozione delle stopwords
        news[campo] = [stemmer.stem(token) for token in news[campo] if not token.isdigit() and token not in stop_words]# stemming

    return news                                                                                     #ritorna l'articolo processato

# ------------------------------------------------------------------------------------------------- #
# effettua il preprocessamento di tutti gli articoli dell'insieme di trainig

def preproc(X):
    #stopwords di textwiller
    #stop = ["a","adesso","ai","al","alla","allo","allora","altre","altri","altro","anche","ancora","avere","aveva","avevano","ben","buono","che","chi","cinque","comprare","con","consecutivi","consecutivo","cosa","cui","da","del","della","dello","dentro","deve","devo","di","doppio","due","e","ecco","fare","fine","fino","fra","gente","giu","ha","hai","hanno","ho","il","indietro 	","invece","io","la","lavoro","le","lei","lo","loro","lui","lungo","ma","me","meglio","molta","molti","molto","nei","nella","no","noi","nome","nostro","nove","nuovi","nuovo","o","oltre","ora","otto","peggio","pero","persone","piu","poco","primo","promesso","qua","quarto","quasi","quattro","quello","questo","qui","quindi","quinto","rispetto","sara","secondo","sei","sembra 	","sembrava","senza","sette","sia","siamo","siete","solo","sono","sopra","soprattutto","sotto","stati","stato","stesso","su","subito","sul","sulla","tanto","te","tempo","terzo","tra","tre","triplo","ultimo","un","una","uno","va","vai","voi","volte","vostro"]    
    stop_words = stopwords.words('italian') + list(punctuation)                                     # definizione delle stopword
    stemmer = SnowballStemmer("italian")                                                            # definizione dello stemmer
    tokenizer = TweetTokenizer(preserve_case=False,strip_handles=True)                              # definizione del tokenizer
    return [news_analyzer(news, tokenizer, stemmer, stop_words) for news in tqdm.tqdm(X)]           # effettuiamo il preprocessamento su tutti gli articoli in X

# ------------------------------------------------------------------------------------------------- #
# ottieni gli articoli dai file .json
def get_news(news_dir):
    data = []
    onlyfiles = [f for f in listdir(news_dir) if isfile(join(news_dir, f))]
    for articolo in onlyfiles:
        docJson = open(news_dir+articolo,"r")
        jsonData = json.loads(docJson.read())
        data.append(jsonData)
        docJson.close()
    return data

# ------------------------------------------------------------------------------------------------- #
def distribuzione_frequenze(docs,y, ngrammi=(1,3), min_df = 4, max_df=0.9):
    no_features = 10000
    vectorizer = CountVectorizer(ngram_range=ngrammi,max_df=max_df,min_df=min_df, max_features=no_features, lowercase=False, binary=True, encoding='utf-8')
    tf = vectorizer.fit_transform(docs)                                                             # term-document matrix(con solo 1 o 0)
    tdf = np.sum(tf.toarray(), axis=0)                                                              # totali di colonna (elemento i di questo vettore e' il numero di documenti in cui e' presente il termine in colonna i) 
    dfdf = pd.DataFrame(tdf,columns=["df"])                                                         # riorganizziamo i dati in modo da poterne fare un plot
    plt.figure(num=1,figsize=(18, 10))
    sns.countplot(x="df", data = dfdf)
    return sorted(tdf)[0]
    '''
    # plottare il grafico con y in scala logaritmica
    df ={}
    # troviamo le frequenze assolute delle frequenze dei termini (quanti termini appaiono in k documenti con k i valori distinti in tdf)
    for i in tdf:
        if i in df:
            df[i] += 1
        else:
            df[i] = 1
    xy = [[],[]]
    for item in sorted(df.keys()):
        xy[0].append(item)
        xy[1].append(df[item])
    #print(xy)
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.grid(None)
    ax.grid(True, which="both",axis="y")
    ax.bar(xy[0], xy[1])
    ax.set_xticks(xy[0])
    ax.set_yscale('log')
    plt.show()
    '''

# ------------------------------------------------------------------------------------------------- #
def conf_mat(pred_cats, test_cats):

    classi = unique_labels(test_cats, pred_cats)
    plt.title("Matrice di confusione")
    cm = confusion_matrix(test_cats, pred_cats)
    # se vogliamo normallizare la matrice, dividendo per riga
    #cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #print("Normalized confusion matrix")
    #print(cm)
    #da notare poi bisognerebbe cambiare il "formato" dei numeri che sarebbero float('.2f' al posto di 'd')
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(fraction=0.04, pad=0.1)#spazi occupati dalla barra a destra
    
    #labels
    tick_marks = range(len(classi))
    plt.xticks(tick_marks, classi, rotation=90)
    plt.yticks(tick_marks, classi)
    
    plt.grid(None)
    
    thresh = cm.max() / 2
    #colore dei numeri dentro i quadrati
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("Classe osservata")
    plt.xlabel("Classe prevista")
    
    return 100 * accuracy_score(test_cats, pred_cats)

# ------------------------------------------------------------------------------------------------- #
def visualizza_topics(lda, feature_names, n):
    mydict = {"Termini piu' probabili per topic":[]}                                                # creiamo dizionario
    for topic_idx, topic in enumerate(lda.components_):                                             # riempiamo il dizionario con i primi n termini per topic termini per topic
        mydict["Termini piu' probabili per topic"].append(" ".join([feature_names[i] for i in topic.argsort()[:-n - 1:-1] ]))
    top = pd.DataFrame(mydict)
    top.index.name = "Topic"
    return top

# ------------------------------------------------------------------------------------------------- #
# grafico per visualizzare l'"ottimizzazione"
def grafico_metrica_iperparametri(risultati, iperparam1, iperparam2, metrica, alpha=0.75, s=300):
    plt.title("{} al variare di {} e {}".format(metrica, iperparam1,
        iperparam2))
    plt.scatter(risultati[iperparam1], risultati[iperparam2],
        c=risultati[metrica], cmap=plt.cm.RdBu_r, alpha=alpha, s=s)
    plt.colorbar()
    plt.xlabel(iperparam1)
    plt.ylabel(iperparam2)
    plt.grid()