#random search su tutti i parametri, alla fine non e' stata utilizzato

from AC import preproc
from AC import get_news
import inspect
import nltk
nltk.download("stopwords")
print("ottenere articoli")
economia = get_news("./articoli_economia/")
cultura = get_news("./articoli_cultura/")
tech = get_news("./articoli_tech/")
politica = get_news("./articoli_politica/")
sport = get_news("./articoli_sport/")
cronaca = get_news("./articoli_cronaca/")
print("preprocessamento")
for articolo in economia:
    articolo['categoria'] = "Economia"
for articolo in cultura:
    articolo['categoria'] = "Cultura"
for articolo in tech:
    articolo['categoria'] = "Tech"
for articolo in politica:
    articolo['categoria'] = "Politica"
for articolo in sport:
    articolo['categoria'] = "Sport"
for articolo in cronaca:
    articolo['categoria'] = "Cronaca"
dati_preprocessati =  preproc(tech + politica + cultura + economia + sport + cronaca)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import ParameterSampler
import tqdm

seed = 0
train_documents,test_documents = train_test_split(dati_preprocessati,random_state=seed, train_size = 0.5)
test_documents,val_documents = train_test_split(test_documents,random_state=seed, train_size = 0.5)

train_texts = [' '.join([word for word in x['testo']] + x['sottotitolo'] + x['titolo_articolo']) for x in train_documents]
test_texts = [' '.join([word for word in x['testo']] + x['sottotitolo'] + x['titolo_articolo']) for x in test_documents]
val_texts = [' '.join([word for word in x['testo']] + x['sottotitolo'] + x['titolo_articolo']) for x in val_documents]

train_cats = [x["categoria"] for x in train_documents]
test_cats = [x["categoria"] for x in test_documents]
val_cats = [x["categoria"] for x in val_documents]
    
ldac = Pipeline([
    ("count_mx",CountVectorizer(encoding='utf-8',max_features=1000000, lowercase=True)),
    ("lda", LatentDirichletAllocation(max_iter=50, learning_method='online',random_state=0)), 
    ("classifier",DecisionTreeClassifier(random_state=0 ))
])

param_grid = ParameterGrid({
    'classifier__max_depth': range(10,70),
    'classifier__min_samples_leaf':2 ** np.arange(9),
    'lda__n_components':[6,12,24,48,96],
    'lda__learning_decay':[0.5,0.7,0.9],
    'count_mx__ngram_range':[(1,i) for i in range(1,4)]+[(2,2),(3,3)],
    'count_mx__min_df': np.arange(3, 30),
    'count_mx__max_df': [60,70,80,100,200,0.5,0.6,0.7,0.8]
})
#print(param_grid.param_grid)
param_list = list(ParameterSampler(param_grid.param_grid[0],n_iter=1000,random_state=0))

#print(param_list)

risultati = []
print("random search")
count = 0
for params in tqdm.tqdm(param_list):
    count += 1
    ldac.set_params(**params)
    ldac.fit(train_texts, train_cats)
    y_pred = ldac.predict(val_texts)
    params["accuracy_score"] = metrics.accuracy_score(val_cats, y_pred)
    risultati.append(params)
    if count%100 == 0:
        res = pd.DataFrame(risultati).sort_values(["accuracy_score",'count_mx__min_df','count_mx__max_df','lda__n_components','lda__learning_decay', "classifier__max_depth"], ascending=[False, True, True, True, True, True])
        res.reset_index(drop=True, inplace=True)
        res.to_csv('./res_'+str(count)+'.csv')

risultati = pd.DataFrame(risultati).sort_values(["accuracy_score",'count_mx__min_df','count_mx__max_df','lda__n_components','lda__learning_decay', "classifier__max_depth"], ascending=[False, True, True, True, True, True])
risultati.reset_index(drop=True, inplace=True)
risultati.to_csv('./res_fin.csv')
