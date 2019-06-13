# Progetto Big Data, Cristi Gutu, Alessandro Stefani.
### Classificazione di news reperite dall'agenzia ANSA.it 

## Descrizione
In questo progetto si è affrontato il problema della classificazione di articoli provenienti dall’agenzia ANSA.
Ci si è concentrati sul confrontare le prestazioni del classificatore utilizzando diverse rappresentazioni dei documenti di testo.
In particolare sono state utilizzate la rappresentazione con term-document matrix e la rappresentazione ottenibile con la tecnica di topic modeling chiamata Latent Dirichlet Allocation(LDA).

## Prerequisiti
Questo progetto comprende programmi scritti in Python e pertanto è necessario avere installato Python3.
Inoltre, per il corretto funzionamento di tutti i programmi è necessario avere scaricato tutti i moduli python elencati in install.txt
Per leggere e modificare i file .ipynb si è utilizzato jupyter notebook(https://jupyter.org/), installato insieme ad Anaconda(https://www.anaconda.com/).

## I file contenuti nella repository
-La repository contiene 6 cartelle con nome che inizia per "articoli_", queste contengono gli articoli in formato json suddivisi a seconda della categoria.

-AC.py è una libreria utilizzata nei vari programmi, contiene alcune funzioni utili

-Le analisi sono state fatte con 5 dei file .ipynb, in ordine sarbbero:
1. Analisi_esplorativa.ipynb
2. Ottimizzazione.ipynb
3. Varia_training_set.ipynb
4. Tabelle.ipynb
5. Tempi.ipynb

-Il sesto file .ipynb, Nuovi_articoli.ipynb, serve per provare a classificare articoli non appartenenti al dataset (funziona per altri articoli dal sito ansa.it inserendo il corrispondente indirizzo web)

-La relazione e la presentazione sono contenute rispettivamente nei file "Gutu_Stefani_BigData.pdf" e "Big Data.pdf"

-Infine, il programma random_search.py contiene il codice per effettuare una random search su tutti i parametri(non è stato utilizzato)

## Note
-Era stata fatta una versione precedente con un dataset più ristretto e si era notato che una piccola parte di articoli era scritta in inglese perciò tali articoli erano statio riclassificati manualmente come una categoria a sé stante(effettivamente non si potevano neanche applicare efficacemente procedure come stemming o rimozione delle stop word siccome cambiano totalmente ,o almeno in parte, a seconda della lingua utilizzata). Per una svista ci siamo dimenticati di ripetere la cosa dopo il ridimensionamento del dataset. 

-I moduli utilizzati erano aggiornati alle versioni più recenti nel momento della scrittura dei programmi(giugno 2019), potrebbe eventualmente essere necessario riscrivere parti di codice a causa di possibili aggiornamenti dei moduli utilizzati.

-I risultati riportati nella relazione riguardo i tempi di esecuzione sono indicativi, inquanto dipendono dalla macchina su cui si fanno girare i programmi. Quei tempi sono stati registrati su una distribuzione di Linux, un processore AMD Athlon 64 3000+(1 CPU core, Frequency 2000 MH) e 3,5 GB di memoria centrale.

## Autori
Gutu Georghe Cristi (https://github.com/mastershef), 
Alessandro Stefani (https://github.com/asteph7cd/)




