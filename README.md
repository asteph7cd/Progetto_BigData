# Progetto Big Data, Cristi Gutu, Alessandro Stefani.
### Classificazione di news reperite dall'agenzia ANSA.it 

## Descrizione
In questo progetto si é affrontato il problema della classificazione di articoli provenienti dall’agenzia ANSA.
Ci si é concentrati sul confrontare le prestazioni del classificatore utilizzando diverse rappresentazioni dei documenti di testo.
In particolare sono state utilizzate la rappresentazione con term-document matrix e la rappresentazione ottenibile con la tecnica di topic modeling chiamata Latent Dirichlet Allocation(LDA).

## Prerequisiti
Questo progetto comprende programmi scritti in Python e pertanto è necessario avere installato Python3.
Inoltre, per il corretto funzionamento di tutti i programmi è necessario avere scaricato tutti i moduli python elencati in install.txt
Per leggere e modificare i file .ipynb si è utilizzato jupyter notebook(https://jupyter.org/), installato insieme ad Anaconda(https://www.anaconda.com/).

## 



## Note
-Era stata fatta una versione precedente con un dataset più ristretto e si era notato che una piccola parte di articoli era scritta in inglese perciò tali articoli erano statio riclassificati manualmente come una categoria a s\'e stante(effettivamente non si potevano neanche applicare efficacemente procedure come stemming o rimozione delle stop word siccome cambiano totalmente ,o almeno in parte, a seconda della lingua utilizzata). Per una svista ci siamo dimenticati di ripetere la cosa dopo il ridimensionamento del dataset. 

-I moduli utilizzati erano aggiornati alle versioni più recenti nel momento della scrittura dei programmi(giugno 2019), potrebbe eventualmente essere necessario riscrivere parti di codice a causa di possibili aggiornamenti dei moduli utilizzati.

-I risultati riportati nella relazione riguardo i tempi di esecuzione sono indicativi, inquanto dipendono dalla macchina su cui si fanno girare i programmi. Quei tempi sono stati registrati su una distribuzione di Linux, un processore AMD Athlon 64 3000+(1 CPU core, Frequency 2000 MH) e 3,5 GB di memoria centrale.

## Autori
Gutu Georghe Cristi (https://github.com/mastershef)
Alessandro Stefani (https://github.com/asteph7cd/)




