# Learning Personality

`Learning Personality` è un progetto formativo svolto durante lo stage di laurea triennale.  
Il lavoro consiste nell'identificazione di una procedura in grado di estrarre, tramite approcci automatici, la personalità del oggetto "target" a cui un determinato testo si riferisce.  
 Vengono esplorati diversi spazi di rappresentazione, si parte da un approccio che sfrutta la rappresentazione *bag-of-words*, fino a giungere alla costruzione di un embedding di parole utilizzando la versione *skip-gram* dell'algoritmo `word2vec` di Tomas Mikolov. 
 In seguito si sfrutteranno tre diversi tipi di reti neurali artificiali.  
 
 ## Dataset
 
Il dataset per questo task è scaricabile da https://www.yelp.com/dataset/challenge

## Obiettivo

La natura di questo progetto di tesi è altamente sperimentale ed è volta a presentare analisi
dettagliate sull’argomento, in quanto allo stato attuale non esistono importanti indagini
che affrontino il problema dell’apprendimento dei tratti di personalità a partire da testo in
linguaggio naturale.

## Addestramento

Sono stati creati una serie di script Python per automatizzare e rendere ripetibile il 
processo di preprocessing, estrazione delle feature e l'addestramento dei modelli. 

Il primo modello implementato è una NN *feed-forward* e *fully-connected*.  

Il secondo modello utilizzata uan classi di algorimi distribuzionali che consistono nell'utilizzo di una rete neurale in grado di imparare, in modo non supervisionato, i contesti delle parole.
L'embedding di parole da qui generato viene utilizzato come input per una NN *convoluzionale*.

Il terzo modello trasforma il problema di regressione in uno di *classificazione binaria multi-label*, in cui per ogni dimensione di personalità l'output sarà 0 o 1.

## In ordine

* Nei file `input_pipeline`, `preprocess_dataset`, `TFRecord_dataset`, `load_ocean` :
    * Viene parsato il file json estraendo solo le recensioni (trasformate in lower case).
    * Vengono generati i due dataset di training e test 80-20: Abbiamo in totale 1243000 sentence nel dataset di test e 4974000 nel dataset di training.
    * Vengono generate le sentences splittando sulla punteggiatura.    
    * Vengono rimosse la stopwords dalle sentences.
    * Vengono eliminate le sentences che non contengono gli aggettivi.
    * Vengono salvati su file i tre zip contenenti l'intero dataset, quello di training e quello di test.

* Nel file `dictionary`, `voc`, `remove_adj` :
    * Viene generato un file .txt contentente per ogni riga una parola per tutte le sentences in ordine. 
    * In seguito eseguiamo un sorting del file, teniamo un contatore per ogni parola in modo da non avere ripetizioni e ordiniamo nuovamente.
    * Vengono eliminati gli aggettivi che compaiono nel dizionario presenti nel dataset ocean.
    * Generiamo un nuovo file compatto in cui abbiamo solo le prime n parole più frequenti, in modo che in seguito venga associato ad essi il token 'UNK'.

#### Modello 1
   
* Nei file `extract_features`, `model_input`, `training` :
    * Viene costruita una lookup-table contentente le 60000 parole più frequenti. Le parole univoche vengono indicizzate con un valore intero univoco (corrispondente al numero della linea), le parole non comprese tra le prime 60000 parole più comuni verranno contrassegnate con "-1". 
    * Viene creata una reverse lookup-table che permette di cercare una parola passando attraverso il suo identificatore univoco. Le parole sconosciute, identificate da '-1', vengono sostituite con il token "UNK".
    * Viene generato il vettore bag of words e ad esso viene associato il vettore ocean.
    * Costruiamo un modello di base, con n layer completamente connessi. Ad ognuno di essi viene applicata la funzione di attivazione non lineare *ReLU*. Dopo ogni layer viene effettuata una *batch-normalization*.
    * Le simulazioni possono essere addestate per n epoche. L'ottimizzatore scelto è *Adagrad* con learning rate 0,001. La funzione obiettivo utilizzata è il *mean squared error* MSE, inoltre si ricorre all'utilizzo della metrica *root mean squared error* RMSE.

#### Modello 2

* Nei file `mikolov_features`, `mikolov_embedding_model`, `mikolov_embedding_training` :
    * Viene effettuata la stessa procedura per costruire il dizionario delle 60000 parole più frequenti. 
    * Vengono generate le feature per la costruzione dell'embedding formando un set di dati composto dall'accoppiamento di ogni parola con il suo contesto. Vengono considerate come contesto la parola a destra e la parola a sinistra del target.  
    È possibile determinare la dimensione dell'embedding e il numero di etichette negative utilizzate per il campionamento.
    * La funzione obiettivo usata dalla rete è lo *Stocastic Gradient Descent* SGD.

* Nei file `mikolov_model`, `mikolov_training` :
    * Costruiamo un modello con n layer convoluzionali e uno finale completamente connesso. Ad ognuno di essi viene applicata la funzione di attivazione non lineare *ReLU*. Dopo ogni layer viene effettuata una *batch-normalization*. Inoltre dopo il primo layer vi è un layer di *pooling*.
    * Le simulazioni possono essere addestate per n epoche. L'ottimizzatore scelto è *Adagrad* con learning rate 0,005. La funzione obiettivo utilizzata è il *mean squared error* MSE, inoltre si ricorre all'utilizzo della metrica *root mean squared error* RMSE.


#### Modello 3

* Nei file `mikolov_features` :
    * Viene effettuata la stessa procedura del modello 2 ma la costruzione dell'embedding aviene formando un set di dati composto dall'accoppiamento di ogni aggettivo di nostro interesse con il suo contesto. Vengono considerate come contesto le due parole a destra e le due parole a sinistra del target.  
    
#### Modello 4

* Nei file `mikolov_multiclass_binary_model`, `mikolov_multiclass_binary_training`  :
    * La procedure di estrazione dell'embedding è la stessa dei due precedenti modelli.    * Costruiamo un modello di base, con n layer. ad ognuno di essi viene applicata la funzione di attivazione non lineare *ReLU*. Dopo ogni layer viene effettuata una *batch-normalization*.
    * Il modello costruito è simile al precedente con la differenza che la funzione obiettivo utilizzata è la *softmax cross entropy*. Inoltre si ricorre all'utilizzo della metrica *accuracy* per ogni tratto di personalità, e vengono plottate tramite *Tensorboard* la matrici di confusione.
