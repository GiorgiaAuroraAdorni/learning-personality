# Learning Personality

`chess_recognition` è una libreria MATLAB che permette di riconoscere 
automaticamente gli schemi degli scacchi de _La Settimana Enigmistica_.


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
        
* Nel file `bag_of_words`, `extract_features` :
    * Viene costruita una lookup-table contentente le 60000 parole più frequenti. Le parole univoche vengono indicizzate con un valore intero univoco (corrispondente al numero della linea), le parole non comprese tra le prime 60000 parole più comuni verranno contrassegnate con "-1". 
    * Viene create una reverse lookup-table che permette di cercare una parola passando attraverso il suo identificatore univoco. Le parole sconosciute, identificate da '-1', vengono sostituite con il token "UNK".
    * Viene generato il vettore on_hot.
    
    * associamo il vettore ocean


## Utilizzo

La funzione principale è `recognize_chess_pieces`, che prende in input 
l'immagine da analizzare, localizza la scacchiera, riconosce i pezzi e li 
ritorna codificati in notazione FEN.

```matlab
  image = imread('path/immagine/desiderata.jpg');
  fen = recognize_chess_pieces(image);
```

Chiamando `recognize_chess_pieces` senza assegnare il valore di ritorno o 
passando esplicitamente `true` come secondo parametro permette di visualizzare 
in una figura di MATLAB la posizione della scacchiera nell'immagine e la 
configurazione di pezzi riconosciuta.

```matlab
  recognize_chess_pieces(image);
```

## Addestramento

Abbiamo creato una serie di script per automatizzare e rendere ripetibile il 
processo di addestramento dei classificatori. La procedure dei tre 
classificatori sono molto simili tra loro, viene mostrata come esempio la 
procedura per `orientation_classifier`:

```matlab
  % Estrai le feature dalle immagini dei tre dataset e crea la tabella che verrà
  % usata per l'addestramento.
  ds = orientation_classifier.training.create_dataset(1:3);

  % Partiziona il dataset in training e test per permettere la validazione.
  cv = orientation_classifier.training.create_cvpartition(ds);

  % La partizione viene fatta sulle immagini, ma il classificatore lavora su una
  % cella singola per volta. Tutte le celle di un'immagine vengono messe insieme.
  test = images(cv.test, :);
  training = images(cv.training, :);

  test_cells = innerjoin(ds, test);
  training_cells = innerjoin(ds, training);

  % Addestra il classificatore.
  model = orientation_classifier.training.train_cubic_svm(training_cells);

  % Esegui la validazione del modello, mostrando la matrice di confusione e 
  % infomazioni sulle performance ottenute aggregando tutte le celle di un'immagine.
  orientation_classifier.training.evaluate_model(model);
```

## Organizzazione delle cartelle

Nella cartella principale del progetto troviamo diverse sottocartelle e funzioni 
utilizzate nel progetto:

* `datasets` contiene i tre dataset, al cui interno abbiamo:
  * cartella `images` con le immagini originali;
  * `labels.csv` e `puzzles.csv` con le configurazioni FEN di ogni scacchiera;
  * `framepoints.mat` con le coordinate degli angoli di ogni scacchiera;
  * cartella `frames` con le componenti connesse estratte dagli edge ed etichettate, usate nell'addestramento di `edge_classifier`;
  * cartella `groundtruth` con le configurazioni FEN nel formato richiesto dalla consegna del progetto;
* `docs` contiene i dettagli del progetto e la relazione;
* `fonts` contiene il font _Chess Mérida_ e i template dei singoli pezzi;
* `+edge_classifier`, `+orientation_classifier` e `+piece_classifier` contengono il codice dei tre classificatori:
  * il modello addestrato, le funzioni utilizzate per l’estrazione delle feature;
  * `+training` con le funzioni utilizzate per l'addestramentoe e la valutazione del modello;.
* `+lib` contiene funzioni di terze parti scaricate da _MATLAB Central_ e altre fonti.
* `+utilities` contiene vari script creati e utilizzati nello sviluppo ma che non fanno parte del normale flusso del progetto.
