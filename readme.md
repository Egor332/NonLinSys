# HIV Molecule Transformer

Projekt wykorzystujący architekturę Transformer do przewidywania właściwości molekuł (dataset HIV).

## Struktura projektu

### `Data/`
* `HIV.csv` – pełny zbiór danych.
* `HIV_test.csv` – zbiór testowy.
* `HIV_train.csv` – zbiór treningowy.
* `HIV_validation.csv` – zbiór walidacyjny.
* `HIVDataset.py` – manager ładowania danych (Dataset) dla modeli PyTorch.
* `split_dataset.py` – skrypt dzielący dataset na podzbiory.

### `DataAnalysis/`
* `data_analysis.ipynb` – notebook z analizą danych.
* `tokenized_data_analysis.ipynb` – notebook z analizą danych po tokenizacji.

### `Tokenization/`
* `SMILESTokenizer.py` – klasa tokenizatora dla formatu SMILES.
* `create_vocabulary.py` – skrypt tworzący i zapisujący słownik tokenizacji.

### `Transformers/`
* `binary_focal_loss.py` – implementacja dodatkowej funkcji straty (Focal Loss).
* `GELU_transformer.py` – model Transformera wykorzystujący aktywację GELU i mechanizmy ważenia.
* `HIVMoleculeTransformer.py` – model Transformera wykorzystujący aktywację ReLU.

### Katalog główny (`/`)
* `HIVPredictor.py` – klasa pośrednicząca używana do testowania i modelu.
* `transformer_trainer.py` – klasa zawierająca logikę pętli treningowej.
* `train_transformer_main.py` – główny skrypt uruchamiający trening.
* `test_transformer.py` – skrypt do testowania wytrenowanego modelu.

---

## Środowisko wirtulane 
Można utworzyć środowisko wirtualne dla projektu
```bash
  python -m venv venv_sciezka
```

Po aktywacji środowiska wirtualnego wymagane biblioteki można pobarać stosując polecenie:
```bash
  pip install -r requirements.txt
```


## Trenowanie nowego transformera

Aby wytrenować nowy model, należy uruchomić plik `train_transformer_main.py`. 
```bash
  python train_transformer_main.py
```
Konfiguracja parametrów znajduje się na początku tego pliku.


### Dostępne parametry:
* `TRAIN_PATH` – ścieżka do zbioru treningowego.
* `VALIDATION_PATH` – ścieżka do zbioru walidacyjnego.
* `NUM_HEADS` – liczba głowic (heads) w mechanizmie attention.
* `NUM_LAYERS` – liczba warstw transformera.
* `EMBEDDING_DIM` – wymiar warstwy embeddingu.
* `NUM_CLASSES` – liczba klas przewidywanych przez model.
    * Wartości: `1` (regresja/binarna z logitami) | `2` (klasyfikacja).
* `MAX_LEN` – długość wektora tokenów (domyślnie: `100`, zmiana niezalecana).
* `BATCH_SIZE` – rozmiar batchu.
* `EPOCHS` – ilość epok.
* `LEARNING_RATE` – learning rate.
* `CRITERION_NAME` – funkcja straty. Dostępne wartości:
    * `'CrossEntropyLoss'` (wymaga `NUM_CLASSES=2`)
    * `'BCEWithLogitsLoss'` (wymaga `NUM_CLASSES=1`)
    * `'FocalLoss'` (wymaga `NUM_CLASSES=1`)
* `SAVING_PATH` – ścieżka do zapisywania wytrenowanego modelu (model należy zapisywać w formacie '.pth')

---

## Testowanie transformera

Aby przetestować model, należy uruchomić plik `test_transformer.py`. 
```bash
  python test_transformer.py
```
Parametry konfiguracyjne znajdują się na początku pliku.

### Dostępne parametry:
* `TESTSET_PATH` – ścieżka do zbioru testowego.
* `MODEL_PATH` – ścieżka do zapisanego modelu.
* `OUTPUT_PLOT_PATH` – ścieżka, pod którą zostanie zapisany wykres wyników.
* `ACTIVITY_THRESHOLD` – próg (threshold) uznawania aktywności; używany do obliczania metryk klasyfikacji binarnej.

## Zmiana typu transformera:
* `train_transformer_main.py`: linie **47-65**
* `HIVPredictor.py`: linie **19-36**

### Konfiguracja:

1.  **Trenowanie:**
    Aby zmienić model używany do treningu, należy odkomentować pożądaną klasę transformera w pliku `train_transformer_main.py` (i zakomentować poprzednią).

2.  **Testowanie:**
    W przypadku testowania (inferencji), architektura zdefiniowana w pliku `HIVPredictor.py` **musi być zgodna** z typem transformera, którego wagi (plik `.pth`) są wczytywane. Niezgodność architektur spowoduje błąd ładowania modelu.