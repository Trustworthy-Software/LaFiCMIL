# LaFiCMIL

LaFiCMIL: Rethinking Large File Classification from the Perspective of Correlated Multiple Instance Learning

## Environment Setup

- Python 3.7.11
- numpy 1.21.6
- torch 1.12.1
- torchvision 0.2.2
- torchmetrics 0.3.2
- pytorch_lightning 1.3.7
- tensorboard 2.9.1
- transformers 4.14.1
- nystrom_attention 0.0.11
- scikit-learn 1.0.2

## Prepare the datasets

### Hyperpartisan News Detection 

* Available at <https://zenodo.org/record/1489920#.YLferh1Olc8>
* Download the datasets

```
mkdir data/hyperpartisan
wget -P data/hyperpartisan/ https://zenodo.org/record/1489920/files/articles-training-byarticle-20181122.zip
wget -P data/hyperpartisan/ https://zenodo.org/record/1489920/files/ground-truth-training-byarticle-20181122.zip
unzip data/hyperpartisan/articles-training-byarticle-20181122.zip -d data/hyperpartisan
unzip data/hyperpartisan/ground-truth-training-byarticle-20181122.zip -d data/hyperpartisan
rm data/hyperpartisan/*zip
```
  
*  Prepare the datasets with the resulting xml files and this preprocessing script (following [Longformer](https://arxiv.org/abs/2004.05150)): <https://github.com/allenai/longformer/blob/master/scripts/hp_preprocess.py>

### 20NewsGroups

* Originally available at <http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz>
* Running `train.py` with the `--data 20news` flag will download and prepare the data available via `sklearn.datasets` (following [CogLTX](https://proceedings.neurips.cc/paper/2020/file/96671501524948bc3937b4b30d0e57b9-Paper.pdf)).
We adopt the train/dev/test split from [this ToBERT paper](https://ieeexplore.ieee.org/document/9003958).
  
### EURLEX-57K

* Available at <https://github.com/iliaschalkidis/lmtc-emnlp2020>
* Download the datasets

```
mkdir data/EURLEX57K
wget -O data/EURLEX57K/datasets.zip http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K/datasets.zip
unzip data/EURLEX57K/datasets.zip -d data/EURLEX57K
rm data/EURLEX57K/datasets.zip
rm -rf data/EURLEX57K/__MACOSX
mv data/EURLEX57K/dataset/* data/EURLEX57K
rm -rf data/EURLEX57K/dataset
wget -O data/EURLEX57K/EURLEX57K.json http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K/eurovoc_en.json
```

* Running `train.py` with the `--data eurlex` flag reads and prepares the data from `data/EURLEX57K/{train, dev, test}/*.json` files
* Running `train.py` with the `--data eurlex --inverted` flag creates Inverted EURLEX data by inverting the order of the sections
* `data/EURLEX57K/EURLEX57K.json` contains label information.

### CMU Book Summary Dataset

* Available at <http://www.cs.cmu.edu/~dbamman/booksummaries.html>

```
wget -P data/ http://www.cs.cmu.edu/~dbamman/data/booksummaries.tar.gz
tar -xf data/booksummaries.tar.gz -C data
```

* Running `train.py` with the `--data books` flag reads and prepares the data from `data/booksummaries/booksummaries.txt`
* Running `train.py` with the `--data books --pairs` flag creates Paired Book Summary by combining pairs of summaries and their labels


##  Model Tranining & Evaluation

Here, we provide some examples for different datasets and settings.
```
python train.py --model_name laficmil-posemb --data books --batch_size 1 --epochs 100 --lr 5e-06 --num_workers 16 --hidden_size 128 --gpu 0
python train.py --model_name laficmil-no-pos --data books --pairs --batch_size 1 --epochs 100 --lr 5e-06 --num_workers 16 --hidden_size 128 --gpu 0
python train.py --model_name laficmil-posemb --data hyperpartisan --batch_size 1 --epochs 100 --lr 5e-06 --num_workers 16 --hidden_size 768 --gpu 1
python train.py --model_name laficmil-posemb --data 20news --batch_size 1 --epochs 40 --lr 5e-07 --num_workers 16 --hidden_size 768 --gpu 1 
python train.py --model_name laficmil-no-pos --data eurlex --batch_size 1 --epochs 60 --lr 5e-06 --num_workers 16 --hidden_size 128 --gpu 0
python train.py --model_name laficmil-no-pos --data eurlex --batch_size 1 --epochs 60 --lr 5e-06 --num_workers 16 --hidden_size 128 --gpu 0 --inverted 
```
