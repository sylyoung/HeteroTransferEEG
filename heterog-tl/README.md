# Heterogeneous Cross-Device Transfer for EEG Classification

## Steps for Usage:

#### 1. Download Datasets

To download datasets, run   
```sh 
python ./download_data.py
```   

For Blankertz2007 (MI1) dataset, you have to download it yourself, see and run
```sh 
python ./process_MI.py
```   

You might encounter problem with Schirrmeister2017 (HighGamma) dataset, in that case you have to download it yourself, see and run
```sh 
python ./process_HighGamma.py
```   

#### 2. Proprocess Datasets

Since since datasets are too big, we prealign data using Euclidean alignment (EA) so we do not have to do it per every transfer run/task.

For MI datasets
```sh 
python ./pre_EA_align_mi.py
```   

and For P300 datasets
```sh 
python ./pre_EA_align_p300.py
```   

Note that the downloaded data and preprocessed data in step 1 uses ./data/ path, while EA alignment and split data uses another path location of your specification (see the two pre_EA files)

#### 3. Run Code

TODO: will update soon

To test any of the algorithm, run from the main folder at this level, e.g.:
```sh 
python ./ml/*.py
```
or 
```sh 
python ./nn/*.py
```   

#### Name Correspondence for Datasets used in the Paper
Set-A  
Tangermann2012: BNCI2014001 (also known as BCI Competition IV-2a) (MOABB)  
Leeb2007: BNCI2014004 (MOABB)  
Blankertz2007: MI1 (also known as BCI Competition IV-1) (BCI Competition)  
Schirrmeister2017: HighGamma (MOABB or self-download)  
Yi2014: Weibo2014 (The author name was Weibo Yi) (MOABB)  

Set-B  
Tangermann2012: BNCI2014001 (MOABB)  
Faller2012: BNCI2015001 (MOABB)  
Schirrmeister2017: HighGamma (MOABB or self-download)  
Yi2014: Weibo2014 (MOABB)  
Zhou2016: Zhou2016 (MOABB)  

P300  
Arico2014: BNCI2014008 (MOABB)  
Guger2009: BNCI2014009 (MOABB)  
Hoffmann2008: BNCI2015003 (MOABB)  
Riccio2013: EPFLP300 (MOABB)  