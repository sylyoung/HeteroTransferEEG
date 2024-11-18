# Heterogeneous Cross-Device Transfer for EEG Classification

## Steps for Usage:

#### 1. Download Datasets

To download datasets, run   
```sh 
python ./download_data.py
```   

#### 2. Run Code

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
Tangermann2012: BNCI2014001 (MOABB)  
Leeb2007: BNCI2014004 (MOABB)  
Blankertz2007: MI1  (BCI Competition)  
Schirrmeister2017: HighGamma (MOABB)  
Yi2014: Weibo2014 (The author name was Weibo Yi) (MOABB)  

Set-B  
Tangermann2012: BNCI2014001 (MOABB)  
Faller2012: BNCI2015001 (MOABB)  
Schirrmeister2017: HighGamma (MOABB)  
Yi2014: Weibo2014 (MOABB)  
Zhou2016: Zhou2016 (MOABB)  

P300  
Arico2014: BNCI2014008 (MOABB)  
Guger2009: BNCI2014009 (MOABB)  
Hoffmann2008: BNCI2015003 (MOABB)  
Riccio2013: EPFLP300 (MOABB)  