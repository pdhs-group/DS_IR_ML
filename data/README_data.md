# Degree of substitution of cellulose acetate and other esters: Raw ATR-FTIR spectra and DS data obtained from IR integration, 1H NMR and 31P NMR  

## Authors
Frank Rhein, Timo Sehn, Michael A.R. Meier

## About
This repository contains the raw data used in *"Efficient and accurate determination of the degree of substitution of cellulose acetate using ATR-FTIR spectrosopy and machine learning"*. For detailed information please refer to the original publication upon acceptance.

## Datasets
The following data sets are compiled in this repository and published together with raw ATR-FTIR spectra:
- `Wolfs.A`: DOI https://doi.org/10.1002/pol.20230220 
    - Raw ATR-FTIR data
    - DS data from IR integration, 1H NMR, 31P NMR methods
- `Wolfs.B`: DOI https://doi.org/10.1039/D1GC01508G
    - Raw ATR-FTIR data
    - DS data from 1H NMR, 31P NMR methods
- `Sehn.A`, `Sehn.B`, `Sehn.C`: DOI https://doi.org/10.1021/acs.biomac.3c00762 
    - Raw ATR-FTIR data
    - DS data from 1H NMR method

An overview is given in `data_overview.xlsx`.

## Structure
Raw ATR-FTIR data is contained in the `IR_data_..` folders, while DS values from different methods are found in the respective `.txt` files. 
> Note: `IR_DS` refers to the integration based method reported in https://doi.org/10.1002/pol.20230220  

## Methods
For more information on data generation and analytical setups please refer to the original publications referenced above. DS data based on 1H NMR data is obtained according to the newly reported evaluation routine.  

## Evaluation Scripts
The Python scripts for data analysis are available on Github under https://github.com/pdhs-group/DS_IR_ML 