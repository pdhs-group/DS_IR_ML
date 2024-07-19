# Efficient and accurate determination of the degree of substitution of cellulose acetate using ATR-FTIR spectrosopy and machine learning

## Authors
Frank Rhein, Timo Sehn, Michael A.R. Meier

## About
This repository contains the data evaluation reported in *"Efficient and accurate determination of the degree of substitution of cellulose acetate using ATR-FTIR spectrosopy and machine learning"*. 

## Data 
Data used in this repository is published under the CC BY-NC 4.0 license here: https://publikationen.bibliothek.kit.edu/1000172511 (DOI: 10.35097/tvwlylbMvDXhEcRt) and simple downloaded and extracted to the `data/` folder in this repository. 

## Reproduce the Results
All code is written in Python. The reported study is contained in `ds_ir_ml.py` and is divided into sections according to the publication. Each section can be run individually by setting the corresponding boolean values in lines 25-30. 

More detailed calculations and the `config` class are contained in `mod/util_functions.py`. In `config`, the default evaluation parameters can be set. Most importantly `N_REPS`, defining the number of repetitions. The study uses `1000`, however for quick access choose a smaller value. Note that exact results are stochastic in nature (random test/train split).

A custom plotter function is used and provided in `mod/custom_plotter.py`.
