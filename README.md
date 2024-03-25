# Anomaly Detection Based on Transformer via Variable and Time dimension

## 1. Introduction

This project focus on Anomaly Detection Based on Transformer via Variable and Time dimension(TranVT).

As far as current researched, vanilla Transformers on Time-series dataset always ignore the function of variable or time dimension on self-attention model.

Hence, this project mainly explore the effects of Transformer on variable and time dimension.

## 2. Usage

```bash
usage: train.py [-h] [--dataset -d] [--model -m] [--lr -l] [--win_size -ws] [--batch_size -bs] [--epochs -e] [--test] [--exp_id -ei]

Time-Series Anomaly Detection on Variable and Time dimension

optional arguments:
  -h, --help        show this help message and exit
  --dataset -d      dataset from ['swat', 'wadi']
  --model -m        model name from [TranAD, TranVT]
  --lr -l           learning rate in training
  --win_size -ws    windows size in splitting dataset to type of time series
  --batch_size -bs  batch size
  --epochs -e       epoch times
  --test            test model
  --exp_id -ei      tested checkpoint id
  --top_k -t        topk algorithm param
```
### 2.1. Train
```bash
python ./train.py --dataset swat --model TranVT --lr 0.0001 --win_size 10 --batch_size 128 --epochs 5
```

### 2.2. Test
```bash
python ./test.py --dataset swat --model TranVT --test --exp_id 2024-03-24_20-09
```

### 2.3 Using launcher
- optional: 
```bash
go build -ldflags "-linkmode external -extldflags '-static'" -o launcher.exe .\launcher.go
```
```bash
./launcher.exe
```
- windows
```bash
./launcher.bat
```
- linux: rename the launcher.bat to launcher.sh
```bash
./launcher.sh
```

# 3. Notice

- This project is not finished, most functions has not come true
- Some names of variables are not formal

# 4. Author
- The owner of github.com/LosFurina