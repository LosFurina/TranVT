# Anomaly Detection Based on Graph Transformer via Variable and Time dimension

## 1. Introduction

This project focus on Anomaly Detection Based on Transformer via Variable and Time dimension(TranVT).

As far as current researched, vanilla Transformers on Time-series dataset always ignore the function of variable or time dimension on self-attention model.

Hence, this project mainly explore the effects of Transformer on variable and time dimension.

## 2. Usage

```bash
(torchgpu) PS D:\git\TranVT> python .\main.py --help
usage: main.py [-h] [--save_pattern -sp] [--dataset -d] [--model -m] [--lr -l] [--win_size -ws] [--g_dim -gd] [--g_out_layer_num -gon] [--g_out_layer_inter_dim -goin] [--g_top_k -gtk] [--batch_size -bs] [--epochs -e]
               [--train_ratio -tr] [--recon] [--test] [--exp_id -ei] [--top_k -t]

Time-Series Anomaly Detection on Variable and Time dimension

optional arguments:
  -h, --help            show this help message and exit
  --save_pattern -sp    the data folder path for experiment result saving
  --dataset -d          dataset from ['swat', 'wadi']
  --model -m            model name from [TranVTV, TranVTP, TranVTS]
  --lr -l               learning rate in training
  --win_size -ws        windows size in splitting dataset to type of time series
  --g_dim -gd           latent dimension of graph
  --g_out_layer_num -gon
                        OutLayer number of graph
  --g_out_layer_inter_dim -goin
                        OutLayer number of graph
  --g_top_k -gtk        TopK of Graph layer
  --batch_size -bs      batch size
  --epochs -e           epoch times
  --train_ratio -tr     Train ratio
  --recon               pred or recon pattern
  --test                test model
  --exp_id -ei          tested checkpoint id
  --top_k -t            the top k score used in evaluation algorithm
```
### 2.1. Train
- Strongly Recommend:
```bash
python random_train.py
```

### 2.2. Test
```bash
python ./test.py --test --save_pattern [you set in training] --exp_id 2024-03-24_20-09 --top_k 1
```

# 3. Author
- The owner of github.com/LosFurina