# -*- coding: utf-8 -*-
# Author：Weijun Li
# Date：2024-03-23
# E-mail：liweijun0302@gmail.com
import argparse
import yaml
import os
import pathlib
import logging
import torch
import numpy as np
import pandas as pd
import random

import src.models

from datetime import datetime
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt

from src.constant import Args
from src.save_model import save_model, load_model
from src.pot import pot_eval
from src.diagnosis import hit_att, ndcg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Main(object):

    def __init__(self):
        self.paser = argparse.Namespace
        self.set_paser()
        now = str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        self.args = Args()
        self.args.run_time = now
        self.args.dataset = self.paser.dataset
        self.args.model = self.paser.model
        self.args.is_recon = self.paser.recon
        self.args.dataset_path = os.path.join("dataset", self.args.dataset)
        self.args.config_path = os.path.join("exp", self.paser.save_pattern, now, "config.yaml")
        self.args.exp_path = os.path.join("exp", self.paser.save_pattern, now)
        self.args.lr = self.paser.lr
        self.args.win_size = self.paser.win_size
        self.args.g_dim = self.paser.g_dim
        self.args.g_out_layer_num = self.paser.g_out_layer_num
        self.args.g_out_layer_inter_dim = self.paser.g_out_layer_inter_dim
        self.args.g_top_k = self.paser.g_top_k
        self.args.temp = self.paser.temp
        self.args.temp_drop_frac = self.paser.temp_drop_frac
        self.args.batch_size = self.paser.batch_size
        self.args.epochs = self.paser.epochs
        self.args.top_k = self.paser.top_k

        if self.paser.test:
            if self.paser.save_pattern is None:
                raise Exception("No save pattern was given!")
            if self.paser.exp_id is None:
                raise Exception("No experiment id was given!")
            config_path = os.path.join("exp", self.paser.save_pattern, self.paser.exp_id, "config.yaml")
            with open(config_path, "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            self.args.run_time = config.get("run_time")
            self.args.dataset = config.get("dataset")
            self.args.model = config.get("model")
            self.is_recon = config.get("is_recon")
            self.args.dataset_path = config.get("dataset_path")
            self.args.config_path = config.get("config_path")
            self.args.save_pattern = config.get("save_pattern")
            self.args.exp_path = config.get("exp_path")
            self.args.exp_id = self.paser.exp_id
            self.args.lr = config.get("lr")
            self.args.win_size = config.get("win_size")
            self.args.g_dim = config.get("g_dim")
            self.args.g_out_layer_num = config.get("g_out_layer_num")
            self.args.g_out_layer_inter_dim = config.get("g_out_layer_inter_dim")
            self.args.g_top_k = config.get("g_top_k")
            self.args.temp = config.get("temp")
            self.args.temp_drop_frac = config.get("temp_drop_frac")
            self.args.batch_size = config.get("batch_size")
            self.args.epochs = config.get("epochs")

        else:
            should_dir = str(pathlib.Path(self.args.config_path).parent.resolve())
            if not os.path.exists(should_dir):
                os.makedirs(should_dir)

            with open(self.args.config_path, "w") as f:
                yaml.dump(self.args.__dict__, f, default_flow_style=False)

        self.writer = None
        self.logger = None
        self.logger: logging.Logger
        self.set_logger()

    def set_logger(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # 添加一个输出到控制台的处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # 添加一个输出到文件的处理器
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_handler = logging.FileHandler(os.path.join(self.args.exp_path, f"{now}.log"))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.writer = SummaryWriter(os.path.join(self.args.exp_path, f"{now}_tensor_log"))

    def set_paser(self):
        parser = argparse.ArgumentParser(description='Time-Series Anomaly Detection on Variable and Time dimension')
        parser.add_argument('--save_pattern',
                            metavar='-sp',
                            type=str,
                            required=False,
                            default='',
                            help="the data folder path for experiment result saving")
        parser.add_argument('--dataset',
                            metavar='-d',
                            type=str,
                            required=False,
                            default='swat',
                            help="dataset from ['swat', 'wadi']")
        parser.add_argument('--model',
                            metavar='-m',
                            type=str,
                            required=False,
                            default='TranVT',
                            help="model name from [TranVTV, TranVTP, TranVTS]")
        parser.add_argument('--lr',
                            metavar='-l',
                            type=float,
                            required=False,
                            default=0.0001,
                            help="learning rate in training")
        parser.add_argument('--win_size',
                            metavar='-ws',
                            type=int,
                            required=False,
                            default=10,
                            help="windows size in splitting dataset to type of time series")
        parser.add_argument('--g_dim',
                            metavar='-gd',
                            type=int,
                            required=False,
                            default=256,
                            help="latent dimension of graph")
        parser.add_argument('--g_out_layer_num',
                            metavar='-gon',
                            type=int,
                            required=False,
                            default=1,
                            help="OutLayer number of graph")
        parser.add_argument('--g_out_layer_inter_dim',
                            metavar='-goin',
                            type=int,
                            required=False,
                            default=256,
                            help="OutLayer number of graph")
        parser.add_argument('--g_top_k',
                            metavar='-gtk',
                            type=int,
                            required=False,
                            default=40,
                            help="TopK of Graph layer")
        parser.add_argument('--temp',
                            metavar='-temp',
                            type=float,
                            required=False,
                            default=1,
                            help="Temperature of GumbelSoftmax")
        parser.add_argument('--temp_drop_frac',
                            metavar='-tdf',
                            type=float,
                            required=False,
                            default=0.999,
                            help="Temperature drop frac of GumbelSoftmax")
        parser.add_argument('--batch_size',
                            metavar='-bs',
                            type=int,
                            required=False,
                            default=128,
                            help="batch size")
        parser.add_argument('--epochs',
                            metavar='-e',
                            type=int,
                            required=False,
                            default=5,
                            help="epoch times")
        parser.add_argument('--train_ratio',
                            metavar='-tr',
                            type=float,
                            required=False,
                            help="Train ratio")
        parser.add_argument('--recon',
                            action='store_true',
                            help="pred or recon pattern")

        parser.add_argument('--test',
                            action='store_true',
                            help="test model")
        parser.add_argument('--exp_id',
                            metavar='-ei',
                            type=str,
                            required=False,
                            help="tested checkpoint id")
        parser.add_argument('--top_k',
                            metavar='-t',
                            type=int,
                            required=False,
                            help="the top k score used in evaluation algorithm")
        self.paser = parser.parse_args()

    def load_dataset(self, train_ratio=0.8):
        if self.args.dataset in ["swat", "wadi", "wadi_less", "tep"]:
            raw_train_path = os.path.join(self.args.dataset_path, "train.csv")
            raw_test_path = os.path.join(self.args.dataset_path, "test.csv")

            df_train = pd.read_csv(raw_train_path)
            df_test = pd.read_csv(raw_test_path)
            df_labels = df_test["attack"]

            # Drop the first and last columns
            df_train = df_train.drop(columns=[df_train.columns[0], df_train.columns[-1]])
            df_test = df_test.drop(columns=[df_test.columns[0], df_train.columns[-1]])

            ts_train = torch.from_numpy(df_train.values)
            ts_test = torch.from_numpy(df_test.values)
            ts_labels = torch.from_numpy(df_labels.values)

            # Randomly select training data
            train_length = min(int(len(ts_train) * train_ratio), len(ts_train))
            # Select a random starting point within the dataset
            start_index = random.randint(0, len(ts_train) - train_length)
            # Select the train_length rows starting from the random start_index
            ts_train = ts_train[start_index:start_index + train_length]

            ts_train_win = Main.convert_to_windows(ts_train, self.args.win_size)
            ts_test_win = Main.convert_to_windows(ts_test, self.args.win_size)
            return ts_train, ts_test, ts_labels, ts_train_win, ts_test_win
        else:
            raise Exception("Unknown dataset")

    @staticmethod
    def convert_to_windows(data: torch.Tensor, w_size: int):
        windows = []
        for i, g in enumerate(data):
            if i >= w_size:
                w = data[i - w_size:i]
            else:
                w = torch.cat([data[0].repeat(w_size - i, 1), data[0:i]])
            windows.append(w)
        return torch.stack(windows)

    @staticmethod
    def plot_pics(np_ori: np.ndarray, np_pred: np.ndarray, args: Args, step=10):
        pdf = PdfPages(os.path.join(args.exp_path, "plotter.pdf"))
        for dim in range(np_ori.shape[1]):
            y_o, y_p = np_ori[::step, dim], np_pred[::step, dim]
            fig, ax1 = plt.subplots(1, 1, sharex=True)
            ax1.set_ylabel('Value')
            ax1.set_title(f'Dimension = {dim}')

            ax1.plot(y_o, linewidth=0.2, label='Original Value')
            ax1.plot(y_p, '-', alpha=0.6, linewidth=0.3, label='Predicted')
            ax1.legend()
            pdf.savefig(fig)
            plt.close()
        pdf.close()

    @staticmethod
    def plotter(y_true, y_pred, ascore, labels, args: Args, threshold):
        step = 20
        star_index = 100
        end_index = 30000
        y_true, y_pred, ascore, labels = y_true[star_index:end_index:step], y_pred[star_index:end_index:step], ascore[
                                                                                                               star_index:end_index:step], labels[
                                                                                                                                           star_index:end_index:step]
        pdf = PdfPages(os.path.join(args.exp_path, "plotter.pdf"))
        for dim in range(y_true.shape[1]):
            y_t, y_p, l, a_s = y_true[:, dim], y_pred[:, dim], labels, ascore[:, dim]
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            ax1.set_ylabel('Value')
            ax1.set_title(f'Dimension = {dim}')
            ax1.plot(y_t, linewidth=0.2, label='True')
            ax1.plot(y_p, '-', alpha=0.6, linewidth=0.3, label='Predicted')
            ax3 = ax1.twinx()
            ax3.plot(l, '--', linewidth=0.3, alpha=0.5)
            ax3.fill_between(np.arange(l.shape[0]), l, color='red', alpha=0.2)

            ax1.legend(ncol=2, bbox_to_anchor=(0.6, 1.02))
            ax2.plot(a_s, linewidth=0.2, color='g')
            ax2.axhline(y=threshold, color='red', linestyle='--', label=f'threshold = {threshold}')
            ax2.set_ylim(0, int(threshold)*2)
            ax2.set_xlabel('Timestamp')
            ax2.set_ylabel('Anomaly Score')
            pdf.savefig(fig)
            plt.close()
        pdf.close()

    def get_available_memory(self, device):
        if device.type == 'cuda':
            total_memory = torch.cuda.get_device_properties(device).total_memory
            reserved_memory = torch.cuda.memory_reserved(device)
            allocated_memory = torch.cuda.memory_allocated(device)
            available_memory = total_memory - (reserved_memory + allocated_memory)
        else:
            import psutil
            available_memory = psutil.virtual_memory().available
        return available_memory

    def calculate_optimal_batch_size(self, available_memory, element_size, safety_factor=0.8):
        # safety_factor is used to avoid using up all memory
        max_elements = int(available_memory * safety_factor / element_size)
        return max(1, max_elements)  # Ensure at least 1

    def train(self, train_ratio=0.8):
        # 1.Load dataset================================================================================================
        ts_train, ts_test, ts_label, ts_train_win, ts_test_win = self.load_dataset(train_ratio=train_ratio)
        self.logger.info("Load dataset finished")
        # 2. Load model=================================================================================================
        model = None
        if self.args.model == "TranVTV":
            model = src.models.TranVTV(ts_train.shape[1], self.args).double().to(device)
        elif self.args.model == "TranVTP":
            model = src.models.TranVTP(ts_train.shape[1], self.args).double().to(device)
        elif self.args.model == "TranVTS":
            model = src.models.TranVTS(ts_train.shape[1], self.args).double().to(device)
        elif self.args.model == "GTranVTV":
            model = src.models.GTranVTV(ts_train.shape[1], self.args).double().to(device)
        elif self.args.model == "GTranVTP":
            model = src.models.GTranVTP(ts_train.shape[1], self.args).double().to(device)
        elif self.args.model == "GTranVTS":
            model = src.models.GTranVTS(ts_train.shape[1], self.args).double().to(device)
        elif self.args.model == "GumbelGraphormer":
            model = src.models.GumbelGraphormer(ts_train.shape[1], self.args).double().to(device)
        else:
            raise NotImplementedError("Unknown model")

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)  # dynamic learning rate
        self.logger.info("Load model finished")
        # 3. Training===================================================================================================
        self.logger.info(f"Training {self.args.model} on {self.args.dataset}")
        self.logger.info(f"Start time: {str(datetime.now())}")

        accuracy_list = []
        num_epochs = self.args.epochs
        model.train()
        if self.args.is_recon:
            self.logger.info("Training pattern is Recon")
        else:
            self.logger.info("Training pattern is Pred")

        global global_step
        global_step = 0
        for e in tqdm(list(range(1, num_epochs + 1))):
            feats = ts_train.shape[1]
            data_x = torch.DoubleTensor(ts_train_win).to(device)
            dataset = TensorDataset(data_x, data_x)  # @TODO: reconstruction methodology
            dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)
            weight = 1  # @TODO: Change weight distribution on two model's put out
            l1s, l2s = [], []

            for d, _ in dataloader:
                # TODO: Change here Pred and Reconstruction
                local_bs = d.shape[0]
                window = d.permute(1, 0, 2).to(device)  # (20, 32, 51)
                if self.args.is_recon:
                    gd = window
                else:
                    gd = window[-1, :, :].view(1, local_bs, feats)
                z = model(window, gd)
                if not isinstance(z, tuple):
                    l1 = nn.functional.mse_loss(z, gd)
                else:
                    l1 = nn.functional.mse_loss(z[0], gd)
                l1s.append(torch.mean(l1).item())
                loss = torch.mean(l1)
                self.writer: SummaryWriter
                self.writer.add_scalar('Loss/train', loss.item(), global_step=global_step)
                global_step += 1
                model.global_step += 1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            self.logger.info(f'Epoch {e},\tL1 = {np.mean(l1s)}')

        self.logger.info("Train finished")
        self.logger.info(f"End time: {str(datetime.now())}")
        # 4. Save model=================================================================================================
        save_model(model, optimizer, scheduler, accuracy_list, self.args, self.logger)
        self.logger.info("Train finished")

    def test(self):
        # 1.Load dataset================================================================================================
        ts_train, ts_test, ts_label, ts_train_win, ts_test_win = self.load_dataset()
        ts_label = ts_label.to(device)
        self.logger.info(f"Load dataset [{self.args.dataset}] finished")
        # 2.Load model==================================================================================================
        model, _, _, _ = load_model(self.args.exp_id, self.paser.save_pattern, ts_train.shape[1],
                                    self.args)  # This 'args' is just an interface
        model = model.double().to(device)
        model.eval()
        self.logger.info(f"Load model [{self.args.model}] finished")
        # 3.Test========================================================================================================
        data_test = torch.DoubleTensor(ts_test_win).to(device)
        dataset_test = TensorDataset(data_test, data_test)  # @TODO: reconstruction methodology
        available_memory = self.get_available_memory(device)
        example_tensor = torch.DoubleTensor(1, *ts_train_win.shape[1:]).to(device)
        element_size = example_tensor.element_size() * example_tensor.numel()
        batch_size = self.calculate_optimal_batch_size(available_memory, element_size)
        self.logger.info(f"Available memory is {available_memory}")
        self.logger.info(f"Element size is {element_size}")
        self.logger.info(f"Optimized batch size is {batch_size}")
        dataloader_test = DataLoader(dataset_test, batch_size=batch_size)

        data_train = torch.DoubleTensor(ts_train_win).to(device)
        dataset_train = TensorDataset(data_train, data_train)  # @TODO: reconstruction methodology
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size)

        dataloader = {
            "train": dataloader_train,
            "test": dataloader_test
        }

        los_f = nn.MSELoss(reduction='none')
        with torch.no_grad():
            pred_1 = {}
            loss_1 = {}
            self.logger.info("Test dataset is going through per-trained model")
            for k, v_da in dataloader.items():
                if k == "train":
                    continue
                # v_da: value of dataloader
                output1 = []
                # output2 = []
                for d, _ in tqdm(v_da):
                    local_bs = d.shape[0]
                    window = d.permute(1, 0, 2)
                    gd = window[-1, :, :].view(1, local_bs, ts_test.shape[1])
                    z = model(window, gd)
                    output1.append(z[0].reshape(-1, ts_test.shape[1]))  # first output, every single batch
                    # output2.append(z[1].reshape(-1, ts_test.shape[1]))

                mod_out1 = torch.cat(output1, dim=0)
                pred_1[k] = mod_out1.detach().cpu().numpy()
                loss_1[k] = los_f(mod_out1, eval(f"ts_{k}").double().to(device)).detach().cpu().numpy()
                # loss_2.append(los_f(mod_out2, eval(f"ts_{i}").double().to(device)).detach().cpu().numpy())
                del output1

        ts_label = ts_label.detach().cpu().numpy()
        ts_test = ts_test.detach().cpu().numpy()
        ts_train = ts_train.detach().cpu().numpy()
        # 4.Anomaly detection from TranAD===============================================================================
        # self.logger.info("Anomaly Detection on SPOT algorithm")
        # df_loss = pd.DataFrame()
        # df_pred = pd.DataFrame()
        # for i in range(loss_1["train"].shape[1]):
        #     loss_train, loss_test, label = loss_1["train"][:, i], loss_1["test"][:, i], ts_label
        #     pred_train, pred_test, label = pred_1["train"][:, i], pred_1["test"][:, i], ts_label
        #     result_los, pred_los = pot_eval(loss_train, loss_test, label)
        #     result_pre, pred_pre = pot_eval(pred_train, pred_test, label)
        #     df_loss = pd.concat([df_loss, pd.DataFrame([result_los])], ignore_index=True)
        #     df_pred = pd.concat([df_pred, pd.DataFrame([result_pre])], ignore_index=True)
        #
        # loss_train_mean, loss_test_mean = np.mean(loss_1["train"], axis=1), np.mean(loss_1["test"], axis=1)
        #
        # result, _ = pot_eval(loss_train_mean, loss_test_mean, ts_label)
        # result.update(hit_att(loss_1["test"], ts_label))
        # result.update(ndcg(loss_1["test"], ts_label))
        # self.logger.info("Anomaly Detection on SPOT algorithm loss pattern")
        # self.logger.info(df_loss)
        # df_loss.to_csv(os.path.join(f"{self.args.exp_path}", "loss-pattern.csv"), index=True)
        # self.logger.info("Anomaly Detection on SPOT algorithm pred pattern")
        # self.logger.info(df_pred)
        # df_pred.to_csv(os.path.join(f"{self.args.exp_path}", "pred-pattern.csv"), index=True)
        # self.logger.info("Anomaly Detection on SPOT algorithm loss mean pattern")
        # self.logger.info(result)
        # 5.Anomaly detection from topK
        self.logger.info("Anomaly Detection on Deviation algorithm")
        from src.topk import get_best_f1_score
        test_result = [pred_1["test"], ts_test, ts_label]
        # val_result = [pred_1["train"], ts_train, ts_label]
        res = get_best_f1_score(test_result, test_result, self.logger, self.args, top_k=self.args.top_k)
        # 6. Draw plot
        Main.plotter(ts_test, pred_1["test"], res[7], ts_label, self.args, res[4])


if __name__ == '__main__':
    main = Main()
    main.test()
