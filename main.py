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

import src.models

from datetime import datetime
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
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

        now = str(datetime.now().strftime("%Y-%m-%d_%H-%M"))
        self.exp_config_path = os.path.join(str(pathlib.Path(__file__).resolve().parents[0]), "exp", now, "config.yaml")

        self.args = Args()
        self.args.run_time = now
        self.args.dataset = self.paser.dataset
        self.args.model = self.paser.model
        self.args.dataset_path = os.path.join(str(pathlib.Path(__file__).resolve().parents[0]), "dataset",
                                              self.args.dataset)
        self.args.config_path = self.exp_config_path
        self.args.exp_path = os.path.join(str(pathlib.Path(__file__).resolve().parents[0]), "exp", now)

        self.args.lr = self.paser.lr
        self.args.win_size = self.paser.win_size
        self.args.batch_size = self.paser.batch_size
        self.args.epochs = self.paser.epochs

        self.args.top_k = self.paser.top_k

        if self.paser.test:
            if self.paser.exp_id is None:
                raise Exception("No experiment id was given!")
            config_path = os.path.join("exp", self.paser.exp_id, "config.yaml")
            with open(config_path, "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            self.args.run_time = config.get("run_time")
            self.args.dataset = config.get("dataset")
            self.args.model = config.get("model")
            self.args.dataset_path = config.get("dataset_path")
            self.args.config_path = config.get("config_path")
            self.args.exp_path = config.get("exp_path")
            self.args.exp_id = self.paser.exp_id

            self.args.lr = config.get("lr")
            self.args.win_size = config.get("win_size")
            self.args.batch_size = config.get("batch_size")
            self.args.epochs = config.get("epochs")

        else:
            should_dir = str(pathlib.Path(self.exp_config_path).parent.resolve())
            if not os.path.exists(should_dir):
                os.makedirs(should_dir)

            with open(self.exp_config_path, "w") as f:
                yaml.dump(self.args.__dict__, f, default_flow_style=False)

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

    def set_paser(self):
        parser = argparse.ArgumentParser(description='Time-Series Anomaly Detection on Variable and Time dimension')
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

    def load_dataset(self):
        if self.args.dataset in ["swat", "wadi"]:
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
    def plotter(y_true, y_pred, ascore, labels, args: Args):
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
            ax3.fill_between(np.arange(l.shape[0]), l, color='blue', alpha=0.3)
            if dim == 0:
                ax1.legend(ncol=2, bbox_to_anchor=(0.6, 1.02))
            ax2.plot(a_s, linewidth=0.2, color='g')
            ax2.set_xlabel('Timestamp')
            ax2.set_ylabel('Anomaly Score')
            pdf.savefig(fig)
            plt.close()
        pdf.close()

    def train(self):
        # 1.Load dataset================================================================================================
        ts_train, ts_test, ts_label, ts_train_win, ts_test_win = self.load_dataset()
        self.logger.info("Load dataset finished")
        # 2. Load model=================================================================================================
        model = None
        if self.args.model == "TranVTV":
            model = src.models.TranVTV(ts_train.shape[1], self.args).double().to(device)
        elif self.args.model == "TranVTP":
            model = src.models.TranVTP(ts_train.shape[1], self.args).double().to(device)
        elif self.args.model == "TranVTS":
            model = src.models.TranVTS(ts_train.shape[1], self.args).double().to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)  # dynamic learning rate
        self.logger.info("Load model finished")
        # 3. Training===================================================================================================
        self.logger.info(f"Training {self.args.model} on {self.args.dataset}")
        self.logger.info(f"Start time: {str(datetime.now())}")

        accuracy_list = []
        num_epochs = self.args.epochs
        model.train()
        for e in tqdm(list(range(1, num_epochs + 1))):
            feats = ts_train.shape[1]
            data_x = torch.DoubleTensor(ts_train_win).to(device)
            dataset = TensorDataset(data_x, data_x)  # @TODO: reconstruction methodology
            dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)
            weight = 1  # @TODO: Change weight distribution on two model's put out
            l1s, l2s = [], []

            for d, _ in dataloader:
                local_bs = d.shape[0]
                window = d.permute(1, 0, 2).to(device)
                gd = window[-1, :, :].view(1, local_bs, feats)
                z = model(window, gd)
                l1 = ((1 / weight) * nn.functional.mse_loss(z[0], gd) +
                      (1 - 1 / weight) * nn.functional.mse_loss(z[1], gd))
                l1s.append(torch.mean(l1).item())
                loss = torch.mean(l1)
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
        model, _, _, _ = load_model(self.args.exp_id, ts_train.shape[1], self.args)  # This 'args' is just an interface
        model = model.double().to(device)
        model.eval()
        self.logger.info(f"Load model [{self.args.model}] finished")
        # 3.Test========================================================================================================
        data_test = torch.DoubleTensor(ts_test_win).to(device)
        dataset_test = TensorDataset(data_test, data_test)  # @TODO: reconstruction methodology
        dataloader_test = DataLoader(dataset_test, batch_size=ts_test.shape[0] // 30)
        # In order to calculate fast, but if your ram is not big enough, please decline the batch size

        data_train = torch.DoubleTensor(ts_train_win).to(device)
        dataset_train = TensorDataset(data_train, data_train)  # @TODO: reconstruction methodology
        dataloader_train = DataLoader(dataset_train, batch_size=ts_train.shape[0] // 30)

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
        Main.plotter(ts_test, pred_1["test"], loss_1["test"], ts_label, self.args)
        # 4.1.Anomaly detection from TranAD=============================================================================
        self.logger.info("Anomaly Detection on SPOT algorithm")
        df_loss = pd.DataFrame()
        df_pred = pd.DataFrame()
        for i in range(loss_1["train"].shape[1]):
            loss_train, loss_test, label = loss_1["train"][:, i], loss_1["test"][:, i], ts_label
            pred_train, pred_test, label = pred_1["train"][:, i], pred_1["test"][:, i], ts_label
            result_los, pred_los = pot_eval(loss_train, loss_test, label)
            result_pre, pred_pre = pot_eval(pred_train, pred_test, label)
            df_loss = pd.concat([df_loss, pd.DataFrame([result_los])], ignore_index=True)
            df_pred = pd.concat([df_pred, pd.DataFrame([result_pre])], ignore_index=True)

        loss_train_mean, loss_test_mean = np.mean(loss_1["train"], axis=1), np.mean(loss_1["test"], axis=1)

        result, _ = pot_eval(loss_train_mean, loss_test_mean, ts_label)
        result.update(hit_att(loss_1["test"], ts_label))
        result.update(ndcg(loss_1["test"], ts_label))
        self.logger.info("Anomaly Detection on SPOT algorithm loss pattern")
        self.logger.info(df_loss)
        df_loss.to_csv(os.path.join(f"{self.args.exp_path}", "loss-pattern.csv"), index=True)
        self.logger.info("Anomaly Detection on SPOT algorithm pred pattern")
        self.logger.info(df_pred)
        df_pred.to_csv(os.path.join(f"{self.args.exp_path}", "pred-pattern.csv"), index=True)
        self.logger.info("Anomaly Detection on SPOT algorithm loss mean pattern")
        self.logger.info(result)
        # 4.2.Anomaly detection from topK
        self.logger.info("Anomaly Detection on topK algorithm")
        from src.topk import get_best_f1_score
        test_result = [pred_1["test"], ts_test, ts_label]
        val_result = [pred_1["train"], ts_train, ts_label]
        get_best_f1_score(test_result, val_result, self.logger, top_k=self.args.top_k)


if __name__ == '__main__':
    main = Main()
    main.test()
