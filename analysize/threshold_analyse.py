from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

path = Path(__file__).parent.parent


class Main(object):
    def __init__(self):
        pass

    def run(self, save_pattern_1, exp_id_1, save_pattern_2, exp_id_2):
        threshold_csv_path_1 = path.joinpath("exp", save_pattern_1, exp_id_1, "thresholds.csv")
        threshold_df_1 = pd.read_csv(threshold_csv_path_1)
        threshold_csv_path_2 = path.joinpath("exp", save_pattern_2, exp_id_2, "thresholds.csv")
        threshold_df_2 = pd.read_csv(threshold_csv_path_2)
        print(threshold_df_1)
        # 绘制折线图
        plt.figure(figsize=(10, 6))
        plt.plot(threshold_df_1['Thresholds'], threshold_df_1['Precision'], label='Precision')
        plt.plot(threshold_df_1['Thresholds'], threshold_df_1['Recall'], label='Recall')
        plt.plot(threshold_df_1['Thresholds'], threshold_df_1['F1 Score'], label='F1 Score')
        plt.xlabel('Threshold')
        plt.ylabel('Scores')
        plt.title('Precision, Recall, and F1 Score vs. Threshold on WADI')
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    main = Main()
    main.run(save_pattern_1="wadi_less_win90_TranVTP_recon_1", exp_id_1="2024-05-13_01-08-43", save_pattern_2="swat_win", exp_id_2="2024-05-04_15-41")
