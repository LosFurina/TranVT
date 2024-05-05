from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

path = Path(__file__).parent


class Main(object):
    def __init__(self):
        pass

    def run(self, save_pattern, exp_id):
        threshold_csv_path = path.joinpath("exp", save_pattern, exp_id, "thresholds.csv")
        threshold_df = pd.read_csv(threshold_csv_path)
        print(threshold_df)
        # 绘制折线图
        plt.figure(figsize=(10, 6))
        plt.plot(threshold_df['Thresholds'], threshold_df['Precision'], label='Precision')
        plt.plot(threshold_df['Thresholds'], threshold_df['Recall'], label='Recall')
        plt.plot(threshold_df['Thresholds'], threshold_df['F1 Score'], label='F1 Score')
        plt.xlabel('Threshold')
        plt.ylabel('Scores')
        plt.title('Precision, Recall, and F1 Score vs. Threshold on WADI')
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    main = Main()
    main.run(save_pattern="", exp_id="2024-04-29_12-58")
