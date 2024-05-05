import logging
import numpy
import numpy as np
import pandas as pd

from scipy.stats import rankdata, iqr, trim_mean
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, mean_squared_error


def get_err_median_and_iqr(predicted, groundtruth):
    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))

    err_median = np.median(np_arr)
    err_iqr = iqr(np_arr)

    return err_median, err_iqr


def get_err_scores(test_res, j, if_print, attack_index_list, norm_index_list):
    test_predict, test_gt = test_res
    # val_predict, val_gt = val_res
    # print("GDN/eva test_predict: ", test_predict)
    # print("GDN/eva test_predict.shape: ", test_predict.shape)
    n_err_mid, n_err_iqr = get_err_median_and_iqr(test_predict, test_gt)

    test_delta = np.abs(np.subtract(
        np.array(test_predict).astype(np.float64),
        np.array(test_gt).astype(np.float64)
    ))
    epsilon = 1e-2
    # TODO:这里的epsilon我改过 1e-2
    # if(if_print):
    #     print("eva Sensor", j,"delta:", '%.4f'%np.mean(test_delta[20000:40000]))
    #     print("GDN/eva test_gt: ",'%.4f'%np.mean(test_gt[20000:40000]))
    # print("GDN/eva test_delta: ", test_delta)
    # print("GDN/eva test_delta.shape: ", len(test_delta))
    err_scores = (test_delta - n_err_mid) / (np.abs(n_err_iqr) + epsilon)
    # print(err_scores.shape)

    smoothed_err_scores = np.zeros(err_scores.shape)
    before_num = 3
    for i in range(before_num, len(err_scores)):
        smoothed_err_scores[i] = np.mean(err_scores[i - before_num:i + 1])

    if if_print:
        return smoothed_err_scores, np.mean([test_delta[i] for i in attack_index_list]), np.mean(
            [test_delta[i] for i in norm_index_list])  # 返回所有的attack误差和normal误差
    return smoothed_err_scores, 0, 0


def eval_scores(scores, true_scores, th_steps, return_thresold=False):
    # print(scores)
    padding_list = [0] * (len(true_scores) - len(scores))
    # print(padding_list)

    if len(padding_list) > 0:
        scores = padding_list + scores

    scores_sorted = rankdata(scores, method='ordinal')
    th_steps = th_steps
    # th_steps = 500
    th_vals = np.array(range(th_steps)) * 1.0 / th_steps
    fmeas = [None] * th_steps
    thresholds = [None] * th_steps
    for i in range(th_steps):
        cur_pred = scores_sorted > th_vals[i] * len(scores)

        fmeas[i] = f1_score(true_scores, cur_pred)

        score_index = scores_sorted.tolist().index(int(th_vals[i] * len(scores) + 1))
        thresholds[i] = scores[score_index]

    if return_thresold:
        return fmeas, thresholds
    return fmeas


def get_best_performance_data(total_err_scores, gt_labels, topk=1, offset=0, is_pearson=True):
    #  total_err_scores： 平滑后的异常分数[[sen0],[sen1],...] @test
    total_features = total_err_scores.shape[0]

    correlation_coefficients = np.corrcoef(total_err_scores.T, gt_labels, rowvar=False)[:total_features, -1]
    # TODO: 有待检验,这里换成一部分test集来获取权重列表即可

    # 使用皮尔逊相关系数作为权重来更新 total_err_score

    weighted_total_err_score = total_err_scores * correlation_coefficients[:, np.newaxis]

    if is_pearson:
        total_err_scores = weighted_total_err_score
    else:
        total_err_scores = total_err_scores
    # topk_indices = np.argpartition(total_err_scores, range(total_features-1-topk, total_features-1), axis=0)[-topk-1:-1]
    topk_indices = np.argpartition(total_err_scores, range(total_features - topk - 1 - offset, total_features - offset),
                                   axis=0)[-topk:]
    # 找到top的索引

    total_topk_err_scores = []
    topk_err_score_map = []

    total_topk_err_scores = np.sum(np.take_along_axis(weighted_total_err_score, topk_indices, axis=0), axis=0)
    # print("total_topk_err_scores:",total_topk_err_scores.shape)
    '''计算数组 total_err_scores 中最大的 topk 个元素的总和。
    它使用 np.take_along_axis 函数来根据索引数组 topk_indices 从 total_err_scores 中获取对应的元素值，
    并使用 np.sum 函数对这些元素进行求和操作。最终得到的结果是一个一维数组 total_topk_err_scores,
    其中每个元素是对应时间步上最大的 topk 个元素的总和。'''

    df_temp = pd.DataFrame(total_topk_err_scores)
    # TODO: total score save hare
    # df_temp.to_csv('./test_result/totalscore.csv')

    final_topk_fmeas, thresolds = eval_scores(total_topk_err_scores, gt_labels, 400, return_thresold=True)
    # final_topk_fmeas 是以1/400粒度测试的阈值列表中，每个阈值对应的f1
    th_i = final_topk_fmeas.index(max(final_topk_fmeas))
    thresold = thresolds[th_i]

    pred_labels = np.zeros(len(total_topk_err_scores))
    pred_labels[total_topk_err_scores > thresold] = 1

    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i])
        gt_labels[i] = int(gt_labels[i])

    pre = precision_score(gt_labels, pred_labels)
    rec = recall_score(gt_labels, pred_labels)
    auc_score = roc_auc_score(gt_labels, total_topk_err_scores)
    my_auc = roc_auc_score(gt_labels, pred_labels)
    my_f1 = f1_score(gt_labels, pred_labels)

    # return max(final_topk_fmeas), pre, rec, auc_score, thresold
    # return f1, pre, rec, auc_score, thresold, total_topk_err_scores
    return_score = np.repeat(np.expand_dims(total_topk_err_scores, axis=1), total_features, axis=1)
    return max(final_topk_fmeas), pre, rec, auc_score, thresold, my_f1, my_auc, return_score


def get_full_err_scores(test_result):
    # np_test_result = np.array(test_result)
    # np_val_result = np.array(val_result)

    all_scores = None
    all_normals = None
    # feature_num = np_test_result.shape[-1]
    # print(" test_result[0].shape", test_result[0].shape)    # test_result[0].shape (44956, 51)
    feature_num = test_result[0].shape[1]
    labels = test_result[2].tolist()
    # labels = np_test_result[2, :, 0].tolist()
    attack_list = []
    norm_list = []

    attack_index_list = [i for i, x in enumerate(labels) if
                         x == 1]  # x==0/1 输出attack的偏差均值/normal的偏差均值   #label中attack的索引列表
    norm_index_list = [i for i, x in enumerate(labels) if x == 0]
    all_print_score = []
    for i in range(feature_num):
        # print("processing channel_",i)
        test_re_list = np.array([test_result[0][:, i], test_result[1][:, i]])  # 对每个传感器,计算其err序列    [[预测值],[实际值]]
        # val_re_list = np.array([val_result[0][:, i], val_result[1][:, i]])
        # test_re_list = np_test_result[:2,:,i]      #对每个传感器,计算其err序列
        # val_re_list = np_val_result[:2,:,i]

        scores, attack_delta, norm_delta = get_err_scores(test_re_list, i, 1, attack_index_list,
                                                          norm_index_list)  # 返回平滑后的异常分数、未平滑的所有attack的err均值，未平滑的所有 norm的err均值
        attack_list.append(attack_delta)
        norm_list.append(norm_delta)
        all_print_score.append(scores)
        # normal_dist, temp1, temp2 = get_err_scores(val_re_list, i, 0, attack_index_list,
        #                                            norm_index_list)  # 返回val集中平滑后的异常分数

        if all_scores is None:
            all_scores = scores
            # all_normals = normal_dist
        else:
            all_scores = np.vstack((  # 把所有channel的scores堆叠起来，[[sen0],[sen1],...]
                all_scores,
                scores
            ))
            all_normals = np.vstack((
                all_normals,
                # normal_dist
            ))
    # for i in range(100):
    #     print("eva sequence", i, "label:", labels[i])
    # delta_list1 = np.array(attack_list).transpose()
    # delta_list1 = pd.DataFrame(delta_list1)
    # delta_list1.to_csv('/home/jpf/lzh/GDN/delta_attack_GDN.csv')
    # delta_list2 = np.array(norm_list).transpose()
    # delta_list2 = pd.DataFrame(delta_list2)
    # delta_list2.to_csv('/home/jpf/lzh/GDN/delta_norm_GDN.csv')
    # df = pd.DataFrame(all_print_score)
    # TODO: change path of smd_scores.csv
    # df.to_csv('./test_result/smd_scores.csv')
    # return all_scores, all_normals  # [[sen0],[sen1],...] @test ,  [[sen0],[sen1],...] @val
    return all_scores


def get_best_f1_score(test_result, val_result, logger: logging.Logger, top_k=1):
    # feature_num = len(test_result[0][0])
    # np_test_result = np.array(test_result)
    test_labels: numpy.ndarray = test_result[2]  # ground_truth_label from test dataset
    # np_val_result = np.array(val_result)

    test_labels = test_labels.tolist()
    # TODO: 可以在这里加入马氏距离试一下
    test_scores = get_full_err_scores(test_result)

    # @TODO: change offset of topK here
    logger.info(f"Top k is now {top_k}")
    top1_best_info = get_best_performance_data(test_scores, test_labels, topk=top_k, offset=0)
    # top1_val_info = get_val_performance_data(test_scores, normal_scores, test_labels, topk=1)

    logger.info("Result from top_k algorithm")

    # info = None
    # if self.env_config['report'] == 'best':
    info = top1_best_info
    # elif self.env_config['report'] == 'val':
    #     info = top1_val_info

    logger.info(f'F1 score:\t{info[0]}')
    logger.info(f'precision:\t{info[1]}')
    logger.info(f'recall:\t{info[2]}')
    logger.info(f'auc:\t{info[3]}')
    logger.info(f'My auc:\t{info[6]}')
    logger.info(f"My F1 score:\t{info[5]}")
    return info[0], info[1], info[2], info[3], info[4], info[5], test_labels, info[7]
    # info: return f1, pre, rec, auc_score, thresold, total_topk_err_scores
