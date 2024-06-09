import os

import pandas as pd


def get_feature_map(dataset_path: str) -> list[str]:
    """

    :param dataset_path: The path of the training dataset
    :return:
    """
    df_dataset = pd.read_csv(dataset_path)
    feature_list = df_dataset.columns[1:-1].to_list()
    return feature_list


# graph is 'fully-connect'
def get_fc_graph_struc(dataset_path: str) -> dict[str, list[str]]:
    """

    :param dataset_path: The path of the training dataset (e.t. train.csv)
    :return:
    """
    df_dataset = pd.read_csv(dataset_path)
    feature_list = df_dataset.columns[1:-1].to_list()
    struc_map = {}
    for ft in feature_list:
        if ft not in struc_map:
            struc_map[ft] = []

        for other_ft in feature_list:
            if other_ft is not ft:
                struc_map[ft].append(other_ft)

    return struc_map


def build_loc_net(struc: dict[str, list[str]], all_features: list[str]):
    index_feature_map = all_features
    edge_indexes = [
        [],
        []
    ]
    for node_name, node_list in struc.items():
        if node_name not in all_features:
            continue

        if node_name not in index_feature_map:
            index_feature_map.append(node_name)

        p_index = index_feature_map.index(node_name)
        for child in node_list:
            if child not in all_features:
                continue

            if child not in index_feature_map:
                print(f'error: {child} not in index_feature_map')
                # index_feature_map.append(child)

            c_index = index_feature_map.index(child)
            # edge_indexes[0].append(p_index)
            # edge_indexes[1].append(c_index)
            edge_indexes[0].append(c_index)
            edge_indexes[1].append(p_index)

    return edge_indexes


if __name__ == '__main__':
    get_fc_graph_struc(dataset_path=os.path.abspath("D:\\git\\TranVT\\dataset\\swat\\train.csv"))
    feature_map = get_feature_map(dataset_path=os.path.abspath("D:\\git\\TranVT\\dataset\\swat\\train.csv"))
    fc_struc = get_fc_graph_struc(dataset_path=os.path.abspath("D:\\git\\TranVT\\dataset\\swat\\train.csv"))
    fc_edge_index = build_loc_net(struc=fc_struc, all_features=feature_map)
    import torch
    fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)
