from operator import index
from pathlib import Path
import pdb
import shutil
from typing import List
import numpy as np
import os
from sklearn import metrics, datasets
import pandas as pd
from .constant import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from pandas import DataFrame


def save_csv_report(report, output_dir, class_names, pos_thresh=0.50):
    """
    class precision recall f1-score support
    OK      ...      ...      ...      ...
    NoOK    ...      ...      ...      ...
    accuracy
    macro
    weighted 

    Args:
        report (_type_): _description_
        output_dir (_type_): _description_
        class_names (_type_): _description_

    Returns:
        _type_: _description_
    """
    repdf = pd.DataFrame(report).transpose()
    repdf.insert(loc=0,
                 column='class',
                 value=class_names + ["accuracy", "macro avg", "weighted avg"])
    repdf['pos_thresh'] = pos_thresh
    repdf.to_csv(output_dir, index=False, float_format='%.4f')
    return repdf


def save_csv_confusion_matrix(confusion_matrix, output_dir, class_names):
    cm = pd.DataFrame(confusion_matrix, columns=class_names)
    cm.insert(loc=0, column=' ', value=class_names)
    cm.to_csv(output_dir, index=False)
    return cm


def vis_confusion_matrix(labels, scores, output_dir, display_labels):
    metrics.ConfusionMatrixDisplay.from_predictions(
        labels, scores, display_labels=display_labels)
    plt.plot()
    plt.show()
    plt.savefig(output_dir)
    plt.close()
    plt.clf()


def save_false_negative(data_folder, pred_data, gt_data, class_names,
                        output_dir):
    """select the false negative samples based on predictions and ground truth

    Args:
        data_folder (_type_): _description_
        pred_data (_type_): _description_
        gt_data (_type_): _description_
        class_names (_type_): _description_
        output_dir (_type_): _description_
    """
    for class_name in class_names[1:]:
        selected_pred = pred_data[gt_data[class_name] == 1]
        false_positive = selected_pred[selected_pred['pred_class'] == 'OK']
        file_names = false_positive['filename']
        class_output_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)
        for filename in file_names:
            file_path = os.path.join(data_folder, filename)
            shutil.copy(file_path, class_output_dir)
            print(f'Copied file {file_path} to {class_output_dir}')
        print(f'{class_name}: {false_positive.shape[0]}')


def generate_report(pred_data, gt_data, pred_filename, output_dir):
    fn_output_dir = os.path.join(output_dir, 'fn')
    os.makedirs(fn_output_dir, exist_ok=True)
    # Exclude first column (filename), and calculate the the prediction results, which is appended to the last column.
    pred_data = pred_data.copy()
    gt_data = gt_data.copy()
    pred_data['pred_class'] = pred_data.iloc[:, 1:].idxmax(axis=1)
    # print(pred_data[:10])
    # pred_filename = os.path.splitext(pred_file)[0]

    # save_false_negative(hparams.data_folder, pred_data, gt_data,  class_names, fn_output_dir)

    # Read ground truth labels from test data.
    # The label format is one-hot vector.
    gt_labels = gt_data.iloc[:, 1:].to_numpy()
    pred_labels = pred_data.iloc[:, 1:-1].to_numpy()

    # N * 2
    thresh_pred_labels = np.concatenate(
        [pred_labels[:, [0]], pred_labels[:, 1:].sum(axis=1, keepdims=True)],
        axis=1)

    # Convert one-hot to class label.
    gt_labels = np.argmax(gt_labels, axis=1)
    pred_labels = np.argmax(pred_labels, axis=1)

    # Convert to binary classification
    bn_gt_labels = gt_labels.copy()
    bn_pred_labels = pred_labels.copy()

    # All the labels of bad eggs are set to 1
    # 0 represents good eggs
    # 1 represents bad eggs
    bn_gt_labels[bn_gt_labels != 0] = 1
    bn_pred_labels[bn_pred_labels != 0] = 1

    th_bn_reports = []
    print(f'Processing binary classification report')
    for th in THRESH:
        print(f'Report by positive threshold {th}')
        # N
        # try:
        # th_pred_labels = np.argmax((thresh_pred_labels > th).astype(int), axis=1)
        th_pred_labels = (~(thresh_pred_labels[:, 0] > th)).astype(int)
        # th_pred_labels[th_pred_labels!=0] = 1
        th_output_dir = os.path.join(output_dir, 'th')
        os.makedirs(th_output_dir, exist_ok=True)
        th_confusion_matrix = metrics.confusion_matrix(bn_gt_labels,
                                                       th_pred_labels)
        # print(bn_gt_labels)
        # print(th_pred_labels)
        th_report = metrics.classification_report(bn_gt_labels,
                                                  th_pred_labels,
                                                  target_names=BN_CLASS_NAMES,
                                                  output_dict=True)
        # th_report['pos_threshold'] = th
        bn_report_path = os.path.join(
            th_output_dir, f'TH_Report_{pred_filename}_th_{th}.csv')
        th_report = save_csv_report(th_report,
                                    bn_report_path,
                                    BN_CLASS_NAMES,
                                    pos_thresh=th)
        th_bn_reports.append(th_report)
        print(f'Report by positive threshold {th} saved into {bn_report_path}')
        # bn_reports['pos_threshold'] = th
        save_csv_confusion_matrix(
            th_confusion_matrix,
            os.path.join(th_output_dir, f'TH_CM_{pred_filename}_th_{th}.csv'),
            BN_CLASS_NAMES)
        vis_confusion_matrix(
            bn_gt_labels, th_pred_labels,
            os.path.join(th_output_dir, f'TH_CM_{pred_filename}_th_{th}.png'),
            VIS_BINARY_LABELS)
        # except Exception as e:
        # print(f'Error while handling {th}')
        # traceback.print_exc()
    th_bn_dfs: pd.DataFrame = pd.concat(th_bn_reports).sort_values(
        by=['class', 'pos_thresh'], ascending=False)
    th_bn_dfs.to_csv(os.path.join(output_dir,
                                   f'TH_Report_summary_{pred_filename}.csv'),
                      index=False,
                      float_format='%.4f')

    confusion_matrix = metrics.confusion_matrix(gt_labels, pred_labels)
    report = metrics.classification_report(gt_labels,
                                           pred_labels,
                                           target_names=CLASS_NAMES,
                                           output_dict=True)

    bn_confusion_matrix = metrics.confusion_matrix(bn_gt_labels,
                                                   bn_pred_labels)
    bn_report = metrics.classification_report(bn_gt_labels,
                                              bn_pred_labels,
                                              target_names=BN_CLASS_NAMES,
                                              output_dict=True)

    report = save_csv_report(report,
                    os.path.join(output_dir, f'Report_{pred_filename}.csv'),
                    CLASS_NAMES)
    save_csv_confusion_matrix(
        confusion_matrix, os.path.join(output_dir, f'CM_{pred_filename}.csv'),
        CLASS_NAMES)
    vis_confusion_matrix(gt_labels, pred_labels,
                         os.path.join(output_dir, f'CM_{pred_filename}.png'),
                         VIS_ALL_LABELS)

    bn_report = save_csv_report(bn_report,
                    os.path.join(output_dir, f'BN_Report_{pred_filename}.csv'),
                    BN_CLASS_NAMES)
    save_csv_confusion_matrix(
        bn_confusion_matrix,
        os.path.join(output_dir, f'BN_CM_{pred_filename}.csv'), BN_CLASS_NAMES)
    vis_confusion_matrix(
        bn_gt_labels, bn_pred_labels,
        os.path.join(output_dir, f'BN_CM_{pred_filename}.png'), VIS_BINARY_LABELS)
    return report, bn_report, th_bn_reports

    # df = pd.DataFrame.from_dict(report)
    # df.to_csv(f'Report_{filename}.csv')
    # classification_report_csv(report, os.path.join('outputs',f'Report_{filename}.csv'))

def remove_rows(df: pd.DataFrame, filter_values):
    for filter_value in filter_values:
        df.drop(df.index[df['class'] == filter_value], inplace=True)
    return df

def compose_report(hparams, report, output_path, match_pattern, embedding_rank=-1, filter_values=['weighted avg', 'macro avg', 'accuracy']):
    report.insert(loc=0,
                column='Rank',
                value=embedding_rank)
    report = remove_rows(report, filter_values)
    bn_reports = [report]
    match_paths = list(Path(hparams.log_dir).glob(match_pattern))
    for rank, path in enumerate(match_paths):
        match_report = pd.read_csv(path)
        match_report.insert(loc=0,
                    column='Rank',
                    value=rank)
        match_report = remove_rows(match_report, filter_values)
        bn_reports.append(match_report)
    df: pd.DataFrame = pd.concat(bn_reports).sort_values(
        by=['class', 'precision', 'Rank'], ascending=False)
    df.to_csv(output_path, index=False, float_format='%.4f')
    return df

def handle_single_test_set(hparams, test_set_idx, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    pred_results = []
    for path in list(Path(hparams.log_dir).glob(f'fold*/test/set/{test_set_idx}/pred.csv')):
        pred_df = pd.read_csv(path).sort_values(by=['filename'])
        pred = pred_df.iloc[:, 1:].to_numpy()
        pred_results.append(pred)
    if len(pred_results) == 0:
        return
    embedding_results = np.mean(pred_results, axis=0)
    label_df = pd.read_csv(list(Path(hparams.log_dir).glob(f'fold*/test/set/{test_set_idx}/label.csv'))[0]).sort_values(by=['filename'])
    embedding_df = label_df.copy()
    embedding_df.iloc[:, 1:] = embedding_results
    report, bn_report, th_reports = generate_report(embedding_df, label_df, 'embedding', os.path.join(output_dir, 'report'))

    embedding_rank = len(pred_results)

    multiclass_df = compose_report(hparams, report, os.path.join(output_dir, 'multiclass_summary_all.csv'),match_pattern=f'fold*/test/set/{test_set_idx}/report/Report_*.csv', embedding_rank=embedding_rank)
    bn_df = compose_report(hparams, bn_report, os.path.join(output_dir, 'bn_summary_all.csv'),match_pattern=f'fold*/test/set/{test_set_idx}/report/BN_Report_*.csv', embedding_rank=embedding_rank)
    th_dfs = []
    for idx, th in enumerate(THRESH):
        th_df = compose_report(hparams, th_reports[idx], os.path.join(output_dir, f'th_{th}_summary_all.csv'),match_pattern=f'fold*/test/set/{test_set_idx}/report/th/TH_Report*{th}.csv', 
        embedding_rank=embedding_rank)
        th_dfs.append(th_df)
    return bn_df, multiclass_df, th_dfs

def get_mean(dfs: List[DataFrame]):
    mean: DataFrame = dfs[0].copy().sort_values(by=['Rank', 'class'], ascending=False)
    dfs: pd.DataFrame = [df.sort_values(by=['Rank', 'class'], ascending=False) for df in dfs ]
    mean.iloc[:, 2:] = np.mean([df.iloc[:, 2:].to_numpy() for df in dfs], axis=0)
    mean = mean.sort_values(by=['class', 'precision', 'Rank'], ascending=False)
    return mean

def excluse_df(dfs, exclude_idx):
    return [df for idx, df in enumerate(dfs) if idx != exclude_idx]

def save_mean_df_for_real_world_sets(bn_dfs, multiclass_dfs, th_df_groups, test_pos_idx, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    bn_dfs = excluse_df(bn_dfs, test_pos_idx)
    multiclass_dfs = excluse_df(multiclass_dfs, test_pos_idx)
    th_df_groups = excluse_df(th_df_groups, test_pos_idx)
    mean_bn_df: DataFrame = get_mean(bn_dfs)
    mean_multiclass_df: DataFrame = get_mean(multiclass_dfs)
    mean_th_dfs: List[DataFrame] = [get_mean(th_df) for th_df in th_df_groups]

    mean_bn_df.to_csv(os.path.join(output_dir, 'bn_summary_all.csv'), index=False, float_format='%.4f')
    mean_multiclass_df.to_csv(os.path.join(output_dir, 'multiclass_summary_all.csv'), index=False, float_format='%.4f')
    mean_th_dfs[-1].to_csv(os.path.join(output_dir, 'th_summary_all.csv'), index=False, float_format='%.4f')


def post_embedding(hparams):
    output_dir = os.path.join(hparams.log_dir, 'embedding')
    os.makedirs(output_dir, exist_ok=True)
    test_set_num = hparams.test_real_world_num + 1
    test_pos_idx = 0 # point the postion idx of exp test set, the remainted sets are real-world test sets.

    bn_dfs, multiclass_dfs, th_df_groups = [], [], []
    for set_id in range(test_set_num):
        data_output_dir = os.path.join(output_dir, str(set_id))
        bn_df, multiclass_df, th_dfs = handle_single_test_set(hparams, set_id, data_output_dir)
        bn_dfs.append(bn_df)
        multiclass_dfs.append(multiclass_df)
        th_df_groups.append(th_dfs)
    save_mean_df_for_real_world_sets(bn_dfs, multiclass_dfs, th_df_groups, test_pos_idx, os.path.join(output_dir, 'mean'))

    

   

    
        




if __name__ == '__main__':
    X, y = datasets.make_classification(n_samples=10000,
                                        n_classes=8,
                                        n_informative=8,
                                        random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    clf = SVC(random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    vis_confusion_matrix(y_test,
                         y_pred,
                         'test.png',
                         display_labels=VIS_ALL_LABELS)

    




if __name__ == '__main__':
    X, y = datasets.make_classification(n_samples=10000,
                                        n_classes=8,
                                        n_informative=8,
                                        random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    clf = SVC(random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    vis_confusion_matrix(y_test,
                         y_pred,
                         'test.png',
                         display_labels=VIS_ALL_LABELS)
