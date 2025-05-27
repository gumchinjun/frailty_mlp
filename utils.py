import csv
import ast
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def str2float(lst):
    return [float(i) if i != "" else -1.0 for i in lst]

def str2vec(lst):
    new_list = []
    for x in lst:
        if x == "" or (isinstance(x, (float, np.float64)) and np.isnan(x)):
            new_list.append(0.0)
            continue
        try: new_list.append(float(x))
        except ValueError:
            one_hot_vec = ast.literal_eval(x)
            for i in one_hot_vec:
                new_list.append(float(i))
    return new_list

def visualize_reg_perf(*args):
    for i in range(3):
        y_preds, y_trues = args[i*2].flatten(), args[i*2+1]
        mean_y_true = np.mean(y_trues.numpy())
        mean_y_pred = np.mean(y_preds.numpy())
        plt.figure(figsize=(10, 6))
        plt.plot(y_trues, label='y_true', marker='o', linewidth=1, linestyle='-', color='blue')
        plt.plot(y_preds, label='y_pred', marker='x', linewidth=1, linestyle='--', color='red')
        plt.axhline(y=mean_y_true, color='blue', linestyle='-', linewidth=2, label=f'Mean y_true = {mean_y_true:.4f}')
        plt.axhline(y=mean_y_pred, color='red', linestyle='--', linewidth=2, label=f'Mean y_pred = {mean_y_pred:.4f}')
        plt.xlabel('Sample Index')
        plt.ylabel('Values')
        plt.title(f'y_true vs y_pred for reg {i+1}')
        plt.legend()
        plt.show()

def eval_metrics(*args):
    y_preds_cls, y_trues_cls = args[0], args[1]
    y_preds_aux_cls_1, y_trues_aux_cls_1 = args[2], args[3]
    y_preds_aux_cls_2, y_trues_aux_cls_2 = args[4], args[5]
    y_preds_reg_1, y_trues_reg_1 = args[6], args[7]
    y_preds_reg_2, y_trues_reg_2 = args[8], args[9]
    y_preds_reg_3, y_trues_reg_3 = args[10], args[11]
    acc = accuracy_score(y_trues_cls.numpy(), y_preds_cls.numpy())
    f1 = f1_score(y_trues_cls.numpy(), y_preds_cls.numpy(), average='weighted')
    acc_aux_1 = accuracy_score(y_trues_aux_cls_1.numpy(), y_preds_aux_cls_1.numpy())
    f1_aux_1 = f1_score(y_trues_aux_cls_1.numpy(), y_preds_aux_cls_1.numpy(), average='weighted')
    acc_aux_2 = accuracy_score(y_trues_aux_cls_2.numpy(), y_preds_aux_cls_2.numpy())
    f1_aux_2 = f1_score(y_trues_aux_cls_2.numpy(), y_preds_aux_cls_2.numpy(), average='weighted')
    conf_mat = confusion_matrix(y_trues_cls, y_preds_cls)
    y_preds_reg_1, y_trues_reg_1 = np.squeeze(y_preds_reg_1.numpy()), y_trues_reg_1.numpy()
    y_preds_reg_2, y_trues_reg_2 = np.squeeze(y_preds_reg_2.numpy()), y_trues_reg_2.numpy()
    y_preds_reg_3, y_trues_reg_3 = np.squeeze(y_preds_reg_3.numpy()), y_trues_reg_3.numpy()
    mae_1 = np.mean(np.absolute(y_preds_reg_1 - y_trues_reg_1))
    mae_2 = np.mean(np.absolute(y_preds_reg_2 - y_trues_reg_2))
    mae_3 = np.mean(np.absolute(y_preds_reg_3 - y_trues_reg_3))
    corr_1 = np.corrcoef(y_preds_reg_1, y_trues_reg_1)[0][1]
    corr_2 = np.corrcoef(y_preds_reg_2, y_trues_reg_2)[0][1]
    corr_3 = np.corrcoef(y_preds_reg_3, y_trues_reg_3)[0][1]
    return acc, f1, acc_aux_1, f1_aux_1, acc_aux_2, f1_aux_2, mae_1, corr_1, mae_2, corr_2, mae_3, corr_3, conf_mat

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def plot_data_distribution(lst, mode):
    counts = {i: lst.count(i) for i in set(lst)}
    plt.bar(counts.keys(), counts.values())
    plt.xlabel('Element')
    plt.ylabel('Count')
    plt.title('Histogram of Element Counts: {}'.format(mode))
    plt.xticks(list(counts.keys()))  # Ensure all elements are shown on x-axis
    plt.show()

def visualize_conf_mat(cm):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

def get_x_axis_feat():
    x_axis = []
    with open('./data/normalized_data.csv', mode='r') as file:
        csvFile = csv.reader(file)
        header = next(csvFile)
        end_of_feat = header.index('BL_sgdsk_score')
        row = next(csvFile)
        header = header[2:end_of_feat + 1]
        for i, feat in enumerate(row[2:end_of_feat + 1]):
            x_axis.append(header[i])
            if '[' in feat:
                skip = len(ast.literal_eval(feat)) - 1
                for _ in range(skip):
                    x_axis.append(None)
    return x_axis

def visualize_attn_map(attention_weights, outputs, num_classes, input_size):
    # Convert attention weights to numpy for visualization
    attention_weights_np = attention_weights.numpy()
    predicted_classes = outputs.numpy()
    x_labels = get_x_axis_feat()
    tick_positions = [i for i, label in enumerate(x_labels) if label is not None]
    tick_labels = [label for label in x_labels if label is not None]

    # Create attention maps for each class
    for class_id in range(num_classes):
        # Get the attention weights for the predicted class
        class_attention = attention_weights_np[predicted_classes == class_id]

        # Average the attention weights for visualization
        mean_attention = class_attention.mean(axis=0)

        plt.figure()
        plt.bar(range(input_size), mean_attention)
        plt.xticks(tick_positions, tick_labels, fontsize=7, rotation=45, ha='right')
        plt.xlim(left=0)
        plt.title(f'Attention Map for Class {class_id}')
        plt.xlabel('Feature Index')
        plt.ylabel('Attention Weight')
        plt.tight_layout()
        plt.show()


def vis_major_feat():
    # major_feat = ['BL_kidney', 'BL_COPD', 'BL_Allergic_rhinitis', 'BL_social_capital_poor']
    major_feat = ['BL_social_capital_poor']
    df = pd.read_csv('./data/Merged_Frailty_Variables_Final_240925.csv')
    cohorts = ['Robust', 'Prefrail', 'Frail']
    x = np.arange(len(cohorts))
    width = 0.2
    fig, ax = plt.subplots()
    for feat in major_feat:
        df0 = df[df['BL_CHS_FRAIL_Three_cat'] == 0][feat]
        df1 = df[df['BL_CHS_FRAIL_Three_cat'] == 1][feat]
        df2 = df[df['BL_CHS_FRAIL_Three_cat'] == 2][feat]
        count_values_0 = df0.value_counts(dropna=False)
        count_nan_0 = df0.isna().sum()
        count_values_1 = df1.value_counts(dropna=False)
        count_nan_1 = df1.isna().sum()
        count_values_2 = df2.value_counts(dropna=False)
        count_nan_2 = df2.isna().sum()
        robust_zero = count_values_0.get(0, 0)
        robust_one = count_values_0.get(1, 0)
        robust_nan = count_nan_0
        prefrail_zero = count_values_1.get(0, 0)
        prefrail_one = count_values_1.get(1, 0)
        prefrail_nan = count_nan_1
        frail_zero = count_values_2.get(0, 0)
        frail_one = count_values_2.get(1, 0)
        frail_nan = count_nan_2
        zeros = [robust_zero, prefrail_zero, frail_zero]
        ones = [robust_one, prefrail_one, frail_one]
        nans = [robust_nan, prefrail_nan, frail_nan]
        bars1 = ax.bar(x - width, zeros, width, label='0')
        bars2 = ax.bar(x, ones, width, label='1')
        bars3 = ax.bar(x + width, nans, width, label='nan')
        ax.set_xlabel('Cohorts')
        ax.set_ylabel('Values')
        ax.set_title('Stats for ' + feat)
        ax.set_xticks(x)
        ax.set_xticklabels(cohorts)
        ax.legend()
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    vis_major_feat()