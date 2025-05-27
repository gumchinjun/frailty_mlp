import csv
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from utils import str2vec, plot_data_distribution


class create_3_cls_dataset(Dataset):
    def __init__(self, args, mode='train'):
        self.args = args
        self.mode = mode
        total_feat, label_cls_3, total_reg_label = self.read_data(args)  # total_label: (N, 8) 8: cls_2,3,6, reg_1,2,3,4,5
        self.X_train, self.X_val, self.X_test, y_train, y_val, y_test, train_indices, val_indices, test_indices = \
            self.get_train_val_test_split(total_feat, label_cls_3)
        # y have the shape of (N, 6) -> 6: 1 for cls, 5 for reg
        self.y_train, self.y_val, self.y_test = \
            self.merge_labels(y_train, y_val, y_test, total_reg_label, train_indices, val_indices, test_indices)

    def read_data(self, args):
        total_feat = []
        label_cls_3 = []
        label_aux_cls_1, label_aux_cls_2, label_reg_1, label_reg_2, label_reg_3 = [], [], [], [], []
        with open(args.data_path, mode='r') as file:
            csvFile = csv.reader(file)
            header = next(csvFile)
            end_of_feat = header.index('BL_sgdsk_score')
            score, three, two = end_of_feat + 1, end_of_feat + 2, end_of_feat + 3
            if args.gender_agnostic:
                reg_1, reg_2, reg_3, aux_cls_1, aux_cls_2 = header.index('BL_HGS_BMI_Norm'), header.index('PA_kcal_Norm'), \
                header.index('BL_GS_height_Norm'), header.index('BL_weight_loss_4_5kg_Norm'), header.index('BL_exhaustion')
            else:
                reg_1, reg_2, reg_3, aux_cls_1, aux_cls_2 = header.index('BL_HGS_max_Norm'), header.index('PA_kcal_Norm'), \
                header.index('BL_gaitspeed_Norm'), header.index('BL_weight_loss_4_5kg_Norm'), header.index('BL_exhaustion')
            for i, row in enumerate(csvFile):
                feat = str2vec(row[2:end_of_feat+1])
                label_cls_3.append(int(row[three]))
                label_aux_cls_1.append(int(row[aux_cls_1]))
                label_aux_cls_2.append(int(row[aux_cls_2]))
                label_reg_1.append(float(row[reg_1]))
                label_reg_2.append(float(row[reg_2]))
                label_reg_3.append(float(row[reg_3]))
                total_feat.append(feat)
            total_reg_label = np.stack((label_aux_cls_1, label_aux_cls_2, label_reg_1, label_reg_2, label_reg_3), axis=1)
        return total_feat, label_cls_3, total_reg_label

    def get_train_val_test_split(self, total_feat, label_cls_3):
        indices = np.arange(len(total_feat))
        X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(total_feat, label_cls_3,
                                                                                         indices,
                                                                                         stratify=label_cls_3,
                                                                                         test_size=0.2,
                                                                                         random_state=self.args.seed)
        X_val, X_test, y_val, y_test, val_indices, test_indices = train_test_split(X_test, y_test,
                                                                                   test_indices,
                                                                                   stratify=y_test,
                                                                                   test_size=0.5,
                                                                                   random_state=self.args.seed)
        return X_train, X_val, X_test, y_train, y_val, y_test, train_indices, val_indices, test_indices

    def merge_labels(self, y_train, y_val, y_test, total_reg_label, train_indices, val_indices, test_indices):
        y_train, y_val, y_test = np.array(y_train).reshape(-1, 1), np.array(y_val).reshape(-1, 1), np.array(y_test).reshape(-1, 1)
        y_train_merged = np.concatenate((y_train, total_reg_label[train_indices]), axis=1)
        y_val_merged = np.concatenate((y_val, total_reg_label[val_indices]), axis=1)
        y_test_merged = np.concatenate((y_test, total_reg_label[test_indices]), axis=1)
        return y_train_merged, y_val_merged, y_test_merged

    def __len__(self):
        if self.mode == 'train':
            return len(self.X_train)
        elif self.mode == 'val':
            return len(self.X_val)
        elif self.mode == 'test':
            return len(self.X_test)
        else:
            print("Select a mode between [train/val/test]")

    def __getitem__(self, idx):
        if self.mode == 'train':
            return torch.FloatTensor(self.X_train[idx]), self.y_train[idx]
        elif self.mode == 'val':
            return torch.FloatTensor(self.X_val[idx]), self.y_val[idx]
        elif self.mode == 'test':
            return torch.FloatTensor(self.X_test[idx]), self.y_test[idx]
        else:
            print("Select a mode between [train/val/test]")


class create_2_cls_dataset(Dataset):
    def __init__(self, args, mode='train'):
        self.args = args
        self.mode = mode
        self.df = self.process_label_space()
        total_feat, label_cls_2, total_reg_label = self.read_data(args)  # total_label: (N, 8) 8: cls_2,3,6, reg_1,2,3,4,5
        self.X_train, self.X_val, self.X_test, y_train, y_val, y_test, train_indices, val_indices, test_indices = \
            self.get_train_val_test_split(total_feat, label_cls_2)
        # y have the shape of (N, 6) -> 6: 1 for main cls task, 5 for aux task
        self.y_train, self.y_val, self.y_test = \
            self.merge_labels(y_train, y_val, y_test, total_reg_label, train_indices, val_indices, test_indices)

    def read_data(self, args):
        total_feat = []
        header = list(self.df)
        end_of_feat = header.index('BL_sgdsk_score')
        if args.gender_agnostic:
            reg_1, reg_2, reg_3, aux_cls_1, aux_cls_2 = 'BL_HGS_BMI_Norm', 'PA_kcal_Norm', 'BL_GS_height_Norm', 'BL_weight_loss_4_5kg_Norm', 'BL_exhaustion'
        else:
            reg_1, reg_2, reg_3, aux_cls_1, aux_cls_2 = 'BL_HGS_max_Norm', 'PA_kcal_Norm', 'BL_gaitspeed_Norm', 'BL_weight_loss_4_5kg_Norm', 'BL_exhaustion'
        feat_df = self.df.iloc[:,2:end_of_feat+1]
        for i in range(len(feat_df)):
            row = list(feat_df.iloc[i])
            feat = str2vec(row)
            total_feat.append(feat)
        label_cls_2 = list(self.df['new_class_2'])
        label_reg_1 = list(self.df[reg_1])
        label_reg_2 = list(self.df[reg_2])
        label_reg_3 = list(self.df[reg_3])
        label_aux_cls_1 = list(self.df[aux_cls_1])
        label_aux_cls_2 = list(self.df[aux_cls_2])
        total_reg_label = np.stack((label_aux_cls_1, label_aux_cls_2, label_reg_1, label_reg_2, label_reg_3), axis=1)
        return total_feat, label_cls_2, total_reg_label

    def process_label_space(self):
        df = pd.read_csv(self.args.data_path)
        df['new_class_2'] = None    # add new column for the new "two class"
        class_1, class_2 = [], []
        indices_0 = df.index[df['BL_CHS_FRAIL_Three_cat'] == 0].tolist()
        indices_1 = df.index[df['BL_CHS_FRAIL_Three_cat'] == 1].tolist()
        indices_2 = df.index[df['BL_CHS_FRAIL_Three_cat'] == 2].tolist()
        if self.args.two_class_split_type == 1:
            class_1.extend(indices_0)
            class_1.extend(indices_1)
            class_2.extend(indices_2)
            df.loc[class_1, 'new_class_2'] = 0  # robust & prefrail
            df.loc[class_2, 'new_class_2'] = 1  # frail
        elif self.args.two_class_split_type == 2:
            class_1.extend(indices_0)
            class_2.extend(indices_1)
            class_2.extend(indices_2)
            df.loc[class_1, 'new_class_2'] = 0  # robust
            df.loc[class_2, 'new_class_2'] = 1  # prefrail & frail
        elif self.args.two_class_split_type == 3:
            class_1.extend(indices_0)
            class_2.extend(indices_2)
            df.loc[class_1, 'new_class_2'] = 0  # robust
            df.loc[class_2, 'new_class_2'] = 1  # frail
            df = df.dropna(subset=['new_class_2'])  # drop "prefrail" rows
        else:
            print("Please select an option between [1/2/3]")
        return df

    def get_train_val_test_split(self, total_feat, label_cls_2):
        indices = np.arange(len(total_feat))
        X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(total_feat, label_cls_2,
                                                                                         indices,
                                                                                         stratify=label_cls_2,
                                                                                         test_size=0.2,
                                                                                         random_state=self.args.seed)
        X_val, X_test, y_val, y_test, val_indices, test_indices = train_test_split(X_test, y_test,
                                                                                   test_indices,
                                                                                   stratify=y_test,
                                                                                   test_size=0.5,
                                                                                   random_state=self.args.seed)
        return X_train, X_val, X_test, y_train, y_val, y_test, train_indices, val_indices, test_indices

    def merge_labels(self, y_train, y_val, y_test, total_reg_label, train_indices, val_indices, test_indices):
        y_train, y_val, y_test = np.array(y_train).reshape(-1, 1), np.array(y_val).reshape(-1, 1), np.array(y_test).reshape(-1, 1)
        y_train_merged = np.concatenate((y_train, total_reg_label[train_indices]), axis=1)
        y_val_merged = np.concatenate((y_val, total_reg_label[val_indices]), axis=1)
        y_test_merged = np.concatenate((y_test, total_reg_label[test_indices]), axis=1)
        return y_train_merged, y_val_merged, y_test_merged

    def __len__(self):
        if self.mode == 'train':
            return len(self.X_train)
        elif self.mode == 'val':
            return len(self.X_val)
        elif self.mode == 'test':
            return len(self.X_test)
        else:
            print("Select a mode between [train/val/test]")

    def __getitem__(self, idx):
        if self.mode == 'train':
            return torch.FloatTensor(self.X_train[idx]), self.y_train[idx]
        elif self.mode == 'val':
            return torch.FloatTensor(self.X_val[idx]), self.y_val[idx]
        elif self.mode == 'test':
            return torch.FloatTensor(self.X_test[idx]), self.y_test[idx]
        else:
            print("Select a mode between [train/val/test]")


def get_dataloader(args):
    if args.cls_type == 3:
        train_set = create_3_cls_dataset(args, mode='train')
        val_set = create_3_cls_dataset(args, mode='val')
        test_set = create_3_cls_dataset(args, mode='test')
    elif args.cls_type == 2:
        train_set = create_2_cls_dataset(args, mode='train')
        val_set = create_2_cls_dataset(args, mode='val')
        test_set = create_2_cls_dataset(args, mode='test')
    else:
        print("Please choose a class type between [3/2]")
    TrainLoader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    ValLoader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)
    TestLoader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    num_train_optimization_steps = (int(len(train_set) / args.batch_size) * args.epochs)
    return TrainLoader, ValLoader, TestLoader, num_train_optimization_steps


if __name__ == "__main__":
    from run import parse_args
    args = parse_args()
    train_set = create_3_cls_dataset(args, mode='train')
    print(train_set[0])
    # plot data distribution of train/val/test datasets
    plot_data_distribution(train_set.y_train, mode='Train')
    plot_data_distribution(train_set.y_val, mode='Valid')
    plot_data_distribution(train_set.y_test, mode='Test')
    TrainLoader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    for i, data in enumerate(TrainLoader):
        feat, label = data[0], data[1]
        print(feat.shape)
        print(label)
        if i == 0:
            break

