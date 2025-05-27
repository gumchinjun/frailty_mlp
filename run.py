import os
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import logging
import argparse
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from loss import FocalLoss, SupConLoss
from models import Attn_MLP
from data_loader import get_dataloader
from utils import eval_metrics, setup_seed, visualize_attn_map, visualize_conf_mat, visualize_reg_perf


class Trainer:
    def __init__(self, args):
        self.args = args
        self.writer = SummaryWriter()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.TrainLoader, self.ValLoader, self.TestLoader, self.num_train_optimization_steps \
            = get_dataloader(args)
        self.model, self.optimizer, self.scheduler = self.prep_for_training(args)
        self.reg_loss_fn = nn.L1Loss()
        if args.loss == "focal":
            self.cls_loss_fn = FocalLoss(gamma=0.7)
        else:
            self.cls_loss_fn = nn.CrossEntropyLoss()
        if args.metric_loss == 'contrastive':
            self.aux_loss_fn = SupConLoss()
        elif args.metric_loss == 'ranking':
            self.aux_loss_fn = nn.MarginRankingLoss()

    def prep_for_training(self, args):
        model = Attn_MLP(args)
        parameters = model.parameters()
        if args.load_model:
            path = os.path.join(args.save_path, args.model + str(args.cls_type) + "_param.pt")
            model.load_state_dict(torch.load(path, weights_only=True))
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(parameters, lr=args.lr, weight_decay=args.l2)
        elif args.optimizer == 'adam':
            optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.l2)
        elif args.optimizer == 'adamw':
            optimizer = optim.AdamW(parameters, lr=args.lr, weight_decay=args.l2)
        else:
            print("Choose a optimizer between [sgd/adam/adamw]")
        if args.scheduler == 'linear':  # scheduler.step() after a BATCH (same as optimizer.step())
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_proportion * self.num_train_optimization_steps,
                num_training_steps=self.num_train_optimization_steps,
            )
        elif args.scheduler == 'step':  # scheduler.step() after whole EPOCH
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        elif args.scheduler == 'cosine':  # scheduler.step() after a BATCH (same as optimizer.step())
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)
        return model, optimizer, scheduler

    def calc_loss(self, outputs, labels):
        cls_loss, reg_loss, aux_loss = 0.0, 0.0, 0.0
        for i in range(3):
            weight = self.args.aux_cls_loss_weight[i]
            cls_loss += weight * self.cls_loss_fn(outputs[i], labels[:, i].long())
        # for reg loss
        for i in range(3, 6):
            weight = self.args.reg_loss_weight[i - 3]
            reg_loss += weight * self.reg_loss_fn(outputs[i], labels[:, i].float().unsqueeze(1))
        # for metric loss
        if args.metric_loss == 'contrastive':
            pass
        total_loss = cls_loss + reg_loss + aux_loss
        return total_loss

    def train(self):
        self.model.train()
        best_valid, wait = 1e8, 0
        for epoch in range(self.args.epochs):
            logger.info(f"Epoch: {epoch}")
            train_loss = 0.0
            with tqdm(self.TrainLoader) as td:
                for batch_data in td:
                    self.optimizer.zero_grad()
                    self.model = self.model.to(self.device)
                    data, labels = batch_data[0].to(self.device), batch_data[1].to(self.device)
                    outputs = self.model(data)
                    loss = self.calc_loss(outputs, labels)
                    loss.backward()
                    train_loss += loss.item()
                    self.optimizer.step()
                    if self.args.scheduler != 'step':
                        self.scheduler.step()
            if self.args.scheduler == 'step':
                self.scheduler.step()
            train_loss /= len(self.TrainLoader)
            cur_valid = self.validate()
            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            self.writer.add_scalar("Loss/Valid", cur_valid, epoch)
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Valid Loss: {cur_valid:.4f}")
            if cur_valid < best_valid:
                best_valid = cur_valid
                torch.save(self.model.state_dict(), \
                           os.path.join(args.save_path, args.model+str(args.cls_type)+"_param.pt"))
                wait = 0
            else:
                wait += 1
            if wait > args.early_stop:    # early stop
                return

    def validate(self):
        self.model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            with tqdm(self.ValLoader) as td:
                for batch_data in td:
                    self.model = self.model.to(self.device)
                    data, labels = batch_data[0].to(self.device), batch_data[1].to(self.device)
                    outputs = self.model(data)
                    loss = self.calc_loss(outputs, labels)
                    eval_loss += loss.item()
        eval_loss /= len(self.ValLoader)
        return eval_loss

    def test(self):
        self.model.eval()
        y_preds_cls = torch.tensor([], dtype=torch.float)
        y_trues_cls = torch.tensor([], dtype=torch.float)
        y_preds_reg_1 = torch.tensor([], dtype=torch.float)
        y_trues_reg_1 = torch.tensor([], dtype=torch.float)
        y_preds_reg_2 = torch.tensor([], dtype=torch.float)
        y_trues_reg_2 = torch.tensor([], dtype=torch.float)
        y_preds_reg_3 = torch.tensor([], dtype=torch.float)
        y_trues_reg_3 = torch.tensor([], dtype=torch.float)
        y_preds_aux_cls_1 = torch.tensor([], dtype=torch.float)
        y_trues_aux_cls_1 = torch.tensor([], dtype=torch.float)
        y_preds_aux_cls_2 = torch.tensor([], dtype=torch.float)
        y_trues_aux_cls_2 = torch.tensor([], dtype=torch.float)
        attention = torch.tensor([], dtype=torch.float)
        with torch.no_grad():
            with tqdm(self.TestLoader) as td:
                for batch_data in td:
                    self.model = self.model.to(self.device)
                    data, labels = batch_data[0].to(self.device), batch_data[1]
                    outputs = self.model(data)
                    attention = torch.cat((attention, outputs[6].cpu()), 0)
                    cls_probs = F.softmax(outputs[0], dim=1)
                    aux_cls_probs_1 = F.softmax(outputs[1], dim=1)
                    aux_cls_probs_2 = F.softmax(outputs[2], dim=1)
                    cls_preds = torch.argmax(cls_probs, dim=1)
                    aux_cls_preds_1 = torch.argmax(aux_cls_probs_1, dim=1)
                    aux_cls_preds_2 = torch.argmax(aux_cls_probs_2, dim=1)
                    y_preds_cls = torch.cat((y_preds_cls, cls_preds.cpu()), 0)
                    y_trues_cls = torch.cat((y_trues_cls, labels[:,0]), 0)
                    y_preds_aux_cls_1 = torch.cat((y_preds_aux_cls_1, aux_cls_preds_1.cpu()), 0)
                    y_trues_aux_cls_1 = torch.cat((y_trues_aux_cls_1, labels[:,1]), 0)
                    y_preds_aux_cls_2 = torch.cat((y_preds_aux_cls_2, aux_cls_preds_2.cpu()), 0)
                    y_trues_aux_cls_2 = torch.cat((y_trues_aux_cls_2, labels[:,2]), 0)
                    y_preds_reg_1 = torch.cat((y_preds_reg_1, outputs[3].cpu()), 0)
                    y_trues_reg_1 = torch.cat((y_trues_reg_1, labels[:,3]), 0)
                    y_preds_reg_2 = torch.cat((y_preds_reg_2, outputs[4].cpu()), 0)
                    y_trues_reg_2 = torch.cat((y_trues_reg_2, labels[:,4]), 0)
                    y_preds_reg_3 = torch.cat((y_preds_reg_3, outputs[5].cpu()), 0)
                    y_trues_reg_3 = torch.cat((y_trues_reg_3, labels[:,5]), 0)
        acc, f1, acc_aux_1, f1_aux_1, acc_aux_2, f1_aux_2, mae_1, corr_1, mae_2, corr_2, mae_3, corr_3, conf_mat = \
            eval_metrics(y_preds_cls, y_trues_cls,
                         y_preds_aux_cls_1, y_trues_aux_cls_1,
                         y_preds_aux_cls_2, y_trues_aux_cls_2,
                         y_preds_reg_1, y_trues_reg_1,
                         y_preds_reg_2, y_trues_reg_2,
                         y_preds_reg_3, y_trues_reg_3)
        mae_list = [float(mae_1), float(mae_2), float(mae_3)]
        corr_list = [float(corr_1), float(corr_2), float(corr_3)]
        mean_mae = sum(mae_list) / len(mae_list)
        mean_corr = sum(corr_list) / len(corr_list)
        logger.info(f"TEST_ACC: {acc:.4f} \t TEST_F1: {f1:.4f}")
        logger.info(f"AUX_TEST_ACC_1: {acc_aux_1:.4f} \t AUX_TEST_F1: {f1_aux_1:.4f}")
        logger.info(f"AUX_TEST_ACC_2: {acc_aux_2:.4f} \t AUX_TEST_F2: {f1_aux_2:.4f}")
        logger.info(f"TEST_Corr: {mean_corr:.4f} \t TEST_MAE: {mean_mae:.4f}")
        visualize_conf_mat(conf_mat)
        visualize_reg_perf(y_preds_reg_1, y_trues_reg_1,
                         y_preds_reg_2, y_trues_reg_2,
                         y_preds_reg_3, y_trues_reg_3)
        if self.args.model == "Attn_MLP":
            visualize_attn_map(attention, y_preds_cls, args.cls_type, args.input_dim)


def parse_args():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--data_path', type=str, default='./data/normalized_data.csv', help='path to csv file')
    parser.add_argument('--cls_type', type=int, default=3, help='[3/2]')
    parser.add_argument('--gender_agnostic', type=bool, default=False,
                        help='use normalized (in terms of gender) data for regression task')
    parser.add_argument('--two_class_split_type', type=int, default=3,
                        help='1: (R,P) vs F / 2: R vs (P,F) / 3: R vs F')
    # train
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--seed', type=int, default=1, help='seed')
    parser.add_argument('--loss', type=str, default='focal', help='[cross_entropy/focal]')
    parser.add_argument('--reg_loss_weight', nargs='+', type=float, help='weight for regression losses')
    parser.add_argument('--aux_cls_loss_weight', type=list, default=[1,0.9,0.9], help='weight for auxiliary losses')
    parser.add_argument('--metric_loss', type=str, default=None, help='[contrastive/ranking]')
    parser.add_argument('--optimizer', type=str, default="adamw", help='[sgd/adam/adamw]')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate: MLP(1e-3), BERT(1e-5)')
    parser.add_argument('--l2', type=float, default=1e-2, help='weight decay for optimizer: [1e-2/1e-3]')
    parser.add_argument('--scheduler', type=str, default="linear", help='[linear/step(for contrastive)]')
    parser.add_argument('--warmup_proportion', type=float, default=0.1, help='for warmup steps in lr scheduler')
    parser.add_argument('--early_stop', type=int, default=10, help='number of iteration until early stop')
    parser.add_argument('--save_path', type=str, default='./model_weight', help='path to model parameters')
    parser.add_argument('--log_path', type=str, default='./logs', help='path to a log file')
    # model
    parser.add_argument('--model', type=str, default='Attn_MLP', help='[Attn_MLP]')
    parser.add_argument('--input_dim', type=int, default=109, help='dimension of input feature') # original data: 72
    parser.add_argument('--hidden_layer', type=list, default=[64, 64, 32], help='dimensions of hidden layer in MLP')
    parser.add_argument('--dropout', type=float, default=0.25, help='dropout')
    parser.add_argument('--attn_dropout', type=float, default=0.25, help='dropout')
    parser.add_argument('--load_model', type=bool, default=False, help="load pretrained model's weight")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    setup_seed(args.seed)
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Log message format
        handlers=[
            logging.StreamHandler(),  # Log to console
            logging.FileHandler(os.path.join(args.log_path, str(args.model)+'_results'+str(args.cls_type)+'.log'))  # Log to a file
        ]
    )
    logger = logging.getLogger(__name__)
    run = Trainer(args)
    run.train()
    run.writer.flush()
    run.writer.close()
    run.test()
