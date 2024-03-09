import argparse
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from qlstm_pennylane import QLSTM
from qlstm_pennylane import QGRU
from qlstm_pennylane import QMGU
from matplotlib import pyplot as plt

import extract_feats.opensmile as of
from utils import parse_opt

from torch.utils.data import Dataset, DataLoader, TensorDataset
import time
import sys
import os

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

writer = SummaryWriter('./log')
from tensorflow.python.keras.utils.np_utils import to_categorical

config = parse_opt()
train_voices, valid_voices, train_labels, valid_labels = of.load_feature(config, train=True)

train_voices = train_voices.astype(np.float32)
train_voices = torch.from_numpy(train_voices)

valid_voices = valid_voices.astype(np.float32)
valid_voices = torch.from_numpy(valid_voices)

train_labels = torch.from_numpy(train_labels)
valid_labels = torch.from_numpy(valid_labels)

# 客户端数量
num_clients = 8

# 划分比例
# train_ratios = [0.4, 0.3, 0.3]
# valid_ratios = [0.4, 0.3, 0.3]
train_ratios = [0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
valid_ratios = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
# train_ratios = [0.2, 0.2, 0.35, 0.25]
# valid_ratios = [0.25, 0.25, 0.25, 0.25]
# train_ratios = [0.4, 0.6]
# valid_ratios = [0.5, 0.5]

# 计算每个客户端的数据集大小
train_size = len(train_voices)
valid_size = len(valid_voices)

train_sizes = [int(train_size * ratio) for ratio in train_ratios]
valid_sizes = [int(valid_size * ratio) for ratio in valid_ratios]

# 划分数据集
train_datasets = []
valid_datasets = []

train_start_idx = 0
valid_start_idx = 0

for i in range(num_clients):
    train_end_idx = train_start_idx + train_sizes[i]
    valid_end_idx = valid_start_idx + valid_sizes[i]

    train_datasets.append((train_voices[train_start_idx:train_end_idx], train_labels[train_start_idx:train_end_idx]))
    valid_datasets.append((valid_voices[valid_start_idx:valid_end_idx], valid_labels[valid_start_idx:valid_end_idx]))

    train_start_idx = train_end_idx
    valid_start_idx = valid_end_idx


class Logger(object):
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

log_path = './PrintLogs/'
if not os.path.exists(log_path):
    os.makedirs(log_path)
# 日志文件名按照程序运行时间设置
log_file_name = log_path + 'log-' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.txt'
# 记录正常的 print 信息
logger = Logger(log_file_name)
sys.stdout = logger
# 记录 traceback 异常信息
sys.stderr = logger

class LSTMTagger(nn.Module):

    def __init__(self, input_dim, hidden_dim,  tagset_size, n_qubits=0, backend='default.qubit'):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        if n_qubits > 0:
            # print(f"Tagger will use Quantum LSTM running on backend {backend}")
            # self.lstm = QLSTM(input_dim, hidden_dim, n_qubits=n_qubits, backend=backend)
            # print(f"Tagger will use Quantum GRU running on backend {backend}")
            # self.lstm = QGRU(input_dim, hidden_dim, n_qubits=n_qubits, backend=backend)
            # logger.write(f"BitPhaseFlip=0.1, Tagger will use Quantum MGU running on backend {backend}\n")
            logger.write(f"Tagger will use Quantum MGU running on backend {backend}\n")
            self.lstm = QMGU(input_dim, hidden_dim, n_qubits=n_qubits, backend=backend)
        else:
            print("Tagger will use Classical LSTM")
            self.lstm = nn.LSTM(input_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, sentences):
        # lstm_out, _ = self.lstm(sentence.view(-1, 1, len(sentence)))
        _, (lstm_out, _) = self.lstm(sentences.view(len(sentences), 5, -1))

        tag_logits = self.hidden2tag(lstm_out.view(len(sentences), -1))

        tag_scores = F.log_softmax(tag_logits, dim=1)

        return tag_scores

def addbatch(data_train,data_test,batchsize):
    """
    设置batch
    :param data_train: 输入
    :param data_test: 标签
    :param batchsize: 一个batch大小
    :return: 设置好batch的数据集
    """
    data = TensorDataset(data_train,data_test)
    data_loader = DataLoader(data, batch_size=batchsize, shuffle=False)
    #shuffle是是否打乱数据集，可自行设置

    return data_loader

def evaluate(model, valid_data):
    model.eval()
    with torch.no_grad():
        sentences, labels = valid_data
        outputs = model(sentences)
        valid_loss = torch.nn.functional.cross_entropy(outputs, labels)
        valid_preds = torch.softmax(outputs, dim=-1).argmax(dim=-1)
        valid_corrects = (valid_preds == labels)
        valid_accuracy = valid_corrects.sum().float() / float(labels.size(0))

        cm = confusion_matrix(labels, valid_preds)
        precision = precision_score(labels, valid_preds, average='weighted')
        recall = recall_score(labels, valid_preds, average='weighted')
        f1 = f1_score(labels, valid_preds, average='weighted')
        logger.write(f'Validation confusion matrix:\n{cm}\n')
        logger.write(f'Validation precision: {precision:.4f}\n')
        logger.write(f'Validation recall: {recall:.4f}\n')
        logger.write(f'Validation F1 score: {f1:.4f}\n')

    return valid_accuracy, valid_loss.item()


if __name__ == '__main__':

    # These will usually be more like 32 or 64 dimensional.
    # We will keep them small, so we can see how the weights change as we train.
    parser = argparse.ArgumentParser("QMGU Example")
    parser.add_argument('-I', '--input_dim', default=64, type=int)
    parser.add_argument('-H', '--hidden_dim', default=64, type=int)
    parser.add_argument('-Q', '--n_qubits', default=4, type=int)
    parser.add_argument('-e', '--n_epochs', default=200, type=int)
    parser.add_argument('-s', '--batch_size', default=128, type=int)
    parser.add_argument('-R', '--num_rounds', default=200, type=int)
    parser.add_argument('-L', '--local_epochs', default=1, type=int)
    parser.add_argument('-B', '--backend', default='default.qubit')
    args = parser.parse_args()

    train_data = addbatch(train_voices, train_labels, args.batch_size)

    logger.write(f"Input dim:    {args.input_dim}\n")
    logger.write(f"LSTM output size: {args.hidden_dim}\n")
    logger.write(f"Number of qubits: {args.n_qubits}\n")
    logger.write(f"Training epochs:  {args.n_epochs}\n")
    logger.write(f"Batch_size:       {args.batch_size}\n")
    logger.write(f"Num_rounds:       {args.num_rounds}\n")
    logger.write(f"Local_epochs:       {args.local_epochs}\n")

    model = LSTMTagger(args.input_dim,
                        args.hidden_dim,
                        tagset_size=3,
                        n_qubits=args.n_qubits,
                        backend=args.backend)

    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    decayRate = 0.98
    # my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
    my_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)

    history = {
        'loss': [],
        'acc': [],
        'valid_loss': [],
        'valid_acc': []
    }



######################
    #
    # # 设置通信轮次
    # num_rounds = args.num_rounds
    #
    # # 初始化全局模型参数
    # global_model_params = copy.deepcopy(model.state_dict())
    #
    # client_models = []
    # for i in range(num_clients):
    #     client_models.append(copy.deepcopy(global_model_params))
    #
    # for r in range(num_rounds):
    #
    #     logger.write(f"Communication round : {r+1}\n")
    #
    #     # 客户端训练
    #     for i in range(num_clients):
    #
    #         logger.write(f"client: {i+1}\n")
    #         # 拷贝模型，确保每个客户端训练的模型是独立的
    #         client_model = copy.deepcopy(model)
    #         client_model.load_state_dict(client_models[i])
    #         client_optimizer = optim.Adam(client_model.parameters(), lr=0.001)
    #
    #         # 获取当前客户端的训练集和验证集
    #         client_train_data, client_train_labels = train_datasets[i]
    #         client_valid_data, client_valid_labels = valid_datasets[i]
    #
    #         train_data = addbatch(client_train_data, client_train_labels, args.batch_size)
    #
    #         # 训练模型
    #         for epoch in range(args.local_epochs):
    #             since = time.time()
    #             losses = []
    #             preds = []
    #             targets = []
    #             for step, data in enumerate(train_data):
    #                 b_since = time.time()
    #                 client_optimizer.zero_grad()
    #                 sentences_in, labels = data
    #                 tag_scores = client_model(sentences_in)
    #                 loss = loss_function(tag_scores, labels)
    #                 # l1_reg = torch.tensor(0.)
    #                 # for name, param in client_model.named_parameters():
    #                 #     if 'weight' in name:
    #                 #         l1_reg += torch.norm(param, p=1)
    #                 # loss += 0.001 * l1_reg
    #                 loss.backward()
    #                 client_optimizer.step()
    #                 losses.append(float(loss))
    #                 batch_time = time.time() - b_since
    #                 logger.write(
    #                     f"batch {step + 1}, loss {float(loss)}, batch_time {(batch_time // 60):.0f}m {(batch_time % 60):.0f}s\n")
    #
    #                 probs = torch.softmax(tag_scores, dim=-1)  # tensor转换为概率分布
    #                 probss = probs.argmax(dim=-1)  # 最大元素的位置
    #                 preds.append(probss)
    #                 targets.append(labels)
    #
    #             epoch_time = time.time() - since
    #             avg_loss = np.mean(losses)
    #
    #             preds = torch.cat(preds)
    #             targets = torch.cat(targets)
    #             corrects = (preds == targets)
    #             accuracy = corrects.sum().float() / float(targets.size(0))
    #
    #             logger.write(f"epoch: {epoch+1} Loss = {avg_loss:.5f} Acc = {accuracy:.5f} epoch_time {(epoch_time // 60):.0f}m {(epoch_time % 60):.0f}s\n")
    #         # 保存模型参数
    #         client_model_params = client_model.state_dict()
    #
    #         # 保存当前客户端的模型参数
    #         client_models[i] = client_model_params
    #
    #
    #     # 验证每个客户端的模型
    #     client_accuracies = []
    #     client_losses = []
    #     for i in range(num_clients):
    #         client_model = copy.deepcopy(model)
    #         client_model.load_state_dict(client_models[i])
    #         client_accuracy, client_loss = evaluate(client_model, valid_datasets[i])
    #         client_accuracies.append(client_accuracy)
    #         client_losses.append(client_loss)
    #         logger.write("Client {}: Accuracy {:.5f}, Loss {:.5f}\n".format(i + 1, client_accuracy, client_loss))
    #
    #     # 输出平均结果
    #     avg_accuracy = np.mean(client_accuracies)
    #     avg_loss = np.mean(client_losses)
    #     logger.write("Average Accuracy: {:.5f}, Average Loss: {:.5f}\n".format(avg_accuracy, avg_loss))
    #
    #     writer.add_scalar('Validation/Average_Accuracy', avg_accuracy, global_step=(r+1))
    #     writer.add_scalar('Validation/Average_Loss', avg_loss, global_step=(r+1))
#######################

    #设置通信轮次
    num_rounds = args.num_rounds

    # 初始化全局模型参数
    global_model_params = copy.deepcopy(model.state_dict())

    for r in range(num_rounds):

        logger.write(f"Communication round : {r+1}\n")

        # 客户端模型参数列表
        client_models = []

        # 客户端训练
        for i in range(num_clients):

            logger.write(f"client: {i+1}\n")
            # 拷贝模型，确保每个客户端训练的模型是独立的
            client_model = copy.deepcopy(model)
            client_optimizer = optim.Adam(client_model.parameters(), lr=0.001)

            # 获取当前客户端的训练集和验证集
            client_train_data, client_train_labels = train_datasets[i]
            client_valid_data, client_valid_labels = valid_datasets[i]

            train_data = addbatch(client_train_data, client_train_labels, args.batch_size)

            # 训练模型
            for epoch in range(args.local_epochs):
                since = time.time()
                losses = []
                preds = []
                targets = []
                for step, data in enumerate(train_data):
                    b_since = time.time()
                    client_optimizer.zero_grad()
                    sentences_in, labels = data
                    tag_scores = client_model(sentences_in)
                    loss = loss_function(tag_scores, labels)
                    # l1_reg = torch.tensor(0.)
                    # for name, param in client_model.named_parameters():
                    #     if 'weight' in name:
                    #         l1_reg += torch.norm(param, p=1)
                    # loss += 0.001 * l1_reg
                    loss.backward()
                    client_optimizer.step()
                    losses.append(float(loss))
                    batch_time = time.time() - b_since
                    logger.write(
                        f"batch {step + 1}, loss {float(loss)}, batch_time {(batch_time // 60):.0f}m {(batch_time % 60):.0f}s\n")

                    probs = torch.softmax(tag_scores, dim=-1)  # tensor转换为概率分布
                    probss = probs.argmax(dim=-1)  # 最大元素的位置
                    preds.append(probss)
                    targets.append(labels)

                epoch_time = time.time() - since
                avg_loss = np.mean(losses)

                preds = torch.cat(preds)
                targets = torch.cat(targets)
                corrects = (preds == targets)
                accuracy = corrects.sum().float() / float(targets.size(0))

                logger.write(f"epoch: {epoch+1} Loss = {avg_loss:.5f} Acc = {accuracy:.5f} epoch_time {(epoch_time // 60):.0f}m {(epoch_time % 60):.0f}s\n")
            # 保存模型参数
            client_model_params = client_model.state_dict()

            # 保存当前客户端的模型
            client_models.append(client_model_params)

        # 计算平均参数
        global_model_params = copy.deepcopy(client_models[0])
        for layer_name in global_model_params:
            for i in range(1, num_clients):
                global_model_params[layer_name] += client_models[i][layer_name]
            global_model_params[layer_name] /= num_clients

        # 更新全局模型
        model.load_state_dict(global_model_params)

        # 验证每个客户端的模型
        client_accuracies = []
        client_losses = []
        for i in range(num_clients):
            client_model = copy.deepcopy(model)
            client_model.load_state_dict(client_models[i])
            client_accuracy, client_loss = evaluate(client_model, valid_datasets[i])
            client_accuracies.append(client_accuracy)
            client_losses.append(client_loss)
            logger.write("Client {}: Accuracy {:.5f}, Loss {:.5f}\n".format(i + 1, client_accuracy, client_loss))

        # 输出平均结果
        avg_accuracy = np.mean(client_accuracies)
        avg_loss = np.mean(client_losses)
        logger.write("Average Accuracy: {:.5f}, Average Loss: {:.5f}\n".format(avg_accuracy, avg_loss))

        writer.add_scalar('Validation/Average_Accuracy', avg_accuracy, global_step=(r+1))
        writer.add_scalar('Validation/Average_Loss', avg_loss, global_step=(r+1))

    # for epoch in range(args.n_epochs):
    #     since = time.time()
    #     losses = []
    #     preds = []
    #     targets = []
    #     # model.zero_grad()
    #     for step, data in enumerate(train_data):
    #         # Step 1. Remember that Pytorch accumulates gradients.
    #         # We need to clear them out before each instance
    #         optimizer.zero_grad()
    #
    #         # Step 2. Get our inputs ready for the network, that is, turn them into
    #         # Tensors of word indices.
    #         sentences_in, labels = data
    #         # sentence_in = train_voices[i]
    #         # labels = train_labels[i]
    #         # labels = labels.long()
    #         # labels = labels.unsqueeze(0)
    #         # print(f"lables:{labels}")
    #         # print(f"sentencein:{sentence_in}")
    #         # Step 3. Run our forward pass.
    #         tag_scores = model(sentences_in)
    #         # print(tag_scores[0])
    #         # Step 4. Compute the loss, gradients, and update the parameters by
    #         #  calling optimizer.step()
    #         loss = loss_function(tag_scores, labels)
    #         l1_reg = torch.tensor(0.)
    #         for name, param in model.named_parameters():
    #             if 'weight' in name:
    #                 l1_reg += torch.norm(param, p=1)
    #         loss += 0.001 * l1_reg
    #
    #         loss.backward()
    #         optimizer.step()
    #
    #         losses.append(float(loss))
    #
    #         # index = torch.argmax(labels)
    #
    #         probs = torch.softmax(tag_scores, dim=-1)
    #         probss = probs.argmax(dim = -1)
    #         preds.append(probss)
    #         targets.append(labels)
    #         batch_time = time.time()-since
    #         logger.write(f"batch {step+1}, loss {float(loss)}, batch_time {(batch_time // 60):.0f}m {(batch_time % 60):.0f}s\n")
    #
    #     # my_lr_scheduler.step()
    #     lr_current = optimizer.state_dict()['param_groups'][0]['lr']
    #     avg_loss = np.mean(losses)
    #     history['loss'].append(avg_loss)
    #
    #     # print("preds", preds)
    #     preds = torch.cat(preds)
    #     # print("targets", targets)
    #     targets = torch.cat(targets)
    #     corrects = (preds == targets)
    #     accuracy = corrects.sum().float() / float(targets.size(0) )
    #     history['acc'].append(accuracy)
    #
    #     # valid_preds = []
    #     # valid_targets = []
    #     with torch.no_grad():
    #         model.eval()
    #         valid_tag_scores = model(valid_voices)
    #         valid_loss = loss_function(valid_tag_scores, valid_labels)
    #         history['valid_loss'].append(valid_loss)
    #         valid_preds = torch.softmax(valid_tag_scores, dim=-1).argmax(dim = -1)
    #         valid_corrects = (valid_preds == valid_labels)
    #         valid_accuracy = valid_corrects.sum().float() / float(valid_labels.size(0))
    #         history['valid_acc'].append(valid_accuracy)
    #         cm = confusion_matrix(valid_labels, valid_preds)
    #         precision = precision_score(valid_labels, valid_preds, average='weighted')
    #         recall = recall_score(valid_labels, valid_preds, average='weighted')
    #         f1 = f1_score(valid_labels, valid_preds, average='weighted')
    #         logger.write(f'Validation confusion matrix:\n{cm}\n')
    #         logger.write(f'Validation precision: {precision:.4f}\n')
    #         logger.write(f'Validation recall: {recall:.4f}\n')
    #         logger.write(f'Validation F1 score: {f1:.4f}\n')
    #         # writer.add_scalar('Precision', precision, (epoch + 1))
    #         # writer.add_scalar('Recall', recall, (epoch + 1))
    #         # writer.add_scalar('F1Score', f1, (epoch + 1))
    #         #
    #         # writer.add_images('Confusion Matrix', img_data, (epoch + 1))
    #
    #     time_elapsed = time.time() - since
    #     logger.write(f"Epoch {epoch+1} / {args.n_epochs}: Loss = {avg_loss:.5f} Acc = {accuracy:.5f}  Valid_acc = {valid_accuracy} lr = {lr_current:.5f} Training complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s'\n")
    #     writer.add_scalars( main_tag='Accuracy',
    #                         tag_scalar_dict = {'Training Accuracy': accuracy,
    #                                            'Validation Accuracy': valid_accuracy},
    #                        global_step = (epoch + 1))
    #
    #     writer.add_scalars(main_tag='Loss',
    #                        tag_scalar_dict={'Training Loss': avg_loss,
    #                                         'Validation Loss': valid_loss},
    #                        global_step=(epoch + 1))


        # tensorboard - -logdir =./ path / to / the / folder - -port 8123


