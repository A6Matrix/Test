from models.TextCNN import textcnn
from sklearn import metrics
from DistillModel_Utils.Bert_utils import teacher_test
# from DistillModel_Utils.Quora_student_data import Quora_Dataset
from DistillModel_Utils.Student_Data_Utils import LCQMC_Dataset
from DistillModel_Utils.DistillBert_data_utils import DataPrecessForSentence
# from DistillModel_Utils.Quora_bert_data import DataPrecessForSentence
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import argparse
from Config import opt
import time
import torch
import numpy as np
import logging
import os
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
bert_tokenizer = BertTokenizer.from_pretrained('./bert_pretrain', do_lower_case=True)
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

def get_loss(t_logits, s_logits, label, a):
    loss1 = nn.CrossEntropyLoss()
    loss2 = nn.MSELoss()
    loss = a * loss1(s_logits, label) + (1-a) * loss2(t_logits, s_logits)
    return loss


def correct_predictions(output_probabilities, targets):
    """
    Compute the number of predictions that match some target classes in the
    output of a model.
    Args:
        output_probabilities: A tensor of probabilities for different output
            classes.
        targets: The indices of the actual target classes.
    Returns:
        The number of correct predictions in 'output_probabilities'.
    """
    _, out_classes = output_probabilities.max(dim=1)
    correct = (out_classes == targets).sum()
    return correct.item()

# 预测学生模型输出结果
# def student_predict(ntm_model,embeddings):
def student_predict(embeddings):
    # student_test_data = Quora_Dataset(tag='test')
    student_test_data = LCQMC_Dataset(tag='test')
    student_test_loader = DataLoader(dataset=student_test_data,
                                      collate_fn = student_test_data.collate_fn_data,
                                      pin_memory=True, batch_size=opt.batch_size, shuffle=False)

    model = textcnn(embeddings=embeddings).to(device)
    model.load_state_dict(torch.load('data/saved_dict/textcnn.ckpt'))
    model.eval()
    time_start = time.time()
    predict_all = []
    prob_all = []
    labels_all = []
    correct_preds = 0
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f'{total_params:,} total parameters.')
    with torch.no_grad():
        for i, (q, h, label) in enumerate(student_test_loader):
            q1 = q.to(device)
            q2 = h.to(device)
            labels = label.to(device)
            # s_logits = model(q1,q2,topic_represent)
            s_logits,prob = model(q1,q2)
            correct_preds += correct_predictions(prob, labels)
            predict_all.extend(prob[:,1].cpu().numpy())
            prob_all.extend(np.argmax(prob.cpu().numpy(),axis=1))
            #print(len(prob_all))
            labels_all.extend(label)
            #print(len(labels_all))
    #         predict_all = np.append(predict_all, predic)
    #         labels_all = np.append(labels_all,label)
    acc = correct_preds / len(student_test_loader.dataset)
    total_time = time.time() - time_start
    print("-> Test. time: {:.4f}s, accuracy: {:.4f}%, auc: {:.4f}, F1:{:.4f}\n".format(total_time, (acc * 100), roc_auc_score(labels_all,predict_all),f1_score(labels_all,prob_all)))


def train_model(bert_model, stu_model, embeddings, patience=5):

    teacher_train_data = DataPrecessForSentence(bert_tokenizer, tag='train')
    teacher_train_loader = DataLoader(teacher_train_data, shuffle=False, batch_size=opt.batch_size)

    teacher_test_data = DataPrecessForSentence(bert_tokenizer, tag='valid')
    teacher_test_loader = DataLoader(teacher_test_data, shuffle=False, batch_size=opt.batch_size)

    student_train_data = LCQMC_Dataset(tag='train')
    student_train_loader = DataLoader(dataset=student_train_data,
                                      collate_fn=student_train_data.collate_fn_data,
                                      pin_memory=True, batch_size=opt.batch_size, shuffle=False)

    _,_,_,_, t_logits = teacher_test(bert_model, teacher_train_loader)
    _,_,_,_, t_test = teacher_test(bert_model, teacher_test_loader)
    # stu_total_params = sum(p.numel() for p in stu_model.parameters())
    # bert_total_params = sum(p.numel() for p in bert_model.parameters())
    # print("stu_model parameter:%.2fM"%(stu_total_params/1e6))
    # print("bert_model parameter:%.2fM"%(bert_total_params/1e6))
     
    parameters  = filter(lambda p: p.requires_grad, stu_model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=opt.learning_rate,weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                           factor=0.85, patience=0)
    best_score = 0.0
    patience_counter = 0
    train_losses = []
    valid_losses = []
    epochs_count = []
    train_acc = []
    for epoch in range(opt.epoch):
        epochs_count.append(epoch)
        print("* Training epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy = student_train(stu_model, student_train_loader, t_logits, optimizer)
        train_losses.append(epoch_loss)
        train_acc.append(epoch_accuracy)
        print("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%"
              .format(epoch_time, epoch_loss, (epoch_accuracy * 100)))
        print("* Validation for epoch {}:".format(epoch))
        _, epoch_loss, epoch_accuracy, epoch_auc = student_evaluate(stu_model,t_test)
        valid_losses.append(epoch_loss)
        print("-> loss: {:.4f}, accuracy: {:.4f}%, auc: {:.4f}\n"
              .format(epoch_loss, (epoch_accuracy * 100), epoch_auc))
        # Update the optimizer's learning rate with the scheduler.
        scheduler.step(epoch_accuracy)
        # Early stopping on validation accuracy.
        if epoch_accuracy < best_score:
            patience_counter += 1
        else:
            best_score = epoch_accuracy
            patience_counter = 0
            torch.save(stu_model.state_dict(),'data/saved_dict/textcnn.ckpt')
        if patience_counter >= patience:
            print("-> Early stopping: patience limit reached, stopping...")
            break
    drawacc(epochs_count,train_acc)
    drawloss(epochs_count,train_losses)
    # student_predict(ntm_model,embeddings)
    # student_predict(embeddings)

def student_train(stu_model, dataloader, t_logits, optimizer, max_gradient_norm=10.0):
    stu_model.train()
    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0
    correct_preds = 0
    tqdm_batch_iterator = tqdm(dataloader)
    for i, (q, h, label) in enumerate(tqdm_batch_iterator):
        batch_start = time.time()
        # Move input and output data to the GPU if it is used.
        q1 = q.to(device)
        q2 = h.to(device)
        labels = label.to(device)
        optimizer.zero_grad()
        s_logits,prob = stu_model(q1, q2)
        loss = get_loss(t_logits[i], s_logits, labels,1)
        loss.backward()
        nn.utils.clip_grad_norm_(stu_model.parameters(), max_gradient_norm)
        optimizer.step()
        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()
        correct_preds += correct_predictions(prob, labels)
        description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}" \
            .format(batch_time_avg / (i + 1), running_loss / (i + 1))
        tqdm_batch_iterator.set_description(description)
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_preds / len(dataloader.dataset)
    return epoch_time, epoch_loss, epoch_accuracy

# def student_evaluate(ntm_model, stu_model, t_logits):
def student_evaluate(stu_model, t_logits):
    # student_dev_data = Quora_Dataset(tag='valid')
    student_dev_data = LCQMC_Dataset(tag='valid')
    student_dev_loader = DataLoader(dataset=student_dev_data,
                                      collate_fn=student_dev_data.collate_fn_data,
                                      pin_memory=True, batch_size=opt.batch_size, shuffle=False)
    stu_model.eval()
    predict_all = []
    labels_all = []
    loss_total = 0.0
    acc = 0.0
    with torch.no_grad():
        for i, (q, h, label) in enumerate(student_dev_loader):
            q1 = q.to(device)
            q2 = h.to(device)
            labels = label.to(device)
            # s_logits = stu_model(q1,q2,topic_represent)
            s_logits,prob = stu_model(q1,q2)
            # loss = F.cross_entropy(cur_pred.squeeze(1), labels.squeeze(1).long())
            loss = get_loss(t_logits[i], s_logits, labels, 1)
            loss_total += loss
            # predic = torch.max(s_logits, 1)[1].cpu().numpy()
            # labels = labels.data.cpu().numpy()
            acc += correct_predictions(prob,labels)
            predict_all.extend(prob[:,1].cpu().numpy())
            labels_all.extend(label)
            # labels_all = np.append(labels_all, labels)
            # predict_all = np.append(predict_all, predic)
    # acc = metrics.accuracy_score(labels_all, predict_all)
    acc /= (len(student_dev_loader.dataset))
    return predict_all, loss_total/len(student_dev_loader.dataset), acc, roc_auc_score(labels_all,predict_all)

def drawacc(epochs,acc):
    
 
    # epochs = [0,1,2,3]
    # acc = [4,8,6,5]
    # loss = [3,2,1,4]
     
    plt.plot(epochs,acc,color='r',label='acc')        # r表示红色
    # plt.plot(epochs,loss,color=(0,0,0),label='loss')  #也可以用RGB值表示颜色
     
    #####非必须内容#########
    plt.xlabel('epochs')    #x轴表示
    plt.ylabel('train accuracy')   #y轴表示
    plt.title("train accuracy")      #图标标题表示
    plt.legend()            #每条折线的label显示
    x_major_locator = MultipleLocator(1)
    # 获得当前坐标图句柄
    ax = plt.gca()
    # 设置横坐标刻度间隔
    ax.xaxis.set_major_locator(x_major_locator)
    # 设置横坐标取值范围
    plt.xlim(1, len(epochs))
    #######################
    plt.savefig('acc1.png')  #保存图片，路径名为test.jpg
    plt.show() 

def drawloss(epochs,loss):
    
 
    # epochs = [0,1,2,3]
    # acc = [4,8,6,5]
    # loss = [3,2,1,4]
     
    plt.plot(epochs,loss,color=(0,0,0),label='loss')        # r表示红色
    # plt.plot(epochs,loss,color=(0,0,0),label='loss')  #也可以用RGB值表示颜色
     
    #####非必须内容#########
    plt.xlabel('epochs')    #x轴表示
    plt.ylabel('train loss')   #y轴表示
    plt.title("train loss")      #图标标题表示
    plt.legend()            #每条折线的label显示
    x_major_locator = MultipleLocator(1)
    # 获得当前坐标图句柄
    ax = plt.gca()
    # 设置横坐标刻度间隔
    ax.xaxis.set_major_locator(x_major_locator)
    # 设置横坐标取值范围
    plt.xlim(1, len(epochs))
    #######################
    plt.savefig('loss.png')  #保存图片，路径名为test.jpg
    plt.show() 
