# -*- coding: utf-8 -*-

import os
import torch
from torch.utils.data import DataLoader
from DistillModel_Utils.DistillBert_data_utils import DataPrecessForSentence
#from DistillModel_Utils.Quora_bert_data import DataPrecessForSentence
from DistillModel_Utils.Bert_utils import teacher_train, teacher_validate,teacher_test
from transformers import BertTokenizer
from models.Bert import BertModel,BertModelTest
from transformers.optimization import AdamW
from sys import platform

def bert_test(pretrained_file, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    bert_tokenizer = BertTokenizer.from_pretrained('./bert_pretrain', do_lower_case=True)
    print(20 * "=", " Preparing for testing ", 20 * "=")
    if platform == "linux" or platform == "linux2":
        checkpoint = torch.load(pretrained_file)
    else:
        checkpoint = torch.load(pretrained_file, map_location=device)
    # checkpoint = torch.load(pretrained_file)
    # Retrieving model parameters from checkpoint.
    print("\t* Loading test data...")
    test_data = DataPrecessForSentence(bert_tokenizer, tag='test')
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
    print("\t* Building model...")
    model = BertModelTest().to(device)
    model.load_state_dict(checkpoint["model"])
    print(20 * "=", " Testing BERT model on device: {} ".format(device), 20 * "=")
    batch_time, total_time, accuracy, auc, _ = teacher_test(model, test_loader)
    print(
        "\n-> Average batch processing time: {:.4f}s, total test time: {:.4f}s, accuracy: {:.4f}%, auc: {:.4f}\n".format(
            batch_time, total_time, (accuracy * 100), auc))


def bert_train(target_dir,
         epochs=20,
         batch_size=32,
         lr=2e-05,
         patience=3,
         max_grad_norm=10.0,
         checkpoint=None):
    bert_tokenizer = BertTokenizer.from_pretrained('./bert_pretrain', do_lower_case=True)
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print(20 * "=", " Preparing for training ", 20 * "=")
    # 保存模型的路径
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    # -------------------- Data loading ------------------- #
    print("\t* Loading training data...")
    train_data = DataPrecessForSentence(bert_tokenizer, tag='train')
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    print("\t* Loading validation data...")
    dev_data = DataPrecessForSentence(bert_tokenizer,  tag='valid')
    dev_loader = DataLoader(dev_data, shuffle=True, batch_size=batch_size)
    # -------------------- Model definition ------------------- #
    print("\t* Building model...")
    model = BertModel().to(device)
    # -------------------- Preparation for training  ------------------- #
    # 待优化的参数
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    # optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                           factor=0.85, patience=0)
    best_score = 0.0
    start_epoch = 1
    # Data for loss curves plot
    epochs_count = []
    train_losses = []
    valid_losses = []
    # Continuing training from a checkpoint if one was given as argument
    if checkpoint:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]
        print("\t* Training will continue on existing model from epoch {}...".format(start_epoch))
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epochs_count = checkpoint["epochs_count"]
        train_losses = checkpoint["train_losses"]
        valid_losses = checkpoint["valid_losses"]
    # Compute loss and accuracy before starting (or resuming) training.
    _, valid_loss, valid_accuracy, auc = teacher_validate(model, dev_loader)
    print("\t* Validation loss before training: {:.4f}, accuracy: {:.4f}%, auc: {:.4f}".format(valid_loss,
                                                                                               (valid_accuracy * 100),
                                                                                               auc))
    # -------------------- Training epochs ------------------- #
    print("\n", 20 * "=", "Training Bert model on device: {}".format(device), 20 * "=")
    patience_counter = 0
    for epoch in range(start_epoch, epochs + 1):
        epochs_count.append(epoch)
        print("* Training epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy,_ = teacher_train(model, train_loader, optimizer, max_grad_norm)
        train_losses.append(epoch_loss)
        print("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%"
              .format(epoch_time, epoch_loss, (epoch_accuracy * 100)))
        print("* Validation for epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy, epoch_auc = teacher_validate(model, dev_loader)
        valid_losses.append(epoch_loss)
        print("-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%, auc: {:.4f}\n"
              .format(epoch_time, epoch_loss, (epoch_accuracy * 100), epoch_auc))
        # Update the optimizer's learning rate with the scheduler.
        scheduler.step(epoch_accuracy)
        # Early stopping on validation accuracy.
        if epoch_accuracy < best_score:
            patience_counter += 1
        else:
            best_score = epoch_accuracy
            patience_counter = 0
            torch.save({"epoch": epoch,
                        "model": model.state_dict(),
                        "best_score": best_score,
                        "epochs_count": epochs_count,
                        "train_losses": train_losses,
                        "valid_losses": valid_losses},
                       os.path.join(target_dir, "quora_best.pth.tar"))
        if patience_counter >= patience:
            print("-> Early stopping: patience limit reached, stopping...")
            break


# if __name__ == "__main__":
#    Teacher_train("../data/LCQMC_train.csv", "../data/LCQMC_dev.csv", "models")
