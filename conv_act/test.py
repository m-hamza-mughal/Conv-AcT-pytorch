#!/usr/bin/env python
import torch
from tqdm import tqdm
import logging
import sys
import numpy as np


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return np.array(res)


def test_model(model, data_loader, model_path):
    device = next(model.parameters()).device
    val_loss = 0.0
    v_correct, v_total = 0, 0
    model.eval()
    accs = np.zeros(2)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    # count = 0

    logging.info(f"Loading checkpoint at {model_path}")
    checkpoint = torch.load(f"logs/{model_path}/best_model.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    logging.info("Testing")
    with torch.no_grad():
        v_loader = iter(data_loader)
        for batch in tqdm(range(1, len(data_loader)+1), desc="Testing", position=0):
            x, y = next(v_loader)
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y) #/ len(x)
            val_loss += loss.detach().cpu().item()
            # count+=1
            
            v_correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            v_total += len(y)

            
            accs += accuracy(y_hat, y, topk=(1,5))

        
        val_acc_top1 = v_correct / v_total * 100

    logging.info(f"Top1 and Top5 accuracy {accs/batch}") 
    logging.info(f"Testing loss: {val_loss/batch:.2f}")
    logging.info(f"Testing Top-1 accuracy: {val_acc_top1:.2f}%")
    return val_loss/batch, val_acc_top1