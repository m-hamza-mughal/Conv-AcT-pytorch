#!/usr/bin/env python
import torch
from tqdm import tqdm
import copy
import logging
import sys
import numpy as np

from .test import accuracy


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train_model(model, train_loader, val_loader, model_path: str, num_epochs:int, learning_rate: float, finetune: bool = False):
    # Training
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100, 140, 180, 220, 250], gamma=0.9, verbose=True)
    device = next(model.parameters()).device

    best_model = None
    best_val_acc = 0

    if finetune:
        checkpoint = torch.load(f"{model_path}/best_model.pt", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    t_loader = iter(train_loader)
    for epoch in range(num_epochs):
            # TRAINING
            train_loss = 0.0
            # count = 0
            model.train()
            t_correct, t_total = 0, 0
            for batch in tqdm(range(1, len(val_loader) +1), desc=f"Epoch {epoch + 1}", position=0):
                try:
                    x, y = next(t_loader)
                except StopIteration:
                    logging.info("One data cycle complete")
                    t_loader = iter(train_loader)
                    x, y = next(t_loader)
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                loss = criterion(y_hat, y) #/ len(x)

                train_loss += loss.detach().cpu().item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                t_correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
                t_total += len(y)
                
                # count+=1
                if batch%1000==0:
                   logging.info(f"Iter {batch}/{len(val_loader) +1} Train Loss: {train_loss/batch} Train accuracy: {t_correct / t_total * 100:.2f}%")
            
            train_acc = t_correct / t_total * 100
            train_loss = train_loss/batch
            logging.info(f"Epoch {epoch + 1}/{num_epochs} train loss: {train_loss:.2f}")
            logging.info(f"Train accuracy: {train_acc:.2f}%")

            scheduler.step()

            val_loss, val_acc = evaluate_model(model, val_loader, criterion)

            # early stopping
            if(best_val_acc < val_acc):
                best_model = copy.deepcopy(model)
                
                best_val_acc = val_acc
                logging.info(f"Best accuracy: {best_val_acc}")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, f"{model_path}/best_model.pt")
                
                best_model = best_model.cpu()
                torch.cuda.empty_cache()
            
            stats = {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            }


    return best_model, stats

def evaluate_model(model, data_loader, criterion):
    device = next(model.parameters()).device
    val_loss = 0.0
    v_correct, v_total = 0, 0
    model.eval()

    accs = np.zeros(2)
    # count = 0

    with torch.no_grad():
        v_loader = iter(data_loader)
        for batch in tqdm(range(1, len(data_loader)+1), desc="Validating", position=0):
            x, y = next(v_loader)
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y) #/ len(x)
            val_loss += loss.detach().cpu().item()
            # count+=1

            # v_correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            # v_total += len(y)

            accs += accuracy(y_hat, y, topk=(1,5))
        
        # val_acc = v_correct / v_total * 100
    accs = accs/batch
    logging.info(f"Evaluation loss: {val_loss/batch:.2f}")
    # logging.info(f"Evaluation accuracy: {val_acc:.2f}%")
    print(f"Evaluation Top1 and Top5 accuracy {accs}") 
    return val_loss/batch, accs[0]
        

