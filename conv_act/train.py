import torch
from tqdm import tqdm
import copy


def train_model(model, train_loader, val_loader, model_name: str, num_epochs:int, learning_rate: float, finetune: bool = False):
    # Training
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) #, weight_decay=0.1)
    device = next(model.parameters()).device

    best_model = None
    best_val_acc = 0

    if finetune:
        checkpoint = torch.load(f"logs/{model_name}/best_model.pt")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    for epoch in range(num_epochs):
            # TRAINING
            train_loss = 0.0
            count = 0
            model.train()
            t_loader = iter(train_loader)
            t_correct, t_total = 0, 0
            for batch in tqdm(range(len(train_loader)), desc=f"Epoch {epoch + 1}", position=0):
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
                
                count+=1
                #if count>3000:
                #    break
            train_acc = t_correct / t_total * 100
            train_loss = train_loss/count
            print(f"Epoch {epoch + 1}/{num_epochs} train loss: {train_loss:.2f}")
            print(f"Train accuracy: {train_acc:.2f}%")

            val_loss, val_acc = evaluate_model(model, val_loader, criterion)

            # early stopping
            if(best_val_acc < val_acc):
                best_model = copy.deepcopy(model)
                
                best_val_acc = val_acc
                print(f"Best accuracy: {best_val_acc}")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, f"logs/{model_name}/best_model.pt")
                
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
    count = 0

    with torch.no_grad():
        v_loader = iter(data_loader)
        for batch in tqdm(range(len(data_loader)), desc="Validating", position=0):
            x, y = next(v_loader)
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y) #/ len(x)
            val_loss += loss.detach().cpu().item()
            count+=1

            v_correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            v_total += len(y)
        
        val_acc = v_correct / v_total * 100
        
    print(f"Evaluation loss: {val_loss/count:.2f}")
    print(f"Evaluation accuracy: {val_acc:.2f}%")
    return val_loss/count, val_acc
        

