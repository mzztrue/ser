import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import shutil

from tools import get_uar

def dadcnn_train(device,source_loader, target_loader, model, criterion, optimizer, epoch, alpha):
    
    model.train()
    
    iter_source = iter(source_loader)
    iter_target = iter(target_loader)
    num_iter = len(source_loader)
    
    corrects = 0
    total = 0
    total_loss = 0.
    total_clf_lss = 0.
    total_mmd_lss = 0.
    
    for i in range(1, num_iter+1):
        source_data, source_label = iter_source.next()
        target_data, _ = iter_target.next()
        if i % len(target_loader) == 0: 
            iter_target = iter(target_loader)
        source_data, source_label = source_data.to(device), source_label.to(device)
        target_data = target_data.to(device)
        
        source_preds, mmd_lss = model(source_data, target_data) 
        clf_lss = criterion(source_preds, source_label)
        
        loss = (clf_lss +  alpha * mmd_lss)

        corrects += source_preds.argmax(dim=1).eq(source_label).sum().item()
        total += len(source_label)
        
        total_loss += loss.item()*len(source_label)
        total_clf_lss += clf_lss.item()*len(source_label)
        total_mmd_lss += mmd_lss.item()*len(source_label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      

    acc = corrects/total
    lss = total_loss / total
    mean_clf_lss = total_clf_lss/total
    mean_mmd_lss = total_mmd_lss/total

    return acc, lss, mean_clf_lss, mean_mmd_lss

def train(device, source_loader, model, criterion, optimizer, epoch):
    
    corrects = 0
    total = 0
    total_loss = 0.

    model.train()
    for source_input in source_loader:
        data, labels = source_input
        data,labels = data.to(device), labels.to(device)

        preds = model(data)        
        loss = criterion(preds, labels)
        corrects += preds.argmax(dim=1).eq(labels).sum().item()
        total += len(labels)
        total_loss += loss.item()*len(labels) 

        optimizer.zero_grad()
        loss.backward()     
        optimizer.step()

    # print("train:",preds.argmax(dim=1), labels)
    acc = corrects/total
    lss = total_loss / total

    return acc,lss
    
def test(device, target_loader, model,da=1):
    corrects = 0
    total = 0
    
    all_preds = torch.tensor([]).long()
    all_labels = torch.tensor([]).long()

    model.eval()
    with torch.no_grad():
        for target_input in target_loader:
            data, labels = target_input
            data, labels = data.to(device), labels.to(device)
            if(da):
                preds, _ = model(data, data)
            else:
                preds = model(data)#.long()

            corrects += preds.argmax(dim=1).eq(labels).sum().item()
            total += len(labels)
            all_labels = torch.cat((all_labels, labels.cpu()),dim=0)
            all_preds = torch.cat((all_preds, preds.cpu()),dim=0)
       
    acc = corrects/total
    # print("test:",all_preds.argmax(dim=1), all_labels)
    cm = confusion_matrix((all_labels), all_preds.argmax(dim=1),normalize=None)
    uar = get_uar(cm)
    return acc,uar,cm

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
        print("best saved")        

def finetune(device, source_loader, model, criterion, optimizer, epoch):
    
    corrects = 0
    total = 0
    total_loss = 0.

    model.train()
    for source_input in source_loader:
        data, labels = source_input
        data,labels = data.to(device), labels.to(device)

        preds = model(data)        
        loss = criterion(preds, labels)
        corrects += preds.argmax(dim=1).eq(labels).sum().item()
        total += len(labels)
        total_loss += loss.item()*len(labels) 

        optimizer.zero_grad()
        loss.backward()     
        optimizer.step()

    # print("train:",preds.argmax(dim=1), labels)
    acc = corrects/total
    lss = total_loss / total

    return acc,lss
