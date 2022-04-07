from __future__ import print_function
from __future__ import division
import time
import copy

import torch
import os
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import audioset
from torch.utils.tensorboard import SummaryWriter
from itertools import product
# from tfln import set_parameter_requires_grad,initialize_model
from sklearn.metrics import confusion_matrix
from tools import get_uar,add_confusion_matrix
from train_test_save import save_checkpoint

import network

#--------------------------------------------------
# CHANGE ***TWO*** THINGS: MODEL AND TRAIN TEST SELECTION
#----------------------------------------------

#-------------------------------------------
# path on local machine
#-------------------------------------------
MYROOT = 'D:/ser_local_repo/ser'
MODELROOT = 'E:/projects/ser/model'
DATAROOT ='E:/projects/ser/database'
TBROOT = 'D:/ser_local_repo/ser/tb'

#-------------------------------------------
# path on colab
#-------------------------------------------
# MYROOT = '/content/drive/MyDrive/ser'
# MODELROOT = '/content/drive/MyDrive/asset'
# DATAROOT ='/content/drive/MyDrive/asset/database'
# TBROOT = '/content/drive/MyDrive/tb'

#-----------------------------------------------------------
# change parameter
#-----------------------------------------------------------
duo_code = ['enter2emodb', 'emodb2enter', 'casia2emodb', 'emodb2casia','enter2casia', 'casia2enter']

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
# model_name = "alexnet"

# Number of classes in the dataset
num_classes = 5

# Number of epochs to train for
num_epochs = 100

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True

#Flag for applying domain adaptation. When false only use target domain to validate.
#When true add mmd_loss as a part of regularization
domain_adaptation = True

#additional info
add_info = 'DA_FE_clfr_unfrozen'


#----------------------------------------
# check the situation without mmd layer
#----------------------------------------
# para = dict(
#     learning_rate = [1e-5]
#     ,batch_size = [16]
#     ,alpha=[0]
#     ,duo = ['enter2emodb', 'emodb2enter', 'casia2emodb', 'emodb2casia','enter2casia', 'casia2enter']
# )

#----------------------------------------
# check the situation with mmd layer
#----------------------------------------
para = dict(
    learning_rate = [1e-5]
    ,batch_size = [16]
    ,alpha=[0.1]
    ,duo = ['casia2enter']
)

para_values = [v for v in para.values()]


#--------------------------------
# iterate parameter
#--------------------------------
for learning_rate, batch_size, alpha, duo in product(*para_values):

    print(learning_rate, batch_size, alpha, duo)


    #----------------------------------------------
    # sort out common labels:
    #    [an, fear, hap, ntr, sad, sur, dis,bore]
    # en:[215, 215, 212,   0, 215, 215, 215,  0]
    # em:[127,  69,  71,  79,  62,   0,  46, 81]  
    # ca:[200, 200, 200, 200, 200, 200,   0,  0]
    #------------------------------------------------
    print("Initializing Datasets and Dataloaders...")

    if duo == duo_code[0]:
        en_text = "enter2emodb.txt"
        em_text = "emodb2enter.txt"
        en_file = "enterface1287_raw"
        em_file = "emodb535_raw"
        en_em_list = [1,2,3,5,7]
        data_classes = ['angry', 'fear', 'happy', 'sad','disgust']
        source_data = audioset.Audioset(DATAROOT, en_text, en_file, en_em_list,'src')
        target_data = audioset.Audioset(DATAROOT, em_text, em_file, en_em_list,'tar')
   
    if duo == duo_code[1]:
        en_text = "enter2emodb.txt"
        em_text = "emodb2enter.txt"
        en_file = "enterface1287_raw"
        em_file = "emodb535_raw"
        en_em_list = [1,2,3,5,7]
        data_classes = ['angry', 'fear', 'happy', 'sad','disgust']
        source_data = audioset.Audioset(DATAROOT, em_text, em_file, en_em_list,'src')
        target_data = audioset.Audioset(DATAROOT, en_text, en_file, en_em_list,'tar')
    
    if duo ==duo_code[2]:
        ca_text = "casia2emodb.txt"
        em_text = "emodb2casia.txt"
        ca_file = "casia1200_raw"
        em_file = "emodb535_raw"
        ca_em_list = [1,2,3,4,5]
        data_classes = ['angry', 'fear', 'happy', 'sad', 'neutral']
        source_data = audioset.Audioset(DATAROOT, ca_text, ca_file, ca_em_list,'src')
        target_data = audioset.Audioset(DATAROOT, em_text, em_file, ca_em_list,'tar')
    
    if duo==duo_code[3]:
        ca_text = "casia2emodb.txt"
        em_text = "emodb2casia.txt"
        ca_file = "casia1200_raw"
        em_file = "emodb535_raw"
        ca_em_list = [1,2,3,4,5]
        data_classes = ['angry', 'fear', 'happy', 'sad','neutral']
        source_data = audioset.Audioset(DATAROOT, em_text, em_file, ca_em_list,'src')
        target_data = audioset.Audioset(DATAROOT, ca_text, ca_file, ca_em_list,'tar')
    
    if duo == duo_code[4]:
        ca_text = "casia2enter.txt"
        en_text = "enter2casia.txt"
        ca_file = "casia1200_raw"
        en_file = "enterface1287_raw"
        ca_en_list = [1,2,3,5,6]
        data_classes = ['angry', 'fear', 'happy', 'sad','surprise']
        source_data = audioset.Audioset(DATAROOT, en_text, en_file, ca_en_list,'src')
        target_data = audioset.Audioset(DATAROOT, ca_text, ca_file, ca_en_list,'tar')

    if duo == duo_code[5]:
        ca_text = "casia2enter.txt"
        en_text = "enter2casia.txt"
        ca_file = "casia1200_raw"
        en_file = "enterface1287_raw"
        ca_en_list = [1,2,3,5,6]
        data_classes = ['angry', 'fear', 'happy', 'sad', 'surprise']
        source_data = audioset.Audioset(DATAROOT, ca_text, ca_file, ca_en_list,'src')
        target_data = audioset.Audioset(DATAROOT, en_text, en_file, ca_en_list,'tar')    

    #------------------------------------------------------------------------------------
    # load model
    #------------------------------------------------------------------------------------
    # Create training and validation datasets
    image_datasets = {'src': source_data,'tar':target_data}
    # Create training and validation dataloaders
    dataloaders_dict = {'src': DataLoader(source_data, batch_size=batch_size, shuffle=True),'tar':DataLoader(target_data, batch_size=batch_size, shuffle=True)}#, num_workers=4

    # Initialize the model for this run
    # model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

    model_name ='da_alexfc3'
    model_ft = network.DA_Alex_FC3(num_classes=len(data_classes))
    pretrained_root = os.path.join(MODELROOT,'pretrained_model')
    alexnet_path = os.path.join(pretrained_root,'alexnet-owt-7be5be79.pth')
    network.load_pretrained_net(model_ft,alexnet_path)
    print('Load pretrained alexnet parameters complete\n')
    #freeze feature layers
    for param in model_ft.features.parameters():
        param.requires_grad = False
    for param in model_ft.classifier.parameters():
        param.requires_grad = True
    for param in model_ft.final_classifier.parameters():
        param.requires_grad = True

    # Print the model we just instantiated
    # print(model_ft)

    # Send the model to GPU   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=learning_rate, momentum=0.9)
    # optimizer = optim.Adam(params_to_update, lr=learning_rate)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    
    def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
        since = time.time()

        val_acc_history = []

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.28

        parameters = add_info + '-' + duo +'-' + model_name + '-' + str(learning_rate)+ '-' + str(alpha) + '-' + str(batch_size)

        #--------------------------------
        # make tensorboard dir
        #--------------------------------
        tb_dir=os.path.join(TBROOT+'/'+duo + '-' +model_name,parameters)
        if not os.path.exists(tb_dir):
            os.makedirs(tb_dir)
        writer = SummaryWriter(log_dir=tb_dir,comment=parameters)

        #--------------------------------
        # make log dir
        #--------------------------------
        log_dir = os.path.join(TBROOT,"log")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_name = parameters
        print("log_name: ", log_name)
        f = open(os.path.join(log_dir, log_name + ".txt"), "a")

        for epoch in range(num_epochs):
            
            all_preds = torch.tensor([]).long()
            all_labels = torch.tensor([]).long()
            uar = 0.0

            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)


            # Each epoch has a training and validation phase
            for domain in ['src', 'tar']:
                if domain == 'src':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                total_clf_lss = 0.
                total_mmd_lss = 0.

                # Iterate over data.
                if domain_adaptation and domain == 'src':

                    iter_source = iter(dataloaders['src'])
                    iter_target = iter(dataloaders['tar'])
                    num_iter = len(dataloaders['src'])
                    
                    for i in range(1, num_iter+1):

                        source_data, source_label = iter_source.next()
                        target_data, _ = iter_target.next()
                        if i % len(dataloaders['tar']) == 0: 
                            iter_target = iter(dataloaders['tar'])
                        source_data, source_label = source_data.to(device), source_label.to(device)
                        target_data = target_data.to(device)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(domain == 'src'):
                            source_preds, mmd_lss = model(source_data, target_data) 
                            clf_lss = criterion(source_preds, source_label)
                            loss = (clf_lss +  alpha * mmd_lss)
                        
                            _, preds = torch.max(source_preds, 1)
                            
                            total_clf_lss += clf_lss.item()*len(source_label)
                            total_mmd_lss += mmd_lss.item()*len(source_label)

                            # backward + optimize only if in training phase     
                            loss.backward()
                            optimizer.step()

                        # statistics
                        running_loss += loss.item() * source_data.size(0)
                        running_corrects += torch.sum(preds == source_label.data)

                else:

                    for inputs, labels in dataloaders[domain]:

                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(domain == 'src'):
                            # Get model outputs and calculate loss
                            # Special case for inception because in training it has an auxiliary output. In train
                            #   mode we calculate the loss by summing the final output and the auxiliary output
                            #   but in testing we only consider the final output.
                            if is_inception and domain == 'src':
                                # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                                outputs, aux_outputs = model(inputs)
                                loss1 = criterion(outputs, labels)
                                loss2 = criterion(aux_outputs, labels)
                                loss = loss1 + 0.4*loss2
                            else:
                                if domain_adaptation and domain == 'tar':
                                    outputs, _ = model(inputs, inputs)
                                else:
                                    outputs = model(inputs)

                                loss = criterion(outputs, labels)

                            _, preds = torch.max(outputs, 1)

                            # backward + optimize only if in training phase
                            if domain == 'src':
                                loss.backward()
                                optimizer.step()
                            
                        #get uar for target domain
                        if domain == 'tar':
                            all_labels = torch.cat((all_labels, labels.cpu()),dim=0)
                            all_preds = torch.cat((all_preds, preds.cpu()),dim=0)

                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)


                epoch_loss = running_loss / len(dataloaders[domain].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[domain].dataset)

                
                # print('epoch:',epoch,'acc:',acc,'lss:',lss)
                if domain == 'src' and not(domain_adaptation):
                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(domain, epoch_loss, epoch_acc))
                    print('train:','epoch:',epoch,'acc:',epoch_acc,'lss:',epoch_loss,file = f)
                    writer.add_scalar("Lss/Epochs", epoch_loss, epoch)
                    writer.add_scalar("Acc/Epochs", epoch_acc, epoch)

                if domain == 'src' and domain_adaptation:
                    mean_clf_lss = total_clf_lss/len(dataloaders[domain].dataset)
                    mean_mmd_lss = total_mmd_lss/len(dataloaders[domain].dataset)
                    len(dataloaders['src'].dataset)
                    print('{} Loss: {:.4f} Acc: {:.4f} MMD_Loss: {:.4f} CLF_Loss: {:.4f}'.format(domain, epoch_loss, epoch_acc,mean_mmd_lss,mean_clf_lss))
                    print('{} Loss: {:.4f} Acc: {:.4f} MMD_Loss: {:.4f} CLF_Loss: {:.4f}'.format(domain, epoch_loss, epoch_acc,mean_mmd_lss,mean_clf_lss),file=f)
                    writer.add_scalar("Lss/Epochs", epoch_loss, epoch)
                    writer.add_scalar("Acc/Epochs", epoch_acc, epoch)  
                    writer.add_scalar("clf_lss/Epochs", mean_clf_lss, epoch)
                    writer.add_scalar("mmd_lss/Epochs", mean_mmd_lss, epoch)

                if domain == 'tar':
                    cm = confusion_matrix((all_labels), (all_preds),normalize=None)
                    uar = get_uar(cm)
                    print('{} Loss: {:.4f} Acc: {:.4f} UAR: {:.4f}'.format(domain, epoch_loss, epoch_acc,uar))
                    print('test:','epoch:',epoch,'acc:',epoch_acc,'uar:',uar,'lss:',epoch_loss,file = f)
                    writer.add_scalar("TEST_ACC/Epochs", epoch_acc, epoch)
                    writer.add_scalar("TESTt_UAR/Epochs", uar, epoch)
                    
                f.flush()

                # deep copy the model
                if domain == 'tar' and uar > best_acc:
                    best_acc = uar
                    best_model_wts = copy.deepcopy(model.state_dict())
                    print('best saved',file = f)
                    add_confusion_matrix(
                writer,cm,num_classes=len(data_classes),class_names=data_classes
                ,tag = parameters +'-'+ str(epoch) +'-'+ 'uar:' + str('%.4f'%uar)
                )
                
                    save_dir = os.path.join(MODELROOT,"best_saved")
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    checkpoint_name =  os.path.join(save_dir,parameters+'.pth.tar')

                    save_checkpoint({
                    'epoch': epoch ,
                    'arch': model_name,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),
                    }, is_best=True,filename=checkpoint_name)

                if domain == 'tar':
                    val_acc_history.append(epoch_acc)

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        f.close()
        writer.close()

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model, val_acc_history

    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))



