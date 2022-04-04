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
from tfln import set_parameter_requires_grad,initialize_model
from sklearn.metrics import confusion_matrix
from tools import get_uar,add_confusion_matrix
from train_test_save import save_checkpoint

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
model_name = "alexnet"

# Number of classes in the dataset
num_classes = 5

# Number of epochs to train for
num_epochs = 100

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True

#additional info
add_info = 'FE_ftr_clfr_unfrozen'


#----------------------------------------
# check the situation without mmd layer
#----------------------------------------
para = dict(
    learning_rate = [1e-5]
    ,batch_size = [16]
    ,alpha=[0]
    ,duo = ['enter2emodb', 'emodb2enter', 'casia2emodb', 'emodb2casia','enter2casia', 'casia2enter']
)

#----------------------------------------
# check the situation with mmd layer
#----------------------------------------
# para = dict(
#     learning_rate = [1e-5]
#     ,batch_size = [16]
#     ,alpha=[1.0]
#     ,duo = ['emodb2casia','enter2casia']
# )

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
    dataloaders_dict = {x: DataLoader(target_data, batch_size=batch_size, shuffle=True) for x in ['src', 'tar']}#, num_workers=4

    # Initialize the model for this run
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

    # Print the model we just instantiated
    print(model_ft)

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

        parameters = add_info + duo +'-' + model_name + '-' + str(learning_rate)+ '-' + str(alpha) + '-' + str(batch_size)

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

                # Iterate over data.
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
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        if domain == 'src':
                            loss.backward()
                            optimizer.step()
                        
                        if domain == 'tar':
                            all_labels = torch.cat((all_labels, labels.cpu()),dim=0)
                            all_preds = torch.cat((all_preds, preds.cpu()),dim=0)

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders[domain].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[domain].dataset)
                
                # print('epoch:',epoch,'acc:',acc,'lss:',lss)
                if domain == 'src':
                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(domain, epoch_loss, epoch_acc))
                    print('train:','epoch:',epoch,'acc:',epoch_acc,'lss:',epoch_loss,file = f)
                    writer.add_scalar("Lss/Epochs", epoch_loss, epoch)
                    writer.add_scalar("Acc/Epochs", epoch_acc, epoch)
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
    #-----------------------------------------------------------------
    # architecture: LeNet without mmd
    #-----------------------------------------------------------------
    # arch ='lenet'
    # da=0
    # model = network.LeNet_finetune(num_classes=len(data_classes))

    # # load the model checkpoint
    # checkpoint = torch.load('E:/projects/ser/model/best_saved/enter2casia-lenet-1e-05-0-512.pth.tar')
    # # load model weights state_dict
    # model.load_state_dict(checkpoint['state_dict'])
    # print('Previously trained model weights state_dict loaded...')

    #-----------------------------------------------------------------
    # architecture: LeNet with mmd
    #-----------------------------------------------------------------
    # arch ='da_lenet_fc1'
    # da=1
    # model = network.DA_LeNet_FC1(num_classes=len(data_classes))

    # #-----------------------------------------------------------------
    # # architecture: pretrained_alexnet + fc layern + mmd + the rest
    # #-----------------------------------------------------------------
    # arch ='da_alexfc3'
    # da=1
    # model = network.DA_Alex_FC3(num_classes=len(data_classes))
    # pretrained_root = os.path.join(MODELROOT,'pretrained_model')
    # alexnet_path = os.path.join(pretrained_root,'alexnet-owt-7be5be79.pth')
    # network.load_pretrained_net(model,alexnet_path)
    # print('Load pretrained alexnet parameters complete\n')

    #-----------------------------------------------------------------
    # architecture: pretrained alexnet without mmd
    #-----------------------------------------------------------------
    # arch ='alex'
    # da=0
    # model = network.Alexnet_finetune(num_classes=len(data_classes))
    # pretrained_root = os.path.join(MODELROOT,'pretrained_model')
    # alexnet_path = os.path.join(pretrained_root,'alexnet-owt-7be5be79.pth')
    # network.load_pretrained_net(model,alexnet_path)
    # print('Load pretrained alexnet parameters complete\n')

    #-----------------------------------------------------------------
    # architecture: pretrained vgg11bn with mmd
    #-----------------------------------------------------------------
    # arch ='vgg11bn_fc2'
    # da=1
    # model = network.DA_VGG11bn_FC2(num_classes=len(data_classes))
    # pretrained_root = os.path.join(MODELROOT,'pretrained_model')
    # vggbn11_path = os.path.join(pretrained_root,'vgg11_bn-6002323d.pth')
    # network.load_pretrained_net(model,vggbn11_path)
    # print('Load pretrained vggbn11 parameters complete\n')

    #-----------------------------------------------------------------
    # architecture: pretrained vgg11bn without mmd
    #-----------------------------------------------------------------
    # arch ='vgg11bn'
    # da=0
    # model = network.VGG11bn_finetune(num_classes=len(data_classes))
    # pretrained_root = os.path.join(MODELROOT,'pretrained_model')
    # vggbn11_path = os.path.join(pretrained_root,'vgg11_bn-6002323d.pth')
    # network.load_pretrained_net(model,vggbn11_path)
    # print('Load pretrained vggbn11 parameters complete\n')

    #------------------------------------------------------------------
    # create optimizer and training criterion
    #------------------------------------------------------------------
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # epochs = 200

    # # load trained optimizer state_dict
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # print('Previously trained optimizer state_dict loaded...')


