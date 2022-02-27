import torch
import os
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import audioset
import network
from tools import add_confusion_matrix
from train_test_save import train,dadcnn_train,test,save_checkpoint
from torch.utils.tensorboard import SummaryWriter
from itertools import product

MYROOT = 'D:/ser_local_repo/ser'
MODELROOT = 'E:/projects/ser/pretrained_model'
DATAROOT ='E:/projects/ser/database'
TBROOT = 'D:/ser_local_repo/ser/tb'

#-----------------------------------------------------------
# change parameter
#-----------------------------------------------------------
duo_code = ['enter2emodb', 'emodb2enter', 'casia2emodb', 'emodb2casia','enter2casia', 'casia2enter']

para = dict(
    batch_size = [8,16,64,512]
    ,alpha = [100.0,1.0,0.01]
    ,learning_rate = [1e-5,1e-4,1e-3]
    ,duo = ['enter2emodb', 'emodb2enter', 'casia2emodb', 'emodb2casia','enter2casia', 'casia2enter']
)

para_values = [v for v in para.values()]

#--------------------------------
# iterate parameter
#--------------------------------
for batch_size, alpha, learning_rate, duo in product(*para_values):
    print(alpha, batch_size,learning_rate, duo)


    #----------------------------------------------
    # sort out common labels:
    # [an, fear, hap, ntr, sad, sur, dis,bore]
    # en:[215, 215, 212,   0, 215, 215, 215,  0]
    # em:[127,  69,  71,  79,  62,   0,  46, 81]  
    # ca:[200, 200, 200, 200, 200, 200,   0,  0]
    #------------------------------------------------
    if duo == duo_code[0]:
        en_text = "enter2emodb.txt"
        em_text = "emodb2enter.txt"
        en_file = "enterface1287_raw"
        em_file = "emodb535_raw"
        en_em_list = [1,2,3,5,7]
        data_classes = ['angry', 'fear', 'happy', 'sad','disgust']
        source_data = audioset.Audioset(DATAROOT, en_text, en_file, en_em_list)
        target_data = audioset.Audioset(DATAROOT, em_text, em_file, en_em_list)
   
    if duo == duo_code[1]:
        en_text = "enter2emodb.txt"
        em_text = "emodb2enter.txt"
        en_file = "enterface1287_raw"
        em_file = "emodb535_raw"
        en_em_list = [1,2,3,5,7]
        data_classes = ['angry', 'fear', 'happy', 'sad','disgust']
        source_data = audioset.Audioset(DATAROOT, em_text, em_file, en_em_list)
        target_data = audioset.Audioset(DATAROOT, en_text, en_file, en_em_list)
    
    if duo ==duo_code[2]:
        ca_text = "casia2emodb.txt"
        em_text = "emodb2casia.txt"
        ca_file = "casia1200_raw"
        em_file = "emodb535_raw"
        ca_em_list = [1,2,3,4,5]
        data_classes = ['angry', 'fear', 'happy', 'sad', 'neutral']
        source_data = audioset.Audioset(DATAROOT, ca_text, ca_file, ca_em_list)
        target_data = audioset.Audioset(DATAROOT, em_text, em_file, ca_em_list)
    
    if duo==duo_code[3]:
        ca_text = "casia2emodb.txt"
        em_text = "emodb2casia.txt"
        ca_file = "casia1200_raw"
        em_file = "emodb535_raw"
        ca_em_list = [1,2,3,4,5]
        data_classes = ['angry', 'fear', 'happy', 'sad','neutral']
        source_data = audioset.Audioset(DATAROOT, em_text, em_file, ca_em_list)
        target_data = audioset.Audioset(DATAROOT, ca_text, ca_file, ca_em_list)
    
    if duo == duo_code[4]:
        ca_text = "casia2enter.txt"
        en_text = "enter2casia.txt"
        ca_file = "casia1200_raw"
        en_file = "enterface1287_raw"
        ca_en_list = [1,2,3,5,6]
        data_classes = ['angry', 'fear', 'happy', 'sad','surprise']
        source_data = audioset.Audioset(DATAROOT, en_text, en_file, ca_en_list)
        target_data = audioset.Audioset(DATAROOT, ca_text, ca_file, ca_en_list)

    if duo == duo_code[5]:
        ca_text = "casia2enter.txt"
        en_text = "enter2casia.txt"
        ca_file = "casia1200_raw"
        en_file = "enterface1287_raw"
        ca_en_list = [1,2,3,5,6]
        data_classes = ['angry', 'fear', 'happy', 'sad', 'surprise']
        source_data = audioset.Audioset(DATAROOT, ca_text, ca_file, ca_en_list)
        target_data = audioset.Audioset(DATAROOT, en_text, en_file, ca_en_list)    

    #-------------------------------------------------------------
    # load model, create optimizer and training criterion
    #-------------------------------------------------------------
    arch ='da_alex1'
    model = network.DA_Alex_FC1(num_classes=len(data_classes))

    alexnet_path = os.path.join(MODELROOT,'alexnet-owt-7be5be79.pth')
    network.load_pretrained_net(model,alexnet_path)
    print('Load pretrained alexnet parameters complete\n')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    epochs = 100

    parameters = duo +'-' + arch + '-' + str(learning_rate)+ '-' + str(alpha) + '-' + str(batch_size)
    #--------------------------------
    # make tensorboard dir
    #--------------------------------
    tb_dir=os.path.join(TBROOT+'/'+duo + '-' +arch,parameters)
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

    #--------------------------------
    # load data
    #--------------------------------
    source_loader = DataLoader(source_data, batch_size, shuffle=True)
    target_loader = DataLoader(target_data, batch_size, shuffle=True)

    #check
    print('source :', len(source_data), len(source_loader))
    print('target :', len(target_data), len(target_loader))  
    print('Load data complete')


    #--------------------------------
    # load model
    #--------------------------------

    #--------------------------------
    # train,test,save
    #--------------------------------
    best_acc = 0.3

    for epoch in range(1, epochs+1):
        #--------------------------------
        # train
        #--------------------------------
        acc, lss, clf_lss, mmd_lss = dadcnn_train(device, source_loader, target_loader, model, criterion, optimizer, epoch, alpha)
        print('epoch:',epoch,'acc:',acc,'lss:',lss,'clf_lss:',clf_lss, "mmd_lss:",mmd_lss)
        print('epoch:',epoch,'acc:',acc,'lss:',lss,'clf_lss:',clf_lss, "mmd_lss:",mmd_lss,file = f)

        writer.add_scalar("Lss/Epochs", lss, epoch)
        writer.add_scalar("Acc/Epochs", acc, epoch)
        writer.add_scalar("clf_lss/Epochs", clf_lss, epoch)
        writer.add_scalar("mmd_lss/Epochs", mmd_lss, epoch)

        #--------------------------------
        # test
        #--------------------------------       
        t_acc,t_uar,cm = test(device, target_loader, model,da=1)
        print('epoch:',epoch,'test_acc:',t_acc,'test_uar:',t_uar)
        print('epoch:',epoch,'test_acc:',t_acc,'test_uar:',t_uar,file = f)
        
        f.flush()

        writer.add_scalar("TEST_ACC/Epochs", t_acc, epoch)
        writer.add_scalar("TESTt_UAR/Epochs", t_uar, epoch)

        #--------------------------------
        #save
        #--------------------------------
        is_best = t_uar > best_acc
        best_acc = max(t_uar, best_acc)
        
        if(is_best):
            print('best saved',file = f)
            add_confusion_matrix(
                writer,cm,num_classes=len(data_classes),class_names=data_classes
                ,tag = parameters +'-'+ str(epoch) + '\n' + 'train_acc:'+ str('%.4f'%acc)+ 'test_acc:'+ str('%.4f'%t_acc) + 'test_uar:' + str('%.4f'%t_uar)
                )
        
        checkpoint_name= MODELROOT+'/'+parameters+'.pth.tar'
        save_checkpoint({
        'epoch': epoch ,
        'arch': arch,
        'state_dict': model.state_dict(),
        'best_acc': best_acc,
        'optimizer' : optimizer.state_dict(),
        }, is_best,checkpoint_name)

   
    f.close()
    writer.close()