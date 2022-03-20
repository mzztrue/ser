from torch.utils.data import DataLoader
import audioset_test
from tools import plot_spectrogram

import matplotlib.pyplot as plt

DATAROOT ='E:/projects/ser/database'

em_text = "emodb2enter.txt"
en_text = "enter2emodb.txt"
em_file = "emodb535_raw"
en_file = "enterface1287_raw"
en_em_list = [1,2,3,5,7]


target_data = audioset_test.Audioset(DATAROOT, em_text, em_file, en_em_list,'tar')
source_data = audioset_test.Audioset(DATAROOT, en_text, en_file, en_em_list,'src')

BATCH_SIZE=1
source_loader = DataLoader(source_data, BATCH_SIZE, shuffle=True)
target_loader = DataLoader(target_data, BATCH_SIZE, shuffle=True)

#check
print('source :', len(source_data), len(source_loader))
print('target :', len(target_data), len(target_loader))  
print('Load data complete')

for i in range(1):
    s_melspec,idx = next(iter(source_loader))
    t_melspec,idx = next(iter(target_loader))
