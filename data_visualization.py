import glob
import os
import numpy as np
import matplotlib.pyplot as plt

DATAROOT ='E:/projects/ser/database'

#----------------------------------------------
    # labels mapping:
    #    [an, fear, hap, ntr, sad, sur, dis,bore]
    # en:[215, 215, 212,   0, 215, 215, 215,  0]
    # em:[127,  69,  71,  79,  62,   0,  46, 81]  
    # ca:[200, 200, 200, 200, 200, 200,   0,  0]

duo_code = ['enter2emodb', 'emodb2enter', 'casia2emodb', 'emodb2casia','enter2casia', 'casia2enter']

data_classes = ['angry', 'fear', 'happy', 'neutral','sad','surprise','disgust','bored']# emotion labels map to number 1~8

en_file = "enterface1287_raw"
em_file = "emodb535_raw"
ca_file = "casia1200_raw"



enter_dict = {"an":1, "fe":2, "ha":3, "sa":5, "su":6, "di":7}
emodb_dict = {"W":1, "A":2, "F":3, "N":4, "T":5, "E":7, "L":8 }
casia_dict = {1,2,3,4,5,6}

en_files = glob.glob(DATAROOT+"/enterface1287_raw/**.wav")
em_files = glob.glob(DATAROOT+"/emodb535_raw/**.wav")
ca_files = glob.glob(DATAROOT+'/casia1200_raw/**.wav')
ca_count = [0,0,0,0,0,0,0,0,0]# count on i when its label i(1-8),index 0 is ignored.
en_count = [0,0,0,0,0,0,0,0,0]
em_count = [0,0,0,0,0,0,0,0,0]

for file in en_files:
    filename = os.path.split(file)[1]
    emo = filename.split(".")[0][4:6]
    en_count[int(enter_dict[emo])]+=1

for file in em_files:
    filename = os.path.split(file)[1]
    emo = filename.split(".")[0][5]
    em_count[int(emodb_dict[emo])]+=1

for file in ca_files:
    filename = os.path.split(file)[1]
    emo = filename.split(".")[0][6]
    ca_count[int(emo)]+=1

# print(en_count)
# print(em_count)
# print(ca_count)


# Create a grouped bar chart

fig, ax = plt.subplots(figsize=(10, 10))

#plot a bar chart
x = np.arange(len(data_classes))
width = 0.25
bar1 = ax.bar(x + width, ca_count[1:], width, label='CASIA')
bar2 = ax.bar(x + 2*width, en_count[1:], width, label='eENTERFACE')
bar3 = ax.bar(x + 3*width, em_count[1:], width, label='EmoDB')

x_tick_label=[]
for i in range(8):
    x_tick_label.append(data_classes[i]+'('+str(i)+')')


plt.xticks(x,x_tick_label)
ax.set_xlabel('Sample labels')
ax.set_ylabel('Sample numbers')
ax.set_title('Sample numbers by label and database')

# For each bar in the chart, add a text label.
for bar in ax.patches:
  # The text annotation for each bar should be its height.
  bar_value = bar.get_height()
  # Format the text with commas to separate thousands. You can do
  # any type of formatting here though.
  text = f'{bar_value:,}'
  # This will give the middle of each bar on the x-axis.
  text_x = bar.get_x() + bar.get_width() / 2
  # get_y() is where the bar starts so we add the height to it.
  text_y = bar.get_y() + bar_value
  # If we want the text to be the same color as the bar, we can
  # get the color like so:
  bar_color = bar.get_facecolor()
  # If you want a consistent color, you can just set it as a constant, e.g. #222222
  ax.text(text_x, text_y, text, ha='center', va='bottom', color=bar_color,
          size=12)

ax.legend()

plt.show()





