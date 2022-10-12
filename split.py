import os
import shutil
from sklearn.model_selection import train_test_split

from PIL import Image

data_dir = "ADC_Dataset/train"
output_dir = "data/ADC_Dataset_Split"
os.makedirs(output_dir, exist_ok=True)
img_size = 128


states = [folder for folder in os.listdir(data_dir)]

train_path = os.path.join(output_dir, "train")
test_path = os.path.join(output_dir, "test")

os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

train_cnt = 0
test_cnt = 0

print(states)
for state in states:
    data_path = os.path.join(data_dir,state)
    print(data_path)
    train_file_path = os.path.join(train_path,state)
    test_file_path = os.path.join(test_path,state)
    os.mkdir(train_file_path)
    os.mkdir(test_file_path)
    
    files = [file for file in os.listdir(data_path)]
    train, test = train_test_split(files, test_size=0.2, random_state=42)
    for file in train:
        img_src = os.path.join(data_path,file)
        img_dst = os.path.join(train_file_path,file)
        
#         img = Image.open(img_src)
#         img_128 = img.resize((img_size,img_size))
#         img_128.save(img_dst)
        shutil.copy(img_src,img_dst)
        train_cnt += 1
        
        
    for file in test:
        img_src = os.path.join(data_path,file)
        img_dst = os.path.join(test_file_path,file)
        
#         img = Image.open(img_src)
#         img_128 = img.resize((img_size,img_size))
#         img_128.save(img_dst)
        shutil.copy(img_src,img_dst)
        test_cnt += 1
        
print(f'Augmented dataset: {test_cnt} test, {train_cnt} train samples')
    
    