import os
init_dir = 'dataset' #'dataset/'

for dirs in os.listdir(init_dir):
    if os.path.basename(dirs) == 'data':
        data_dir = os.path.join(init_dir, dirs)
        for sub_dirs in os.listdir(data_dir):
            list_dir = os.listdir(os.path.join(data_dir,sub_dirs))
            rgb_dir = list_dir[0]
            length = len(os.listdir(list_dir, rgb_dir))
            name = (sub_dirs) + '/ ' + str(length) + '\n'
            with open('dataset_index.txt', mode='a') as f:
                f.write(name)