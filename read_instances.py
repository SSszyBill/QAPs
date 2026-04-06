import os
from main_GI import generate_isomorphic
# list all files in the current directory


graph_type = 'paley'
files = os.listdir(f'/home/xjx/A-xjx/QAPs/qaplibs/GI/{graph_type}')
files = [f.split('.')[0] for f in files]
# files.sort()

# sort by the numeric part after 'latin-'
files.sort(key=lambda x: int(x.split('-')[1]))

file_list = []
for file in files:
    n, _, _, _, _ = generate_isomorphic(f'GI/{graph_type}/{file}')
    if n <= 500:
        file_list.append(file)

with open(f'./qaplibs/GI/{graph_type}.txt', 'w') as f:
    f.write('\n'.join(file_list))

# files = os.listdir(f'/home/xjx/A-xjx/QAPs/qaplibs/QAPLIB')

# files  = [f.split('.')[0] for f in files]
# files = list(set(files))
# files.sort()


# with open(f'qaplib.txt', 'w') as f:
#     f.write('\n'.join(files))