import os

# list all files in the current directory
files = os.listdir('/home/xjx/A-xjx/QAPs/qaplibs/QAPLIB')
files = [f.split('.')[0] for f in files]
files.sort()

with open('qaplib.txt', 'w') as f:
    f.write('\n'.join(files))