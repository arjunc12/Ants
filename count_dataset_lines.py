import os
from sys import argv

dataset_str = []
for dataset in argv[1:]:
    dataset_str.append('reformated_counts%s.csv' % dataset)
    
dataset_str = ' '.join(dataset_str)
command_str = 'wc -l %s' % dataset_str
print command_str
os.system(command_str)