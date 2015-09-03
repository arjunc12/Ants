import pandas as pd
import datetime
import numpy as np
from sys import argv

in_file = open(argv[1])
out_file = open('counts.csv', 'a')

for line in in_file:
    line = line.strip('\n')
    line = line.split(',')
    if line[0] == 'Year':
        continue
    year = int(line[0])
    day = int(line[1])
    hour = int(line[2])
    minute = int(line[3])
    second = float(line[4])
    dt = datetime.datetime(int(line[0]), 1, 1)
    delta = datetime.timedelta(days=day, hours=hour, minutes=minute, seconds=second)
    dt += delta
    dt = str(dt)
    edge = line[6]
    edge = edge.split('-')
    if len(edge) < 2:
        continue
    source = edge[0].strip()
    dest = edge[1].strip()
    out_file.write('%s, %s, %s\n' % (source, dest, dt))

in_file.close()
out_file.close()