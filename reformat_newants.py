import pandas as pd

columns = ['explore', 'decay', 'journey_time'] 
df = pd.read_csv('new_ant_deviate10000.csv', header=None, names = columns)
'''
outfile = open('new_ant_deviate10000a.csv', 'w')
for name, group in df.groupby(['explore', 'decay']):
    explore, decay = name
    i = 0
    for row in group.iterrows():
        outfile.write('%0.2f, %0.2f, %d\n' % (explore, decay, row[1]['journey_time']))
        i += 1
    while i < 30:
        outfile.write('%0.2f, %0.2f, %d\n' % (explore, decay, -1))
        i += 1
outfile.close()
'''

'''
for name, group in df.groupby(['explore', 'decay']):
    explore, decay = name
    i = 0
    for row in group.iterrows():
        i += 1
    print name, i'''
for name, group in df.groupby(['explore', 'decay']):
    group2 = group[group['journey_time'] != -1]
    print len(group.index), len(group2.index)