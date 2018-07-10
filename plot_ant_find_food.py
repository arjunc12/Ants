import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sys import argv
import pylab
import seaborn as sns

FIND_FOOD_FILE = '/iblsn/data/Arjun/Ants/ant_find_food.csv'

def find_food_heat(df, max_steps, steps_label, decay):
    df = df[(df['max steps'] == max_steps) & (df['decay'] == decay)]
    print df
    for name, group in df.groupby(['graph', 'strategy', 'decay type']):
        graph, strategy, decay_type, = name
        group = group[['explore', 'food dist', 'steps']]
        group = group.groupby(['explore', 'food dist'], as_index=False).agg(pylab.mean)
        data = group.pivot('explore', 'food dist', 'steps')
        pylab.figure()
        sns.heatmap(data)
        outname = 'ant_find_food_%s_%s_%s%s.pdf' % (strategy, graph, decay_type, steps_label)
        pylab.savefig('figs/find_food/%s' % outname, format='pdf')
        pylab.close()

if __name__ == '__main__':
    df = pd.read_csv(FIND_FOOD_FILE, skipinitialspace=True)
    find_food_heat(df, 100000, '100k', 0.01)
    find_food_heat(df, 10000, '10k', 0.01)
