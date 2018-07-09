import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sys import argv
import pylab
import seaborn as sns

FIND_FOOD_FILE = '/iblsn/data/Arjun/Ants/ant_find_food.csv'

def find_food_heat(df):
    df = df[['explore', 'food dist', 'steps']]
    df = df.groupby(['explore', 'food dist'], as_index=False).agg(pylab.mean)
    data = df.pivot('explore', 'food dist', 'steps')
    pylab.figure()
    sns.heatmap(data)
    pylab.savefig('figs/find_food/ant_find_food.pdf', format='pdf')
    pylab.close()

if __name__ == '__main__':
    df = pd.read_csv(FIND_FOOD_FILE, skipinitialspace=True)
    find_food_heat(df)
