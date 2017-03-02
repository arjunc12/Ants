import pandas as pd
from sys import argv
import pylab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_find_food3d(df):
    x = []
    y = []
    z = []
    for name, group in df.groupby(['explore', 'distance']):
        explore, distance = name
        x.append(explore)
        y.append(distance)
        z.append(pylab.median(group['time']))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(x, y, z)
    ax.set_xlabel('explore probability')
    ax.set_ylabel('distance')
    ax.set_zlabel('time to find food')
    pylab.savefig("find_food3d.png", format="png")
    pylab.close()
    
def plot_find_food2d(df):
    pylab.figure()
    for name1, group1 in df.groupby('explore'):
        x = []
        y = []
        explore = name1
        for name2, group2 in group1.groupby('distance'):
            distance = name2
            x.append(distance)
            y.append(pylab.median(group2['time']))
        #print explore, explore % 0.15
        if explore % 0.15 < 6e-17:
            pylab.plot(x, y, label=str(explore))
            pylab.xlabel('distance')
            pylab.ylabel('time')
            pylab.legend(loc=2)
            pylab.savefig("find_food/find_food_e%f.png" % explore, format="png")
    
if __name__ == '__main__':
    columns = ['ants', 'explore', 'decay', 'distance', 'time']
    filename = argv[1]
    
    df = pd.read_csv(filename, header=None, names = columns, na_values='nan', skipinitialspace=True)
    
    # plot_find_food3d(df)
    plot_find_food2d(df)