import pandas as pd
import pylab

def walk_heat(df):
    x = df['add'].unique()
    y = df['decay'].unique()
    z = pylab.zeros((len(y), len(x)))
    grouped = df.groupby(['add', 'decay'])
    pos = 0
    for name, group in grouped:
        mu = pylab.mean(group['length'])
        i, j = pos % len(y), pos / len(y)
        z[i, j] = mu
        pos += 1
    assert pos == len(x) * len(y)
    #fig, ax = pylab.subplots(1, 1)
    pylab.pcolormesh(z, cmap='Reds')
    #ax.xaxis.set_tick_params(labeltop='on')
    #ax.yaxis.set_tick_params(
    ax = pylab.gca()
    ax.tick_params(top = True, labeltop = True, right=True, labelright = True)
    pylab.xticks(pylab.arange(len(x)) + 0.5, x)
    pylab.yticks(pylab.arange(len(y)) + 0.5, y)
    pylab.xlabel("pheromone add")
    pylab.ylabel("pheromone decay")
    pylab.show()
    
def main():
   df = pd.read_csv('ant_walks.csv', header=None, names = ['ants', 'add', 'decay', 'length', \
                                                           'first', 'last', 'revisits', 'finished'])
   walk_heat(df)
   
if __name__ == '__main__':
    main() 