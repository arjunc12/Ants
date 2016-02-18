import pandas as pd
import pylab

df = pd.read_csv('pure_random_walk.csv', header=None, names = ['steps'])
pylab.hist(df['steps'])
pylab.title("mean = %0.2f, median = %d" % (pylab.mean(df['steps']), pylab.median(df['steps'])))
pylab.savefig('pure_rand_walk.png', format='png')