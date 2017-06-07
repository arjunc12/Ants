import matplotlib as mpl
mpl.use('agg')
import pandas as pd
import pylab

def plot_repellant(repellant_df, label):
    df1 = repellant_df[repellant_df['lay_back'] == 1]
    df2 = repellant_df[((repellant_df['lay_back'] == 0) & (repellant_df['pluck_rate'] > 0))]
    df3 = repellant_df[((repellant_df['lay_back'] == 0) & (repellant_df['pluck_rate'] == 0))]

    dfs = [df1, df2, df3]
    colors = ['r', 'b', 'c']

    pylab.figure()
    for i in xrange(3):
        df = dfs[i]
        df = df.groupby('pluck_rate', as_index=False).agg(pylab.mean)
        color = colors[i]
        pylab.scatter(df['pluck_rate'], df['steps'], c=color)
    pylab.savefig('test_repellant%s.pdf' % label, format='pdf')

def main():
    #repellant_df1 = pd.read_csv('test_repellant.csv', names=['steps', 'pluck_rate', 'lay_back'])
    #plot_repellant(repellant_df1, '1') 
    
    repellant_df2 = pd.read_csv('test_repellant2.csv', names=['steps', 'pluck_rate', 'lay_back'])
    #print repellant_df2
    plot_repellant(repellant_df2, '2') 

if __name__ == '__main__':
    main()
