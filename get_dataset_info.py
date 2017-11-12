import argparse
import pandas as pd
from repair_ml import NOCUT, NOCUT2, CUT, ALL, ALL2, datasets_dict
import pylab
from pylab import mean

def get_df(sheet):
    choices = 'datasets/reformated_csv/reformated_counts%s.csv' % sheet
    df = pd.read_csv(choices, header=None, names=['source', 'dest', 'dt'], skipinitialspace=True)
    df['dt'] = pd.to_datetime(df['dt'])
    df.sort_values(by='dt', inplace=True)

    return df

def dataset_time(df):
    times = list(df['dt'])
    starttime = times[0]
    endtime = times[-1]
    return (endtime - starttime) / pylab.timedelta64(1, 'm')

def data_info(sheets):
    times = []
    lengths = []
    for sheet in sheets:
        df = get_df(sheet)
        time = dataset_time(df)
        times.append(time)
        length = len(df.index)
        lengths.append(length)

    print "times"
    print "min", min(times), "max", max(times), "mean", mean(times)

    print "lengths"
    print "min", min(lengths), "max", max(lengths), "mean", mean(times), "total", sum(lengths)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-sh', '--sheets', default=ALL, nargs='+')
    
    args = parser.parse_args()
    sheets = args.sheets
    if len(sheets) == 1:
        sheets = sheets[0]
        assert sheets in datasets_dict
        sheets = datasets_dict[sheets]

    print sheets

    data_info(sheets)

if __name__ == '__main__':
    main()
