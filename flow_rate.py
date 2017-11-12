import matplotlib as mpl
mpl.use('agg')
import pylab
import pandas as pd
import os

NOCUT = ['6', '7a', '7c', '8a', '8b', '9b', '14','15', '16', '17', '21a', '21b', '23a', '23d']
CUT = ['7b', '9a', '9c', '9d', '10', '11', '12', '13', '18', '19', '20', '21c', '22', '23b', '23c', '23e', '23f']
ALL = NOCUT + CUT

NOCUT2 = ['6', '7a', '7c', '9b', '14','15', '16', '21a', '21b', '23a', '23d']
ALL2 = NOCUT2 + CUT

def flow_rate(sheet):
    df = pd.read_csv(sheet, skipinitialspace=True, names=['source', 'dest', 'time'])
    df['time'] = pd.to_datetime(df['time'])
    df.sort_values(by='time', inplace=True)
    times = list(df['time'])
    deltas = []
    starttime = times[0]
    for time in times:
        deltas.append((time - starttime) / pylab.timedelta64(1, 's'))
    
    total_count = 0
    prev_delta = None
    ants = []
    seconds = []
    for delta in deltas:
        if prev_delta != None and delta != prev_delta:
            seconds.append(prev_delta)
            ants.append(total_count)
        prev_delta = delta
        total_count += 1

    return ants[-1], seconds[-1], len(seconds)

def flow_rate_hist(sheets):
    ant_rates = []
    weights = []
    for sheet in sheets:
        ants, seconds, weight = flow_rate(sheet)
        ant_rate = seconds / ants
        #ant_rate = ants / seconds
        ant_rates.append(ant_rate)
        weights.append(float(weight))
        #weights.append(seconds)

    weights = pylab.array(weights)
    weights /= sum(weights)

    #print "ants per second"
    print "seconds per ant"
    mu = pylab.mean(ant_rates)
    print "mean", pylab.mean(ant_rates)
    wmean = pylab.average(ant_rates, weights=weights)
    print "weighted mean", wmean
    print "median", pylab.median(ant_rates)
    print "std", pylab.std(ant_rates, ddof=1)
    ant_rates = pylab.array(ant_rates)
    werror = (ant_rates - mu) * weights
    print "weighted std", ((sum(werror ** 2))) ** 0.5
    print "weighted std 2", (pylab.average((ant_rates - mu)**2, weights=weights)) ** 0.5
    pylab.figure()
    pylab.hist(ant_rates)
    pylab.savefig('ant_flow_rates.pdf', format='pdf')
    pylab.close()

def main():
    prefix = 'datasets/reformated_csv'
    sheets = []
    '''
    for csv_file in os.listdir(prefix):
        sheet = '%s/%s' % (prefix, csv_file)
        sheets.append(sheet)
    '''
    for sheet in CUT:
        sheets.append('%s/reformated_counts%s.csv' % (prefix, sheet))
    flow_rate_hist(sheets)

if __name__ == '__main__':
    main()
