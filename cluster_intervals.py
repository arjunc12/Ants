from interval_utils import get_times
import pylab
from sys import argv

def scatter_times(name, sheets):
    means = []
    medians = []
    delays = []
    mean_points = []
    med_points = []
    for sheet, delay in sheets:
        delays.append(delay)
        times = get_times(sheet)
        mean = pylab.mean(times)
        median = pylab.median(times)
        means.append(mean)
        medians.append(median)
        mean_points.append((mean, sheet))
        med_points.append((median, sheet)) 
    
    print "----------mean points-----------"    
    for mean, sheet in sorted(mean_points):
        print mean, sheet
    print "----------median points-----------"
    for median, sheet in sorted(med_points):
        print median, sheet
          
    pylab.scatter(delays, means, color='r')
    pylab.scatter(delays, medians, color='b')
    print "show"
    pylab.show()
    
if __name__ == '__main__':
    name = argv[1]
    datasets = argv[2:]
    assert len(datasets) % 2 == 0
    sheets = []
    for i in xrange(0, len(datasets) - 1, 2):
        sheet = datasets[i]
        delay = float(datasets[i + 1])
        sheets.append((sheet, delay))
    scatter_times(name, sheets)