from repair_ml import *
import pylab

EXPLORE_PROB = 0.05
DECAY_RATE = 0.01

def exponent_likelihood(choices, exponent, G=None, explore=EXPLORE_PROB,\
                        decay=DECAY_RATE, decay_type='exp', ghost=True): 
    likelihood_func = lambda G, source, dest, explore :\
                             uniformn_likelihood(G, source, dest, explore,\
                                                 exponent)

    return param_likelihood(choices, explore, decay, likelihood_func, decay_type, G, ghost)


def exponent_likelihood_list(sheet, exponents, explore=EXPLORE_PROB,\
                             decay=DECAY_RATE, decay_type='exp', ghost=True):
    G = None
    choices = '%s/reformated_counts%s.csv' % (DATASETS_DIR, sheet)
    likelihoods = []

    for exponent in exponents:
        likelihood, G = exponent_likelihood(choices, exponent, G, explore,\
                                            decay, decay_type, ghost)

        likelihoods.append(likelihood)

    return likelihoods

def exponent_ml_analysis(sheets, exponents, explore=EXPLORE_PROB,\
                               decay=DECAY_RATE, decay_type='exp', ghost=True):
    f = open('%s/uniform_ml.csv' % OUT_DIR, 'a')
    for sheet in sheets:
        print sheet
        likelihoods = exponent_likelihood_list(sheet, exponents, explore,\
                                               decay, decay_type, ghost)
        assert len(likelihoods) == len(exponents)
        for i, (exponent, likelihood) in enumerate(zip(exponents, likelihoods)):
            f.write('%s, %s, %d, %f, %f, %f, %f\n' % (sheet, decay_type,\
                                                      ghost, exponent,\
                                                      decay, explore,\
                                                      likelihood))
    f.close()
        

def main():
    #strategy_choices = ['uniform', 'max', 'maxz', 'rank']
    #decay_choices = ['linear', 'const', 'exp']
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--label', required=True)
    parser.add_argument('-sh', '--sheets', nargs='+', required=True)
    
    parser.add_argument('-c', '--cumulative', action='store_true')
   
   
    parser.add_argument('--decay', type=float, default=DECAY_RATE)
    parser.add_argument('--explore', type=float, default=EXPLORE_PROB)
    parser.add_argument('-t', '--threshold', type=float, default=MIN_DETECTABLE_PHEROMONE)

    parser.add_argument('-kmin', type=float, default=1)
    parser.add_argument('-kmax', type=float, default=2)
    parser.add_argument('-kstep', type=float, default=0.01)

    parser.add_argument('-o', '--out', action='store_true')
    
    parser.add_argument('-g', '--ghost', action='store_true')
    
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--heat', action='store_true')
    parser.add_argument('--write_file', action='store_true')
    
    args = parser.parse_args()
    label = args.label
    sheets = args.sheets
    if sheets == ['nocut']:
        sheets = NOCUT
    elif sheets == ['nocut2']:
        sheets = NOCUT2
    elif sheets == ['cut']:
        sheets = CUT
    elif sheets == ['all']:
        sheets = ALL
    elif sheets == ['all2']:
        sheets = ALL2
    
    cumulative = args.cumulative
    out = args.out
    ghost = args.ghost
    if ghost:
        label += '_ghost' 
    
    heat = args.heat
    plot = args.plot
    write_file = args.write_file

    kmin, kmax, kstep = args.kmin, args.kmax, args.kstep
    exponents = pylab.arange(kmin, kmax, kstep) 

    if not (heat or plot or write_file):
        print "error: must select an action"
        return None
    exponent_ml_analysis(sheets, exponents, ghost=ghost)

if __name__ == '__main__':
    main()
