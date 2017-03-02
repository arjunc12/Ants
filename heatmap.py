#!/usr/bin/env python

from matplotlib import pylab

from numpy import random
decays   = [0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95]
explores = [0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95]
likelihoods = random.randn(19,19)


title_str = "something"



pylab.figure()
pylab.title(title_str)
hm = pylab.pcolormesh(likelihoods, cmap='nipy_spectral')
ax = pylab.gca()
ax.axis('tight')
cb = pylab.colorbar(hm)
cb.ax.set_ylabel('log-likelihood')
pylab.tick_params(which='both', bottom='off', top='off', left='off', right='off', \
labeltop='off', labelbottom='off', labelleft='off', labelright='off')

pylab.xlabel("explore probability (%0.2f - %0.2f)" % (min(explores), max(explores)))
pylab.ylabel("pheromone decay (%0.2f-%0.2f)" % (min(decays), max(decays)))

print "show"
pylab.show()

pylab.close()