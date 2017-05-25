# Ants
Set of modules for analyzing turtle ant data to find parameters that best explain 
choices made by real turtle ants, as well as modules for implementing and testing
various turtle ant-inspired algorithms.

All of the code is written in Python. In addition to the standard python modules,
running the code will require the python packages Numpy, Matplotlib/Pylab, Pandas,
Scipy, and Networkx. We suggest installing the lastest version of the Anaconda 
distribution of Python in order to easily obtain and maintain all the required
packages and dependencies.  Additionally, the code relies on the command line
tools imagemagick and ffmpeg; we suggest installing these through Homebrew.

To test some of the ant algorithms with a variety of networks and parameter values,
one can run the following sequence of commands on a remote server:

python make_bash.py -s rank uniform -g minimal full -dt exp -m 1000 -l 1k -emin 0.1 -emax 0.4 -dmin 0.01 -dmax 0.3 -estep 0.01 -dstep 0.01

python run_simulations.py -s rank uniform -g minimal full -dt exp -l 1k -x 5

python plot_results.py -s rank uniform -g minimal full -dt exp -m 1000 -l 1k

Then to download and view results locally:

python download_results.py -s rank uniform -g minimal full -dt exp -l 1k -me path_success_rate
