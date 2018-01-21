import os
from optparse import OptionParser

def getOptions():
    usage = "python pg-plot.py <data>"
    parser = OptionParser( usage=usage )
    (options, args) = parser.parse_args()
    if len(args) != 1:
        print usage
        exit()

    return (options, args)

#options, args = getOptions()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import stats
from collections import deque
import ast

plt.figure(1,figsize=(16, 18), dpi=80)
ax1=plt.subplot(211)
plt.xlabel('Episodes')
plt.ylabel('Best Reward')

#cat current_run.txt 
#tail -50 CartPole-v0_ac_won.txt 

count = 100
if True:
    if os.path.exists('current_run.txt'):
        with open('current_run.txt', 'r') as f:
            runtxt = f.read()
            try:
                cnt = int(runtxt[runtxt.find('Iteration')+len('Iteration'):])
                if cnt > 0 and cnt < 10000:
                    count = cnt + 1
            except:
                print "Nothing in current_run.txt, defaulting to 100: " + runtxt

#score_ind = 13 # for pg-pole.py
#abbr = "pg"
score_ind = 5 # for ac-pole.py
abbr = "ac"

#[0.4870887644984899, 0.01731657794205047, 0.06378070828897703, 0.9948356417679789, 0.000766760240096467, 24.75, 5000.0]
with open('CartPole-v0_'+abbr+'_won.txt', 'r') as f:
    lines = deque(f, maxlen=count)
    y = []
    sorted_lines = []
    for line in lines:
        line = line.rstrip()
        resd = ast.literal_eval(line)
        if isinstance(resd,list):
            sorted_lines.append(resd)
            best = resd[score_ind]
            y.append(best)


    sorted_lines = sorted( sorted_lines, key=lambda a_entry: a_entry[score_ind] )
    for line in sorted_lines:
        print line

    print "# of runs: %d" % (len(y),)
    print "Min/Max: %f/%f" % (np.min(y),np.max(y))
    print "Mean/stdev: %f/%f" % (np.mean(y),np.std(y))

    slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(y)), y)
    print "Slope/intercept: %f/%f" % (slope, intercept)

#plt.plot(data[-500:,0],data[-500:,1])
    ax1.plot(y)
#plt.figure(2)
    plt.show()
#    plt.pause(10)

