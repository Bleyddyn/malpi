from optparse import OptionParser

def getOptions():
    usage = "python pg-plot.py <data>"
    parser = OptionParser( usage=usage )
    (options, args) = parser.parse_args()
    if len(args) != 1:
        print usage
        exit()

    return (options, args)

options, args = getOptions()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import stats

plt.figure(1,figsize=(16, 18), dpi=80)
ax1=plt.subplot(211)
plt.xlabel('Episodes')
plt.ylabel('Running Reward')

if os.path.exists('karpathy_full.png'):
    img = mpimg.imread('karpathy_full.png')
    plt.subplot(212)
    imgplot = plt.imshow(img)
#plt.ion()

if True:
    data = np.loadtxt(args[0],delimiter=',')
    #data = data[100:,:]
    print "Mean/stdev: %f/%f" % (np.mean(data[:,1]),np.std(data[:,1]))
#print data.shape

    slope, intercept, r_value, p_value, std_err = stats.linregress(data)
    print slope, intercept

#plt.plot(data[-500:,0],data[-500:,1])
    ax1.plot(data[:,0],data[:,1])
#plt.figure(2)
    plt.show()
#    plt.pause(10)

