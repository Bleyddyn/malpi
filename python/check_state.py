import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import stats

with open('one_experience.pickle', 'rb') as pf:
    exp = pickle.load(pf)

states, actions, rewards, batch_done, new_states = exp

print states.shape
print actions
print rewards
print batch_done
print new_states.shape

#states.shape = (32, 4, 84, 84)
si = 0
frame1 = new_states[si,0,:,:]
frame2 = new_states[si,1,:,:]
frame3 = new_states[si,2,:,:]
frame4 = new_states[si,3,:,:]


plt.figure(1,figsize=(16, 18), dpi=80)
ax1=plt.subplot(411)
plt.imshow(frame1,cmap='gray')
ax1=plt.subplot(412)
plt.imshow(frame2,cmap='gray')
ax1=plt.subplot(413)
plt.imshow(frame3,cmap='gray')
ax1=plt.subplot(414)
plt.imshow(frame4,cmap='gray')

#plt.xlabel('Episodes')
#plt.ylabel('Running Reward')
#
#if os.path.exists('karpathy_full.png'):
#    img = mpimg.imread('karpathy_full.png')
#    plt.subplot(212)
#    imgplot = plt.imshow(img)
##plt.ion()
#
#if True:
#    data = np.loadtxt(args[0],delimiter=',')
#    #data = data[100:,:]
#    print "Mean/stdev: %f/%f" % (np.mean(data[:,1]),np.std(data[:,1]))
##print data.shape
#
#    slope, intercept, r_value, p_value, std_err = stats.linregress(data)
#    print slope, intercept
#
##plt.plot(data[-500:,0],data[-500:,1])
#    ax1.plot(data[:,0],data[:,1])
##plt.figure(2)
plt.show()
##    plt.pause(10)
#
