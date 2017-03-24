import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import stats

img=mpimg.imread('karpathy.png')

data = np.loadtxt('save.txt',delimiter=',')
#print data.shape

slope, intercept, r_value, p_value, std_err = stats.linregress(data)
print slope, intercept

plt.figure(1,figsize=(16, 18), dpi=80)
plt.subplot(211)
plt.xlabel('Episodes')
plt.ylabel('Running Reward')
#plt.plot(data[-500:,0],data[-500:,1])
plt.plot(data[:,0],data[:,1])
#plt.figure(2)
plt.subplot(212)
imgplot=plt.imshow(img)
plt.show()

