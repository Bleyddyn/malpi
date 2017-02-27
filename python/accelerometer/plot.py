import matplotlib.pyplot as plt
import pickle
import numpy as np
from optparse import OptionParser
import os

parser = OptionParser()
(options, args) = parser.parse_args()

def norm(data):
    return (data - np.mean(data)) / np.std(data)

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def cropPeaks():
    files = ['accel_20170221-184036.pickle', 'accel_20170221-184349.pickle', 'accel_20170221-184459.pickle', 'accel_20170221-185037.pickle', 'accel_20170221-185117.pickle', 'accel_20170221-185443.pickle', 'accel_20170221-185555.pickle']
    examples = [ [], [], [], [206,602], [203,600,1000], [212,515], [202, 501] ]
#    examples = [ [], [], [], [], [], [212,514], [] ]

    count = 1
    slen = 25
    avg = np.zeros(slen)
    for idx, fname in enumerate(files):
        if len(examples[idx]) > 0:
            print fname
            with open(fname) as f:
                data = pickle.load(f)
            times, x, y, z = extract(data)
            for sample in examples[idx]:
                y1 = y[sample:sample+slen]
                avg += y1
                with open('forward' + str(count) + '.pickle', 'wb') as f:
                    pickle.dump( y1, f, pickle.HIGHEST_PROTOCOL)
                count += 1
    avg /= count
    with open('forward_avg.pickle', 'wb') as f:
        pickle.dump( avg, f, pickle.HIGHEST_PROTOCOL)
    return count

def plotPeaks(count):
    lines = []
    labels = []
    for i in range(1,count):
        with open('forward' + str(i) + '.pickle') as f:
            f2 = pickle.load(f)
        line, = plt.plot(f2)
        lines.append(line)
        labels.append("line " + str(i))
    with open('forward_avg.pickle') as f:
        f2 = pickle.load(f)
    line, = plt.plot(f2, 'o')
    lines.append(line)
    labels.append("Avg")

    plt.legend( lines, labels)
    plt.show()

def convolve( times, x, y, z ):
    with open('forward1.pickle') as f:
        f1 = pickle.load(f)
    with open('forward2.pickle') as f:
        f2 = pickle.load(f)

    f1 = np.array(f1)
    f2 = np.array(f2)
    f = (f1 + f2) / 2.0

    cnv = np.convolve(y, f)
    #plt.plot(cnv, 'r--', y, 'g^')
    plt.plot(f)
    plt.show()

def extract(data, do_mavg=False, n=3, do_norm=False):

    times = [d[0] for d in data]
    times = np.array(times)
    times = times - times[0]
    if do_mavg:
        times = times[(n-1):]

    x = [d[1] for d in data]
    if do_norm:
        x = norm(x)
    if do_mavg:
        x = moving_average(x, n=n)

    y = [d[2] for d in data]
    if do_norm:
        y = norm(y)
    if do_mavg:
        y = moving_average(y, n=n)

    z = [d[3] for d in data]
    if do_norm:
        z = norm(z)
    #z = z - np.mean(z)
    if do_mavg:
        z = moving_average(z, n=n)
    return (times, x, y, z)

def baseline( gen=False ):
    if gen:
        with open('accel_20170221-184036.pickle') as f:
            b1 = pickle.load(f)
        with open('accel_20170221-184459.pickle') as f:
            b2 = pickle.load(f)
        with open('accel_20170221-184349.pickle') as f:
            b3 = pickle.load(f)

        _, b1_x, b1_y, b1_z = extract(b1)
        _, b2_x, b2_y, b2_z = extract(b2)
        _, b3_x, b3_y, b3_z = extract(b3)

        x = np.concatenate( (b1_x, b2_x, b3_x) )
        y = np.concatenate( (b1_y, b2_y, b3_y) )
        z = np.concatenate( (b1_z, b2_z, b3_z) )

        print "x: %f %f" % (np.mean(x), np.std(x))
        print "y: %f %f" % (np.mean(y), np.std(y))
        print "z: %f %f" % (np.mean(z), np.std(z))
        base = {}
        base['x'] = (np.mean(x), np.std(x), np.min(x), np.max(x))
        base['y'] = (np.mean(y), np.std(y), np.min(y), np.max(y))
        base['z'] = (np.mean(z), np.std(z), np.min(z), np.max(z))
        with open('baseline.pickle', 'wb') as f:
            pickle.dump( base, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open('baseline.pickle', 'rb') as f:
            base = pickle.load(f)
    return base

def moving_rms( times, x, y, z ):
    with open('forward_avg.pickle', 'rb') as f:
        f1 = pickle.load( f )

    f1 = np.array(f1)
    rms = np.zeros( (len(times)) )
    for i in range( len(times) - len(f1) ):
        samp = y[i:i+len(f1)]
        rms[i] = rmse(samp,f1)
    return rms

def stopped_rms( times, x, y, z ):
    f1 = np.zeros(5)
    rms = np.zeros( (len(times)) )
    for i in range( len(times) - len(f1) ):
        samp = y[i:i+len(f1)]
        rms[i] = rmse(samp,f1)
    return rms

def label1( times, x, y, z ):
    res = np.abs(x) + np.abs(y) + (np.abs(z) * 2)
    n = 3
    res_avg = moving_average(res, n=n)
    np.append(res_avg,  [0] * n )

    labels = np.zeros( (len(times)) ) + 0.5

    labels[res_avg <= 0.06] = 0.0
    labels[res >= 1.0] = 1.0
    return labels

if len(args) != 1:
    print "Usage: plot.py <file.pickle>"
    exit()

with open(args[0]) as f:
    data = pickle.load(f)

if type(data) is dict:
    actions = np.array(data['actions'])
    act_times = np.array(data['action_times'])
    data = data['accelerometer']

if type(data) is not list:
    print "Invalid data type in %s: %s" % (args[0],str(type(data)))

base = baseline(gen=False)
t0 = data[0][0]
act_times = act_times - t0
actions = ((actions + 1) * -1.0) / 3.0

times, x, y, z = extract(data, do_norm=False)
#x = x - np.mean(x)
#y = y - np.mean(y)
z = z - np.mean(z)

labels = np.zeros( (len(times)) )

rms_f = moving_rms( times, x, y, z )
rms_s = stopped_rms( times, x, y, z )

labels[rms_s < 0.025] = 0.1

state = 'stopped'
prev = -100
for idx, yi in enumerate(y):
    if rms_f[idx] < 0.08 and state is 'stopped':
        state = 'check'
        prev = rms_f[idx]
    elif state is 'check':
        if rms_f[idx] > prev:
            state = 'forward'
            prev = -100
            labels[idx] = 0.2
        else:
            prev = rms_f[idx]
    elif state is 'forward':
        if abs(labels[idx] - 0.1) < 0.0001:
            state = 'stopped'
        else:
            labels[idx] = 0.2

crash = np.abs(x) + np.abs(y) + (np.abs(z) * 2)
n = 3
crash = moving_average(crash, n=n)
crash = np.append(crash,  [0] * (n-1) )
labels[crash >= 1.1] = 0.3
# expand crashes
for i in range(2,len(labels)-2):
    if abs(labels[i+2] - 0.3) < 0.0001:
        labels[i] = 0.3
    elif abs(labels[i+1] - 0.3) < 0.0001:
        labels[i] = 0.3
for i in range(len(labels)-2,2,-1):
    if abs(labels[i-1] - 0.3) < 0.0001:
        labels[i] = 0.3
    elif abs(labels[i-2] - 0.3) < 0.0001:
        labels[i] = 0.3

plt.plot(times, labels, 'ko', times, y, 'bs', times, x, 'r--', times, z, 'yx', act_times, actions, 'gx' )
plt.show()

#labels = label1( times, x, y, z )
#plt.plot(times, labels, 'ko', times, x, 'r--', times, y, 'bs', times, z, 'g^')
#plt.show()

#x = np.cumsum(x)
#y = np.cumsum(y)
#z = np.cumsum(z)

#plt.plot(times, x, 'r--', times, y, 'bs', times, z, 'g^')
#plt.show()

#count = cropPeaks()
#plotPeaks(count)

#convolve( times, x, y, z )

#bname = os.path.splitext(args[0])[0]
#plt.savefig( bname + '.svg', bbox_inches='tight' )
#plt.savefig( bname + '.png', bbox_inches='tight' )
