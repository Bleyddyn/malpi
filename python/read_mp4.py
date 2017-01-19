import numpy as np
import pylab
import imageio

def display2(vid):
    nums = [10, 33]
    for num in nums:
        image = vid.get_data(num)
        print image.shape
        fig = pylab.figure()
        fig.suptitle('image #{}'.format(num), fontsize=20)
        pylab.imshow(image)
    pylab.show()

def displayEnum(vid):
    try:
        for num, image in enumerate(vid.iter_data()):
            if num % int(vid._meta['fps']):
                continue
            else:
                fig = pylab.figure()
                pylab.imshow(image)
                timestamp = float(num)/ vid.get_meta_data()['fps']
                print(timestamp)
                fig.suptitle('image #{}, timestamp={}'.format(num, timestamp), fontsize=20)
                pylab.show()
    except RuntimeError:
        print('something went wrong')

def getDataFromVideo(vid):
    frames = []
    for num, image in enumerate(vid.iter_data()):
        frames.append(image)

    # Normalize the data: subtract the mean image
    mean_image = np.mean(frames, axis=0)
    frames -= mean_image
    
    # Transpose so that channels come first
    X_train = frames.transpose(0, 3, 1, 2).copy()
    return X_train

def getVid(filename):
    vid = imageio.get_reader(filename,  'ffmpeg')
    return vid

if __name__ == "__main__":
    vid = getVid('test_data/video_test1.mp4')
#print vid.get_meta_data()
#displayEnum(vid)
#display2(vid)
    x_test = getDataFromVideo(vid)
    print x_test.shape

