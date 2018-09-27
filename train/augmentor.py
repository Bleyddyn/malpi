from random import random, randint
from skimage import draw
import math
import numpy as np

class ImageAugmentor():
    """ Modify input images for data augmentation.
    Draw ellipse_count ellipses, each with a random color in the part of the image given by 'area'.
    Area is height x width
    """
    def __init__(self, percent, image_shape=(120,120,3), area=(30,120) ):
        self.percent = percent / 100.0
        self.image_shape = image_shape
        self.area = area
        self.area_shape = (area[0], area[1], image_shape[2])
        self.ellipse_count = 10

    def __call__(self, images):
        for img in images:
            if random() < self.percent:
                self.ellipses(img)
                #before = np.count_nonzero(img)
                self.noisy(img)
                #after = np.count_nonzero(img)
                #print( "Diff: {}".format( after - before ) )

        return images

    def random_color(self):
        return (random(), random(), random())

    def _highlight(self, image):
        """ Test to make sure we're augmenting in the right place """
        print( "Area shape: {}".format( (self.area_shape[0], self.area_shape[1]) ) )
        rr, cc = draw.rectangle( (0,0), end=(self.area_shape[0]-1, self.area_shape[1]-1) )
        image[rr,cc,:] = (0.5,0.5,0.5)

    def ellipses(self, image):
        #for i in range(randint(1,self.ellipse_count)):
        for i in range(self.ellipse_count):
            row = randint(0,self.area_shape[0])
            col = randint(0,self.area_shape[1])
            r_rad = randint(5,self.area_shape[0]/5)
            c_rad = randint(5,self.area_shape[1]/5)
            rot = (random() * math.pi * 2.0) - math.pi
            #self._highlight(img)
            rr, cc = draw.ellipse(row, col, r_rad, c_rad, self.area_shape, rot)
            image[rr, cc, :] = self.random_color()
        return image

    def noisy(self, image, noise_typ="gaussian"):
        """
        Parameters
        ----------
        image : ndarray
            Input image data. Will be converted to float.
        mode : str
            One of the following strings, selecting the type of noise to add:

            'gaussian'     Gaussian-distributed additive noise.
            'poisson'   Poisson-distributed noise generated from the data.
            's&p'       Replaces random pixels with 0 or 1.
            'speckle'   Multiplicative noise using out = image + n*image,where
                        n is uniform noise with specified mean & variance.
        From: https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
        """

        if noise_typ == "gaussian":
            row,col,ch= image.shape
            mean = 0.0
            var = 0.01
            sigma = var**0.5
            gauss = np.random.normal(mean,sigma,(row,col,ch))
            image += gauss
            np.clip(image, 0.0, 1.0, image)
            return image
        elif noise_typ == "s&p":
          row,col,ch = image.shape
          s_vs_p = 0.5
          amount = 0.004
          out = np.copy(image)
          # Salt mode
          num_salt = np.ceil(amount * image.size * s_vs_p)
          coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
          out[coords] = 1

          # Pepper mode
          num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
          coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
          out[coords] = 0
          return out
        elif noise_typ == "poisson":
          vals = len(np.unique(image))
          vals = 2 ** np.ceil(np.log2(vals))
          noisy = np.random.poisson(image * vals) / float(vals)
          return noisy
        elif noise_typ =="speckle":
          row,col,ch = image.shape
          gauss = np.random.randn(row,col,ch)
          gauss = gauss.reshape(row,col,ch)        
          noisy = image + image * gauss
          return noisy

def runTests():
    import matplotlib.pyplot as plt
    import numpy as np

    n = 10
    images = np.zeros( (n,120,120,3) )

    aug = ImageAugmentor( 50.0 )
    aug(images)

    plt.figure(figsize=(20, 4))
    plt.suptitle( "Sample Images", fontsize=16 )
    for i in range(n):
        ax = plt.subplot(1, n, i+1)
        plt.imshow(images[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

if __name__ == "__main__":
    runTests()
