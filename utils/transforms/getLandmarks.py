from utils.utils import *

class GetLandmarksImage(object):
    """
    :returns a landmark image for a given image. returns a black image if no landmarks found
    """
    def __call__(self, img):
        return landmark(img)
