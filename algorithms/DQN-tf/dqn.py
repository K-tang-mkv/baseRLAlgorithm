import tensorflow as tf

class DQN(object):
    def __init__(self, image_height, image_width):
        self.image_height = image_height
        self.image_width = image_width

        