import tensorflow as tf
from .converter import Converter
from .data import Data


class TensorflowPack(object):

    converter = Converter
    data = Data




if __name__ == '__main__':

    TensorflowPack.frozengraph2function('../frozen_graph.pb',['x:0'],['Identity:0'],True)