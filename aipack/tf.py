import tensorflow as tf
from .converter import Converter
from .data import Data
from .util import Util


class TensorflowPack(object):

    converter = Converter
    data = Data
    util = Util



if __name__ == '__main__':

    TensorflowPack.frozengraph2function('../frozen_graph.pb',['x:0'],['Identity:0'],True)