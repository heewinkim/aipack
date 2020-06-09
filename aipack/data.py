import tensorflow as tf
import os
import random
from utilpack.util import *
from utilpack.core import *


class Data(object):

    @staticmethod
    def _int64_feature(value):
        """Wrapper for inserting int64 features into Example proto."""
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def _float_feature(value):
        """Wrapper for inserting float features into Example proto."""
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    @staticmethod
    def _bytes_feature(value):
        """Wrapper for inserting bytes features into Example proto."""
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    @staticmethod
    def _serialize_example(featureDict):


        for k,v in featureDict.items():
            type_ = type(v)
            if type_==list:
                types = list({type(e) for e in v})
                if len(types)==1:
                    type_ = types[0]
                else:
                    raise ValueError('every values in one key are have to same!!')

            if type_ == int:
                featureDict[k] = Data._int64_feature(v)
            elif type_ == float:
                featureDict[k] = Data._float_feature(v)
            elif type_ == str:
                featureDict[k] = Data._bytes_feature(v.encode('utf-8'))
            elif type_ == bytes:
                featureDict[k] = Data._bytes_feature(v)
            else:
                raise ValueError('Unsupported format value offered - {}:{}[{}]'.format(k,v,type_))
        example = tf.train.Example(features=tf.train.Features(feature=featureDict))
        return example.SerializeToString()

    @staticmethod
    def processing(dataset, dataset_name='dataset', output_directory='.',preprocessFunc=None,shuffle=True):
        """
        all keys are lowercase required.
        all value types in [ str, int, float, bytes ] ! not support list/array type
        if you want to convert image path to image bytes , use keyname 'image'
        example of dataset
          dataset :
          [
              {
                "filename" : 'a.jpg',
                "image" : image path,
                "class" : 3,
                "score" : 4.7,
              },
              {
                "filename" : 'a.jpg',
                "image" : image path,
                "class" : 3,
                "score" : 4.7,
              }
              ,
            ...
          ]

        :param dataset: list of dict, support value types are str,int,float,bytes
        :param dataset_name: a name for the dataset
        :param output_directory: path to a directory to write the tfrecord files
        :param preprocessFunc: image preprocessing Function : function(img_cv) -> img_cv
        :param shuffle:  bool, should the image examples be shuffled or not prior to creating the tfrecords.
        :return: exceptImages , which is fail to read images (when convertImage is True)
        """

        # Images in the tfrecords set must be shuffled properly
        if shuffle:
            random.shuffle(dataset)

        save_path = os.path.expanduser(os.path.abspath(os.path.join(output_directory,dataset_name+'.tfrecord')))

        # tf.Example 데이터를 tfrecord 파일에 write
        options = tf.io.TFRecordOptions(compression_type="GZIP")
        writer = tf.io.TFRecordWriter(save_path,options=options)

        exceptImages=[]
        for dataElem in dataset:
            for k,v in dataElem.items():
                if k.lower()=='image':
                    try:
                        bytes_data = PyImage.read_bytedata(v,'filepath')
                        if preprocessFunc:
                            img_cv = preprocessFunc(PyImage.bytes2cv(bytes_data))
                            bytes_data = PyImage.cv2bytes(img_cv)
                        dataElem['image']=bytes_data
                    except Exception:
                        exceptImages.append(v)
            # serialize
            example = Data._serialize_example(dataElem)

            # write
            writer.write(example)
        writer.close()
        print("=" * 80)
        print('{} saved successfully!'.format(save_path))
        if len(exceptImages):
            print('{} images excepted cause by read error.')
            print('excepted images are :')
            [print(v) for v in exceptImages]
        print("=" * 80)

        return exceptImages

    @staticmethod
    def _featureDict_from_tfrecord(tfrecord_path):
        """
        {featureName : tf.io.FixedLenFeature([], featureDataType)} 형태의 featureDict를 얻습니다

        :param tfrecord_path: tfrecord filepath
        :return: featureDict
        """
        serialized_examples = tf.data.TFRecordDataset(tfrecord_path, compression_type = "GZIP")
        example = tf.train.Example()
        featureDict={}
        for serialized_example in serialized_examples.take(1):
            example.ParseFromString(serialized_example.numpy())
            for k,v in dict(example.features.feature).items():
                if bool(v.bytes_list.value):
                    featureDict[k] = tf.io.FixedLenFeature([], tf.string)
                elif bool(v.float_list.value):
                    featureDict[k] = tf.io.FixedLenFeature([],float)
                elif bool(v.int64_list.value):
                    featureDict[k] = tf.io.FixedLenFeature([], tf.int64)
        return featureDict

    @staticmethod
    def read_tfrecord(tfrecord_path):
        """
        tfrecord파일을 읽어 직렬화된 데이터셋과, 파싱 함수를 제공합니다.

        :param tfrecord_path: tfrecord filepath
        :return: serialized dataset, parse_function
        """
        serialized_examples = tf.data.TFRecordDataset(tfrecord_path, compression_type = "GZIP")
        featureDict = Data._featureDict_from_tfrecord(tfrecord_path)

        def _parse_function(example_proto,eager=False):

            parsed_example = tf.io.parse_single_example(example_proto, featureDict)
            # Get the image as raw bytes.
            result = {}
            if eager and 'image' in parsed_example:
                result['image'] = PyImage.bytes2cv(parsed_example['image'].numpy())
                parsed_example.pop('image')

            for k, v in parsed_example.items():
                dtype = featureDict[k].dtype
                result[k] = tf.cast(v,dtype)
                if eager:
                    result[k] = result[k].numpy() if dtype!=tf.string else result[k].numpy().decode()
            return result

        return serialized_examples,_parse_function


if __name__ == '__main__':
    dataset =[
        {
            'index': 0,
            'image':'/Users/hian/picture/hian.jpg',
            'score':3.4,
            'label':'hian',
        },
        {
            'index': 1,
            'image': '/Users/hian/picture/faces.jpg',
            'score': 3.4,
            'label': 'faces',
        }
    ]
    import cv2
    def preFunc(img_cv):
        img_cv = cv2.resize(img_cv,(300,300))
        img_cv = cv2.cvtColor(img_cv,cv2.COLOR_BGR2RGB)
        return img_cv

    Data.processing(dataset,dataset_name='dataset',output_directory='.', preprocessFunc=preFunc)

    tfrecord_path='dataset.tfrecord'
    raw_dataset, decode = Data.read_tfrecord(tfrecord_path)
    for raw_data in raw_dataset:
        data = decode(raw_data,eager=True)
        print(data)
        [print(v) for k,v in data.items() if k!='image']
        PyImageUtil.plot(data['image'],color_mode='rgb')



