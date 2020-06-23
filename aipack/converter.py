import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


class Converter(object):

    @staticmethod
    def pb2function(pb_path, inputs, outputs, print_graph=False):

        with tf.io.gfile.GFile(pb_path, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        def _imports_graph_def():
            tf.compat.v1.import_graph_def(graph_def, name="")

        wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
        import_graph = wrapped_import.graph

        if print_graph == True:
            print("-" * 50)
            print("Frozen model layers: ")
            layers = [op.name for op in import_graph.get_operations()]
            for layer in layers:
                print(layer)
            print("-" * 50)

        return wrapped_import.prune(tf.nest.map_structure(import_graph.as_graph_element, inputs),
                                    tf.nest.map_structure(import_graph.as_graph_element, outputs))

    @staticmethod
    def model2function(model, input_name='x', freeze=False, verbose=True):
        full_model = tf.function(lambda x: model(x))
        full_model = full_model.get_concrete_function(
            tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name=input_name))
        if freeze:
            # Get frozen ConcreteFunction
            frozen_func = convert_variables_to_constants_v2(full_model)
            frozen_func.graph.as_graph_def()
            if verbose:
                print("Frozen model inputs: ")
                print(frozen_func.inputs)
                print("Frozen model outputs: ")
                print(frozen_func.outputs)
            return frozen_func
        return full_model

    @staticmethod
    def model2tflite(model, save_path='./model.tflite'):
        tflite_model = tf.lite.TFLiteConverter.from_keras_model(model)
        open(save_path, 'wb').write(tflite_model)

    @staticmethod
    def model2frozengraph(model, save_dir='./frozen_models'):

        frozen_func = Converter.model2function(model, freeze=True)
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                          logdir=save_dir,
                          name="frozen_graph.pb",
                          as_text=False)

    @staticmethod
    def pb2tflite(pb_path, inputs, outputs, input_shapes, save_path='./model.tflite'):
        converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
            graph_def_file=pb_path,
            input_arrays=inputs,
            output_arrays=outputs,
            input_shapes=input_shapes
        )
        tflite_model = converter.convert()
        tflite_model.write_bytes(save_path)

    @staticmethod
    def pb2coreml(pb_path='./frozen_graph.pb',save_path='./model.mlmodel',input_tensor_shapes={'x': [1, 640, 640, 3]},image_scale=1.0 / 255.0,output_tensor_names=['Identity'],ios_version='13'):

        import tfcoreml
        """
        example        

        pb_path = './frozen_graph.pb'
        input_tensor_shapes = {"x:0": [1, 32, 32, 9]}
        # Output CoreML model path
        save_path = './model.mlmodel'
        output_tensor_name = ['Identity:0']
        """

        tfcoreml.convert(
            tf_model_path=pb_path,
            mlmodel_path=save_path,
            input_name_shape_dict=input_tensor_shapes,
            output_feature_names=output_tensor_names,
            image_scale=image_scale,
            minimum_ios_deployment_target=ios_version

        )
        print("converted!")
