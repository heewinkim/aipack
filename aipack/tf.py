import tensorflow as tf


class TensorflowPack(object):

    @staticmethod
    def frozengraph2function(frozengraph_path, inputs, outputs, print_graph=False):

        with tf.io.gfile.GFile(frozengraph_path, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        def _imports_graph_def():
            tf.compat.v1.import_graph_def(graph_def, name="")

        wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
        import_graph = wrapped_import.graph

        print("-" * 50)
        print("Frozen model layers: ")
        layers = [op.name for op in import_graph.get_operations()]
        if print_graph == True:
            for layer in layers:
                print(layer)
        print("-" * 50)

        return wrapped_import.prune(tf.nest.map_structure(import_graph.as_graph_element, inputs),tf.nest.map_structure(import_graph.as_graph_element, outputs))

    @staticmethod
    def model2frozengraph(model,save_dir='./frozen_models'):
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
        full_model = tf.function(lambda x: model(x))
        full_model = full_model.get_concrete_function(
            tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

        # Get frozen ConcreteFunction
        frozen_func = convert_variables_to_constants_v2(full_model)
        frozen_func.graph.as_graph_def()

        layers = [op.name for op in frozen_func.graph.get_operations()]
        print("-" * 50)
        print("Frozen model layers: ")
        for layer in layers:
            print(layer)

        print("-" * 50)
        print("Frozen model inputs: ")
        print(frozen_func.inputs)
        print("Frozen model outputs: ")
        print(frozen_func.outputs)

        # Save frozen graph from frozen ConcreteFunction to hard drive
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                          logdir=save_dir,
                          name="frozen_graph.pb",
                          as_text=False)


if __name__ == '__main__':

    TensorflowPack.frozengraph2function('../frozen_graph.pb',['x:0'],['Identity:0'],True)