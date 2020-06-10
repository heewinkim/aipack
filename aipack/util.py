import numpy as np


class Util(object):

    @staticmethod
    def summary(model):
        print("_" * 89)
        print('  idx   layer name                       input shape              output shape            ')
        print("=" * 89)
        for i, layer in enumerate(model.layers):
            if i != 0:
                print("_" * 89)
            inputs = list(layer.input) if type(layer.input) in [list, tuple] else [layer.input]
            outputs = list(layer.output) if type(layer.output) in [list, tuple] else [layer.output]
            max_length = max(len(inputs), len(outputs))

            if max(len(inputs), len(outputs)) > 1:
                inputs = [str(v.shape) for v in inputs] + [''] * (max_length - len(inputs))
                outputs = [str(v.shape) for v in outputs] + [''] * (max_length - len(outputs))
                print('  {:3}   {:30}   {:22}   {:22}  '.format(i, str(layer.name), str(inputs[0]), str(outputs[0])),
                      end='')
                for inp, oup in zip(inputs[1:], outputs[1:]):
                    print('\n  {:3}   {:30} '.format('', ''), end='')
                    print('  {:22}   {:22}'.format(str(inp), str(oup)), end='')
                print('  ')
            else:
                print('  {:3}   {:30}   {:22}   {:22}  '.format(i, str(layer.name), str(inputs[0].shape),
                                                                str(outputs[0].shape)))
        print("=" * 89)
        print('Total params : {}'.format(sum([np.product(v.shape) for v in list(model.variables)])))
        print('Trainable params : {}'.format(sum([np.product(v.shape) for v in list(model.trainable_variables)])))
        print('Non-trainable params : {}'.format(sum([np.product(v.shape) for v in list(model.non_trainable_variables)])))
        print("_" * 89)