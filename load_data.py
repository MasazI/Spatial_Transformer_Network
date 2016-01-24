# encoding: utf-8
'''
load data
'''
import numpy as np
import lasagne

def load_data(mnist_clutterd, dim):
    '''
    arguments;
        mnist_clutterd: ノイズありMNISTデータのファイルパス
        dim: 入力レイヤーの次元数

    return;
        dict: X_train, y_train, X_valid, y_valid, X_test, y_test, num_examples_train, num_examples_valid, num_examples_test, input_height, input_width, output_dim
    '''
    data = np.load(mnist_clutterd)
    X_train, y_train = data['x_train'], np.argmax(data['y_train'], axis=-1)
    X_valid, y_valid = data['x_valid'], np.argmax(data['y_valid'], axis=-1)
    X_test, y_test = data['x_test'], np.argmax(data['y_test'], axis=-1)


    # dimにリサイズ
    X_train = X_train.reshape((X_train.shape[0], 1, dim, dim))
    X_valid = X_valid.reshape((X_valid.shape[0], 1, dim, dim))
    X_test = X_test.reshape((X_test.shape[0], 1, dim, dim))

    print("examples: the number of samples, the number of channels, height, width")
    print("Train examples: %s", X_train.shape)
    print("Validatoin examples: %s", X_valid.shape)
    print("test examples: %s", X_test.shape)

    return dict(
                X_train=lasagne.utils.floatX(X_train),
                y_train=y_train.astype('int32'),
                X_valid=lasagne.utils.floatX(X_valid),
                y_valid=y_valid.astype('int32'),
                X_test=lasagne.utils.floatX(X_test),
                y_test=y_test.astype('int32'),
                num_examples_train=X_train.shape[0],
                num_examples_valid=X_valid.shape[0],
                num_examples_test=X_test.shape[0],
                input_height=X_train.shape[2],
                input_width=X_train.shape[3],
                output_dim=10,
)
    

def test():
    data = load_data("mnist_cluttered_60x60_6distortions.npz", 60)


if __name__ == '__main__':
    test()
