# encoding: utf-8
'''
spatial transfoer network
'''
import os
os.environ['THEANO_FLAGS']='device=gpu0'

# numpy
import numpy as np
np.random.seed(123)

# load data
import load_data as load

# matplotlib
import matplotlib
import matplotlib.pyplot as plt

# dnn framework
import lasagne
import theano
import theano.tensor as T

# cnn functions
conv = lasagne.layers.Conv2DLayer
pool = lasagne.layers.MaxPool2DLayer

# training parameters
NUM_EPOCHS = 500
BATCH_SIZE = 256
LEARNING_RATE = 0.001
DIM = 60
NUM_CLASSES = 10

# mnist clutterd data
mnist_cluttered = "mnist_cluttered_60x60_6distortions.npz"
# download address
# https://s3.amazonaws.com/lasagne/recipes/datasets/mnist_cluttered_60x60_6distortions.npz
# this is mnist clutterd as 0 to 1 float.

def model(input_width, input_height, output_dim, batch_size=BATCH_SIZE):
    ini = lasagne.init.HeUniform()    
    l_in = lasagne.layers.InputLayer(shape=(None, 1, input_width, input_height),)

    # Localization Network
    # b 2x3
    b = np.zeros((2, 3), dtype=theano.config.floatX)
    b[0, 0] = 1
    b[1, 1] = 1
    # b 1x6 (flatten)
    b = b.flatten()

    loc_l1 = pool(
                l_in,
                pool_size=(2, 2)
            )
    loc_l2 = conv(
                loc_l1,
                num_filters=20,
                filter_size=(5, 5),
                W=ini   
            )
    loc_l3 = pool(
                loc_l2,
                pool_size=(2, 2),
            )
    loc_l4 = conv(
                loc_l3,
                num_filters=20,
                filter_size=(5, 5),
                W=ini
            )
    loc_l5 = lasagne.layers.DenseLayer(
                loc_l4,
                num_units=50,
                W=lasagne.init.HeUniform('relu')
            )
    loc_out = lasagne.layers.DenseLayer(
                loc_l5,
                num_units=6,
                b=b,
                W=lasagne.init.Constant(0.0),
                nonlinearity=lasagne.nonlinearities.identity
            )

    # Sample grid and Sampling
    l_trans1 = lasagne.layers.TransformerLayer(l_in, loc_out, downsample_factor=3.0)
    # None, 1, 20, 20 predicted shape
    print("Transformer network output shape: %s" % (l_trans1.output_shape,))
    
    class_l1 = conv(
                l_trans1,
                num_filters=32,
                filter_size=(3, 3),
                nonlinearity=lasagne.nonlinearities.rectify,
                W=ini,    
            )
    class_l2 = pool(
                class_l1,
                pool_size=(2, 2)
            )
    class_l3 = conv(
                class_l2,
                num_filters=32,
                filter_size=(3, 3),
                nonlinearity=lasagne.nonlinearities.rectify,
                W=ini,
            )
    class_l4 = pool(
                class_l3,
                pool_size=(2, 2)
            )
    class_l5 = lasagne.layers.DenseLayer(
                class_l4,
                num_units=256,
                nonlinearity=lasagne.nonlinearities.rectify,
                W=ini,
            )
    l_out = lasagne.layers.DenseLayer(
                class_l5,
                num_units=output_dim,
                nonlinearity=lasagne.nonlinearities.softmax,
                W=ini,
            )

    return l_out, l_trans1


def visualization(train_accs, valid_accs):
    plt.figure(figsize=(9,9))
    plt.plot(1-np.array(train_accs), label='Training Error')
    plt.plot(1-np.array(valid_accs), label='Validation Error')
    plt.legend(fontsize=20)
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Error', fontsize=20)
    plt.show()


def transpose_visualization(data, transform):
    plt.figure(figsize=(7,14))
    for i in range(3):
        plt.subplot(321+i*2)
        plt.imshow(data['X_test'][i].reshape(DIM, DIM), cmap='gray', interpolation='none')
        if i == 0:
            plt.title('Original 60x60', fontsize=20)
        plt.axis('off')
        plt.subplot(322+i*2)
        plt.imshow(transform[i].reshape(DIM//3, DIM//3), cmap='gray', interpolation='none')
        if i == 0:
            plt.title('Transformed 20x20', fontsize=20)
        plt.axis('off')
    plt.tight_layout()


def stn():
    # load data
    data = load.load_data(mnist_cluttered, DIM)

    # show sample
    plt.figure(figsize=(7,7))
    plt.imshow(data['X_train'][100].reshape(DIM, DIM), cmap='gray', interpolation='none')
    plt.title('Cluttered MNIST', fontsize=20)
    plt.axis('off')
    plt.show()

    # build model
    network_model, l_transform = model(DIM, DIM, NUM_CLASSES)
    parameters = lasagne.layers.get_all_params(network_model, trainable=True)

    # training
    X = T.tensor4() # minibatch inputs
    y = T.ivector() # minibatch labels

    # training output
    output_train = lasagne.layers.get_output(network_model, X, deteministic=False)
    
    # evaluation output
    output_eval, transform_eval = lasagne.layers.get_output([network_model, l_transform], X, deterministic=True)
    
    # shared variables
    sh_lr = theano.shared(lasagne.utils.floatX(LEARNING_RATE))
    cost = T.mean(T.nnet.categorical_crossentropy(output_train, y))
    updates = lasagne.updates.adam(cost, parameters, learning_rate=sh_lr)

    # create theano function
    train = theano.function([X, y], [cost, output_train], updates=updates)
    eval = theano.function([X], [output_eval, transform_eval])

    # training function by epoch
    def train_by_epoch(X, y):
        num_samples = X.shape[0]
        num_batches = int(np.ceil(num_samples/float(BATCH_SIZE)))
        costs = []
        correct = 0
        for i in range(num_batches):
            idx = range(i*BATCH_SIZE, np.minimum((i+1)*BATCH_SIZE, num_samples))
            X_batch = X[idx]
            y_batch = y[idx]
            cost_batch, output_train = train(X_batch, y_batch)
            costs += [cost_batch]
            preds = np.argmax(output_train, axis=-1)
            correct += np.sum(y_batch == preds)
    
        # 平均コスト, 平均精度
        return np.mean(costs), correct / float(num_samples)
    
    # evalation function by epoch
    def eval_by_epoch(X, y):
        output_eval, transform_eval = eval(X)
        preds = np.argmax(output_eval, axis=-1)
        acc = np.mean(preds == y)
        return acc, transform_eval
  
    print("training start.")
    valid_accs, train_accs, test_accs = [], [], []
    try:
        for n in range(NUM_EPOCHS):
            train_cost, train_acc = train_by_epoch(data['X_train'], data['y_train'])
            valid_acc, valid_trainsform = eval_by_epoch(data['X_valid'], data['y_valid'])
            test_acc, test_transform = eval_by_epoch(data['X_test'], data['y_test'])
            valid_accs += [valid_acc]
            test_accs += [test_acc]
            train_accs += [train_acc]
    
            # 20epochごとに学習率を更新
            if (n+1) % 20 == 0:
                new_lr = sh_lr.get_value() * 0.7
                print "New LR:", new_lr
                sh_lr.set_value(lasagne.utils.floatX(new_lr))
                transpose_visualization(data, test_transform)           
    
            print("Epoch %d: Train cost %f, Train acc %f, val acc %f, test acc %f" % (n, train_cost, train_acc, valid_acc, test_acc))
    except KeyboardInterrupt:
        pass
    print("training finish.")

    # visialization
    visualization(train_accs, valid_accs)


if __name__ == '__main__':
    stn()
