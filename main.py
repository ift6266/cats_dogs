#Copyright Yifan Nie
#Built on the template of LeNet Mnist
import numpy
from theano import tensor
#Import blocks modules
from blocks.algorithms import GradientDescent, Scale, Adam
from blocks.bricks import (MLP, Rectifier, Initializable, FeedforwardSequence, Softmax, Activation)
from blocks.bricks.conv import (Convolutional, ConvolutionalSequence, Flattener, MaxPooling)
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.monitoring import (DataStreamMonitoring, TrainingDataMonitoring)
from blocks.extensions.saveload import Checkpoint
from blocks.graph import ComputationGraph
from blocks.initialization import Constant, Uniform
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks_extras.extensions.plot import Plot
#Import Fuel modules
from toolz.itertoolz import interleave
from fuel.streams import ServerDataStream
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream

#hyper-parameters setting
num_epochs = 100
num_channels = 3
conv_step = (1,1)
border_mode = 'full'
image_shape = (128, 128)
filter_sizes = [(5,5),(5,5)]
feature_maps = [20, 60]
pooling_sizes = [(2,2),(2,2)]
mlp_hiddens = [1000]
output_size = 2
conv_activations = [Rectifier() for _ in feature_maps] # Use ReLUs everywhere
mlp_activations = [Rectifier() for _ in mlp_hiddens] + [Softmax()] #softmax for the final prediction

#Generating input and target variables
x = tensor.tensor4('image_features')
y = tensor.lmatrix('targets')


class LeNet(FeedforwardSequence, Initializable):
    '''
    ----------
    conv_activations : list of :class:`.Brick`
        Activations for convolutional network.
    num_channels : int
        Number of channels in the input image.
    image_shape : tuple
        Input image shape.
    filter_sizes : list of tuples
        Filter sizes of :class:`.blocks.conv.ConvolutionalLayer`.
    feature_maps : list
        Number of filters for each of convolutions.
    pooling_sizes : list of tuples
        Sizes of max pooling for each convolutional layer.
    top_mlp_activations : list of :class:`.blocks.bricks.Activation`
        List of activations for the top MLP.
    top_mlp_dims : list
        Numbers of hidden units and the output dimension of the top MLP.
    conv_step : tuples
        Step of convolution (similar for all layers).
    border_mode : str
        Border mode of convolution (similar for all layers).
    '''
    
    def __init__(self, conv_activations, num_channels, image_shape,
                 filter_sizes, feature_maps, pooling_sizes,
                 top_mlp_activations, top_mlp_dims,
                 conv_step=None, border_mode='valid', **kwargs):
        if conv_step is None:
            self.conv_step = (1, 1)
        else:
            self.conv_step = conv_step
        self.num_channels = num_channels
        self.image_shape = image_shape
        self.top_mlp_activations = top_mlp_activations
        self.top_mlp_dims = top_mlp_dims
        self.border_mode = border_mode

        conv_parameters = zip(filter_sizes, feature_maps)

        # Construct convolutional layers with corresponding parameters
        self.layers = list(interleave([
            (Convolutional(filter_size=filter_size,
                           num_filters=num_filter,
                           step=self.conv_step,
                           border_mode=self.border_mode,
                           name='conv_{}'.format(i))
             for i, (filter_size, num_filter)
             in enumerate(conv_parameters)),
            conv_activations,
            (MaxPooling(size, name='pool_{}'.format(i))
             for i, size in enumerate(pooling_sizes))]))

        self.conv_sequence = ConvolutionalSequence(self.layers, num_channels,
                                                   image_size=image_shape)

        # Construct a top MLP
        self.top_mlp = MLP(top_mlp_activations, top_mlp_dims)

        # We need to flatten the output of the last convolutional layer.
        # This brick accepts a tensor of dimension (batch_size, ...) and
        # returns a matrix (batch_size, features)
        self.flattener = Flattener()
        application_methods = [self.conv_sequence.apply, self.flattener.apply,
                               self.top_mlp.apply]
        super(LeNet, self).__init__(application_methods, **kwargs)
    
    @property
    def output_dim(self):
        return self.top_mlp_dims[-1]

    @output_dim.setter
    def output_dim(self, value):
        self.top_mlp_dims[-1] = value

    def _push_allocation_config(self):
        self.conv_sequence._push_allocation_config()
        conv_out_dim = self.conv_sequence.get_dim('output')

        self.top_mlp.activations = self.top_mlp_activations
        self.top_mlp.dims = [numpy.prod(conv_out_dim)] + self.top_mlp_dims

#Load Data
stream_train = ServerDataStream(('image_features','targets'), False, port=5556)
stream_valid = ServerDataStream(('image_features','targets'), False, port=5557)

# Init an instance of the convnet
convnet = LeNet(conv_activations, num_channels , image_shape,
                    filter_sizes=filter_sizes,
                    feature_maps=feature_maps,
                    pooling_sizes= pooling_sizes,
                    top_mlp_activations=mlp_activations,
                    top_mlp_dims=mlp_hiddens + [output_size],
                    border_mode=border_mode,
                    weights_init=Uniform(width=0.2),
                    biases_init=Constant(0))
########## hyper parameters###########################################
# We push initialization config to set different initialization schemes
convnet.push_initialization_config()
convnet.layers[0].weights_init = Uniform(width=0.2)
convnet.layers[1].weights_init = Uniform(width=0.09)
convnet.top_mlp.linear_transformations[0].weights_init = Uniform(width=0.08)
convnet.top_mlp.linear_transformations[1].weights_init = Uniform(width=0.11)
convnet.initialize()
#########################################################333
#Generate output and error signal
predict = convnet.apply(x)

cost = CategoricalCrossEntropy().apply(y.flatten(), predict).copy(name='cost')
error = MisclassificationRate().apply(y.flatten(), predict)
#Little trick to plot the error rate in two different plots (We can't use two time the same data in the plot for a unknow reason)
error_rate = error.copy(name='error_rate')
error_rate2 = error.copy(name='error_rate2')
cg = ComputationGraph([cost, error_rate])
########### ALGORITHM of training#############
algorithm = GradientDescent(cost=cost, parameters=cg.parameters, step_rule=Scale(learning_rate=0.1))
extensions = [Timing(),
              FinishAfter(after_n_epochs=num_epochs),
              DataStreamMonitoring([cost, error_rate, error_rate2], stream_valid, prefix="valid"),
              TrainingDataMonitoring([cost, error_rate, aggregation.mean(algorithm.total_gradient_norm)], prefix="train", after_epoch=True),
              Checkpoint("catsVsDogs128.pkl"),
              ProgressBar(),
              Printing()]

#Adding a live plot with the bokeh server
'''
extensions.append(Plot(
    'CatsVsDogs_128',
    channels=[['train_error_rate', 'valid_error_rate'],['valid_cost', 'valid_error_rate2'],['train_total_gradient_norm']], after_epoch=True))
'''
model = Model(cost)
main_loop = MainLoop(algorithm,stream_train,model=model,extensions=extensions)
main_loop.run()
