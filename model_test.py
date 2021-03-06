#Copyright Yifan Nie
#Built on the template of LeNet Mnist
import theano
import numpy
import pickle
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

from blocks.serialization import load, dump, load_parameter_values
from blocks.filter import VariableFilter

#hyper-parameters setting
num_epochs = 150
num_channels = 3
conv_step = (1,1)
border_mode = 'full'
image_shape = (256, 256)
filter_sizes = [(5,5),(5,5),(5,5),(4,4),(4,4),(4,4)]
feature_maps = [20, 40, 70, 140, 256, 512]
pooling_sizes = [(2,2),(2,2),(2,2),(2,2),(2,2),(2,2)]
mlp_hiddens = [1000]
output_size = 2
conv_activations = [Rectifier() for _ in feature_maps] # Use ReLUs everywhere
mlp_activations = [Rectifier() for _ in mlp_hiddens] + [Softmax()] #softmax for the final prediction




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


#Generating input and target variables
x = tensor.tensor4('image_features')
y = tensor.lmatrix('targets')

#Load Data
#stream_train = ServerDataStream(('image_features','targets'), False, port=5556)
#stream_valid = ServerDataStream(('image_features','targets'), False, port=5557)
stream_test  = ServerDataStream(('image_features','targets'), False, port=5558)



# Init an instance of the convnet
convnet = LeNet(conv_activations, num_channels , image_shape,
                    filter_sizes=filter_sizes,
                    feature_maps=feature_maps,
                    pooling_sizes= pooling_sizes,
                    top_mlp_activations=mlp_activations,
                    top_mlp_dims=mlp_hiddens + [output_size],
		    conv_step=conv_step,
                    border_mode=border_mode,
                    weights_init=Uniform(width=0.2),
                    biases_init=Constant(0))


########## hyper parameters###########################################
# We push initialization config to set different initialization schemes
'''
convnet.push_initialization_config()

convnet.layers[0].weights_init = Uniform(width=0.2)
convnet.layers[1].weights_init = Uniform(width=0.2)
convnet.layers[2].weights_init = Uniform(width=0.2)
convnet.top_mlp.linear_transformations[0].weights_init = Uniform(width=0.2)
convnet.top_mlp.linear_transformations[1].weights_init = Uniform(width=0.2)
convnet.initialize()
'''
#########################################################
#Generate output and error signal
predict = convnet.apply(x)
cost = CategoricalCrossEntropy().apply(y.flatten(), predict).copy(name='cost')
cg = ComputationGraph(cost)

#Load the parameters of the model
params = load_parameter_values('catsVsDogs256.pkl')
mo = Model(predict)
mo.set_parameter_values(params)
print mo.inputs
print dir(mo.inputs)
print mo.outputs

f = theano.function(mo.inputs, mo.outputs, allow_input_downcast=True)
predictions=[]
k=0
for batch in stream_test.get_epoch_iterator():
	example=numpy.array([batch[0]])
	batch_predictions=f(example)
	#batch_predictions=batch_predictions[0] #get the array
	for result in batch_predictions:
		res=numpy.argmax(result)
		predictions.append(res)
		k+=1
		print "example",k,"predicted",'\n'

print predictions
#save file
f=open('predictions','wb')
pickle.dump(predictions,f)
#construct CSV file to submit to Kaggle
with open('submission.csv', 'w') as f1:
	f1.write('id,label\n')
	for i in range(len(predictions)):
		f1.write('{},{}\n'.format(i+1,predictions[i]))
f1.close()

