#Import modules
from fuel.datasets.dogs_vs_cats import DogsVsCats
from fuel.streams import DataStream, ServerDataStream
from fuel.schemes import ShuffledScheme
from fuel.transformers.image import RandomFixedSizeCrop, MinimumImageDimensions, Random2DRotation
from fuel.transformers import Flatten, Cast, ScaleAndShift
from fuel.server import start_server
from argparse import ArgumentParser

#Settings
image_size = (128,128)
batch_size = 64

#load_data_function
def create_data(dataset, size, batch_size):
    if dataset == "train": #choose which dataset to process:train_data
        data = DogsVsCats(('train',), subset=slice(0, 20000))
        port = 5556
    elif dataset == "valid": #choose which dataset to process:valid_data
        data = DogsVsCats(('train',), subset=slice(20000, 25000))
        port = 5557
    elif dataset == "test": #choose which dataset to process:test_data
        data = DogsVsCats(('test',), subset=slice(0, 12500))
        port = 5558
    stream = DataStream(data, iteration_scheme=ShuffledScheme(data.num_examples, batch_size))
    stream_upscaled = MinimumImageDimensions(stream, size, which_sources=('image_features',))
    stream_downscaled = RandomFixedSizeCrop(stream_upscaled, size, which_sources=('image_features',))    
    stream_data = Cast(stream_downscaled, dtype='float32', which_sources=('image_features',))
    start_server(stream_data, port=port)


if __name__ == "__main__":
    create_data("test", image_size, batch_size)
    print "create data completed\n"
