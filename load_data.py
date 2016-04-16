#Import modules
from fuel.datasets.dogs_vs_cats import DogsVsCats
from fuel.streams import DataStream, ServerDataStream
from fuel.schemes import ShuffledScheme
from fuel.transformers.image import RandomFixedSizeCrop, MinimumImageDimensions, Random2DRotation, MaximumImageDimensions
from fuel.transformers import Flatten, Cast, ScaleAndShift
from fuel.server import start_server
from argparse import ArgumentParser

#Settings
image_size = (256,256)
batch_size = 64

#load_data_function
def create_data(dataset, size, batch_size):
    if dataset == "train": #choose which dataset to process:train_data
        data = DogsVsCats(('train',), subset=slice(0, 20000))
        port = 5556
    elif dataset == "valid": #choose which dataset to process:valid_data
        data = DogsVsCats(('train',), subset=slice(20000, 20200))
        port = 5557
    elif dataset == "test": #choose which dataset to process:test_data
        data = DogsVsCats(('test',), subset=slice(0, 12500))
        port = 5558
    stream = DataStream(data, iteration_scheme=ShuffledScheme(data.num_examples, batch_size))
    stream_upscaled = MinimumImageDimensions(stream, size, which_sources=('image_features',))
    stream_downscaled = MaximumImageDimensions(stream_upscaled, size, which_sources=('image_features',))
    stream_rotate = Random2DRotation(stream_downscaled, which_sources=('image_features',))
    stream_scale = ScaleAndShift(stream_rotate, 1.0/255, 0, which_sources=('image_features',))    
    stream_data = Cast(stream_scale, dtype='float32', which_sources=('image_features',))
    start_server(stream_data, port=port)


if __name__ == "__main__":
    parser = ArgumentParser("Run the fuel data stream server.")
    parser.add_argument("--type", type=str, default="train", help="Type of the dataset (train, valid, test)")
    args = parser.parse_args()
    create_data(args.type, image_size, batch_size)
