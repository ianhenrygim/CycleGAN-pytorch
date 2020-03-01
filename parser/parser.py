import argparse

class CustomParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--dataroot', type=str, default='/datasets/horse2zebra/', help='path to images')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
        self.parser.add_argument('--batch_size', type=int, default=1, help='input batch size')

    def get_parser(self):
        return self.parser.parse_args()