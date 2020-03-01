import argparse

class CustomParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--dataroot', type=str, default='./datasets/horse2zebra/', help='path to images')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
        self.parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        self.parser.add_argument('--img_size', type=int, default=256, help='input rectangular image size')
        self.parser.add_argument('--img_buffer_size', type=int, default=50, help='input image buffer size')
        self.parser.add_argument('--mode', type=str, default="train", help="choose the mode of network('train'/'test')")
        self.parser.add_argument('--identity_loss', action='store_true', help="use identity loss")

    def get_parser(self):
        return self.parser.parse_args()