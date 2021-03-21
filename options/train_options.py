import argparse

def get_opt():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data', type = str, default = 'pet',
                        help = 'Which dataset? (pet OR mri)', dest = 'data')
    parser.add_argument('-s', '--save', type = bool, default = True,
                         help = 'Save checkpoints', dest = 'save_net')
    parser.add_argument('-e', '--epoch', type = int, default = 50,
                         help = 'Number of epochs', dest = 'num_epoch')
    parser.add_argument('-b', '--batch', type = int, default = 8,
                        help = 'Batch size', dest = 'batch_size')
    parser.add_argument('-l', '--learning-rate', type = float, default = 1e-3,
                        help = 'Learning Rate', dest = 'lr')
    parser.add_argument('-n', '--net', type = str, default = 'besnet',
                        help = 'Type of network to train (unet OR besnet)', 
                        dest = 'net')
    parser.add_argument('--height', type = int, default = 128,
                        help = 'Height of input image', dest = 'height')
    parser.add_argument('--width', type = int, default = 128,
                        help = 'Width of input image', dest = 'width')
    parser.add_argument('--alpha', type = float, default = 0.5,
                        help = 'Alpha value for BECE loss (for BESNet)', dest = 'alpha')
    parser.add_argument('--beta', type = float, default = 1,
                        help = 'Beta value for BECE loss (for BESNet)', dest = 'beta')
    parser.add_argument('--bece-loss', type = bool, default = True,
                        help = 'Loss for MDP in BESNet (BECE loss or BCE)', dest = 'bece_loss')
    return parser.parse_args()