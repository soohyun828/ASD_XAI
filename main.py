# main.py
import torch
import argparse
from train import train_model

def parse_args():
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('--data_root', default='/local_datasets/ASD/ASD-final-annotations/asd_ver2_all_5folds_annotation', help='Path to img and annotation')
    parser.add_argument('--num_epochs', default=20, type=int, help='# of epochs')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate')

    return parser.parse_args()

def main():
    args = parse_args()

    train_model(args.data_root, num_epochs=args.num_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate)


if __name__ == "__main__":
    main()
