# main.py
import torch
import argparse
from train import train

def parse_args():
    parser = argparse.ArgumentParser(description='Train ASD/TD classification model')
    parser.add_argument('--data_root', required=True, help='Path to the image data directory')
    parser.add_argument('--annotations_root', default='part_proportion.csv', help='Path to the CSV file with part proportions')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--image_model', choices=['resnet18', 'efficientnetb0'], required=True, help='Choose image model: resnet18 or efficientnetb1')
    parser.add_argument('--part_model', choices=['lstm', 'linear'], required=True, help='Choose part embedding model: lstm or linear')

    return parser.parse_args()

def main():
    args = parse_args()

    train(args.data_root, args.annotations_root, args.num_epochs, args.batch_size, args.learning_rate, args.image_model, args.part_model)

if __name__ == "__main__":
    main()
