import argparse
import numpy as np
import pandas as pd
import os
import random
from sklearn.model_selection import train_test_split

def add_parser(parser):
    """Add arguments."""
    # argument for loading data
    parser.add_argument(
        '--metadata-file',
        type=str,
        default=None,
        help='path to the metadatafile')

    # argument for splitting tests
    parser.add_argument(
        '--ratio',
        type=float,
        default=0.2,
        help='the ratio of train test split')

    parser.add_argument(
        '--method',
        type=str,
        default='random',
        help='the attribute to split the dataset by')
    
    parser.add_argument(
        '--seed',
        type=int,
        default= 42)

    # argument for saving
    parser.add_argument(
        '--save-dir',
        type=str,
        default='/MAFAT-Satellite-Vision-Challenge/data/split',
        help='to save ')


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description='Splitting dataset')
    add_parser(parser)
    args = parser.parse_args()
    return args


def main():
    """Main function of train test split."""
    args = parse_args()
    random.seed(args.seed)

    metadata_path = os.path.join(os.getcwd(), "data/metadata_train.csv")
    metadata = pd.read_csv(metadata_path)

    if args.method == 'random':
        train, test = train_test_split(metadata, test_size=args.ratio, random_state=args.seed, stratify=metadata['AOI'])
    elif args.method == 'AOI':
        gb = metadata.groupby('AOI')
        groups = [gb.get_group(x) for x in gb.groups]
        n = len(groups) #number of AOIs 
        indices = list(range(n))
        random.shuffle(indices)
        k = int(args.ratio * n)  #number of elements you need
        test = pd.concat([groups[idx] for idx in indices[:k] ])
        train = pd.concat([groups[idx] for idx in indices[k:] ])
    else:
        metadata = metadata.sort_values(args.method)
        n = len(metadata)
        k = int(args.ratio * n)  
        test = metadata.iloc[:k,:]   
        train = metadata.iloc[k:,:]
    train, test = train['Frame'], test['Frame']

    #split_name = f'{args.method}_r{args.ratio}_S{args.seed}'
    #dir_path = os.path.join(args.save_dir, split_name)
    dir_path = args.save_dir
    dir_path = os.path.join(os.getcwd(), "data/split")
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    train.to_csv( os.path.join(dir_path, 'train.csv'), index=False, header=False)
    test.to_csv( os.path.join(dir_path, 'test.csv'), index=False,  header=False)
    
if __name__ == '__main__':
    main()
