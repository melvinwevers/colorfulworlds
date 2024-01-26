#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import pandas as pd
import pickle 
from pathlib import Path
import sys
sys.path.insert(0, '../src/')
from helper import colorz_in_bucket

class ColorBuckets:
    def __init__(self, data_path, meta_data, n_colors, n_pixels_dim, model_path):
        self.data_path = data_path
        self.meta_data = meta_data
        self.n_colors = n_colors
        self.n_pixels_dim = n_pixels_dim
        self.model_path = model_path

    @staticmethod
    def load_df(meta_data):
        meta_ac = pd.read_csv(meta_data / 'autochrome_metadata.csv', delimiter='\t')
        meta_pc = pd.read_csv(meta_data / 'photochrome_metadata.csv', delimiter='\t')

        meta_ac['location'] = meta_ac['location'].str.lower()
        meta_pc['location'] = meta_pc['location'].str.lower()

        meta_ac['type'] = 'ac'
        meta_pc['type'] = 'pc'

        return pd.concat([meta_ac, meta_pc], axis=0)
    
    def calculate_buckets(self):
        meta = self.load_df(self.meta_data)

        buckets = []
        n_files = meta.shape[0]
        print(f'total number of files: {n_files}')
        for idx, filename in enumerate(meta['filename'], start=1):
            if idx % 100 == 0:
                print(idx / n_files * 100)
            buckets.append(colorz_in_bucket(self.data_path, filename, self.n_colors, self.n_pixels_dim))
        
        self.save_buckets(buckets, name='buckets')

    def save_buckets(self, buckets, name):
        model_file_path = self.model_path / f'{name}_{self.n_colors}_{self.n_pixels_dim}.pkl'
        with model_file_path.open('wb') as f:
            pickle.dump(buckets, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../../surfdrive/datasets/Colors/OrientalColorData/images_all')
    parser.add_argument('--meta_data', default='./data/processed')
    parser.add_argument('--n_colors', type=int, default=8)
    parser.add_argument('--n_pixels_dim', type=int, default=4)
    parser.add_argument('--model_path', type=str, default='./models/')
    args = parser.parse_args()

    data_path = Path(args.data_path)
    meta_data = Path(args.meta_data)
    model_path = Path(args.model_path)

    model_path.mkdir(parents=True, exist_ok=True)

    color_bucket = ColorBuckets(data_path, meta_data, args.n_colors, args.n_pixels_dim, model_path)
    color_bucket.calculate_buckets()

