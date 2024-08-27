#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import pandas as pd
import pickle 
from pathlib import Path
import sys
from typing import List, Dict

sys.path.insert(0, '../src/')
from helper import colorz_in_bucket


class ColorBuckets:
    """A class to process and store color buckets from image data."""

    def __init__(self,
                 data_path: Path,
                 meta_data: Path,
                 n_colors: int,
                 n_pixels_dim: int,
                 model_path: Path,
                 color_space: str):
        """
        Initialize ColorBuckets object.

        :param data_path: Path to image data
        :param meta_data: Path to metadata
        :param n_colors: Number of colors to use in analysis
        :param n_pixels_dim: Number of pixels per dimension in color buckets
        :param model_path: Path to save model
        :param color_space: Color space to use (e.g., 'RGB' or 'LAB')
        """
        self.data_path = data_path
        self.meta_data = meta_data
        self.n_colors = n_colors
        self.n_pixels_dim = n_pixels_dim
        self.model_path = model_path
        self.color_space = color_space

    @staticmethod
    def load_df(meta_data: Path) -> pd.DataFrame:
        """
        Load and process metadata.

        :param meta_data: Path to metadata files
        :return: Processed DataFrame
        """
        try:
            meta_ac = pd.read_csv(meta_data / 'autochrome_metadata.csv', delimiter='\t')
            meta_pc = pd.read_csv(meta_data / 'photochrome_metadata.csv', delimiter='\t')
        except FileNotFoundError as e:
            print(f"Error loading metadata: {e}")
            sys.exit(1)
        
        for df in [meta_ac, meta_pc]:
            df['location'] = df['location'].str.lower()

        meta_ac['type'] = 'ac'
        meta_pc['type'] = 'pc'

        combined_df = pd.concat([meta_ac, meta_pc], axis=0)
        sampled_df = combined_df.sample(n=min(1000, len(combined_df)), random_state=42)
        return sampled_df
        
    
    def calculate_buckets(self):
        """Calculate color buckets for all images in the dataset."""
        meta = self.load_df(self.meta_data)
        buckets = []
        n_files = meta.shape[0]
        print(f'total number of files: {n_files}')
        
        for idx, filename in enumerate(meta['filename'], start=1):
            if idx % 100 == 0:
                print(f"Progress: {idx / n_files * 100:.2f}%")
            try:
                buckets.append(colorz_in_bucket(
                    self.data_path,
                    filename,
                    self.color_space,
                    self.n_colors,
                    self.n_pixels_dim,
                ))
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        
        self.save_buckets(buckets, name='buckets')


    def save_buckets(self, buckets, name):
        """
        Save calculated buckets to a file.

        :param buckets: List of color buckets
        :param name: Name prefix for the saved file
        """
        model_file_path = self.model_path / f'{name}_{self.n_colors}_{self.n_pixels_dim}_{self.color_space}.pkl'
        try:
            with model_file_path.open('wb') as f:
                pickle.dump(buckets, f)
            print(f"Buckets saved to {model_file_path}")
        except IOError as e:
            print(f"Error saving buckets: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process images and calculate color buckets.")
    parser.add_argument('--data_path', default='../../ownCloud/datasets/Colors/OrientalColorData/images_all')
    parser.add_argument('--meta_data', default='./data/processed')
    parser.add_argument('--n_colors', type=int, default=8)
    parser.add_argument('--n_pixels_dim', type=int, default=4)
    parser.add_argument('--model_path', type=str, default='./models/')
    parser.add_argument('--color_space', type=str, default='RGB', choices=['RGB', 'LAB'])
    args = parser.parse_args()

    data_path = Path(args.data_path)
    meta_data = Path(args.meta_data)
    model_path = Path(args.model_path)
    model_path.mkdir(parents=True, exist_ok=True)

    color_bucket = ColorBuckets(
        data_path,
        meta_data,
        args.n_colors,
        args.n_pixels_dim,
        model_path,
        args.color_space)
    color_bucket.calculate_buckets()

