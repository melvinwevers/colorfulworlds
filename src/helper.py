import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import random
import os
import pandas as pd
from PIL import Image
from pathlib import Path
from sklearn.cluster import KMeans
from colorsys import rgb_to_hsv, hsv_to_rgb
from typing import Union, List, Tuple, Any, Dict

class Constants:
    THUMB_SIZE = (200, 200)
    SCALE = 256.0
    BAR_DIMENSIONS = (50, 300, 3)

class ImageProcessing:
    @staticmethod
    def get_colors(img: Image) -> List[Tuple[int, int, int]]:
        """
        Returns a list of all the image's colors.
        """
        w, h = img.size
        return [color for count, color in img.convert('RGB').getcolors(w * h)]
    
    def clamp(color: Tuple[int, int, int], min_v: int, max_v: int) -> Tuple[int, int, int]:
        """
        Clamps a color such that the value (lightness) is between min_v and max_v
        """
        h, s, v = rgb_to_hsv(*map(ColorProcessing.down_scale, color))
        min_v, max_v = map(ColorProcessing.down_scale, (min_v, max_v))
        v = min(max(min_v ,v), max_v)
        return tuple(map(ColorProcessing.up_scale, hsv_to_rgb(h, s, v)))
    
    @staticmethod
    def load_and_convert_image(img_path: str, color_scheme='RGB') -> Tuple[np.array, np.array]:
        img = cv2.imread(img_path)
        if color_scheme == 'RGB':
            converted_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            channels = cv2.split(img)
        elif color_scheme == 'LAB':
            converted_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            channels = cv2.split(converted_img)
        else:
            raise ValueError(f'Invalid color scheme {color_scheme}. Choose either "RGB" or "LAB"')
        
        return converted_img, channels

class ColorProcessing:
    @staticmethod
    def down_scale(x: int) -> float:
        return x / Constants.SCALE


    def up_scale(x: int) -> float:
        return int(x * Constants.SCALE)

    @staticmethod
    def order_by_hue(colors: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        """
        Orders colors by hue.
        """
        hsvs = [rgb_to_hsv(*map(ColorProcessing.down_scale, color)) for color in colors]
        hsvs.sort(key=lambda t: t[0])
        return [tuple(map(ColorProcessing.up_scale, hsv_to_rgb(*hsv))) for hsv in hsvs]


class ImageAnalysis:
    @staticmethod
    def country_count(df: pd.DataFrame, country: str) -> None:
        '''
        Count the number of countries in dataframe
        df: dataframe to input
        country: country to be counted 
        '''
        n_countries = df[df['location'].str.contains(country)].shape[0]
        print(f'The collection contains {n_countries} images of {country}')

    @staticmethod
    def get_dominant_colors(image_path: Union[str, Path],
                            num_colors: int = 8,
                            min_v: int = 0,
                            max_v: int = 256,
                            order_colors: bool = True,
                            return_clusters: bool = False) -> Union[np.array, Tuple[np.array, KMeans]]:
        """
        Get the n most dominant colors of an image, with optional color ordering by hue.

        Args:
            image_path (Union[str, Path]): Path to the image file.
            num_colors (int, optional): Number of dominant colors to return. Defaults to 8.
            min_v (int, optional): Minimum value to clamp colors to. Defaults to 0.
            max_v (int, optional): Maximum value to clamp colors to. Defaults to 256.
            order_colors (bool, optional): Whether to order colors by hue. Defaults to True.
            return_clusters (bool, optional): Whether to return the color clusters. Defaults to False.

        Returns:
            Union[np.array, Tuple[np.array, KMeans]]: Dominant colors in the image, and optionally the color clusters.
        """       

        image = Image.open(image_path)
        image.thumbnail(Constants.THUMB_SIZE)
        colors = ImageProcessing.get_colors(image)

        # Adjust the value of each color based on the chosen min and max values
        clamped_colors = [ImageProcessing.clamp(color, min_v, max_v) for color in colors]
        clamped_colors_array = np.array(clamped_colors).astype(float)

        # Perform KMeans clustering to find the dominant colors
        kmeans_model = KMeans(n_clusters=num_colors).fit(clamped_colors_array)
        dominant_colors = kmeans_model.cluster_centers_

        if order_colors:
            dominant_colors = ColorProcessing.order_by_hue(dominant_colors)

        if return_clusters:
            return dominant_colors, kmeans_model
        else:
            return dominant_colors
        
    
    @staticmethod
    def plot_color_histogram(img_path: str, fig_path: str, title: str = '', n_bins: int = 256, 
                             n_dominant_colors: int = 8, color_scheme: str ='RGB'):
        converted_img, channels = ImageProcessing.load_and_convert_image(img_path, color_scheme)

        fig = plt.figure(figsize=(5, 7))
        #fig.suptitle(title, fontsize=20)

        gs = gridspec.GridSpec(2, 1, width_ratios=[1])
        # Plot original image
        #ax1 = fig.add_subplot(221)
        ax1 = plt.subplot(gs[0])
        ax1.axis("off")
        ax1.set_title('Original Image')
        ax1.imshow(converted_img)

        # # Plot RGB histogram
        # colors = ("b", "g", "r")
        # #ax2 = fig.add_subplot(222)
        # ax2 = plt.subplot(gs[0, 1])
        # ax2.set_title(f'{color_scheme} Histogram')
        # ax2.set_xlabel("Bins")
        # ax2.set_ylabel("# of Pixels")

        # for (chan, color) in zip(channels, colors):
        #     hist = cv2.calcHist([chan], [0], None, [n_bins], [0, n_bins])
        #     ax2.plot(hist, color=color)
        #     ax2.set_xlim([0, n_bins])
        
        # Plot dominant colors
        image_array, dominant_colors = ImageAnalysis.get_dominant_colors(img_path, n_dominant_colors, return_clusters=True)
        bar_hist = calculate_normalized_histogram(dominant_colors)
        bar = ColorBar.create_relative_color_bar(bar_hist, dominant_colors.cluster_centers_)

        
        #ax3 = fig.add_subplot(212)
        ax2 = plt.subplot(gs[1])
        ax2.set_title(f'{len(dominant_colors.cluster_centers_)} Dominant Colors (size relative to occurrence)')
        ax2.axis("off")
        ax2.imshow(bar)

        plt.subplots_adjust(hspace=-.2)

        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        plt.savefig(os.path.join(fig_path, title), transparent=True, dpi=300)   

        plt.show()  

class ColorBar:

    @staticmethod
    def create_relative_color_bar(hist: np.array, centroids: List[Tuple[int, int, int]]) -> np.array:
        """
        Create a bar chart representing the relative frequency of each color in an image.

        Args:
            hist (np.array): Normalized histogram of color frequencies.
            centroids (List[Tuple[int, int, int]]): RGB values of color centroids.

        Returns:
            np.array: A 3D NumPy array representing the color bar.
        """

        color_bar = np.zeros(Constants.BAR_DIMENSIONS, dtype="uint8")
        start_x = 0

        for percent, color in zip(hist, centroids):
            end_x = start_x + (percent * Constants.BAR_DIMENSIONS[1])
            cv2.rectangle(color_bar,
                          pt1=(int(start_x), 0),
                          pt2=(int(end_x), Constants.BAR_DIMENSIONS[0]),
                          color=color.astype("uint8").tolist(),
                          thickness=-1)
            start_x = end_x

        return color_bar        



def calculate_normalized_histogram(cluster_labels: np.array) -> np.array:
    """
    Calculate a normalized histogram for a given set of cluster labels.
    
    Args:
        cluster_labels (np.array): Cluster labels for each data point.
        
    Returns:
        hist (np.array): Normalized histogram representing the frequency of each cluster.
    """
    unique_labels = np.unique(cluster_labels.labels_)
    num_labels = np.arange(0, len(unique_labels) + 1)
    hist, _ = np.histogram(cluster_labels.labels_, bins=num_labels)

    # convert the histogram to float and normalize it
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist


def plot_feature_colors(d, title, figures_path, verbose=False):
    # todo add filter based on summary plot
    # have to figure out how to get values from shap summary plot
    features = list()
    for x in d:
        try:
            for k, v in x.items():
                if k == 'img':
                    pass
                elif k in features:
                    pass
                else:
                    features.append(k)
                    feature_color = []
                    feature_color.append(v['color'])
                    avg_feature_color = np.average(feature_color, axis=0)
                    plt.clf()
                    if verbose:
                        ### add title with RGB info
                        plt.title(f'Feature {k} Color \n RGB: {avg_feature_color}')
                    
                    plt.axis('off')
                    plt.imshow([[(avg_feature_color[0] / 255, avg_feature_color[1] / 255, avg_feature_color[2] / 255)]])
                    plt.savefig(os.path.join(figures_path, 
                            f'{title}_{str(k)}.png'),
                            dpi=300,
                            bbox_inches='tight')
        except Exception:
            pass


class DominantColorAnalysis:
    def __init__(self, datapath: Union[str, Path], img_list: List[str], 
                 title: str, output: str, n_sample: int, n_colors: int):
        self.datapath = datapath
        self.img_list = img_list
        self.title = title
        self.output = output
        self.n_sample = n_sample
        self.n_colors = n_colors

    def validate_sample_size(self):
        if self.n_sample > len(self.img_list):
            raise ValueError(f'n_sample larger than list of images. List of images is: {len(self.img_list)}')
        self.img_list = random.sample(self.img_list, self.n_sample)

    def get_image_colors(self):
        all_arrays = []
        self.img_list = [str(self.datapath) + str(x) for x in self.img_list]
        for img in self.img_list:
            image_array = ImageAnalysis.get_dominant_colors(img, self.n_colors, return_clusters=False)
            all_arrays.append(image_array)
        return np.concatenate(all_arrays, axis=0)
    
    def analyze_colors(self, all_images):
        df = pd.DataFrame(all_images)
        y = KMeans(n_clusters=self.n_colors).fit(df)
        hist = calculate_normalized_histogram(y)
        bar = ColorBar.create_relative_color_bar(hist, y.cluster_centers_)
        return y.cluster_centers_, bar

    def save_and_display_results(self, bar):
        plt.figure()
        plt.axis("off")
        plt.title(self.title)
        plt.imshow(bar)
        plt.savefig(self.output, dpi=300, bbox_inches='tight')
        plt.show()

    def run(self):
        self.validate_sample_size()
        all_images = self.get_image_colors()
        cluster_centers, bar = self.analyze_colors(all_images)
        self.save_and_display_results(bar)
        return cluster_centers, all_images
    


def dominant_color_collection(datapath, img_list, title, output, n_sample, n_colors):
    allArrays = []
    img_list = [str(datapath) + str(x) for x in img_list]
    if n_sample > len(img_list):
        print('n_sample larger than list of images')
        print('list of images is: {}'.format(len(img_list)))
    else:
        img_list = random.sample(img_list, n_sample)
        
    for _ in img_list:
        image_array = colorz_in_bucket(_, n_colors, cluster=False)
        allArrays.append(image_array)


    all_images = np.concatenate(allArrays, axis=0)
    df = pd.DataFrame(all_images)
    kmeans = KMeans(n_clusters = n_colors)
    y = kmeans.fit(df)

    # colors = order_by_hue(y.cluster_centers_) 
      
    hist = calculate_normalized_histogram(y)
    bar = ColorBar.create_relative_color_bar(hist, y.cluster_centers_)
    plt.figure()
    plt.axis("off")
    plt.title(title)
    plt.imshow(bar)
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.show()

    return y.cluster_centers_, all_images


def calc_bucket(color: np.array, shared_pixels_per_dim: int) -> int:
    """
    Calculate bucket for a given color.
    """
    return np.dot(color // (256 / shared_pixels_per_dim), [1, shared_pixels_per_dim, shared_pixels_per_dim**2]).astype(int)


def process_colors(colors: np.array, percentages: np.array, n_pixels_dim: int) -> Dict[Any, Dict[str, Any]]:
    """
    Proces the colors and percentages (taken from weights in histogram) into a dictionary
    """
    result = {}
    for color, perc in zip(colors, percentages):
        num_bucket = calc_bucket(color, n_pixels_dim)
        result.setdefault(num_bucket, {'perc': 0, 'color': 0})
        result[num_bucket]['perc'] += perc
        result[num_bucket]['color'] += color
    return result

def colorz_in_bucket(data_path: str, x: str, n_colors: int = 16, n_pixels_dim: int = 4, 
                     min_v: int = 0, max_v: int = 256, thumb_size: tuple = Constants.THUMB_SIZE, debug: bool = False) -> Dict[str, Any]:
    """
    Get the n most dominant colors of an image and categorize them into buckets.
    """
    try:
        img_path = os.path.join(data_path, x)
        img = Image.open(img_path)
        img.thumbnail(Constants.THUMB_SIZE)
        colors = ImageProcessing.get_colors(img)

        clamped = [ImageProcessing.clamp(color, min_v, max_v) for color in colors]
        X = np.array(clamped).astype(float)
        clusters = KMeans(n_clusters=n_colors).fit(X)

        numLabels = np.arange(0, len(np.unique(clusters.labels_)) + 1)
        hist, _ = np.histogram(clusters.labels_, bins = numLabels)
        hist = hist.astype("float")
        hist /= hist.sum()

        color_dict = process_colors(clusters.cluster_centers_, hist, n_pixels_dim)
        color_dict['img'] = x
        return color_dict
    
    except Exception as e:
        if debug:
            print(f'Failed to process image {x}: e')
            raise e