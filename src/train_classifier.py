# #!/usr/bin/env python
# # -*- coding: utf-8 -*-


import argparse
import joblib
import pickle
import yaml
from pathlib import Path
import sys
sys.path.insert(0, '../src/')
from helper import *
from calculate_buckets import prepare_meta_data

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn import preprocessing

from shap import TreeExplainer, summary_plot, decision_plot

from typing import Dict, Any


class Classifier:
    """A class for training and evaluating image classification models."""
    def __init__(self, model_path, data_path, output_path, figures_path, test_size, cv, method, config_path):
        """
        Initialize the Classifier.

        Args:
            model_path (str): Path to the model file.
            data_path (str): Path to the data directory.
            output_path (str): Path to save output files.
            figures_path (str): Path to save figure files.
            test_size (float): Proportion of the dataset to include in the test split.
            cv (int): Number of cross-validation folds.
            method (str): Classification method to use.
            config_path (str): Path to config file detailing which countries to use.
        """
        self.model_path = model_path
        self.data_path = data_path
        self.output_path = output_path
        self.figures_path = figures_path
        self.test_size = test_size
        self.cv = cv
        self.method = method
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as config_file:
                return yaml.safe_load(config_file)
        except FileNotFoundError:
            print(f"Error: Configuration file not found at {self.config_path}")
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"Error parsing YAML configuration file: {e}")
            sys.exit(1)

    # def prepare_meta_data(self):
    #     #TODO REUSE FUNCTION FROM calculate-buckets
    #     try:
    #         meta_ac = pd.read_csv('./data/processed/autochrome_metadata.csv', delimiter='\t')
    #         meta_pc = pd.read_csv('./data/processed/photochrome_metadata.csv', delimiter='\t')
    #     except FileNotFoundError as e:
    #         print(f"Error: Metadata file not found. {e}")
    #         sys.exit(1)
    #     except pd.errors.EmptyDataError:
    #         print("Error: One of the metadata files is empty.")
    #         sys.exit(1)
        
    #     meta_ac['location'] = meta_ac['location'].str.lower()
    #     meta_pc['location'] = meta_pc['location'].str.lower()

    #     meta_ac['type'] = 'ac'
    #     meta_pc['type'] = 'pc'

    #     return pd.concat([meta_ac, meta_pc], axis=0)
    
    @staticmethod
    def prepare_model(data):
        cleaned_data, filenames = [], []
        for item in data:
            if item:
                # Extract percentage data, excluding the 'img' key
                cleaned_item = {k: v['perc'] for k, v in item.items() if k != 'img'}
                cleaned_data.append(cleaned_item)
                filenames.append(item.get('img'))
        return cleaned_data, filenames
    

    def train_classifier(self, meta, df, target):
        X_train, X_test, y_train, y_test = train_test_split(df, meta[target[0]].values, 
                                                            test_size=self.test_size, 
                                                            random_state=42, 
                                                            shuffle=True)
        
        model_path = self.output_path / f'clf_{target[0]}_{target[1]}.pkl'
        if os.path.exists(model_path):
            # check if model exists, if so then load it, else train it.
            print('model exists. Loading!')
            best_clf = joblib.load(model_path)
            print(f'Accuracy: {best_clf.score(X_test, y_test)}')

            # calculate F1-score
            y_pred = best_clf.predict(X_test)
            f1 = f1_score(y_test, y_pred, average='weighted') 
            print(f'F1-score: {f1}')
            return best_clf, X_train

        else:
            param_grid = {'n_estimators': [5, 10, 15, 20, 50, 100, 200], 'max_depth': [2, 5, 7, 9]}
            clf = RandomForestClassifier(class_weight = 'balanced')
            grid_clf = GridSearchCV(clf, param_grid, cv=self.cv, n_jobs=-1, verbose=1)
            grid_clf.fit(X_train, y_train)
            print(f'Accuracy: {grid_clf.score(X_test, y_test)}')



            best_clf = grid_clf.best_estimator_
             # calculate F1-score
            y_pred = best_clf.predict(X_test)
            f1 = f1_score(y_test, y_pred, average='weighted') 
            print(f'F1-score: {f1}')
            joblib.dump(best_clf, self.output_path / f'clf_{target[0]}_{target[1]}.pkl', compress = 1)
            return best_clf, X_train
    
    def plot_avg_bucket_colors(feature_dicts: Dict[str, Any], title: str, figures_path: str, verbose: bool = False):
        """
        Plot average colors associated with various buckets.

        Args:
            title (str): Title prefix for the figures.
            figures_path (str): Path where the figures are saved.
            feature_dicts (Dict[str, Any]): Dictionary containing feature information.
            verbose (bool): Flag for printing additional information.
        """
        features = []
        for feature_dict in feature_dicts:
            try:
                for k, v in feature_dict.items():
                    if k == 'img':
                        pass
                    elif k in features:
                        pass
                    else:
                        features.append(k)
                        feature_color = [v['color']]
                        avg_feature_color = np.average(feature_color, axis=0)
                        Classifier.plot_and_save_bucket_color(avg_feature_color, k, title, figures_path, verbose)
            except Exception:
                pass

        

    def plot_and_save_bucket_color(color: np.array, feature: str, title: str, figures_path: str, verbose: bool):
        """
        Create a swath for a given feature's color and save it to a file.

        Args:
            color (np.array): RGB color to plot.
            feature (str): Feature name.
            title (str): Title prefix for the figure.
            figures_path (str): Path where the figure is saved.
            verbose (bool): Flag for printing additional information.
        """

        plt.figure()
        plt.clf()

        if verbose:
            plt.title(f'Feature {feature} Color \n RGB: {color}')
        
        plt.axis('off')
        plt.imshow([[color / 255]])

        figure_path = os.path.join(figures_path, f'{title}_{feature}.png')
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        plt.close()

    
    def explain_results(self, clf, X_train, buckets):
        explainer = TreeExplainer(clf)
        shap_values = explainer.shap_values(X_train)
        summary_plot(shap_values[0], X_train, max_display=10, show=False)
        plt.savefig(self.figures_path / f"{self.method}_summary_plot.png", dpi=300, bbox_inches='tight')
        Classifier.plot_avg_bucket_colors(buckets, self.method, self.figures_path)

    def load_buckets(self) -> List[Dict]:
        """Load bucket data from the model file"""
        try:
            with open(self.model_path, 'rb') as f:
                print(f"Laoding buckets from {self.model_path}")
                return pickle.load(f)
        except FileNotFoundError:
            print(f"Error: Model file not found at {self.model_path}")
            sys.exit(1)
        except pickle.PickleError:
            print(f"Error: Unable to load pickle file at {self.model_path}")
            sys.exit(1)
            
    def process_data(self, buckets: List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """"Process bucket data and metadata"""
        meta_data = self.prepare_meta_data()
        cleaned_buckets, filenames = self.prepare_model(buckets)
        
        meta_data = meta_data[meta_data['filename'].isin(filenames)]
        df = pd.DataFrame.from_dict(cleaned_buckets).fillna(0)
        df = df.reindex(sorted(df.columns), axis=1)
        df.index = filenames
        
        return meta_data, df
    
    def filter_data(self, meta_data: pd.DataFrame, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """Filter data based on method and location."""
        occident = self.config['locations']['occident']
        orient = self.config['locations']['orient']
        #occident = ["uk", "germany", "switzerland", "belgium", "italy", "austro-hungary", "scotland"]
        #orient = ['saudi arabia', 'holy land', 'egypt', 'algeria', 'turkey', 'afghanistan', 'iraq', 'syria', 'tunesia', 'israel']

        if self.method == 'all':
            target = ['type', 'ac_pc']
        else:
            meta_data = meta_data[meta_data['location'].isin(occident) | meta_data['location'].isin(orient)]
            meta_data['origin'] = np.where(meta_data['location'].isin(orient), 'orient', 'occident')
            meta_data = meta_data[meta_data['type'] == self.method]
            target = ['origin', self.method]

        selected_files = meta_data['filename'].values
        df = df.loc[selected_files]
        
        return meta_data, df, target
    
    def run(self):
        """Execute the classification process."""
        buckets = self.load_buckets()
        meta_data, df = self.process_data(buckets)
        meta_data, df, target = self.filter_data(meta_data, df)
        
        clf, X_train = self.train_classifier(meta_data, df, target)
        #self.explain_results(clf, X_train, buckets)     

    
if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/buckets_16_8.pkl')
    parser.add_argument('--data_path', default='~/work/surfdrive/datasets/Colors/OrientalColorData/')
    parser.add_argument('--output_path', type=str, default='./models/')
    parser.add_argument('--figures_path', type=str, default='./figures/')
    parser.add_argument('--test_size', type=int, default=0.2)
    parser.add_argument('--cv', type=int, default=5)
    # method `all` refers to classification based on autochrome / photochrome for both categories
    # method `ac` refers to classifcation between orient and occident in autochrome
    # method `pc` refers to classification between orient and occident in photochrome
    parser.add_argument('--method', type=str, default='all')
    parser.add_argument('--config_path', type=str, default='./config.yaml',  help='Path to the configuration file')
    
    args = parser.parse_args()

    model_path = Path(args.model_path)
    sub_path = args.model_path[:-4].split('/')[-1][8:]
    data_path = Path(args.data_path)
    output_path = Path(args.output_path) / sub_path
    figures_path = Path(args.figures_path) / sub_path


    output_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)

    print(f'Training Classifier: {args.method}, Colors-Buckets {sub_path}')

    classifier = Classifier(model_path, data_path, output_path, figures_path, 
                            args.test_size, args.cv, args.method, args.config_path)
    classifier.run()
    sys.exit()