# Coloring in the World of Others

## Project Overview

This project examines the influence of color and materiality on geographic imaginaries in late nineteenth-century visual representations of the Orient. We analyze two collections of color(ed) pictures: photochromes (colorized by printers) and autochromes (color derived from light).

Our study employs machine learning techniques to differentiate between:
1. Photochromes and autochromes
2. Photochromes depicting the Orient and the Occident
3. Autochromes portraying the Orient and the Occident

By using random-forest classification algorithms and SHAP (SHapley Additive exPlanations) for interpretation, we aim to uncover insights into the role of color in visual Orientalism and its mediation through material conditions.

## Repository Structure

- `data/`: Contains cropped versions of the images used in the study.
- `models/`: Includes pickles for the prepared buckets and classifiers.
- `figures/`: Contains generated figures and visualizations.
- `src/`: Source code for data processing and analysis.
- `notebooks/`: Jupyter notebooks for data exploration and analysis.

## Data and Model Availability
The models and images used in this project can be downloaded from [Zenodo](https://zenodo.org/records/13888782)

Please download these files and place them in the appropriate directories (data/ for images and models/ for model files) before running the analysis scripts.

## Key Findings

Our analysis reveals three interconnected conclusions:
1. The presence and absence of color were vital aspects of visual Orientalism.
2. Color use became mediated and constrained by material conditions during the late-nineteenth century.
3. Dominant colors serve as an interpretive feature for distinguishing geographic imaginaries.

## Methodology

1. **Data Collection**: For details on data harvesting and processing, refer to the notebooks in the `notebooks/` directory.
2. **Color Extraction**: We extract dominant colors from images and place them into RGB buckets using `src/calculatebuckets.py`.
3. **Classification**: A random forest classifier is trained using the color buckets as features (`src/train_classifier.py`).
4. **Interpretation**: SHAP explainer is applied to produce interpretability plots.

## Usage

1. **Extract Dominant Colors**:
   ```
   ./run_calculating_buckets.sh
   ```

2. **Train Classifier and Generate SHAP Plots**:
   ```
   ./run_classification.sh
   ```

3. **Exploratory Data Analysis**:
   Open and run `notebooks/run_descriptive_analysis.ipynb` for basic functions to explore the collections.

## Requirements

- Python 3.7.11
- For specific library dependencies, see `requirements.txt`

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/melvin/colorfulworlds.git
   cd colorfulworlds
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```


## Contact

- Thomas Smits (t.p.smits(at)uva.nl)
- Melvin Wevers (melvin.wevers(at)uva.nl)

## Acknowledgments

- We would like to thank Mike Kestemont for his feedback on an early draft.
- We would also like the Albert Kahn Museum for making their collection available in digitized form. 
