# Clash of Colorful Worlds Code
In the late nineteenth century, the visual representation of the Orient was characterized not only by specific content, such as harems, camels, and deserts, but also by an ``Orientalist aesthetic'', in which the use of color played a crucial role. However, technological limitations and material constraints constrained which colors artists could use. This study examines the influence of color and materiality on geographic imaginaries by analyzing two collections of late-nineteenth-century color(ed) pictures. The collections consist of two types of images: photochromes, which were colorized by printers, and autochromes, where color was derived from light. By training random-forest classification algorithms with dominant colors exrtracted from these digitized images, we aim to differentiate between (1) photochromes and autochromes; (2) photochromes depicting the Orient and the Occident; and (3) autochromes portraying the Orient and the Occident. To interpret the classifier's output, we employ SHAP (SHapley Additive exPlanations) and discover that the classifier effectively distinguishes between the two worlds in the photochrome collection but underperforms for the autochromes. This observation leads to three interconnected conclusions: (1) the presence and absence of color were vital aspects of visual Orientalism; (2) color use became mediated and constrained by material conditions during the late-nineteenth century; and (3) dominant colors serve as an interpretive feature for distinguishing geographic imaginaries. Traditional analyses of visual materials often focus on specific objects or associated colors, thereby neglecting the absence of dominant colors. Our approach allows scholars to consider both the presence and absence of dominant colors, enabling a more comprehensive understanding of the visual landscape without being distracted by individual objects.

## Data
The folder data contains the cropped versions of the images. 
For the harvesting and processing of the data see the notebooks: 

## Models
Contains pickles for the prepared buckets and the classifiers.

## Figures
Contains the Figures

## Manuscript
Contains the paper manuscript

## Code
- `run_calculating_buckets.sh` fires up `src/calculatebuckets.py`to extract the dominant colors from the images and places them into RGB buckets.
- `run_classification.sh` uses `src/train_classifier.py` to train a random forest classifier using the buckets as features. The script also applies SHAP explainer to produce plots for interpretability. 
- `notebooks/run_descriptive_analysis.ipynb` provides basic functions to explore the collections. 

## Requirements
Test on Python 3.7.11. 
For specific libraries see `requirements.txt`

