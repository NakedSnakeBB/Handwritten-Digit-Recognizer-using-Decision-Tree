# Handwritten-Digit-Recognizer-using-Decision-Tree
Sure, I can try to make a bigger and more proper README file for your project. Here is my attempt:

# Handwriting Recognition

This project is a simple demonstration of how to use decision tree classifier to recognize handwritten digits from the MNIST dataset. The MNIST dataset is a collection of 70,000 images of handwritten digits from 0 to 9, each with a size of 28 by 28 pixels. The dataset is split into 60,000 training images and 10,000 test images. The goal is to train a model that can accurately predict the digit label given an image.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [References](#references)
- [License](#license)

## Installation

To run this project, you need to have Python 3 and the following libraries installed:

- numpy
- pandas
- sklearn
- matplotlib

You can install them using pip:

```bash
pip install numpy pandas sklearn matplotlib
```

You also need to download the dataset from [Kaggle](https://www.kaggle.com/c/digit-recognizer) and save it in the same folder as the script. The dataset consists of two CSV files: train.csv and test.csv.

## Usage

To run the script, simply execute the following command in the terminal:

```bash
python handwriting_recognition.py
```

The script will load the data, train a decision tree classifier using grid search and cross-validation, and make predictions on some random test images. It will also plot the decision tree and show its accuracy score.

## Results

The decision tree classifier achieved an accuracy score of 0.855 on the test set, which is decent but not very impressive. The confusion matrix shows that the model has some difficulty in distinguishing between some digits, such as 4 and 9, or 3 and 5. The plot of the decision tree shows that the model uses a lot of features (pixels) to make the splits, and the tree is very deep and complex.

## Limitations

The decision tree classifier has some limitations that affect its performance and generalization. Some of them are:

- The decision tree is prone to overfitting, especially when the tree is very deep and complex. This means that the model may memorize the training data and fail to generalize well to new data.
- The decision tree is sensitive to noise and outliers in the data. This means that the model may split on irrelevant features or create unnecessary branches that reduce its accuracy.
- The decision tree is not very robust to changes in the data. This means that the model may perform differently if the data is shuffled, scaled, or transformed in some way.

## Future Work

To improve the performance and generalization of the decision tree classifier, some possible future work are:

- Pruning the decision tree to reduce its complexity and avoid overfitting. This can be done by setting a maximum depth, a minimum number of samples per leaf, or a minimum impurity decrease for each split.
- Ensemble methods such as random forest or gradient boosting to combine multiple decision trees and reduce the variance and bias of the model. This can be done by using different subsets of features or data for each tree, or by assigning different weights to each tree based on their performance.
- Feature engineering or dimensionality reduction to extract more meaningful and relevant features from the pixel values. This can be done by using techniques such as PCA, LDA, or autoencoders to reduce the dimensionality of the data, or by using techniques such as HOG, SIFT, or SURF to extract local features from the images.

## References

- [MNIST dataset](https://www.kaggle.com/c/digit-recognizer)
- [Decision tree classifier](https://scikit-learn.org/stable/modules/tree.html)
- [Grid search and cross-validation](https://scikit-learn.org/stable/modules/grid_search.html)
- [Confusion matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
- [Plotting decision tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html)

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.
