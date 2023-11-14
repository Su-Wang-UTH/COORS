# COORS

## Python script usage
### nn_classifier.py
This Python file loads one reference dataset and one test dataset. 
It selects overlapped genes and trains a neural network model as a cell-type classifier based on the reference dataset.
Then it predicts the cell-type probabilities and extracts features for the test dataset.
Finally, it evaluates the model using samples from the test dataset by SHAP.


