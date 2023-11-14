# COORS


### Usage of nn_classifier.py
This Python script loads one reference dataset and one test dataset both provided by the user.
It selects overlapped genes and trains a neural network model as a cell-type classifier based on the reference dataset.
Then it predicts the cell-type probabilities and extracts features of the test dataset.
Finally, it evaluates the model using samples from the test dataset by SHAP analysis.

```
$ python nn_classifier.py [ref_name] [ref_directory] [ref_class_col] [test_name] [test_directory] --marker [filename] --shap
```

Example 1: Use cerebellum as reference and medulloblastoma as test, with reference marker genes provided, opt to run SHAP
```
$ python nn_classifier.py cerebellum cerebellum CellType medulloblastoma medulloblastoma -m CellTypeMarker_DevelopingHumanData.xlsx -s
```

Example 2: Use codex as reference and glioma as test, with reference marker genes provided, not to run SHAP
```
$ python nn_classifier.py codex codex CellType glioma glioma -m codex_cluster_markers.xlsx
```

Example 3: Use bhaduri as reference and DIPG as test, without reference marker genes, opt to run SHAP
```
$ python nn_classifier.py bhaduri bhaduri CellType DIPG DIPG -s
```

### Prerequisites
The algorithm is implemented in Python 3.7.3. 
Install core packages with specific versions:
```
$ pip install tensorflow==1.13.1
$ pip install keras==2.2.4
```

Install other necessary packages:
```
$ pip install pandas
$ pip install scanpy
$ pip install scipy
$ pip install scikit-learn
$ pip install joblib
$ pip install shap
$ pip install openpyxl
```
