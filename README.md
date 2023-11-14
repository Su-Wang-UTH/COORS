# COORS


### Usage of nn_classifier.py
This Python file loads one reference dataset and one test dataset. 
It selects overlapped genes and trains a neural network model as a cell-type classifier based on the reference dataset.
Then it predicts the cell-type probabilities and extracts features for the test dataset.
Finally, it evaluates the model using samples from the test dataset by SHAP.

```
$ python nn_classifier.py [ref_name] [ref_directory] [ref_class_col] [test_name] [test_directory] --marker [filename] --shap
```  

Example 1: Use cerebellum as reference and meningioma as test, with reference marker genes provided, opt to run SHAP
```
$ python nn_classifier.py cerebellum ./developing_human_cerebellum/ Cluster meningioma ./meningioma/ --marker CellTypeMarker_DevelopingHumanData.xlsx --shap
```  

Simpler command for Example 1:
```
$ python nn_classifier.py cerebellum developing_human_cerebellum Cluster meningioma meningioma -m CellTypeMarker_DevelopingHumanData.xlsx -s
```  

Example 2: Use codex as reference and glioma as test, with reference marker genes provided, not to run SHAP
```
$ python nn_classifier.py codex codex Cluster glioma glioma -m codex_cluster_markers.xlsx
```  

Example 3: Use codex as reference and DIPG as test, reference marker genes not provided, opt to run SHAP
```
$ python nn_classifier.py codex codex Cluster DIPG DIPG -s
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
