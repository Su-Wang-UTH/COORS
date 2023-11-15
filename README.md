# COORS
**COORS** (**C**ell **O**f **OR**igin like Cell**S**) is a computational tool trained on developmental human brain single-cell datasets, enabling annotation of developmental-like cell states in brain tumor cells. COORS can be applied to various brain cancer datasets, including medulloblastoma and glioma, to uncover developmental-like cells and potential therapeutic targets.

# Model Description
COORS uses cell-type transcriptional annotation using machine-learned neural network models (NNMs). COORS NNMs are trained from  previously published scRNA-seq developing human brain datasets (totaling ~1M cells), such as developing human neocortical and cerebellum scRNA-Seq data.

# Application
nn_classifier.py loads one **reference** dataset and one **test** dataset both provided by the user. 
The reference cell-type **column** needs to be designated.
The reference **_marker_** genes can be _optionally_ provided.
Overlapped genes are selected and a neural network model is trained as a cell-type classifier based on the reference dataset.
Then the model predicts the cell-type probabilities and extracts features of the test dataset.
At last, the model is _optionally_ evaluated by `SHAP` analysis with samples from the test dataset.

```
$ python nn_classifier.py [ref_name] [ref_directory] [ref_class_col] [test_name] [test_directory] --marker [filename] --shap
```

Example 1: Using **cerebellum** as reference and **medulloblastoma** as test, **_with_** reference marker genes, **_running_** `SHAP`
```
$ python nn_classifier.py cerebellum cerebellum CellType medulloblastoma medulloblastoma -m CellTypeMarker_DevelopingHumanData.xlsx -s
```

Example 2: Using **codex** as reference and **glioma** as test, **_with_** reference marker genes, **_not running_** `SHAP`
```
$ python nn_classifier.py codex codex CellType glioma glioma -m codex_cluster_markers.xlsx
```

Example 3: Using **bhaduri** as reference and **DIPG** as test, **_without_** reference marker genes, **_running_** `SHAP`
```
$ python nn_classifier.py bhaduri bhaduri CellType DIPG DIPG -s
```

# Installation
The algorithm is implemented in `Python 3.7.3`. 
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
