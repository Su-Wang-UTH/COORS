# COORS
**COORS** (**C**ell **O**f **OR**igin like Cell**S**) is a computational tool trained on developmental human brain single-cell datasets, enabling annotation of developmental-like cell states in brain tumor cells. COORS can be applied to various brain cancer datasets, including medulloblastoma and glioma, to uncover developmental-like cells and potential therapeutic targets.

## Model Description
COORS uses cell-type transcriptional annotation using machine-learned neural network models (NNMs). COORS NNMs are trained from  previously published scRNA-seq developing human brain datasets (totaling ~1M cells), such as developing human neocortical and cerebellum scRNA-seq data.
  
The overall workflow of COORS consists of two steps. In the initial step, we train neural network models for cell of origin classification and cell age regression using developing brain scRNA-seq datasets. Assuming we have reference data with two cell origins, A and B, we train a neural network-based cell of origin classifier using this reference data, saving the model in our repository. Concurrently, we train two neural network-based cell age regressors, one for cell origin A and another for cell origin B, also saving these trained models in the repository.  

In the second step, we map scRNA-seq tumor cells to developing healthy brain cells by using the pre-trained models. We predict the cell of origin for the testing dataset using the pre-trained cell of origin classifier. For each cell of origin, we further predict cell age using the corresponding pre-trained cell age regressor. Additionally, we conduct **SH**apley **A**dditive ex**P**lanations (**SHAP**) analysis to extract essential features from our machine-learning neural network models, identifying tumor-specific developmental-like gene markers for each cell type and age within our training datasets.

## Application
### nn_classifier.py
This Python script loads one **reference** dataset and one **test** dataset both provided by the user.  
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

### age_predictor_fit.py


### age_predictor_transform.py


## Installation
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
