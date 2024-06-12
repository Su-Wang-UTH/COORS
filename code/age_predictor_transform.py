import os
import random
import numpy as np
import scanpy as sc
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from joblib import load
from tqdm import tqdm
from scipy.sparse import issparse
import argparse


# Set random seeds for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)


# **** User command line input ****
# **** Example inputs are provided in help messages, with cerebellum used as reference and medulloblastoma as test ****
parser = argparse.ArgumentParser(description='neural network-based regression of cell ages')

# Positional inputs for reference dataset
parser.add_argument('ref_name', help='Input the reference dataset name for model training. e.g., cerebellum')

# Positional inputs for test dataset
parser.add_argument('test_name', help='Input the test dataset name for cell age regression. e.g., medulloblastoma')

# **** User input finished ****


# **** Parse input arguments ****
args = parser.parse_args()

# Arguments for reference dataset
ref_name = args.ref_name

# Arguments for test dataset
test_name = args.test_name

# **** Input arguments parsed ****


colnames = {
    # 'cerebellum': dict(ref_class_col='CellType', ref_age_col='age'),
    # 'codex': dict(ref_class_col='CellType', ref_age_col='Gestation_week'),
    # 'bhaduri': dict(ref_class_col='CellType', ref_age_col='Age'),
    # 'bhaduri_d2': dict(ref_class_col='CellType', ref_age_col='age'),
    'medulloblastoma': dict(col_subgroup='subgroup', col_type='coarse_cell_type', col_subpopulation='tumor_subpopulation'),
    'glioma': dict(col_subgroup='tumorType', col_type='isTumor', col_subpopulation='location'),
    'DIPG': dict(col_subgroup='Sample', col_type='Type', col_subpopulation='Sample'), 
    'dipg_tumor_pons': dict(col_subgroup='Group', col_type='Malignant_normal_consensus_Jessa2022', col_subpopulation='Sample')
}

col_subgroup = colnames[test_name]['col_subgroup']
col_type = colnames[test_name]['col_type']
col_cluster = 'neuralNetwork_' + ref_name + '_cluster'
col_score = 'neuralNetwork_' + ref_name + '_clusterScore'

# Set working paths
model_dir = f'{test_name}_{ref_name}_ageClassifier_ageRegressor/'
output_dir = f'{test_name}_{ref_name}_neuralNetwork/'

# Load adata_test with scores
adata_test = sc.read_h5ad(output_dir + 'adata_scores.h5ad')
print('test_n_samples', adata_test.n_obs, 'test_n_genes', adata_test.n_vars, 'test_minValue', adata_test.X.min(), 'test_maxValue', adata_test.X.max())

adata_test_raw = adata_test.copy()
adata_test_raw.obs['classified_age'] = '-1'
adata_test_raw.obs['regressed_age'] = -1

# Focus on tumor cells
condition1 = ~adata_test.obs[col_cluster].isin(['Outlier_bhaduri', 'outlier_bhaduri_d2'])
condition2 = adata_test.obs[col_score] >= 0.0
condition3 = adata_test.obs[col_type].isin(['Malignant', 'malignant', 'tumor'])
condition4 = ~adata_test.obs[col_subgroup].isin(['Normal'])
adata_test = adata_test[condition1 & condition2 & condition3 & condition4].copy()

# Group by celltype
for celltype, df in tqdm(adata_test.obs.groupby(col_cluster)):
    cells_this_type = df.index.tolist()
    if len(cells_this_type) < 5:
        continue
    adata_group = adata_test[cells_this_type, :].copy()

    # Preprocessing
    print('Preprocessing adata_group...')
    sc.pp.normalize_total(adata_group)
    sc.pp.log1p(adata_group)
    sc.pp.scale(adata_group, zero_center=True, max_value=10)

    # Load common genes
    if not os.path.isfile(model_dir + f'common_genes {celltype}.tsv'):
        continue
    with open(model_dir + f'common_genes {celltype}.tsv') as f:
        common_genes = f.read().rstrip().split('\n')

    adata_group = adata_group[:, common_genes].copy()
    
    # Scale X_test
    scaler = load(model_dir + f'{ref_name}_{celltype}_scaler_x.joblib')
    X_test = adata_group.X
    if issparse(X_test):
        X_test = X_test.toarray()
    X_test = scaler.transform(X_test)

    del adata_group

    # --------------
    # Age classifier
    # --------------
    label_encoder = load(model_dir + f'{ref_name}_{celltype}_labelEncoder.joblib')
    model = load_model(model_dir + f'{ref_name}_{celltype}_ageClassifier.h5')

    y_predict_prob = model.predict(X_test, batch_size=32, verbose=0)
    y_predict_int = np.argmax(y_predict_prob, axis=1)
    y_predict_class = label_encoder.inverse_transform(y_predict_int)

    age_dict = {k: v for k, v in zip(cells_this_type, y_predict_class)}
    for cell in age_dict:
        adata_test_raw.obs.at[cell, 'classified_age'] = age_dict[cell]

    del model
    K.clear_session()
    
    # -------------
    # Age regressor
    # -------------
    scaler = load(model_dir + f'{ref_name}_{celltype}_scaler_y.joblib')
    model = load_model(model_dir + f'{ref_name}_{celltype}_ageRegressor.h5')

    y_predict_scaled = model.predict(X_test, batch_size=32, verbose=0)
    y_predict_norm = scaler.inverse_transform(y_predict_scaled)
    y_predict_orig = np.exp(y_predict_norm) - 1
    y_predict_orig = y_predict_orig.flatten().round(1)
    # if ref_name == 'GSE155121':
    #     y_predict_orig = ['W' + str(x).replace('.', '-') for x in y_predict_orig]

    age_dict = {k: v for k, v in zip(cells_this_type, y_predict_orig)}
    for cell in age_dict:
        adata_test_raw.obs.at[cell, 'regressed_age'] = age_dict[cell]

    del model
    K.clear_session()

# Save age prediction
adata_test_raw.obs.to_csv(output_dir + f'{test_name}_{ref_name}_neuralNetwork_scores_ages.csv', sep=',')
del adata_test_raw.raw
adata_test_raw.write_h5ad(output_dir + 'adata_scores_ages.h5ad')
