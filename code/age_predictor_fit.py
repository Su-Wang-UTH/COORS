import os
import random
import numpy as np
import pandas as pd
from scipy.io import mmread
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import dump
import scanpy as sc
from scipy.sparse import issparse
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import time
import argparse


# Set random seeds for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)


def create_classifier(activation, hidden_layers, dropout_rate, learning_rate):
    # Get data dimension
    input_shape = (X_train.shape[1],)
    output_size = num_classes

    # Input layer
    inputs = Input(shape=input_shape)
    x = inputs
    
    # Hidden layers
    for layer_size in hidden_layers:
        x = Dense(layer_size)(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        x = Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = Dense(output_size, activation='softmax')(x)
    
    # Build model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model


def create_regressor(activation, hidden_layers, dropout_rate, learning_rate):
    # Get data dimension
    input_shape = (X_train.shape[1],)
    output_size = 1

    # Input layer
    inputs = Input(shape=input_shape)
    x = inputs
    
    # Hidden layers
    for layer_size in hidden_layers:
        x = Dense(layer_size)(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        x = Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = Dense(output_size, activation='sigmoid')(x)
    
    # Build model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='mse', optimizer=optimizer)
    
    return model


def decay_learning_rate(initial_learning_rate, final_learning_rate, epochs, batch_size):
    decay_steps = X_train.shape[0] // batch_size
    decay_rate = (final_learning_rate / initial_learning_rate) ** (1 / epochs)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=initial_learning_rate,
                    decay_steps=decay_steps,
                    decay_rate=decay_rate,
                    staircase=True)
    return lr_schedule


def recall_learning_rate(lr_schedule, epochs):
    if isinstance(lr_schedule, tf.keras.optimizers.schedules.ExponentialDecay):
        initial_learning_rate = lr_schedule.initial_learning_rate
        decay_rate = lr_schedule.decay_rate
        final_learning_rate = (decay_rate ** epochs) * initial_learning_rate
    else:
        initial_learning_rate = lr_schedule
        final_learning_rate = lr_schedule
    return initial_learning_rate, final_learning_rate


# **** User command line input ****
# **** Example inputs are provided in help messages, with cerebellum used as reference and medulloblastoma as test ****
parser = argparse.ArgumentParser(description='training neural network-based cell age regressors')

# Positional inputs for reference dataset
parser.add_argument('ref_name', help='Input the reference dataset name for model training. e.g., cerebellum')
parser.add_argument('ref_directory', help='Input the reference directory of the reference dataset. e.g., cerebellum')
parser.add_argument('ref_class_col', help='Input the metadata column name of cell types. e.g., CellType')
parser.add_argument('ref_age_col', help='Input the metadata column name of cell ages. e.g., age')

# Positional inputs for test dataset
parser.add_argument('test_name', help='Input the test dataset name for cell age regression. e.g., medulloblastoma')
parser.add_argument('test_directory', help='Input the test directory of the test dataset. e.g., medulloblastoma')

# **** User input finished ****


# **** Parse input arguments ****
args = parser.parse_args()

# Arguments for reference dataset
ref_name = args.ref_name
ref_directory = args.ref_directory if args.ref_directory.endswith('/') else args.ref_directory + '/'
ref_class_col = args.ref_class_col
ref_age_col = args.ref_age_col

# Arguments for test dataset
test_name = args.test_name
test_directory = args.test_directory if args.test_directory.endswith('/') else args.test_directory + '/'

# **** Input arguments parsed ****


# **** Load all datasets according to user input ****
print('**** Loading all datasets according to user input... ****')
print()

# ----------------------
# Load reference dataset
# ----------------------
print(f'Loading {ref_name} as reference dataset...')

# Assume reference directory directly contains adata.h5ad OR { genes.tsv, barcodes.tsv, matrix.mtx }
filename = 'adata.h5ad'
if os.path.isfile(ref_directory + filename):
    print(f'Loading {filename}...')
    adata_ref = sc.read_h5ad(ref_directory + filename)
    print(f'{filename} loaded.')

    if ref_name == 'GSE155121':
        df_ref_meta = pd.read_csv('GSE155121/Fig1_allweek_cluster_90_metadata.csv', index_col='barcode')
        common_cells = set(adata_ref.obs.index) & set(df_ref_meta.index)
        common_cells = sorted(list(common_cells))
        print('len(common_cells)', len(common_cells))
        adata_ref = adata_ref[common_cells, :].copy()
        df_ref_meta = df_ref_meta.loc[common_cells, :].copy()
        adata_ref.obs = df_ref_meta.copy()

else:
    print('Loading genes.tsv, barcodes.tsv, and matrix.mtx...')
    for filename in ['genes.tsv', 'barcodes.tsv', 'matrix.mtx']:
        assert os.path.isfile(ref_directory + filename), \
            f'FileNotFoundError: {filename} not found in {ref_directory}'

    with open(ref_directory + 'genes.tsv') as f:
        genes = f.read().rstrip().split('\n')

    with open(ref_directory + 'barcodes.tsv') as f:
        barcodes = f.read().rstrip().split('\n')

    mat = mmread(ref_directory + 'matrix.mtx')
    df = pd.DataFrame.sparse.from_spmatrix(mat, index=genes, columns=barcodes).fillna(0)
    adata_ref = sc.AnnData(df.T)
    del genes, barcodes, mat, df
    print('genes.tsv, barcodes.tsv, and matrix.mtx loaded.')

    # Load reference metadata
    # **** Assume cell ID column name is CellID, otherwise please re-assign it here ****
    index_col = 'CellID'

    # Assume the file is in reference directory, filename is expected as meta.tsv, meta.csv, or meta.xlsx
    # If filename is meta.tsv, assume sep='\t', else if filename is meta.csv, assume sep=',' 
    if os.path.isfile(ref_directory + 'meta.tsv'):
        df_ref_meta = pd.read_csv(ref_directory + 'meta.tsv', sep='\t', index_col=index_col) 
    elif os.path.isfile(ref_directory + 'meta.csv'): 
        df_ref_meta = pd.read_csv(ref_directory + 'meta.csv', sep=',', index_col=index_col)
    elif os.path.isfile(ref_directory + 'meta.xlsx'):
        df_ref_meta = pd.read_excel(ref_directory + 'meta.xlsx', index_col=index_col)
    else:
        df_ref_meta = None
    assert df_ref_meta is not None, \
        f'FileNotFoundError: None of meta.tsv, meta.csv, or meta.xlsx found in {ref_directory}'
    
    # Attach metadata to adata
    df_ref_meta = df_ref_meta.loc[adata_ref.obs_names, :]
    adata_ref.obs = df_ref_meta.copy()
    del df_ref_meta

print(f'{ref_name} loaded as reference dataset.')
print('--------')

# In metadata cell type strings, replace each '/' with '.' as they will appear in output filenames
assert ref_class_col in adata_ref.obs_keys(), \
    f'IndexNotFoundError: {ref_class_col} column not found in {ref_name} metadata'
assert ref_age_col in adata_ref.obs_keys(), \
    f'IndexNotFoundError: {ref_age_col} column not found in {ref_name} metadata'
adata_ref = adata_ref[adata_ref.obs[ref_class_col].notna()].copy()
adata_ref = adata_ref[adata_ref.obs[ref_age_col].notna()].copy()
adata_ref.obs[ref_class_col] = adata_ref.obs[ref_class_col].map(lambda x: x.replace('/', '.'))

# -----------------
# Load test dataset
# -----------------
print(f'Loading {test_name} as test dataset...')

# Assume test directory directly contains adata.h5ad OR { genes.tsv, barcodes.tsv, matrix.mtx }
filename = 'adata.h5ad'
if os.path.isfile(test_directory + filename):
    print(f'Loading {filename}...')
    adata_test = sc.read_h5ad(test_directory + filename)
    print(f'{filename} loaded.')

else:
    print('Loading genes.tsv, barcodes.tsv, and matrix.mtx...')
    for filename in ['genes.tsv', 'barcodes.tsv', 'matrix.mtx']:
        assert os.path.isfile(test_directory + filename), \
            f'FileNotFoundError: {filename} not found in {test_directory}'

    with open(test_directory + 'genes.tsv') as f:
        genes = f.read().rstrip().split('\n')

    with open(test_directory + 'barcodes.tsv') as f:
        barcodes = f.read().rstrip().split('\n')

    mat = mmread(test_directory + 'matrix.mtx')
    df = pd.DataFrame.sparse.from_spmatrix(mat, index=genes, columns=barcodes).fillna(0)
    adata_test = sc.AnnData(df.T)

    index_col = 'CellID'
    if os.path.isfile(test_directory + 'meta.csv'):
        df_test_meta = pd.read_csv(test_directory + 'meta.csv', sep=',', index_col=index_col)
    elif os.path.isfile(test_directory + 'meta.tsv'):
        df_test_meta = pd.read_csv(test_directory + 'meta.tsv', sep='\t', index_col=index_col)
    elif os.path.isfile(test_directory + 'meta.xlsx'):
        df_test_meta = pd.read_excel(test_directory + 'meta.xlsx', index_col=index_col)
    else:
        df_test_meta = None
    
    if df_test_meta is not None:
        adata_test.obs = df_test_meta.loc[barcodes, :]
    
    del genes, barcodes, mat, df
    print('genes.tsv, barcodes.tsv, and matrix.mtx loaded.')

print(f'{test_name} loaded as test dataset.')
print('--------')

# **** Datasets all loaded according to user input ****
print('**** Datasets all loaded according to user input. ****')
print()

# keep test genes
test_genes = adata_test.var_names
del adata_test

# colnames = {
#     'cerebellum': dict(ref_class_col='CellType', ref_age_col='age'),
#     'codex': dict(ref_class_col='CellType', ref_age_col='Gestation_week'),
#     'bhaduri': dict(ref_class_col='CellType', ref_age_col='Age'),
#     'bhaduri_d2': dict(ref_class_col='CellType', ref_age_col='age'),
#     'medulloblastoma': dict(col_subgroup='subgroup', col_type='coarse_cell_type', col_subpopulation='tumor_subpopulation'),
#     'glioma': dict(col_subgroup='tumorType', col_type='isTumor', col_subpopulation='location'),
#     'DIPG': dict(col_subgroup='Sample', col_type='Type', col_subpopulation='Sample')
# }

# ref_class_col = colnames[ref_name]['ref_class_col']
# ref_age_col = colnames[ref_name]['ref_age_col']


adata_ref.obs[ref_class_col] = adata_ref.obs[ref_class_col].map(lambda x: x.replace('/', '.'))
adata_ref.obs['age_str'] = adata_ref.obs[ref_age_col].astype(str)
adata_ref.obs['age_str'] = adata_ref.obs['age_str'].map(lambda x: 'E14.0' if x == 'E12.5, E15.5' else x)
if ref_name == 'GSE155121':
    adata_ref.obs['age_int'] = adata_ref.obs['age_str'].map(lambda x: x.replace('W', '').replace('-', '.')).astype(float)
elif ref_name in ['forebrain', 'pons']:
    adata_ref.obs['age_int'] = adata_ref.obs['age_str'].map(lambda x: float(x.replace('E', '')) if x.startswith('E') else 20 + float(x.replace('P', '')))
else:
    adata_ref.obs['age_int'] = adata_ref.obs['age_str'].astype(int)

df_ref_markers = None
# df_ref_markers = pd.read_excel(f'{ref_name}/CellTypeMarker_DevelopingHumanData.xlsx', index_col='Gene')
# df_ref_markers = pd.read_excel(f'{ref_name}/codex_cluster_markers.xlsx', index_col='Gene')
# df_ref_markers = pd.read_excel(f'{ref_name}/bhaduri_clusters_combined.markers.xlsx', index_col='Gene')
# df_ref_markers = pd.read_excel(f'{ref_name}/bhaduri_d2_clusters_combined.markers.xlsx', index_col='Gene')


print('ref_n_samples', adata_ref.n_obs, 'ref_n_genes',  adata_ref.n_vars, 'ref_minValue', adata_ref.X.min(), 'ref_maxValue', adata_ref.X.max())

# Create a directory for output
model_dir = f'{test_name}_{ref_name}_ageClassifier_ageRegressor/'
os.mkdir(model_dir)

# For each cell type, train one age classifier and one age regressor
for celltype, group in tqdm(adata_ref.obs.groupby(ref_class_col)):
    if celltype in ['Outlier_bhaduri', 'outlier_bhaduri_d2']:
        continue
    if group.shape[0] < 100:
        continue

    adata_group = adata_ref[group.index.tolist(), :]

    ages = [age for age, grp in adata_group.obs.groupby('age_str') if grp.shape[0] >= 20]
    if len(ages) < 2:
        continue
    adata_group = adata_group[adata_group.obs['age_str'].isin(ages)]

    # Preprocessing 
    print('Preprocessing adata_group...')
    sc.pp.normalize_total(adata_group)
    sc.pp.log1p(adata_group)
    sc.pp.highly_variable_genes(adata_group, n_top_genes=2000)
    sc.pp.scale(adata_group, zero_center=True, max_value=10)

    # deg
    sc.tl.rank_genes_groups(adata_group, groupby='age_str', use_raw=False, reference='rest', method='wilcoxon')
    num_age_group = len(adata_group.obs['age_str'].drop_duplicates())
    deg = []
    for i in range(num_age_group):
        rank_genes_groups = {key: [tup[i] for tup in arr] for key, arr in adata_group.uns['rank_genes_groups'].items() if key != 'params'}
        df = pd.DataFrame(rank_genes_groups)
        df = df[df['pvals_adj'] < 0.05]
        deg += df['names'].tolist()

    # Select common genes that overlap among training data, testing data, and (optional) training marker genes
    markers = set(adata_group.var_names[adata_group.var.highly_variable]) | set(deg)
    if df_ref_markers is not None:
        markers = markers | set(df_ref_markers.index)
        del df_ref_markers

    common_genes = markers & set(adata_group.var_names) & set(test_genes)
    common_genes = sorted(list(common_genes))
    with open(model_dir + f'common_genes {celltype}.tsv', 'w') as f:
        f.write('\n'.join(common_genes) + '\n')
        print(f'common_genes {celltype} saved.')

    adata_group = adata_group[:, common_genes]

    # Split adata_group into 80% training and 20% validation
    print('Split adata_group into 80% training and 20% validation...')
    cells_train, cells_valid = [], []
    counts = adata_group.obs['age_str'].value_counts()
    num_subsample = counts[counts >= 20].quantile(0.25, interpolation='lower')
    for age, df in adata_group.obs.groupby('age_str'):
        if df.shape[0] < 20:
            print(f'**** Note: age {age} is excluded for too few samples. ****')
            time.sleep(1)
            continue

        cells_this_type = df.index.tolist()
        np.random.shuffle(cells_this_type)

        # Subsample to same level
        cells_this_type = cells_this_type[:num_subsample]

        i = int(len(cells_this_type) * 0.8)
        cells_train += cells_this_type[:i]
        cells_valid += cells_this_type[i:]

    # Shuffle the sample order to mix up cell types
    if len(cells_valid) < 10:
        continue
    np.random.shuffle(cells_train)
    np.random.shuffle(cells_valid)
    
    adata_train = adata_group[cells_train, :]
    adata_valid = adata_group[cells_valid, :]
    del adata_group
    
    # Scale X_train
    scaler = MinMaxScaler()
    X_train = adata_train.X
    if issparse(X_train):
        X_train = X_train.toarray()
    X_train = scaler.fit_transform(X_train)
    dump(scaler, model_dir + f'{ref_name}_{celltype}_scaler_x.joblib')

    # Scale X_valid
    X_valid = adata_valid.X
    if issparse(X_valid):
        X_valid = X_valid.toarray() 
    X_valid = scaler.transform(X_valid)

    # Define decayed learning rate
    epochs = 100
    batch_size = 32
    lr_schedule1 = decay_learning_rate(initial_learning_rate=0.1, final_learning_rate=0.01, epochs=epochs, batch_size=batch_size)
    lr_schedule2 = decay_learning_rate(initial_learning_rate=0.1, final_learning_rate=0.001, epochs=epochs, batch_size=batch_size)
    lr_schedule3 = decay_learning_rate(initial_learning_rate=0.01, final_learning_rate=0.001, epochs=epochs, batch_size=batch_size)

    # Define hyperparameters to tune
    activation = ['relu']
    # hidden_layers = [
    #     [256], [128], [64], [32], 
    #     [256, 128], [256, 64], [256, 32], [128, 64], [128, 32], [64, 32], 
    #     [256, 128, 64], [256, 128, 32], [256, 64, 32], [128, 64, 32], 
    #     [256, 128, 64, 32]
    #     ]
    hidden_layers = [[256, 64, 32]]
    # dropout_rate = [0.1, 0.2]
    dropout_rate = [0.1]
    # learning_rate = [0.1, 0.01, 0.001, lr_schedule1, lr_schedule2, lr_schedule3]
    learning_rate = [lr_schedule2]

    # Create a dictionary of hyperparameters
    param_grid = dict(
        activation=activation,
        hidden_layers=hidden_layers,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate
    )

    # --------------
    # Age classifier
    # --------------
    # One-hot encode y_train
    label_encoder = LabelEncoder()
    y_train_int = label_encoder.fit_transform(adata_train.obs['age_str'].values)
    dump(label_encoder, model_dir + f'{ref_name}_{celltype}_labelEncoder.joblib')
    num_classes = len(np.unique(y_train_int))
    y_train_one_hot = to_categorical(y_train_int, num_classes=num_classes)

    # One-hot encode y_valid
    y_valid_int = label_encoder.transform(adata_valid.obs['age_str'].values)
    y_valid_one_hot = to_categorical(y_valid_int, num_classes=num_classes)

    # # Search for best parameters
    # model = KerasClassifier(build_fn=create_classifier, epochs=epochs, batch_size=batch_size, verbose=2)
    # grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='f1_macro', n_jobs=-1, cv=StratifiedKFold(n_splits=2), error_score='raise')

    # # Run the parameter tuning loop
    # grid.fit(X_train, y_train_int)

    # # Save the best parameters
    # initial_learning_rate, final_learning_rate = recall_learning_rate(grid.best_params_['learning_rate'], epochs=epochs)

    # with open(model_dir + f'{ref_name}_{celltype}_best_classifier_parameters.txt', 'w') as f:
    #     message = f'Best: {grid.best_score_:.4f} using {grid.best_params_}' + '\n'
    #     message += f'initial_learning_rate: {initial_learning_rate}, final_learning_rate: {final_learning_rate}' + '\n'
    #     f.write(message)
    #     print(message)

    # # Use the best parameters to create the best model
    # best_model = create_classifier(
    #     activation=grid.best_params_['activation'],
    #     hidden_layers=grid.best_params_['hidden_layers'],
    #     dropout_rate=grid.best_params_['dropout_rate'],
    #     learning_rate=grid.best_params_['learning_rate']
    # )

    best_model = create_classifier(
        activation=activation[0],
        hidden_layers=hidden_layers[0],
        dropout_rate=dropout_rate[0],
        learning_rate=learning_rate[0]
    )

    # Train the best model
    history = best_model.fit(X_train, y_train_one_hot, 
                            validation_data=(X_valid, y_valid_one_hot), 
                            epochs=epochs, 
                            batch_size=batch_size, 
                            callbacks=[EarlyStopping('val_accuracy', patience=10)], 
                            verbose=2)

    # Save the best model
    best_model.save(model_dir + f'{ref_name}_{celltype}_ageClassifier.h5')
    del best_model
    K.clear_session()

    # Plot history
    for metric in ['loss', 'accuracy']:
        plt.figure()
        plt.plot(history.history[metric])
        plt.plot(history.history['val_' + metric])
        plt.title(f'{ref_name} {celltype} ageClassifier {metric} history')
        plt.xlabel('epoch')
        plt.ylabel(metric)
        plt.legend(['train', 'valid'])
        figure = plt.gcf()
        figure.patch.set_facecolor('white')
        for ex in ['png', 'pdf']:
            figure.savefig(model_dir + f'{ref_name}_{celltype}_ageClassifier_{metric}_history.{ex}', bbox_inches='tight', dpi=300)
        plt.close('all')

    
    # -------------
    # Age regressor
    # -------------
    # Scale y_train
    scaler = MinMaxScaler()
    y_train = np.log(adata_train.obs[['age_int']].values + 1)
    y_train = scaler.fit_transform(y_train)
    dump(scaler, model_dir + f'{ref_name}_{celltype}_scaler_y.joblib')

    # Scale y_valid
    y_valid = np.log(adata_valid.obs[['age_int']].values + 1)
    y_valid = scaler.transform(y_valid)

    # # Search for best parameters
    # model = KerasRegressor(build_fn=create_regressor, epochs=epochs, batch_size=batch_size, verbose=2)
    # grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', n_jobs=-1, cv=2, error_score='raise')

    # # Run the parameter tuning loop
    # grid.fit(X_train, y_train)

    # # Save the best parameters
    # initial_learning_rate, final_learning_rate = recall_learning_rate(grid.best_params_['learning_rate'], epochs=epochs)

    # with open(model_dir + f'{ref_name}_{celltype}_best_regressor_parameters.txt', 'w') as f:
    #     message = f'Best: {grid.best_score_:.4f} using {grid.best_params_}' + '\n'
    #     message += f'initial_learning_rate: {initial_learning_rate}, final_learning_rate: {final_learning_rate}' + '\n'
    #     f.write(message)
    #     print(message)

    # # Use the best parameters to create the best model
    # best_model = create_regressor(
    #     activation=grid.best_params_['activation'],
    #     hidden_layers=grid.best_params_['hidden_layers'],
    #     dropout_rate=grid.best_params_['dropout_rate'],
    #     learning_rate=grid.best_params_['learning_rate']
    # )

    best_model = create_regressor(
        activation=activation[0],
        hidden_layers=hidden_layers[0],
        dropout_rate=dropout_rate[0],
        learning_rate=learning_rate[0]
    )

    # Train the best model
    history = best_model.fit(X_train, y_train, 
                            validation_data=(X_valid, y_valid), 
                            epochs=epochs, 
                            batch_size=batch_size, 
                            callbacks=[EarlyStopping('val_loss', patience=10)], 
                            verbose=2)

    # Save the best model
    best_model.save(model_dir + f'{ref_name}_{celltype}_ageRegressor.h5')
    del best_model
    K.clear_session()

    # Plot history
    for metric in ['loss']:
        plt.figure()
        plt.plot(history.history[metric])
        plt.plot(history.history['val_' + metric])
        plt.title(f'{ref_name} {celltype} ageRegressor {metric} history')
        plt.xlabel('epoch')
        plt.ylabel(metric)
        plt.legend(['train', 'valid'])
        figure = plt.gcf()
        figure.patch.set_facecolor('white')
        for ex in ['png', 'pdf']:
            figure.savefig(model_dir + f'{ref_name}_{celltype}_ageRegressor_{metric}_history.{ex}', bbox_inches='tight', dpi=300)
        plt.close('all')
