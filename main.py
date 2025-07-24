# --- FINAL EXPERIMENT: Highly Optimized Filter-Bank Riemannian SVC ---

import os
import glob
import mne
import numpy as np
import warnings
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

# --- Suppress Warnings ---
warnings.filterwarnings("ignore", category=RuntimeWarning, module='mne')

# --- 1. SETUP: Use Neuro-informed bands and optimal time window ---
DATA_PATH = r'C:\Users\bukka\Downloads\BCICIV_2a_gdf' 
# Recommendation 2: Smarter, neurophysiologically-informed bands
FILTER_BANDS = [(8, 13), (13, 20), (20, 30)] # Alpha, Low Beta, High Beta
TMIN, TMAX = 3.0, 5.5

# --- 2. OPTIMIZED DATA LOADING ---
def load_bci_data_filterbank(data_path, filter_bands, tmin, tmax):
    """
    Loads data once and pre-computes all filtered versions for efficiency.
    Returns a dictionary of data arrays, one for each band.
    """
    subject_files = sorted(glob.glob(os.path.join(data_path, 'A0*T.gdf')))
    
    # Recommendation 1: Pre-compute filtered data
    # Create a dictionary to hold data for each band
    # e.g., precomputed_data[subject_id][(8,13)] = numpy_array
    precomputed_data = {}
    y_dict, groups_dict = {}, {}

    for subject_file in subject_files:
        subject_id_str = os.path.basename(subject_file).split('T.')[0]
        print(f"Preprocessing {subject_id_str}...")
        
        raw = mne.io.read_raw_gdf(subject_file, preload=True, verbose=False)
        raw.pick_types(eeg=True)
        
        events, event_id_map = # ... (Same event loading logic as the champion script) ...
        # ... (Same event loading logic as the champion script) ...
        # ... (Same event loading logic as the champion script) ...
        events_from_annot, event_dict_from_annot = mne.events_from_annotations(raw, verbose=False)
        id_to_description = {v: k for k, v in event_dict_from_annot.items()}
        target_descriptions = ['769', '770', '771', '772'] 
        description_to_new_label = {'769': 0, '770': 1, '771': 2, '772': 3}
        
        mi_events = []
        for event in events_from_annot:
            event_description = id_to_description.get(event[2], '')
            for target in target_descriptions:
                if target in event_description:
                    new_label = description_to_new_label[target]
                    mi_events.append([event[0], event[1], new_label])
                    break
                    
        mi_events = np.array(mi_events)
        event_id_map = {'left_hand': 0, 'right_hand': 1, 'feet': 2, 'tongue': 3}

        
        epochs_base = mne.Epochs(raw, mi_events, event_id=event_id_map, tmin=tmin, tmax=tmax, 
                                 proj=False, baseline=None, preload=True, verbose=False, event_repeated='drop')
        
        # Store labels and groups once
        subject_idx = int(subject_id_str.replace('A0',''))
        y_dict[subject_idx] = epochs_base.events[:, -1]
        groups_dict[subject_idx] = np.full(len(epochs_base), subject_idx)
        
        # Filter and store data for each band
        precomputed_data[subject_idx] = {}
        for fmin, fmax in filter_bands:
            epochs_filtered = epochs_base.copy().filter(fmin, fmax, verbose=False)
            precomputed_data[subject_idx][(fmin, fmax)] = epochs_filtered.get_data()

    # Concatenate all subjects' data
    X_dict = {}
    for band in filter_bands:
        X_dict[band] = np.concatenate([precomputed_data[sid][band] for sid in sorted(precomputed_data.keys())])
    
    y = np.concatenate([y_dict[sid] for sid in sorted(y_dict.keys())])
    groups = np.concatenate([groups_dict[sid] for sid in sorted(groups_dict.keys())])

    return X_dict, y, groups

# --- 3. OPTIMIZED FILTER-BANK TRANSFORMER ---
class EfficientFilterBank(BaseEstimator, TransformerMixin):
    def __init__(self, filter_bands):
        self.filter_bands = filter_bands
        self.pipelines_ = {band: make_pipeline(Covariances(estimator='lwf'), TangentSpace(metric='riemann')) for band in filter_bands}

    def fit(self, X_dict, y=None):
        # Fit a separate pipeline for each band's data
        for band in self.filter_bands:
            self.pipelines_[band].fit(X_dict[band], y)
        return self

    def transform(self, X_dict):
        # Transform each band and concatenate features
        features = [self.pipelines_[band].transform(X_dict[band]) for band in self.filter_bands]
        return np.concatenate(features, axis=1)

# --- 4. MAIN SCRIPT LOGIC ---
print("Loading and pre-filtering data for all bands...")
X_dict, y, groups = load_bci_data_filterbank(DATA_PATH, FILTER_BANDS, TMIN, TMAX)

# Create the full pipeline including feature reduction
fb_pipeline = Pipeline([
    ('filterbank', EfficientFilterBank(filter_bands=FILTER_BANDS)),
    ('scaler', StandardScaler()),
    # Recommendation 3: Add dimensionality reduction
    ('pca', PCA()), 
    ('classifier', SVC(probability=True)) # Probability=True needed for some metrics if you add them
])

# Recommendation 4: More sophisticated hyperparameter search
param_grid = {
    'pca__n_components': [50, 100, 150], # How many features to keep
    'classifier__C': [0.1, 1, 10],
    'classifier__kernel': ['linear', 'rbf']
}

logo = LeaveOneGroupOut()
# Using n_jobs=-1 will parallelize the grid search across folds
clf = GridSearchCV(fb_pipeline, param_grid, cv=logo, n_jobs=-1, scoring='accuracy', verbose=2)

print("\n--- Running Highly Optimized Filter-Bank Riemannian SVC ---")
clf.fit(X_dict, y, groups=groups)

# --- 5. FINAL RESULTS ---
print("\n\n--- FINAL RESULTS FOR OPTIMIZED FILTER-BANK ---")
print(f"Best Mean Cross-Val Accuracy: {clf.best_score_:.4f}")
print(f"Best Params: {clf.best_params_}")

print("\n--- For Comparison ---")
print("Champion Benchmark (Single Band R-SVC): 0.3208")

if clf.best_score_ > 0.3208:
    print(f"✓✓✓ BREAKTHROUGH: +{clf.best_score_ - 0.3208:.4f} accuracy gain!")
else:
    print(f"✗ No improvement over the simpler single-band model.")
