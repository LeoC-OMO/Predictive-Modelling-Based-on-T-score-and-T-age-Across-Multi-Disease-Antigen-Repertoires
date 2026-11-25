import os
import json
import math
import argparse
from collections import defaultdict
import pandas as pd
import numpy as np
from scipy.stats import f_oneway,chi2_contingency
from typing import List, Dict, Tuple, Optional

REQUIRED_COLS = ['GeneV', 'GeneJ', 'Protein', 'Percent']

def validate_sample_df(df: pd.DataFrame, sample_name: Optional[str] = None):
    """Ensure the DataFrame has required columns with proper types."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        label = f" [{sample_name}]" if sample_name else ""
        raise ValueError(f"Missing required columns{label}: {missing}")
    # Coerce Percent to numeric; strip strings
    df['GeneV'] = df['GeneV'].astype(str).str.strip()
    df['GeneJ'] = df['GeneJ'].astype(str).str.strip()
    df['Protein'] = df['Protein'].astype(str).str.strip()
    df['Percent'] = pd.to_numeric(df['Percent'], errors='coerce').fillna(0.0)
    # Drop rows with missing identifiers
    df.dropna(subset=['GeneV', 'GeneJ', 'Protein'], inplace=True)

def read_sample(file_path: str) -> pd.DataFrame:
    """
    Read a TCR repertoire sample file (CSV/TSV) and normalize columns.

    Keeps: Number, GeneV, GeneJ, Protein, Percent, Total, annotation
    Ignores other columns (e.g., Epitope_species_DB_V1.0, sapiens/...).
    """
    # Auto-detect delimiter
    with open(file_path, 'r', encoding='utf-8') as f:
        header = f.readline()
    delimiter = ',' if header.count(',') >= header.count('\t') else '\t'

    # Read as strings first; coerce later
    df = pd.read_csv(file_path, delimiter=delimiter, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    # Map columns by semantics
    alt_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ('number', 'cloneid', 'id'):
            alt_map[c] = 'Number'
        elif 'genev' in cl:
            alt_map[c] = 'GeneV'
        elif 'genej' in cl:
            alt_map[c] = 'GeneJ'
        elif 'protein' in cl or 'cdr3' in cl:
            alt_map[c] = 'Protein'
        elif 'percent' in cl or 'freq' in cl or 'frequency' in cl:
            alt_map[c] = 'Percent'
        elif cl == 'total' or 'reads' in cl or 'count' in cl:
            alt_map[c] = 'Total'
        elif cl == 'annotation':  # only explicit 'annotation' column
            alt_map[c] = 'annotation'
    df = df.rename(columns=alt_map)

    # Essential columns for this pipeline (we do not use annotation for features)
    essential = ['GeneV', 'GeneJ', 'Protein', 'Percent']
    missing_essential = [c for c in essential if c not in df.columns]
    if missing_essential:
        raise ValueError(f"Missing essential columns in {file_path}: {missing_essential}")

    # Optional
    if 'Number' not in df.columns:
        df['Number'] = range(1, len(df) + 1)
    if 'Total' not in df.columns:
        df['Total'] = pd.NA
    if 'annotation' not in df.columns:
        df['annotation'] = 'NO'

    # Reorder to a stable set
    df = df[['Number', 'GeneV', 'GeneJ', 'Protein', 'Percent', 'Total', 'annotation']]

    # Coerce types
    df['GeneV'] = df['GeneV'].astype(str).str.strip()
    df['GeneJ'] = df['GeneJ'].astype(str).str.strip()
    df['Protein'] = df['Protein'].astype(str).str.strip()
    df['Percent'] = pd.to_numeric(df['Percent'], errors='coerce').fillna(0.0)
    df['Total'] = pd.to_numeric(df['Total'], errors='coerce')
    df['annotation'] = df['annotation'].fillna('NO').astype(str).str.strip()
    df.loc[df['annotation'] == '', 'annotation'] = 'NO'

    # Drop rows missing identifiers
    df = df.dropna(subset=['GeneV', 'GeneJ', 'Protein'])

    return df


# -----------------------------
# Thresholded Triplet Features
# -----------------------------

def select_high_clones(df: pd.DataFrame,
                       mode: str = 'topN',
                       N: int = 50,
                       q: float = 0.95) -> pd.DataFrame:
    """
    Select high-abundance clones per sample.
    mode='topN': keep top-N by Percent
    mode='quantile': keep clones with Percent >= quantile q
    """
    perc = pd.to_numeric(df['Percent'], errors='coerce').fillna(0.0)
    df = df.assign(_Percent=perc)

    if mode == 'topN':
        order = df.sort_values('_Percent', ascending=False)
        return order.head(min(N, len(order)))
    elif mode == 'quantile':
        thr = float(np.quantile(df['_Percent'].values, q))
        return df.loc[df['_Percent'] >= thr]
        #return df.loc[df['_Percent'] >= thr]
    else:
        raise ValueError("mode must be 'topN' or 'quantile'.")

def triplet_bin_features(df_sel: pd.DataFrame, sample_total: float) -> Dict[str, float]:
    """
    Aggregate selected clones into (V, J, CDR3) triplet features.
    Feature key: TRIPLET_{GeneV}_{GeneJ}_{Protein}
    Feature value: sum of Percent for that triplet / total Percent of full sample.
    """
    if sample_total <= 0:
        sample_total = 1.0
    grouped = df_sel.groupby(['GeneV', 'GeneJ', 'Protein'])['Percent'].sum()
    feats = {f"{v}_{j}_{cdr3}": float(p / sample_total) for (v, j, cdr3), p in grouped.items()}
    return feats


def load_selected_features(csv_path: str) -> pd.DataFrame:
    """
    Load the selected features DataFrame from a CSV file.

    Args:
        csv_path (str): Path to the CSV file (e.g., "selected_features_2026-01-19_01-10-00.csv").

    Returns:
        pd.DataFrame: The reloaded DataFrame with feature names as index.
    """
    try:
        df = pd.read_csv(csv_path, index_col=0)  # Set first column as index (feature names)
        print(f"Successfully loaded {len(df)} features from {csv_path}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")
    except Exception as e:
        raise ValueError(f"Error loading CSV: {str(e)}")


def threshold_to_features(df: pd.DataFrame,
                          mode: str = 'topN',
                          N: int = 50,
                          q: float = 0.95) -> Dict[str, float]:
    """
    From a sample DataFrame, threshold clones and return triplet features (fraction).
    """
    df_sel = select_high_clones(df, mode=mode, N=N, q=q)
    sample_total = float(pd.to_numeric(df['Percent'], errors='coerce').fillna(0.0).sum())
    return triplet_bin_features(df_sel, sample_total=sample_total)



def build_dataframe_verification(dir_path: str) -> list[pd.DataFrame]:  # create all dataframs for one category

    """
    Read all CSV/TSV files in a directory (non-recursive) using read_sample()
    and return a list of DataFrames.
    """
    dataframes: list[pd.DataFrame] = []
    if not os.path.isdir(dir_path):
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    for filename in os.listdir(dir_path):
            full_path = os.path.join(dir_path, filename)
            try:
                df = read_sample(full_path)
                df.attrs['filename'] = filename
                dataframes.append(df)
                print(f"[INFO] Loaded: {filename}, shape: {df.shape}")
            except Exception as e:
                print(f"[WARN] Skipped {filename}: {e}")
    print(f"[INFO] Total files loaded from '{dir_path}': {len(dataframes)}")
    return dataframes


def build_tscore_model_from_df(df: pd.DataFrame) -> dict: # for building model when loading in CSV feature files
    """
    Rebuild the t-score model dict from a loaded DataFrame (e.g., from CSV).
    Assumes DF has index as features, 'combined_score' for weights,
    and per-group columns like 'mean_healthy', 'std_healthy', etc.
    """
    # Assume fixed groups (sorted as in your code)
    groups = sorted(['disease', 'healthy', 'subhealthy'])

    sel_features = list(df.index)
    weights_raw = df['combined_score']
    w = weights_raw / (weights_raw.sum() + 1e-12)

    params = {
        'features': sel_features,
        'weights': w.to_dict(), # weight of each triplet
        'groups': groups,
        'per_group': {} # the different statistics (mean, std) about the triplet
    }
    for g in groups:
        stats = {}
        for f in sel_features:
            mean_col = f'mean_{g}'
            std_col = f'std_{g}'
            if mean_col not in df.columns or std_col not in df.columns:
                raise ValueError(f"Missing column '{mean_col}' or '{std_col}' in DataFrame for group '{g}'")
            stats[f] = {
                'mean': float(df.loc[f, mean_col]),
                'std': float(df.loc[f, std_col]) if float(df.loc[f, std_col]) > 0 else 1.0
            }
        params['per_group'][g] = stats

    return params


def tscore_predict_group(sample_df: pd.DataFrame, model: dict, threshold_mode: str = 'topN', N: int = 200, q: float = 0.95) -> Tuple[str, Dict[str, float]]:
    validate_sample_df(sample_df)
    sample_feats = threshold_to_features(sample_df, mode=threshold_mode, N=N, q=q)
    features = model['features']
    weights = model['weights']
    scores = {}
    for g, stats in model['per_group'].items(): # loop over all groups and stats(std, mean) in the model
        s = 0.0
        for f in features: # loop over every features in the model
            mu = stats[f]['mean']
            sd = stats[f]['std'] if stats[f]['std'] > 0 else 1.0
            x = float(sample_feats.get(f, 0.0)) # gets value of feature f from sample
            z = (x - mu) / sd # How many standard deviations away is the sample’s value from this group’s typical value for this feature
            s += weights.get(f, 0.0) * z # Multiplies the z-score by the learned weight for that feature
        scores[g] = float(s) # stores the final score for group g
    predicted_group = max(scores, key=scores.get) # Looks at all the scores and picks the group that has the highest numerical value
    return predicted_group, scores 

def build_tscore_model_binary_from_df_disease(df: pd.DataFrame) -> dict:
    """
    Rebuild the binary t-score model dict from a loaded DataFrame (e.g., from CSV).
    Assumes DF has index as features, 'combined_score' for weights,
    and per-group columns like 'mean_disease', 'std_disease', 'mean_other', 'std_other', etc.
    """
    groups = sorted(['disease', 'other'])

    sel_features = list(df.index)
    weights_raw = df['combined_score']
    w = weights_raw / (weights_raw.sum() + 1e-12)

    params = {
        'features': sel_features,
        'weights': w.to_dict(),
        'groups': groups,
        'per_group': {}
    }
    for g in groups:
        stats = {}
        for f in sel_features:
            mean_col = f'mean_{g}'
            std_col = f'std_{g}'
            if mean_col not in df.columns or std_col not in df.columns:
                raise ValueError(f"Missing column '{mean_col}' or '{std_col}' in DataFrame for group '{g}'")
            stats[f] = {
                'mean': float(df.loc[f, mean_col]),
                'std': float(df.loc[f, std_col]) if float(df.loc[f, std_col]) > 0 else 1.0
            }
        params['per_group'][g] = stats

    return params



def build_tscore_model_binary_from_df_healthy(df: pd.DataFrame) -> dict:
    """
    Rebuild the binary t-score model dict from a loaded DataFrame (e.g., from CSV).
    Assumes DF has index as features, 'combined_score' for weights,
    and per-group columns like 'mean_disease', 'std_disease', 'mean_other', 'std_other', etc.
    """
    groups = sorted(['healthy', 'other'])

    sel_features = list(df.index)
    weights_raw = df['combined_score']
    w = weights_raw / (weights_raw.sum() + 1e-12)

    params = {
        'features': sel_features,
        'weights': w.to_dict(),
        'groups': groups,
        'per_group': {}
    }
    for g in groups:
        stats = {}
        for f in sel_features:
            mean_col = f'mean_{g}'
            std_col = f'std_{g}'
            if mean_col not in df.columns or std_col not in df.columns:
                raise ValueError(f"Missing column '{mean_col}' or '{std_col}' in DataFrame for group '{g}'")
            stats[f] = {
                'mean': float(df.loc[f, mean_col]),
                'std': float(df.loc[f, std_col]) if float(df.loc[f, std_col]) > 0 else 1.0
            }
        params['per_group'][g] = stats

    return params

def build_tscore_model_binary_from_df_shealthy(df: pd.DataFrame) -> dict:
    """
    Rebuild the binary t-score model dict from a loaded DataFrame (e.g., from CSV).
    Assumes DF has index as features, 'combined_score' for weights,
    and per-group columns like 'mean_disease', 'std_disease', 'mean_other', 'std_other', etc.
    """
    groups = sorted(['subhealthy', 'other'])

    sel_features = list(df.index)
    weights_raw = df['combined_score']
    w = weights_raw / (weights_raw.sum() + 1e-12)

    params = {
        'features': sel_features,
        'weights': w.to_dict(),
        'groups': groups,
        'per_group': {}
    }
    for g in groups:
        stats = {}
        for f in sel_features:
            mean_col = f'mean_{g}'
            std_col = f'std_{g}'
            if mean_col not in df.columns or std_col not in df.columns:
                raise ValueError(f"Missing column '{mean_col}' or '{std_col}' in DataFrame for group '{g}'")
            stats[f] = {
                'mean': float(df.loc[f, mean_col]),
                'std': float(df.loc[f, std_col]) if float(df.loc[f, std_col]) > 0 else 1.0
            }
        params['per_group'][g] = stats

    return params

def load_selected_features_binary_shealthy(csv_path: str) -> pd.DataFrame:
    """
    Load the selected features DataFrame from a CSV file for binary model.
    """
    try:
        df = pd.read_csv(csv_path, index_col=0)  # Set first column as index (feature names)
        print(f"Successfully loaded {len(df)} features from {csv_path}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")
    except Exception as e:
        raise ValueError(f"Error loading CSV: {str(e)}")
    

def load_selected_features_binary_healthy(csv_path: str) -> pd.DataFrame:
    """
    Load the selected features DataFrame from a CSV file for binary model.
    """
    try:
        df = pd.read_csv(csv_path, index_col=0)  # Set first column as index (feature names)
        print(f"Successfully loaded {len(df)} features from {csv_path}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")
    except Exception as e:
        raise ValueError(f"Error loading CSV: {str(e)}")
    
def load_selected_features_binary_disease(csv_path: str) -> pd.DataFrame:
    """
    Load the selected features DataFrame from a CSV file for binary model.
    """
    try:
        df = pd.read_csv(csv_path, index_col=0)  # Set first column as index (feature names)
        print(f"Successfully loaded {len(df)} features from {csv_path}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")
    except Exception as e:
        raise ValueError(f"Error loading CSV: {str(e)}")


def tscore_predict_binary(sample_df: pd.DataFrame, model: dict, threshold_mode: str = 'topN', N: int = 200, q: float = 0.95) -> Tuple[str, Dict[str, float]]:
    """
    Compute T-scores for each binary group for a single sample DataFrame and predict the group with the highest score.
    """
    validate_sample_df(sample_df)
    sample_feats = threshold_to_features(sample_df, mode=threshold_mode, N=N, q=q)
    features = model['features']
    weights = model['weights']
    scores = {}
    for g, stats in model['per_group'].items():
        s = 0.0
        for f in features:
            mu = stats[f]['mean']
            sd = stats[f]['std'] if stats[f]['std'] > 0 else 1.0
            x = float(sample_feats.get(f, 0.0))
            z = (x - mu) / sd
            s += weights.get(f, 0.0) * z
        scores[g] = float(s)
    predicted_group = max(scores, key=scores.get)
    return predicted_group, scores

def compute_final_scores(multi_scores: Dict[str, float], binary_scores: List[Dict[str, float]], w_multi: float = 0.4, w_b: float = 0.2) -> Dict[str, float]:
    """
    Compute final scores S_c for each class using the ensemble equation.

    Args:
        multi_scores (Dict[str, float]): T-scores from the multi-class model, e.g., {'healthy': 0.6, 'subhealthy': 0.3, 'disease': 0.1}
        binary_scores (List[Dict[str, float]]): List of T-scores from the 3 binary models. Each dict has keys 'positive_class' (e.g., 'healthy') and 'other'.
            Order: [healthy_binary, subhealthy_binary, disease_binary]
        w_multi (float): Weight for multi-class scores (default 0.4)
        w_b (float): Weight for each binary (default 0.2; assumes same for all, total 0.6 for 3 binaries)

    Returns:
        Dict[str, float]: Final S_c scores for each class {'healthy': S, 'subhealthy': S, 'disease': S}
    """
    classes = ['healthy', 'subhealthy', 'disease']
    binary_positives = classes  # Assumes binaries in order: healthy, subhealthy, disease

    S = {c: 0.0 for c in classes}

    for c in classes:
        # Multi-class contribution
        S[c] += w_multi * multi_scores.get(c, 0.0)

        # Binary contributions
        for idx, b_scores in enumerate(binary_scores):
            b_positive = binary_positives[idx] # get all current class scores from the model scores
            I = 1 if b_positive == c else 0 # gets  
            T_b_c = b_scores.get(b_positive, 0.0)  # T_{b,c} if matches, else 0
            T_b_other = b_scores.get('other', 0.0)
            S[c] += w_b * (T_b_c * I + T_b_other * (1 - I))

    return S

def predict_from_final_scores(S: Dict[str, float]) -> str:
    """Predict the class with the highest final score."""
    return max(S, key=S.get)


# ensemble for verdict on group
# def predict_df_class(dfs:list[pd.DataFrame], model_full_external: dict, model_binary_shealthy: dict, model_binary_healthy: dict, model_binary_disease: dict):
#   for i in range(len(dfs)):
#     # 1. get group using the two models
#     group, score = tscore_predict_group(dfs[i],model_full_external)
#     #print("result of full model is:", group, score)


#     #group_7k, score_7k = tscore_predict_group(dataframe_verify_disease[i],model_7k)
#     #print("Result of 7k model is:", group_7k, score_7k)

#     # 2. get groups using the binary models
#     group_bin_sh, score_bin_sh = tscore_predict_binary(dfs[i],model_binary_shealthy)
#     #print("Result of binary model shealthy is:", group_bin_sh, score_bin_sh)
#     group_bin_h, score_bin_h = tscore_predict_binary(dfs[i],model_binary_healthy)
#     #print("Result of binary model shealthy is:", group_bin_h, score_bin_h)
#     group_bin_d, score_bin_d = tscore_predict_binary(dfs[i],model_binary_disease)
#     #print("Result of binary model shealthy is:", group_bin_d, score_bin_d)

#     # 3. function for final verdict:
#     final_s = compute_final_scores(score,[score_bin_h,score_bin_sh,score_bin_d])
#     predicted = predict_from_final_scores(final_s)
#     print("The class of file " ,dfs[i].attrs['filename'], " is: ", predicted)
#     return predicted, final_s

def predict_df_class(
    dfs: list[pd.DataFrame],
    model_full_external: dict,
    model_binary_shealthy: dict,
    model_binary_healthy: dict,
    model_binary_disease: dict,
    top_n:int,
    threshold_mode_used: str
) -> list[dict]:
    """
    Predict class for each sample and return structured results.
    Does NOT print — returns data for GUI or further processing.
    """
    results = []

    for df in dfs:
        filename = df.attrs.get('filename', 'unknown_file')

        # Multi-class
        group_multi, score_multi = tscore_predict_group(df, model_full_external,N = top_n, threshold_mode=threshold_mode_used)

        # Binary models
        _, score_bin_h   = tscore_predict_binary(df, model_binary_healthy,N = top_n, threshold_mode=threshold_mode_used)
        _, score_bin_sh  = tscore_predict_binary(df, model_binary_shealthy,N = top_n, threshold_mode=threshold_mode_used)
        _, score_bin_d   = tscore_predict_binary(df, model_binary_disease,N = top_n, threshold_mode=threshold_mode_used)

        # Ensemble
        final_s = compute_final_scores(
            score_multi,
            [score_bin_h, score_bin_sh, score_bin_d]
        )
        predicted = predict_from_final_scores(final_s)

        # Store everything useful
        result = {
            'filename': filename,
            'predicted_class': predicted,
            'final_scores': {k: round(v, 4) for k,v in final_s.items()},
            'df':df,
            'multi_group': group_multi,
            'multi_scores': {k: round(v, 4) for k,v in score_multi.items()},
            'bin_healthy_score': round(score_bin_h.get('healthy', 0.0), 4),
            'bin_subhealthy_score': round(score_bin_sh.get('subhealthy', 0.0), 4),
            'bin_disease_score': round(score_bin_d.get('disease', 0.0), 4),
        }

        results.append(result)

    return results

def tscore_predict_with_biophysics(sample_df: pd.DataFrame, model: dict, weight_bio: float = 0.3) -> tuple:
    """
    Updated prediction that uses both abundance features and new biophysical properties.
    weight_bio: how much weight to give to the new biophysical features (0.0 ~ 1.0)
    """
    # Original abundance-based features
    sample_feats = threshold_to_features(sample_df)
    
    scores = {}
    for g in model['groups']:
        s = 0.0
        
        # 1. Original abundance contribution
        for f in model['features']:
            mu = model['per_group'][g][f]['mean']
            sd = model['per_group'][g][f]['std']
            x = sample_feats.get(f, 0.0)
            z = (x - mu) / sd if sd > 0 else 0.0
            w = model['weights'].get(f, 0.0)
            s += w * z
        
        # 2. New biophysical contribution (if available in sample_df)
        if hasattr(sample_df, 'attrs') and 'biophysical' in sample_df.attrs:
            bio_data = sample_df.attrs['biophysical']
            for prop, value in bio_data.items():
                # You need to have pre-computed mean/std for these properties per group
                # For simplicity, we can add a simple bonus if pathology matches
                if prop == 'pathology' and value != "none":
                    s += weight_bio * 0.5   # example bonus
        
        scores[g] = s
    
    predicted_group = max(scores, key=scores.get)
    return predicted_group, scores

def tscore_predict_binary_with_biophysics(sample_df: pd.DataFrame, model: dict, weight_bio: float = 0.3) -> tuple:
    """
    Updated binary prediction with biophysical features.
    """
    sample_feats = threshold_to_features(sample_df)
    
    scores = {}
    for g in model['groups']:
        s = 0.0
        
        # Original abundance part
        for f in model['features']:
            mu = model['per_group'][g][f]['mean']
            sd = model['per_group'][g][f]['std']
            x = sample_feats.get(f, 0.0)
            z = (x - mu) / sd if sd > 0 else 0.0
            w = model['weights'].get(f, 0.0)
            s += w * z
        
        # Add biophysical contribution (simple version)
        if hasattr(sample_df, 'attrs') and 'biophysical' in sample_df.attrs:
            bio = sample_df.attrs['biophysical']
            if bio.get('pathology') != "none":
                s += weight_bio * 0.4   # bonus if pathology is present
        
        scores[g] = s
    
    predicted_group = max(scores, key=scores.get)
    return predicted_group, scores