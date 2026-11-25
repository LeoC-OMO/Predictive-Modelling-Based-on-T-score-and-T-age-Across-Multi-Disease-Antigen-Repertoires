import os
import json
import math
import argparse
from collections import defaultdict
import pandas as pd
import numpy as np
from scipy.stats import f_oneway,chi2_contingency
from typing import List, Tuple

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

def build_dataframe(dir_path: str) -> list[pd.DataFrame]:  # create all dataframs for one category

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
                dataframes.append(df)
                print(f"[INFO] Loaded: {filename}, shape: {df.shape}")
            except Exception as e:
                print(f"[WARN] Skipped {filename}: {e}")
    print(f"[INFO] Total files loaded from '{dir_path}': {len(dataframes)}")
    return dataframes

#-------
# Binary model healthy
#------

# binary_tcr_triplet_inmemory_pipeline.py


# It classifies sample into either healthy or other group.
import math
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy.stats import f_oneway

# -----------------------------
# Validation Helpers
# -----------------------------

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
    else:
        raise ValueError("mode must be 'topN' or 'quantile'.")

def triplet_bin_features(df_sel: pd.DataFrame, sample_total: float) -> Dict[str, float]:
    """
    Aggregate selected clones into (V, J, CDR3) triplet features.
    Feature key: {GeneV}_{GeneJ}_{Protein}
    Feature value: sum of Percent for that triplet / total Percent of full sample.
    """
    if sample_total <= 0:
        sample_total = 1.0
    grouped = df_sel.groupby(['GeneV', 'GeneJ', 'Protein'])['Percent'].sum()
    feats = {f"{v}_{j}_{cdr3}": float(p / sample_total) for (v, j, cdr3), p in grouped.items()}
    return feats

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

# -----------------------------
# Build Feature Matrix (from lists) - Binary Version
# -----------------------------

def build_feature_matrix_from_loaded_triplets_binary(
    healthy_dfs: List[pd.DataFrame],
    subhealthy_dfs: List[pd.DataFrame],
    disease_dfs: List[pd.DataFrame],
    mode: str = 'topN',
    N: int = 50,
    q: float = 0.95,
    validate: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Construct X (features) and y (labels) from in-memory DataFrames (triplet features after thresholding).
    Binary: 'other' for healthy + subhealthy, 'disease' for disease.
    Returns:
      X: shape (num_samples, num_features) with triplet columns; values are fractions
      y: Series of group labels in ['other', 'disease']
    """
    rows: List[Dict[str, float]] = []
    labels: List[str] = []
    sample_ids: List[str] = []
    triplet_union: set = set()

    def process_group(dfs: List[pd.DataFrame], label: str):
        for i, df in enumerate(dfs):
            if validate:
                validate_sample_df(df, sample_name=f"{label}|{i:02d}")
            feats = threshold_to_features(df, mode=mode, N=N, q=q)
            triplet_union.update(feats.keys())
            rows.append(feats)
            labels.append(label)
            sample_ids.append(f"{label}|{i:02d}")

    # Combine disease and subhealthy as 'other'
    process_group(disease_dfs + subhealthy_dfs, 'other')
    process_group(healthy_dfs, 'healthy')

    cols = sorted(triplet_union)
    X = pd.DataFrame([{c: row.get(c, 0.0) for c in cols} for row in rows], index=sample_ids).fillna(0.0)
    y = pd.Series(labels, name='Group', index=sample_ids)
    return X, y

# -----------------------------
# Feature Selection (MI + ANOVA)
# -----------------------------

def discretize_series(x: pd.Series, n_bins: int = 5) -> pd.Series:
    """
    Hybrid discretization: 
    - If data is sparse (>80% zeros), use Binary (Presence/Absence).
    - Otherwise, use Quantile binning with a fallback to Uniform binning.
    """
    if (x == 0).mean() > 0.8:
        return (x > 0).astype(int)
    
    try:
        # labels=False returns integers, which is faster for crosstab
        return pd.qcut(x, q=n_bins, duplicates='drop', labels=False)
    except Exception:
        return pd.cut(x, bins=n_bins, labels=False)

def mutual_information_discrete(x: pd.Series, y: pd.Series) -> float:
    """
    Empirical Mutual Information for two discrete variables.
    x should be discretized; y can be categorical labels.
    """
    x_codes = pd.Categorical(x).codes
    y_codes = pd.Categorical(y).codes
    joint = pd.crosstab(x_codes, y_codes)
    joint = joint / joint.values.sum()
    
    if joint.empty or joint.values.sum() == 0:
        return 0.0
    
    px = joint.sum(axis=1).to_numpy()
    py = joint.sum(axis=0).to_numpy()
    mi = 0.0
    for i in range(joint.shape[0]):
        for j in range(joint.shape[1]):
            pxy = joint.iat[i, j]
            if pxy > 0 and px[i] > 0 and py[j] > 0:
                mi += pxy * math.log(pxy / (px[i] * py[j] + 1e-12) + 1e-12)
    return float(mi)

def anova_f_test(feature: pd.Series, groups: pd.Series) -> float:
    """Return ANOVA p-value across multiple groups for a numeric feature."""
    arrays = [feature[groups == g].astype(float) for g in sorted(groups.unique())]
    if sum(len(a) > 1 for a in arrays) < 2:
        return 1.0
    try:
        stat, pval = f_oneway(*arrays)
        return float(pval)
    except Exception:
        return 1.0

# -----------------------------
# One-vs-Rest Group Feature Ranking (Adapted for Binary)
# -----------------------------

def prevalence(series: pd.Series, threshold: float = 0.0) -> float:
    """Fraction of entries above threshold (default > 0)."""
    vals = series.astype(float).values
    return float(np.mean(vals > threshold))

def anova_p_for_two(feature: pd.Series, mask_a: pd.Series) -> float:
    """ANOVA-like p for one-vs-rest (2 groups). If too few, return 1.0."""
    a = feature[mask_a].astype(float)
    b = feature[~mask_a].astype(float)
    if len(a) < 2 or len(b) < 2:
        return 1.0
    try:
        stat, pval = f_oneway(a, b)
        return float(pval)
    except Exception:
        return 1.0

def cohens_d_one_vs_rest(feature: pd.Series, mask_a: pd.Series) -> float:
    """Cohen's d for one-vs-rest effect size."""
    a = feature[mask_a].astype(float)
    b = feature[~mask_a].astype(float)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    ma, mb = float(a.mean()), float(b.mean())
    sa, sb = float(a.std(ddof=1)), float(b.std(ddof=1))
    na, nb = len(a), len(b)
    denom = ((na - 1) * sa**2 + (nb - 1) * sb**2)
    sp = math.sqrt(denom / (na + nb - 2)) if (na + nb - 2) > 0 else 1.0
    if sp == 0:
        return 0.0
    return (ma - mb) / sp

def rank_features_one_vs_rest(X: pd.DataFrame,
                              y: pd.Series,
                              group_name: str,
                              n_bins_mi: int = 5,
                              min_prevalence_in: float = 0.3,
                              max_prevalence_out: float = 0.2) -> pd.DataFrame:
    """
    Rank features for a given group using:
      - prevalence in-group vs out-group
      - Cohen's d (effect size)
      - MI (one-vs-rest, with discretization)
      - ANOVA p (one-vs-rest)
    Filters features by prevalence criteria before ranking.
    """
    mask_a = (y == group_name)
    prev_in = X.loc[mask_a].apply(prevalence, axis=0)
    prev_out = X.loc[~mask_a].apply(prevalence, axis=0)

    keep = (prev_in >= min_prevalence_in) & (prev_out <= max_prevalence_out)
    Xf = X.loc[:, keep.index[keep]]
    if Xf.shape[1] == 0:
        return pd.DataFrame(columns=['feature','prev_in','prev_out','cohens_d','mi','anova_p','combined'])

    # Stats
    d_scores = Xf.apply(lambda s: cohens_d_one_vs_rest(s, mask_a), axis=0)
    y_bin = y.apply(lambda v: group_name if v == group_name else 'other')
    mi_scores = Xf.apply(lambda s: mutual_information_discrete(discretize_series(s, n_bins_mi), y_bin), axis=0)
    pvals = Xf.apply(lambda s: anova_p_for_two(s, mask_a), axis=0)

    # Normalize d and MI; convert p to -log10
    d_norm = (d_scores - d_scores.min()) / (d_scores.max() - d_scores.min() + 1e-12)
    mi_norm = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min() + 1e-12)
    p_score = -np.log10(pvals + 1e-12)
    p_norm = (p_score - p_score.min()) / (p_score.max() - p_score.min() + 1e-12)

    combined = 0.5 * d_norm + 0.3 * mi_norm + 0.2 * p_norm

    out = pd.DataFrame({
        'feature': Xf.columns,
        'prev_in': prev_in[Xf.columns].values,
        'prev_out': prev_out[Xf.columns].values,
        'cohens_d': d_scores.values,
        'mi': mi_scores.values,
        'anova_p': pvals.values,
        'combined': combined.values
    }).sort_values('combined', ascending=False)

    return out

# -----------------------------
# Feature-Group Association Utilities - Binary Version
# -----------------------------

def compute_per_group_means(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Return a DataFrame (features x groups) of per-group means.
    Index: feature names (columns of X)
    Columns: sorted group names ('disease', 'other')
    """
    groups = sorted(y.astype(str).unique())
    means = {}
    for g in groups:
        means[g] = X.loc[y.astype(str) == g].astype(float).mean(axis=0)
    means_df = pd.DataFrame(means)
    means_df.index.name = 'feature'
    return means_df

def compute_per_group_stds(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Return a DataFrame (features x groups) of per-group stds (ddof=1).
    Index: feature names (columns of X)
    Columns: sorted group names ('disease', 'other')
    """
    groups = sorted(y.astype(str).unique())
    stds = {}
    for g in groups:
        stds[g] = X.loc[y.astype(str) == g].astype(float).std(ddof=1, axis=0).fillna(1.0)
    stds_df = pd.DataFrame(stds)
    stds_df.index.name = 'feature'
    return stds_df

def compute_one_vs_rest_effects(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Return a DataFrame (features x groups) with Cohen's d (one-vs-rest) for each group.
    Positive values mean higher in the target group than rest.
    """
    groups = sorted(y.astype(str).unique())
    d_mat = {}
    mask_cache = {g: (y.astype(str) == g) for g in groups}
    for g in groups:
        mask = mask_cache[g]
        d_scores = X.apply(lambda s: cohens_d_one_vs_rest(s, mask), axis=0)
        d_mat[g] = d_scores
    d_df = pd.DataFrame(d_mat)
    d_df.index.name = 'feature'
    return d_df

def assign_feature_associated_group_binary(
    X: pd.DataFrame,
    y: pd.Series,
    selected_df: pd.DataFrame,
    strategy: str = "max_mean",
    min_effect: float = 0.0
) -> pd.DataFrame:
    """
    Annotate selected features with an 'associated_group' column for binary ('disease' or 'other').

    strategy:
      - 'max_mean'     : choose the group with the highest mean for the feature.
      - 'one_vs_rest'  : choose the group with the highest positive Cohen's d.
                         If all d <= min_effect, mark as 'ambiguous'.
    """
    sel = selected_df.copy()
    feats = list(sel.index)

    # Compute stats on only the selected features for efficiency
    means_df = compute_per_group_means(X[feats], y)
    d_df     = compute_one_vs_rest_effects(X[feats], y)
    stds_df  = compute_per_group_stds(X[feats], y)

    assoc_group = []
    assoc_score = []
    assoc_strategy = []

    if strategy == "max_mean":
        for f in feats:
            g = means_df.loc[f].idxmax()
            assoc_group.append(g)
            assoc_score.append(float(means_df.loc[f, g]))
            assoc_strategy.append("max_mean")
    elif strategy == "one_vs_rest":
        for f in feats:
            best_g = d_df.loc[f].idxmax()
            best_d = float(d_df.loc[f, best_g])
            if best_d > min_effect:
                assoc_group.append(best_g)
            else:
                assoc_group.append("ambiguous")
            assoc_score.append(best_d)
            assoc_strategy.append("one_vs_rest")
    else:
        raise ValueError("strategy must be 'max_mean' or 'one_vs_rest'.")

    # Attach summary columns
    sel['associated_group'] = assoc_group
    sel['associated_score'] = assoc_score
    sel['assoc_strategy']   = assoc_strategy

    # Append per-group means, Cohen's d, and stds (prefix columns)
    means_prefixed = means_df.rename(columns={c: f"mean_{c}" for c in means_df.columns})
    d_prefixed     = d_df.rename(columns={c: f"d_{c}" for c in d_df.columns})
    stds_prefixed  = stds_df.rename(columns={c: f"std_{c}" for c in stds_df.columns})

    sel = sel.join(means_prefixed)
    sel = sel.join(d_prefixed)
    sel = sel.join(stds_prefixed)

    # Reorder columns: key metrics first, then per-group stats
    front_cols = ['mi', 'anova_p', 'combined_score', 'associated_group', 'associated_score', 'assoc_strategy']
    other_cols = [c for c in sel.columns if c not in front_cols]
    sel = sel.loc[:, front_cols + other_cols]

    return sel

# -----------------------------
# Feature Selection (adds association) - Binary Version
# -----------------------------

def select_features_binary(
    X: pd.DataFrame,
    y: pd.Series,
    n_top: int = 50,
    n_bins_mi: int = 5,
    attach_association: bool = True,
    association_strategy: str = "max_mean",
    min_effect_for_association: float = 0.0
) -> pd.DataFrame:
    """
    Compute MI (discretized) + ANOVA p-values; return top-N features with combined score:
      combined = 0.6 * normalized_MI + 0.4 * normalized(-log10(ANOVA p))

    If attach_association=True, annotate each selected feature with an associated group ('disease' or 'other').
    """
    mi_scores: Dict[str, float] = {}
    f_pvalues: Dict[str, float] = {}

    y_str = y.astype(str)

    for col in X.columns:
        x = X[col]
        xd = discretize_series(x, n_bins=n_bins_mi)      # MI discretization
        mi_scores[col] = mutual_information_discrete(xd, y_str)
        f_pvalues[col] = anova_f_test(x, y_str)          # ANOVA p-value

    mi_series = pd.Series(mi_scores, name='mi')
    f_series = pd.Series(f_pvalues, name='anova_p')

    mi_norm = (mi_series - mi_series.min()) / (mi_series.max() - mi_series.min() + 1e-12)
    f_score = -np.log10(f_series + 1e-12)
    f_norm  = (f_score - f_score.min()) / (f_score.max() - f_score.min() + 1e-12)

    combined = 0.6 * mi_norm + 0.4 * f_norm

    rank_df = pd.DataFrame({'mi': mi_series, 'anova_p': f_series, 'combined_score': combined})
    rank_df = rank_df.sort_values(by='combined_score', ascending=False)
    selected = rank_df

    if attach_association and len(selected) > 0:
        selected = assign_feature_associated_group_binary(
            X, y_str,
            selected_df=selected,
            strategy=association_strategy,
            min_effect=min_effect_for_association
        )

    return selected

# -----------------------------
# T-score Model - Binary Version
# -----------------------------

def build_tscore_model_binary(X: pd.DataFrame, y: pd.Series, selected: pd.DataFrame) -> dict:
    """
    Store per-group mean/std for selected features and normalized weights from combined scores.
    T-score for a new sample against group g:
        T_g = sum_f w_f * (x_f - mean_{f,g}) / std_{f,g}
    Binary groups: 'disease', 'other'
    """
    groups = sorted(y.astype(str).unique())
    sel_features = list(selected.index)
    weights_raw = selected['combined_score']
    w = weights_raw / (weights_raw.sum() + 1e-12)

    params = {
        'features': sel_features,
        'weights': w.to_dict(),
        'groups': [str(g) for g in groups],
        'per_group': {}
    }
    for g in groups:
        gmask = (y.astype(str) == g)
        stats = {}
        for f in sel_features:
            vals = X.loc[gmask, f].astype(float)
            stats[f] = {
                'mean': float(vals.mean()),
                'std': float(vals.std(ddof=1)) if len(vals) > 1 else 1.0
            }
        params['per_group'][str(g)] = stats

    return params

def tscore_predict_binary(sample_df: pd.DataFrame, model: dict, threshold_mode: str = 'topN', N: int = 50, q: float = 0.95) -> Tuple[str, Dict[str, float]]:
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

# -----------------------------
# Load and Build Model from CSV - Binary Version
# -----------------------------

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

# -----------------------------
# Orchestration (In-Memory) - Binary Version
# -----------------------------

def run_binary_pipeline_from_loaded_triplets_healthy(
    healthy_dfs: List[pd.DataFrame],
    subhealthy_dfs: List[pd.DataFrame],
    disease_dfs: List[pd.DataFrame],
    n_top: int = 50,
    n_bins_mi: int = 15,
    threshold_mode: str = 'topN',
    N: int = 50,
    q: float = 0.95,
    validate: bool = True,
    association_strategy: str = "max_mean",
    min_effect_for_association: float = 0.0
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, dict]:
    """
    End-to-end binary pipeline: build X, y ('other' vs 'disease'), select features (with association),
    and build T-score model.
    Returns: (X, y, selected_features_df, tscore_model_dict)
    """
    print("Training model...")
    X, y = build_feature_matrix_from_loaded_triplets_binary(
        healthy_dfs, subhealthy_dfs, disease_dfs,
        mode=threshold_mode, N=N, q=q, validate=validate
    )
    
    X = np.log1p(X)
    
    selected = select_features_binary(
        X, y.astype(str),
        n_top=n_top,
        n_bins_mi=n_bins_mi,
        attach_association=True,
        association_strategy=association_strategy,
        min_effect_for_association=min_effect_for_association
    )
    model = build_tscore_model_binary(X, y, selected)
    return X, y, selected, model


# === Main Function ===
dataframes_disease = build_dataframe(r"tcr_full_csv_train\disease_7k_train_full")
dataframes_subhealth = build_dataframe(r"tcr_full_csv_train\subhealthy_7k_train_full")
dataframes_healthy = build_dataframe(r"tcr_full_csv_train\healthy_7k_train_full")

# healthy binary model, feature selection
X_h, y_h, selected_h, model_h = run_binary_pipeline_from_loaded_triplets_healthy(
        dataframes_healthy,
        dataframes_subhealth,
        dataframes_disease,
        n_top=5000,          # select top 5000 features by combined score
        n_bins_mi=10,       # quantile bins for MI
        threshold_mode='quantile',
        N=5000,              # keep top-N clones per sample
        q=0.90,            # only used if mode='quantile'
        validate=True,
        association_strategy = "one_vs_rest"
    )


# Export to CSV 
print("exporting model to CSV")
selected_h.to_csv(f"7k_binary_healthy_testing_new_mi.csv")