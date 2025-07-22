import numpy as np
import torch

def calibrate_conformal(uos_scores_calib, alpha=0.05):
    n = len(uos_scores_calib)
    # Calculate the quantile for the UOS scores
    qhat = np.quantile(uos_scores_calib, np.ceil((n + 1) * (1 - alpha)) / n, method='higher')
    return qhat

def conformal_mask(uos_map, qhat):
    # Return a binary mask where UOS scores are greater than or equal to qhat
    return (uos_map >= qhat)