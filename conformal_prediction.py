import numpy as np
import torch

def calibrate_conformal(uos_scores_calib, alpha=0.05):
    """
    Calibrate the conformal prediction threshold based on the UOS scores from the calibration set.
    :param uos_scores_calib: Array of UOS scores from the calibration set.
    :param alpha: Significance level for the conformal prediction.  
    :return: The calibrated threshold qhat.
    """
    n = len(uos_scores_calib)
    qhat = np.quantile(uos_scores_calib, np.ceil((n + 1) * (1 - alpha)) / n, method='higher')
    return qhat

def conformal_mask(uos_map, qhat):
    return (uos_map >= qhat)

def extract_uos_scores_for_calibration(model, loader):
    model.eval()
    scores = []
    with torch.no_grad():
        for imgs, _ in loader:
            out = model(imgs.cuda())["uos"]
            scores.append(out.flatten().cpu().numpy())
    return np.concatenate(scores)
