from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import numpy as np
import torch

# Function to compute AUROC, AP, and FPR95
def evaluate_metrics(scores, gts):
    auroc = roc_auc_score(gts, scores)
    ap = average_precision_score(gts, scores)
    fpr, tpr, _ = roc_curve(gts, scores)
    fpr95 = fpr[np.argmin(np.abs(tpr - 0.95))]
    return {"AUROC": auroc, "AP": ap, "FPR95": fpr95}

# Function computes mean IoU
def calculate_mIoU(model, val_loader, n_classes=7):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    hist = np.zeros((n_classes, n_classes))
    
    with torch.no_grad():
        for imgs, labels in val_loader:
            # Extract predictions from the model
            preds = model(imgs.to(device))["probs"].argmax(dim=1).cpu().numpy()
            labels = labels.numpy()
            for p, l in zip(preds, labels):
                # Ignore pixels with label 255
                mask = (l != 255)
                # Update histogram for IoU calculation
                hist += np.bincount(n_classes * l[mask].astype(int) + p[mask].astype(int),
                                    minlength=n_classes ** 2).reshape(n_classes, n_classes)
    # Calculate IoU for each class
    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-6)
    # Return mean IoU, ignoring NaNs
    return np.nanmean(iou)