import os
import json
import glob
import numpy as np
from collections import defaultdict

def load_metrics(path):
    with open(path, 'r') as f:
        return json.load(f)

def main():
    root_dir = "MIA/outputs"
    # Structure: algo -> eps -> list of metrics
    aggregated = defaultdict(lambda: defaultdict(list))

    # Pattern: algo_epsX_runY
    # algo can be "dro1" or "dro2"
    
    subdirs = glob.glob(os.path.join(root_dir, "*_eps*_run*"))
    
    for d in subdirs:
        folder_name = os.path.basename(d)
        parts = folder_name.split('_')
        
        # Handle "dro1_eps5_run1" -> algo=dro1, eps=5
        # Handle "Baseline_SGDA_eps5_run1" -> algo=Baseline_SGDA, eps=5
        
        algo = None
        eps = None
        
        if len(parts) == 3:
            # dro1_eps5_run1
            algo = parts[0]
            eps_str = parts[1]
        elif len(parts) == 4:
             # Baseline_SGDA_eps5_run1
             algo = f"{parts[0]}_{parts[1]}"
             eps_str = parts[2]
        
        if algo and eps_str:
            eps = eps_str.replace('eps', '')
            
            metrics_path = os.path.join(d, "attack_metrics.json")
            if os.path.exists(metrics_path):
                data = load_metrics(metrics_path)
                aggregated[algo][eps].append(data)

    print(f"{'Algorithm':<10} | {'Epsilon':<8} | {'Metric':<15} | {'Mean':<8} | {'Std':<8}")
    print("-" * 70)

    for algo in sorted(aggregated.keys()):
        for eps in sorted(aggregated[algo].keys(), key=lambda x: float(x)):
            runs = aggregated[algo][eps]
            count = len(runs)
            
            # Extract metrics for Confidence and Loss attacks
            # Confidence
            conf_aucs = [r['confidence']['auc'] for r in runs]
            conf_tpr1 = [r['confidence']['tpr_at_1%_fpr'] for r in runs]
            conf_tpr01 = [r['confidence']['tpr_at_0.1%_fpr'] for r in runs]

            # Loss
            loss_aucs = [r['loss']['auc'] for r in runs]
            loss_tpr1 = [r['loss']['tpr_at_1%_fpr'] for r in runs]
            loss_tpr01 = [r['loss']['tpr_at_0.1%_fpr'] for r in runs]
            
            print(f"{algo:<10} | {eps:<8} | {'CONF_AUC':<15} | {np.mean(conf_aucs):.4f}   | {np.std(conf_aucs):.4f}")
            print(f"{'':<10} | {'':<8} | {'CONF_TPR@1%':<15} | {np.mean(conf_tpr1):.4f}   | {np.std(conf_tpr1):.4f}")
            print(f"{'':<10} | {'':<8} | {'LOSS_AUC':<15} | {np.mean(loss_aucs):.4f}   | {np.std(loss_aucs):.4f}")
            print(f"{'':<10} | {'':<8} | {'LOSS_TPR@1%':<15} | {np.mean(loss_tpr1):.4f}   | {np.std(loss_tpr1):.4f}")
            print("-" * 70)

if __name__ == "__main__":
    main()
