import os
import json
import glob
import numpy as np
from collections import defaultdict

def load_metrics(path):
    with open(path, 'r') as f:
        return json.load(f)



def load_accuracy_from_dir(path, algo):
    # Logic copied/adapted from run_dro.sh python block
    import json
    
    if "dro1" in algo:
        log_path = os.path.join(path, "train.log")
        if os.path.exists(log_path):
             with open(log_path, "r") as f:
                for line in f:
                    if "Best Test Accuracy:" in line:
                         try:
                            # "Best Test Accuracy: 55.060%"
                            val = float(line.split("Best Test Accuracy:")[-1].split("%")[0].strip())
                            return val / 100.0
                         except:
                            pass
    elif "dro2" in algo:
        json_path = os.path.join(path, "summary.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                d = json.load(f)
                return d.get("test_accuracy", 0.0)
    elif "Baseline" in algo:
        json_path = os.path.join(path, "results.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                d = json.load(f)
                return d.get("best_accuracy", 0.0)
    
    return 0.0

def main():
    root_dir = "MIA/outputs"
    # Structure: algo -> eps -> list of (metrics_dict, dir_path)
    aggregated = defaultdict(lambda: defaultdict(list))
    
    # We also need to map from MIA output dir back to multi_results dir to find accuracy logs?
    # Actually, MIA outputs are in MIA/outputs, but training logs are in multi_results.
    # The MIA output folder name is "dro1_eps0p1_run1", which corresponds to "multi_results/dro1_new_eps0p1_run1".
    
    # Let's iterate over multi_results directly to get accuracy, and MIA outputs for MIA.
    
    results = defaultdict(lambda: defaultdict(lambda: {"acc": [], "mia": []}))

    # 1. Parse multi_results for Accuracy
    multi_root = "multi_results"
    for d in glob.glob(os.path.join(multi_root, "*_eps*_run*")):
        folder_name = os.path.basename(d)
        parts = folder_name.split('_')
        
        algo = None
        eps = None
        
        # dro1_new_eps0p1_run1
        if parts[0] == "dro1":
             algo = "dro1"
             eps_str = parts[2]
        elif parts[0] == "dro2":
             algo = "dro2"
             eps_str = parts[2]
        elif parts[0] == "Baseline":
             algo = f"{parts[0]}_{parts[1]}"
             eps_str = parts[2]
        
        if algo and eps_str:
             eps = eps_str.replace('eps', '').replace('p', '.')
             acc = load_accuracy_from_dir(d, algo)
             if acc > 0:
                # Normalization check: ensure we store valid 0-1 float or consistent value
                # load_accuracy_from_dir returns:
                # - dro1: val/100.0 (e.g. 0.55)
                # - dro2: raw value from json (e.g. 58.57) which is percentage
                
                # Consolidate to 0-1 range
                if algo.startswith("dro2") and acc > 1.0:
                    acc = acc / 100.0
                    
                results[algo][eps]["acc"].append(acc)

    # 2. Parse MIA outputs for Privacy
    mia_root = "MIA/outputs"
    for d in glob.glob(os.path.join(mia_root, "*_eps*_run*")):
         folder_name = os.path.basename(d)
         parts = folder_name.split('_')
         
         algo = None
         eps = None
         
         # MIA folders: dro1_eps0p1_run1 (Note: no '_new' usually)
         # Script uses: dro1_eps${eps}_run${run}
         
         if parts[0] == "dro1":
              algo = "dro1"
              eps_str = parts[1]
         elif parts[0] == "dro2":
              algo = "dro2"
              eps_str = parts[1]
         elif parts[0] == "Baseline":
              algo = f"{parts[0]}_{parts[1]}"
              eps_str = parts[2]
              
         if algo and eps_str:
              eps = eps_str.replace('eps', '').replace('p', '.')
              metrics_path = os.path.join(d, "attack_metrics.json")
              if os.path.exists(metrics_path):
                   data = load_metrics(metrics_path)
                   results[algo][eps]["mia"].append(data)
                   
    # Print Table
    print(f"{'Algorithm':<15} | {'Epsilon':<8} | {'Test Acc (Mean±Std)':<25} | {'MIA AUC (Mean±Std)':<25}")
    print("-" * 80)
    
    for algo in sorted(results.keys()):
        for eps in sorted(results[algo].keys(), key=lambda x: float(x)):
             data = results[algo][eps]
             
             accs = data["acc"]
             acc_str = "N/A"
             if accs:
                 acc_str = f"{np.mean(accs)*100:.2f} ± {np.std(accs)*100:.2f}%"
                 
             mia_runs = data["mia"]
             auc_str = "N/A"
             if mia_runs:
                 aucs = [r['loss']['auc'] for r in mia_runs]
                 auc_str = f"{np.mean(aucs):.4f} ± {np.std(aucs):.4f}"
            
             print(f"{algo:<15} | {eps:<8} | {acc_str:<25} | {auc_str:<25}")

if __name__ == "__main__":
    main()
