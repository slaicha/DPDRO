
import os
import re

def patch_sgda():
    path = "Baseline/SGDA/main.py"
    if not os.path.exists(path):
        print(f"Skipping {path} (not found)")
        return
    
    with open(path, "r") as f:
        content = f.read()

    if "--save-model" not in content:
        content = content.replace(
            'parser.add_argument("--output_dir", default=None, type=str, help="Directory to store result summary")',
            'parser.add_argument("--output_dir", default=None, type=str, help="Directory to store result summary")\n    parser.add_argument("--save-model", action="store_true", help="Save the trained model checkpoint")'
        )

    if "checkpoint.pt" not in content:
        pattern = r'(}, fh, indent=2\)\s+)' # json dump block end
        replacement = r'\1\n    if args.save_model and args.output_dir:\n        ckpt_path = os.path.join(args.output_dir, "checkpoint.pt")\n        torch.save(setup["model"].state_dict(), ckpt_path)\n        print(f"Checkpoint saved to {ckpt_path}")\n\n'
        content = re.sub(pattern, replacement, content)

    with open(path, "w") as f:
        f.write(content)
    print(f"Patched {path}")

def patch_diff():
    path = "Baseline/Diff/main.py"
    if not os.path.exists(path):
        print(f"Skipping {path} (not found)")
        return
    with open(path, "r") as f:
        content = f.read()

    if "--save-model" not in content:
        content = content.replace(
            'parser.add_argument("--output_dir", default=None, type=str, help="Directory to store result summary")',
            'parser.add_argument("--output_dir", default=None, type=str, help="Directory to store result summary")\nparser.add_argument("--save-model", action="store_true", help="Save the trained model checkpoint")'
        )

    if "checkpoint.pt" not in content:
        pattern = r'(}, fh, indent=2\)\s+)'
        replacement = r'\1\n    if args.save_model and args.output_dir:\n        ckpt_path = os.path.join(args.output_dir, "checkpoint.pt")\n        torch.save(model.state_dict(), ckpt_path)\n        print(f"Checkpoint saved to {ckpt_path}")\n\n'
        content = re.sub(pattern, replacement, content)

    with open(path, "w") as f:
        f.write(content)
    print(f"Patched {path}")

def patch_dro1():
    path = "dro1_new/algorithm.py"
    if not os.path.exists(path):
        print(f"Skipping {path} (not found)")
        return
    with open(path, "r") as f:
        content = f.read()

    if "import os" not in content:
        content = "import os\n" + content

    if "self.linear" in content:
        content = content.replace("self.linear", "self.fc")
        print(f"Patched {path} (renamed linear to fc)")

    if "--save-model" not in content:
        content = content.replace(
            'parser.add_argument("--run-dp", action="store_true")',
            'parser.add_argument("--run-dp", action="store_true")\n    parser.add_argument("--output-dir", type=str, default=".")\n    parser.add_argument("--save-model", action="store_true")'
        )

    if "checkpoint.pt" not in content:
        logic = '''
    if args.save_model:
        checkpoint_path = os.path.join(args.output_dir, "checkpoint.pt")
        os.makedirs(args.output_dir, exist_ok=True)
        final_model = model
        if args.run_dp and 'dp_model' in locals():
            final_model = dp_model
        torch.save(final_model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
'''

        pattern = r'(print\("\\nDP training not run.*"\))'
        replacement = r'\1\n' + logic
        content = re.sub(pattern, replacement, content)

    with open(path, "w") as f:
        f.write(content)
    print(f"Patched {path}")

def patch_dro2():
    path = "dro2_new/train_rsdro.py"
    if not os.path.exists(path):
        print(f"Skipping {path} (not found)")
        return
    with open(path, "r") as f:
        content = f.read()
    
    broken_str = 'n must be >= max{:.2f, {:.2f}}'
    fixed_str = 'n must be >= max({:.2f}, {:.2f})'
    
    if broken_str in content:
        content = content.replace(broken_str, fixed_str)
        print(f"Patched {path} (Fixed format string)")

    with open(path, "w") as f:
        f.write(content)

def patch_runner():
    path = "run_all_projects.sh"
    if not os.path.exists(path):
        print(f"Skipping {path} (not found)")
        return
    with open(path, "r") as f:
        content = f.read()

    if "baseline_sgda" in content and "save-model" not in content.split("run_baseline_sgda")[1].split("}")[0]:
        content = content.replace(
            '--total_epochs 30 \\',
            '--total_epochs 30 \\\n        --save-model \\'
        )

    if "baseline_diff" in content and "save-model" not in content.split("run_baseline_diff")[1].split("}")[0]:
         content = content.replace(
            '--lr 0.2 \\',
            '--save-model \\\n        --lr 0.2 \\'
        )

    if "dro1_new" in content:
        if "--run-dp" not in content.split("run_dro1_new")[1].split("}")[0]:
             content = content.replace(
                '--epsilon "${eps}" \\',
                '--epsilon "${eps}" \\\n        --run-dp \\\n        --save-model \\'
            )

    if "dro2_new" in content and "--save-model" not in content.split("run_dro2_new")[1].split("}")[0]:
        content = content.replace(
            'bash new.sh \\',
            'bash new.sh \\\n        --save-model \\'
        )

    with open(path, "w") as f:
        f.write(content)
    print(f"Patched {path}")

if __name__ == "__main__":
    patch_sgda()
    patch_diff()
    patch_dro1()
    patch_dro2()
    patch_runner()
    print("All patches applied!")
