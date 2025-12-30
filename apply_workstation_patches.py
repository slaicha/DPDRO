
import os
import re

def patch_sgda():
    path = "Baseline/SGDA/main.py"
    if not os.path.exists(path):
        print(f"Skipping {path} (not found)")
        return
    
    with open(path, "r") as f:
        content = f.read()

    # Add save-model argument
    if "--save-model" not in content:
        content = content.replace(
            'parser.add_argument("--output_dir", default=None, type=str, help="Directory to store result summary")',
            'parser.add_argument("--output_dir", default=None, type=str, help="Directory to store result summary")\n    parser.add_argument("--save-model", action="store_true", help="Save the trained model checkpoint")'
        )

    # Add saving logic
    if "checkpoint.pt" not in content:
        # Saving likely happens after json.dump
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

    # Add import os
    if "import os" not in content:
        content = "import os\n" + content

    # Rename linear to fc for consistency
    if "self.linear" in content:
        content = content.replace("self.linear", "self.fc")
        print(f"Patched {path} (renamed linear to fc)")

    # Add arguments
    if "--save-model" not in content:
        content = content.replace(
            'parser.add_argument("--run-dp", action="store_true")',
            'parser.add_argument("--run-dp", action="store_true")\n    parser.add_argument("--output-dir", type=str, default=".")\n    parser.add_argument("--save-model", action="store_true")'
        )

    # Add saving logic at the end
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
        # Append to end of file, assuming structure ends with 'test_standard(...)' or similar
        # Safest to just append inside the if __name__ block
        # We can look for the last print statement
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

    # Fix formatting bug
    # Look for broken format string if it exists, or just ensure correct one
    # Note: I am writing the CORRECT one. The user's original repo has the bug.
    # Buggy version usually had "max{:.2f, {:.2f}}"
    
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

    # 1. Add --save-model to SGDA
    if "baseline_sgda" in content and "save-model" not in content.split("run_baseline_sgda")[1].split("}")[0]:
        content = content.replace(
            '--total_epochs 30 \\',
            '--total_epochs 30 \\\n        --save-model \\'
        )

    # 2. Add --save-model to Diff
    # Note: Diff replacement might overlap with SGDA if patterns are identical. 
    # Use context.
    # We can rely on just replacing the text globally if unique enough, or careful regex.
    # Actually, simplistic replace of the function body is hard.
    # Let's assume standard structure.
    
    # run_baseline_diff replacement handled by generic replace above? No, arguments differ slightly.
    # SGDA: --lr_w
    # Diff: --lr 0.2
    if "baseline_diff" in content and "save-model" not in content.split("run_baseline_diff")[1].split("}")[0]:
         content = content.replace(
            '--lr 0.2 \\',
            '--save-model \\\n        --lr 0.2 \\'
        )

    # 3. Add --save-model and --run-dp to dro1
    if "dro1_new" in content:
        # Inject --run-dp if missing
        if "--run-dp" not in content.split("run_dro1_new")[1].split("}")[0]:
             content = content.replace(
                '--epsilon "${eps}" \\',
                '--epsilon "${eps}" \\\n        --run-dp \\\n        --save-model \\'
            )
        # Inject save-model if strictly missing (handled above combined)

    # 4. Add --save-model to dro2
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
