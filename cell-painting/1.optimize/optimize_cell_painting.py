import subprocess
dataset = "cell-painting"
cmd = [
    "python",
    "../../scripts/optimize.py",
    "--dataset",
    dataset,]
subprocess.call(cmd)