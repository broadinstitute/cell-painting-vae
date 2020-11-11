import subprocess
dataset = "L1000"
cmd = [
    "python",
    "../../scripts/optimize.py",
    "--dataset",
    dataset,]
subprocess.call(cmd)