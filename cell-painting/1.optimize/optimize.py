import subprocess

directory = "parameter_sweep"
learning_rate = [1e-2, 1e-3, 1e-4, 1e-5]
min_latent_dim = 5
max_latent_dim = 100
min_beta = 1
max_beta = 5
architectures = ["onelayer", "twolayer", "threelayer"]
batch_norm = [True, False]

# Create commands
learning_rate_arg = ["--learning_rate"] + [str(x) for x in learning_rate]

params = [
    "python",
    "optimize-hyperparameters.py",
    "--overwrite",
    "--directory",
    directory,
    "--min_latent_dim",
    str(min_latent_dim),
    "--max_latent_dim",
    str(max_latent_dim),
    "--min_beta",
    str(min_beta),
    "--max_beta",
    str(max_beta),
] + learning_rate_arg

all_commands = []
for architecture in architectures:
    project_params = params + [
        "--project_name",
        architecture,
        "--architecture",
        architecture,
    ]
    all_commands.append(project_params)

if __name__ == "__main__":
    for cmd in all_commands:
        subprocess.call(cmd)
