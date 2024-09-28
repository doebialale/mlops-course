# Your Code Goes here:

# Step 1: Install
# pip install wandb

# Step 2: Setup the WandB library
# Importing the WandB library
import wandb

# Step 3: Initialize a new W&B run
wandb.init(project="first-project-mlops-course")

# Step 4: Log a sample metric
wandb.log({"accuracy": 0.793232})

# Step 5: Finish the run
wandb.finish()

