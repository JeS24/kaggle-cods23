import numpy as np, pandas as pd, argparse

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--exp", type=int, required=True, help="Experiment number") # NOTE: 83 for the submission
args = parser.parse_args()

MODELS = "./submissions/models"
EXP = args.exp
src = f"{MODELS}/{EXP}"

# Generate submission.csv
ids = np.arange(1, 615)
raw_sub = pd.read_csv(src + "/test_pred.csv", header=None)
sub = pd.concat([pd.Series(ids), raw_sub], axis=1)
sub.columns = ["ID", "label"]

# Save submission.csv to disk
sub.to_csv(f"./submissions/{EXP}_GNN_submission.csv", index=False) # NOTE: Uncomment to save | Careful with overwriting
