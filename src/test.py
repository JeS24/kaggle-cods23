import os, json, random
import pandas as pd, numpy as np
import torch
from ast import literal_eval

# Local imports
from models import EdgeConvGNNClassifier


# Models will be stored here
MODELS = "./submissions/models"

# Fix random states - for reproducibility
SEED = 42 # NOTE:
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


@torch.no_grad()
def test(model, data, edge_label_index, exp):
    model.cuda()
    model.eval()
    z = model(data)
    decoder = model.decode_linear if model.use_linear else model.decode
    out = decoder(z, edge_label_index).view(-1).sigmoid()
    out = (out > 0.5).int()

    # Create csv
    print(f"Creating csv for {exp}...")
    # NOTE: Not the final submission.csv - see `gen_sub.py`.
    pd.DataFrame(out.cpu().numpy()).to_csv(f"{MODELS}/{exp}/test_pred.csv", index=False, header=False)


if __name__ == "__main__":
    # Auto exp number specification
    EXP_PATH = f"{MODELS}/"
    EXP = sorted([int(x) for x in os.listdir(EXP_PATH)])[-1]

    # Dataset
    # Load pre_embeddings from csv
    pe = pd.read_csv("../data/kagdata/emb_stdnormed_combined.csv", converters={'pre_embed_vec': literal_eval, 'doc_emb_unproc': literal_eval, 'doc_emb': literal_eval})
    # Converting each list to np.array
    pe['doc_emb'] = pe['doc_emb'].apply(lambda x: np.array(x, dtype=float))
    pe['doc_emb_unproc'] = pe['doc_emb_unproc'].apply(lambda x: np.array(x, dtype=float))
    pe['pre_embed_vec'] = pe['pre_embed_vec'].apply(lambda x: np.array(x))
    # Stacking all arrays to get a matrix
    pre_embeddings = np.c_[np.vstack(pe['pre_embed_vec'].values), np.vstack(pe['doc_emb'].values), np.vstack(pe['doc_emb_unproc'].values)]

    train_idx = 2797 # NOTE: Hardcoded

    # Load [edge_indices, labels] from disk
    labels = np.load("../data/kagdata/edge_indices_labels_combined.npy")[train_idx:] # NOTE: (614, 3)

    X = pre_embeddings
    X = torch.from_numpy(X).float().to('cuda')

    # Find the model with best val_f1 in the last exp
    # Getting all .pt files in the last exp
    models = sorted([x for x in os.listdir(f"{MODELS}/{EXP}") if x.endswith(".pt")])
    model_path = models[-1]
    print(f"Best model found for {EXP} at: {model_path}")

    # Loading from disk
    with open(f"{MODELS}/{EXP}/{EXP}_model_props.json", "r") as f:
        model_props = json.load(f)

    model = EdgeConvGNNClassifier(**model_props)
    
    model.load_state_dict(torch.load(f"{MODELS}/{EXP}/{model_path}"))
    model.eval()

    # For individual runs
    test(model, X, labels[:, :2].T, EXP)
