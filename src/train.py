import os, json
import pandas as pd, numpy as np, random
from sklearn.metrics import accuracy_score, f1_score

# import wandb
import torch
import torch.nn as nn
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

def train(model, data, edge_label_index, edge_label, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    z = model(data)
    decoder = model.decode_linear if model.use_linear else model.decode
    out = decoder(z, edge_label_index).view(-1)
    loss = criterion(out, edge_label)
    loss.backward()
    optimizer.step()

    return loss


@torch.no_grad()
def val(model, data, edge_label_index, edge_label):
    model.eval()
    z = model(data)
    decoder = model.decode_linear if model.use_linear else model.decode
    out = decoder(z, edge_label_index).view(-1).sigmoid()
    out = (out > 0.5).int()

    return f1_score(edge_label.cpu().numpy(), out.cpu().numpy(), average="weighted") # NOTE: Weighted F1 score


def trainer(
    model,
    optimizer,
    criterion,
    data,
    labels,
    data_split_idx,
    num_epochs,
    lr,
    exp,
    model_props,
    comments,
    seed
):
    # wandb.init(
    #     entity="...",
    #     project="...",

    #     config={
    #         "exp-num": exp,
    #         "architecture": model.__class__.__name__,
    #         "learning_rate": lr,
    #         "epochs": num_epochs,
    #         "model_props": model_props,
    #         "data_split_idx": data_split_idx,
    #         "comments": comments,
    #         "seed": seed,
    #     }
    # )

    train_eli = labels[:data_split_idx, :2]
    val_eli = labels[data_split_idx:, :2]
    train_lbl = labels[:data_split_idx, 2]
    val_lbl = labels[data_split_idx:, 2]

    # Save train_eli, val_eli, train_lbl, val_lbl to disk
    np.save(f"{MODELS}/{exp}/train_eli.npy", train_eli)
    np.save(f"{MODELS}/{exp}/val_eli.npy", val_eli)
    np.save(f"{MODELS}/{exp}/train_lbl.npy", train_lbl)
    np.save(f"{MODELS}/{exp}/val_lbl.npy", val_lbl)

    train_eli = torch.Tensor(train_eli).cuda().long().t()
    val_eli = torch.Tensor(val_eli).cuda().long().t()
    train_lbl = torch.Tensor(train_lbl).cuda()
    val_lbl = torch.Tensor(val_lbl).cuda()

    model = model.to('cuda')
    best_val_f1 = 0
    best_val_epoch = 0
    for epoch in range(num_epochs):
        loss = train(
            model,
            data,
            train_eli,
            train_lbl,
            optimizer,
            criterion
        )
        val_f1 = val(model, data, val_eli, val_lbl)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_epoch = epoch

            # Save model if val_f1 is better than previous best
            torch.save(model.state_dict(), f"{MODELS}/{exp}/{best_val_f1:.3}-F1_ep{epoch}_{model.__class__.__name__}.pt")

        # wandb.log({"BCEloss": loss.item(), "val_f1": val_f1, "best_val_f1": best_val_f1, "best_val_epoch": best_val_epoch,})
        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {loss.item():.6f} | Val_F1: {val_f1:.6f} | Best Val F1: {best_val_f1:.6f}")

    print(f"Best Val F1: {best_val_f1:.6f} | Best Val F1 Epoch: {best_val_epoch}")
    # wandb.finish()


if __name__ == "__main__":
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
    labels = np.load("../data/kagdata/edge_indices_labels_combined.npy")[:train_idx] # NOTE: (3411, 3)

    # Randomly shuffle labels at the start of training
    np.random.shuffle(labels) # NOTE: effect on F1 - starts with lower F1 - final F1 is similar.

    # Some parameters
    num_epochs = 600
    lr = 0.0001
    data_split_idx = 2597
    # NOTE: First 2597 edges for training, 200 for validation | 1113 positive labels, 1684 negative labels
    # NOTE: Rest 614 for testing.

    model_props = {
        "in_feat": pre_embeddings.shape[1], # NOTE: 768 * 2 (from LM) + 981 (given) = 2517 input features
        "hid_feat": 1024,
        "out_feat": 2, # n_classes
        "k": 16, # Empirically based on the maxmimum node degree in the training data
        "aggr": "max",
        "use_linear": False
    }
    model = EdgeConvGNNClassifier(**model_props)
    criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X = pre_embeddings
    X = torch.from_numpy(X).float().to('cuda')

    # Automatic exp number specification
    EXP_PATH = f"{MODELS}/"
    curr_max_exp_num = sorted([int(x) for x in os.listdir(EXP_PATH)])[-1]
    EXP = curr_max_exp_num + 1 # Experiment number

    os.makedirs(f"{MODELS}/{EXP}", exist_ok=True)
    # Save model_props to disk
    with open(f"{MODELS}/{EXP}/{EXP}_model_props.json", "w") as f:
        json.dump(model_props, f, indent=4)

    # For individual runs
    trainer(
        model,
        optimizer,
        criterion,
        data=X,
        labels=labels,
        data_split_idx=data_split_idx,
        num_epochs=num_epochs,
        lr=lr,
        exp=EXP,
        model_props=model_props,
        comments="...",
        seed=SEED,
    )
