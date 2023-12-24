import torch
from torch import nn
from torch_geometric import nn as pyg_nn


def print_shape(x):
    print(x.shape)


class EdgeConvGNNClassifier(nn.Module):
    def __init__(self, in_feat, hid_feat=1024, out_feat=2, k=8, aggr="max", use_linear=False):
        super().__init__()
        # Pass in pre_embeddings as node features to a Linear layer
        self.linear1 = nn.Sequential( # NOTE: This has a compression effect -> (*, 981) -> usually (*, 1024) (*, 512)
            nn.Linear(in_feat, hid_feat),
            nn.ReLU(),
            nn.Linear(hid_feat, 256),
        )
        self.econv1 = pyg_nn.DynamicEdgeConv( # (7272x512 and 256x512)
            nn.Sequential(
                nn.Linear(256 * 2, 1024), # NOTE: x2 because pair of nodes per edge
                nn.ReLU(),
                nn.Linear(1024, 2048)
            ),
            k=k,
            aggr=aggr # NOTE: `max` performs best (set for all DEConvs)
        )
        self.econv2 = pyg_nn.DynamicEdgeConv(
            nn.Sequential(
                nn.Linear(2048 * 2, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512)
            ),
            k=k,
            aggr=aggr
        )
        # self.econv3 = pyg_nn.DynamicEdgeConv(
        #     nn.Sequential(
        #         nn.Linear(512 * 2, 1024),
        #         nn.ReLU(),
        #         nn.Linear(1024, 512)
        #     ),
        #     k=k,
        #     aggr=aggr
        # )

        self.use_linear = use_linear
        if use_linear:
            self.linear2_decode = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, out_feat)
            )

    def forward(self, x): # NOTE: AKA encode()
        # NOTE: x.shape = [n_nodes, in_feat]
        # print_shape(x) # (*, 981)
        # Apply Linear layer to pre_embeddings
        x = self.linear1(x)
        # print_shape(x) # (*, 256)
        # Apply EdgeConv layers to get edge embeddings
        x = self.econv1(x)
        # print_shape(x) # (*, 2048)
        x = self.econv2(x)
        # x = self.econv3(x)
        # print_shape(x) # (*, 512)
        # Apply Linear layer to get logits
        # x = self.linear2(x)
        # x = torch.softmax(x, dim=1) # Not with BCEWithLogitsLoss
        # print_shape(x) # (*, 2)

        return x

    def decode(self, z, edge_label_index): # z - (*, 512)
        res = (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)
        return res

    def decode_all(self, z):
        prob_adj = z @ z.t()

        return (prob_adj > 0).nonzero(as_tuple=False).t()

    def decode_linear(self, z, edge_label_index):
        """
        Use a MLP to decode the edge labels

        """
        out1 = self.linear2_decode(z[edge_label_index[0]])
        out2 = self.linear2_decode(z[edge_label_index[1]])
        
        # print("dlinear: ", (out1 * out2).sum(dim=-1).shape)

        return (out1 * out2).sum(dim=-1)
