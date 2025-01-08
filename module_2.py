import torch
import numpy as np
import ot
from scipy.spatial.distance import pdist, squareform
from heapq import nlargest, nsmallest
import itertools
import random
import json

# Load pre-trained protein language model and alphabet from Facebook's ESM repository
model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")

# Get batch converter to convert sequences into tokenized format
batch_converter = alphabet.get_batch_converter()

# Prepare input data with masked mutation sites
data = [("GB1", "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNG<mask><mask><mask>EWTYDDATKTFT<mask>TE")]

# Convert input data into tokenized format
batch_labels, batch_strs, batch_tokens = batch_converter(data)

# Create a mapping of natural amino acids to their corresponding indices in the model
natural_amino_acids = {k: v - 4 for k, v in alphabet.tok_to_idx.items() if 4 <= v <= 23}

# Create a reverse mapping from index to token
idx_to_tok = {v: k for k, v in natural_amino_acids.items()}

# Create a mapping from token to index
tok_to_idx = natural_amino_acids

# Extract embeddings for amino acids from the model's token embeddings
AA_embed = model.embed_tokens.weight[4:24]

# Get model representations for the input data with no gradient calculation
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[33], return_contacts=True)

# Extract token representations from the output
# Layer 33 is used to get the final representation of tokens
token_representations = results["representations"][33]

# Compute pairwise Euclidean distances between amino acid embeddings
Dist = squareform(pdist(AA_embed.detach().numpy(), metric='euclidean'))

# Define a class to create subsets based on model output
class CreateSubset:
    def __init__(self, n, sinka, sinkb, is_top, dist, idx2tok, results):
        self.n = n  # Number of amino acids to select
        self.sinka = sinka  # Regularization parameter for Sinkhorn distance
        self.sinkb = sinkb  # Another regularization parameter for Sinkhorn distance
        self.is_top = is_top  # Whether to select the top or bottom subsets
        self.dist = dist  # Distance matrix for amino acids
        self.idx2tok = idx2tok  # Mapping from index to token
        self.results = results  # Model results for input data
        self.mask_positions = [39, 40, 41, 54]  # Positions of the masked amino acids
    
    # Get probabilities for each mask position
    def get_mask_probs(self):
        return [torch.nn.functional.softmax(self.results["logits"][0][pos])[4:24] for pos in self.mask_positions]
    
    # Find the subsets of amino acids with the highest or lowest scores
    def find_flag(self):
        mask_probs = self.get_mask_probs()  # Get probabilities for each mask position
        subsets = list(itertools.combinations(range(20), self.n))  # Generate all possible subsets of size n
        
        # Compute Sinkhorn distances for each subset of amino acids
        pos_dists = [{subset: float(ot.unbalanced.sinkhorn_unbalanced2(
            torch.tensor(mask_prob[list(subset)]), mask_prob, self.dist[list(subset)], self.sinka, self.sinkb))
            for subset in subsets} for mask_prob in mask_probs]
        
        # Select the subsets with either the highest or lowest Sinkhorn distances
        select_func = nlargest if self.is_top else nsmallest
        selected_ids = [list(select_func(1, pos_dist, key=pos_dist.get)[0]) for pos_dist in pos_dists]
        
        # Convert selected indices to amino acid tokens
        selected_tokens = [[self.idx2tok[id] for id in ids] for ids in selected_ids]
        return [''.join(combo) for combo in itertools.product(*selected_tokens)]

# Create an instance of CreateSubset to find the top 4 combinations
top_4 = CreateSubset(n=4, sinka=1, sinkb=1, is_top=True, dist=Dist, idx2tok=idx_to_tok, results=results)

# Get the top 4 combinations of amino acids
top_4_list = top_4.find_flag()

# Randomly select 96 combinations from the top 4 list
select_96 = random.sample(top_4_list, 96)

# Save the selected combinations to a JSON file
with open('select_96.json', 'w') as file:
    json.dump(select_96, file)
