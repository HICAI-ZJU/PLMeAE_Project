import torch
import numpy as np


# Helper function to split data into batches of a specified size
def batch_data(data, batch_size):
    # Iterate over the data in chunks of size 'batch_size'
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


# Function to calculate the likelihood of sequences using a pre-trained model
def calculate_likelihood(sequences, batch_size=8):
    all_likelihoods = []
    # Process sequences in batches
    for batch in batch_data(sequences, batch_size):
        # Prepare batch data as tuples of ("seq", sequence)
        data = [("seq", seq) for seq in batch]
        # Convert sequences to token format using batch_converter
        _, _, batch_tokens = batch_converter(data)
        # Move the tokenized batch to the specified device (GPU or CPU)
        batch_tokens = batch_tokens.to(device=device, non_blocking=True)

        # Disable gradient computation for inference
        with torch.no_grad():
            # Get model predictions, specifically the token representations from layer 33
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        # Extract representations from layer 33
        token_representations = results["representations"][33]

        # Calculate the mean of the token representations for each sequence
        likelihoods = token_representations.mean(dim=1).mean(dim=1).cpu().numpy()
        all_likelihoods.extend(likelihoods)

    return all_likelihoods


# Function to select the top 'n' sequences based on calculated likelihoods
def select_top_sequences(sequences, top_n=96):
    # Compute likelihoods for the input sequences
    likelihoods = calculate_likelihood(sequences)
    # Sort indices of sequences based on likelihood in descending order
    sorted_indices = np.argsort(likelihoods)[::-1]
    # Select the top 'n' sequences based on sorted likelihoods
    top_sequences = [sequences[i] for i in sorted_indices[:top_n]]
    return top_sequences


# Function to generate a mutation library by creating all possible single mutations of a sequence
def generate_mutation_library(sequence, amino_acids):
    mutation_library = []
    # Loop through each position in the sequence
    for i in range(len(sequence)):
        # Replace the amino acid at position 'i' with every other amino acid in the set
        for aa in amino_acids:
            if aa != sequence[i]:
                # Create a new sequence with a single mutation
                mutated_sequence = sequence[:i] + aa + sequence[i + 1:]
                mutation_library.append(mutated_sequence)
    return mutation_library


# Function to identify single mutations by comparing an original sequence to a set of top sequences
def find_single_mutations(original_sequence, top_sequences):
    single_mutations = []
    # Compare each top sequence with the original sequence
    for seq in top_sequences:
        if len(seq) != len(original_sequence):
            continue

        # Find positions where the amino acids differ between the sequences
        differences = [(i, original_sequence[i], seq[i]) for i in range(len(seq)) if original_sequence[i] != seq[i]]

        # Only record single mutations (sequences with exactly one difference)
        if len(differences) == 1:
            single_mutations.append(differences[0])

    return single_mutations


# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the pre-trained ESM model and its alphabet for sequence processing
model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
model = model.eval().to(device)  # Set the model to evaluation mode and move it to the correct device

# Obtain the batch converter from the alphabet for tokenizing sequences
batch_converter = alphabet.get_batch_converter()

# Define the standard set of 20 amino acids
amino_acids = "ACDEFGHIKLMNPQRSTVWY"

# Define the original protein sequence to work with
original_sequence = "MDEFEMIKRNTSEIISEEELREVLKKDEKSALIGFEPSGKIHLGHYLQIKKMIDLQNAGFDIIIVLADLHAYLNQKGELDEIRKIGDYNKKVFEAMGLKAKYVYGSEWMLDKDYTLNVYRLALKTTLKRARRSMELIAREDENPKVAEVIYPIMQVNGAHYLGVDVAVGGMEQRKIHMLARELLPKKVVCIHNPVLTGLDGEGKMSSSKGNFIAVDDSPEEIRAKIKKAYCPAGVVEGNPIMEIAKYFLEYPLTIKRPEKFGGDLTVNSYEELESLFKNKELGCMKLKNAVAEELIKILEPIRKRL"

# Generate a mutation library by making single mutations at each position in the original sequence
mutation_library = generate_mutation_library(original_sequence, amino_acids)
print("Total number of sequences in the mutation library:", len(mutation_library))

# Select the top sequences from the mutation library based on their likelihood scores
top_sequences = select_top_sequences(mutation_library)

# Identify the single mutation differences between the original sequence and the top sequences
single_mutations = find_single_mutations(original_sequence, top_sequences)
# Print out the differences found in the top sequences
for position, target_aa, seq_aa in single_mutations:
    print(f"Sequence difference found at position {position}: target '{target_aa}' vs sequence '{seq_aa}'")
