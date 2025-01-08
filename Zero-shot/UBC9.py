import torch
import numpy as np
import pandas as pd
import random


# Helper function to split data into smaller batches of a specified size
def batch_data(data, batch_size):
    # Iterates over the data in chunks of batch_size and yields each chunk
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


# Function to calculate the likelihood of each sequence in a list using a pre-trained model
def calculate_likelihood(sequences, batch_size=8):
    all_likelihoods = []  # List to store likelihoods of all sequences
    # Process sequences in batches to handle memory efficiently
    for batch in batch_data(sequences, batch_size):
        # Prepare each sequence in batch for model input
        data = [("seq", seq) for seq in batch]
        _, _, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device=device, non_blocking=True)

        # Calculate representations using the model in evaluation mode (no gradient calculation)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]

        # Calculate likelihood by averaging across tokens and sequences
        likelihoods = token_representations.mean(dim=1).mean(dim=1).cpu().numpy()
        all_likelihoods.extend(likelihoods)  # Add to likelihood list

    return all_likelihoods  # Return likelihoods of all sequences


# Function to select the top 'n' sequences based on likelihood scores
def select_top_sequences(sequences, top_n=96):
    likelihoods = calculate_likelihood(sequences)  # Calculate likelihood for each sequence
    sorted_indices = np.argsort(likelihoods)[::-1]  # Sort indices by descending likelihood
    # Select the top 'n' sequences based on highest likelihood
    top_sequences = [sequences[i] for i in sorted_indices[:top_n]]
    return top_sequences


# Function to generate all possible single amino acid mutations for a given sequence
def generate_mutation_library(sequence, amino_acids):
    mutation_library = []
    # Loop through each position in the sequence and replace with each amino acid
    for i in range(len(sequence)):
        for aa in amino_acids:
            if aa != sequence[i]:  # Skip if it's the same amino acid
                mutated_sequence = sequence[:i] + aa + sequence[i + 1:]
                mutation_library.append(mutated_sequence)  # Add mutated sequence to the library
    return mutation_library


# Function to generate a specified number of random mutations for a given sequence
def generate_random_mutations(sequence, amino_acids, num_mutations=96):
    mutation_library = []
    sequence_length = len(sequence)

    # Generate random mutations until num_mutations is reached
    for _ in range(num_mutations):
        mutation_pos = random.randint(0, sequence_length - 1)  # Random mutation position
        original_aa = sequence[mutation_pos]  # Original amino acid
        # Randomly select a different amino acid for mutation
        new_aa = random.choice([aa for aa in amino_acids if aa != original_aa])
        mutated_sequence = sequence[:mutation_pos] + new_aa + sequence[mutation_pos + 1:]
        mutation_library.append(mutated_sequence)  # Add the mutated sequence

    return mutation_library


# Function to identify the mutation position and amino acid change in a mutated sequence
def find_mutation_details(original_sequence, mutated_sequence):
    # Compare each position in original and mutated sequence to find the mutation
    for i in range(len(original_sequence)):
        if original_sequence[i] != mutated_sequence[i]:
            return i + 1, mutated_sequence[i]  # Return 1-indexed position and new amino acid
    return None, None  # If no mutation, return None (unlikely here)


# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load pre-trained ESM model and alphabet for sequence tokenization
model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
model = model.eval().to(device)  # Set model to eval mode and move to appropriate device

# Obtain batch converter from the alphabet for tokenizing sequences
batch_converter = alphabet.get_batch_converter()

# Define the standard set of 20 amino acids
amino_acids = "ACDEFGHIKLMNPQRSTVWY"

# Load the dataset, assuming it has 'sequence' in first column and 'label' in second column
dataset = pd.read_csv("./dataset/UBC9/normalized_dataset.csv")

# Extract sequences and labels
sequences = dataset['sequence'].tolist()
labels = dataset['label'].tolist()

# Create a dictionary to quickly look up labels by sequence
label_dict = dict(zip(sequences, labels))

# Initialize the original sequence
original_sequence = "MSGIALSRLAQERKAWRKDHPFGFVAVPTKNPDGTMNLMNWECAIPGKKGTPWEGGLFKLRMLFKDDYPSSPPKCKFEPPLFHPNVYPSGTVCLSILEEDKDWRPAITIKQILLGIQELLNEPNIQDPAQAEAYTIYCQNRVEYEKRVRAQAKKFAPS"

# Generate a library of all possible single amino acid mutations of the original sequence
mutation_library = generate_mutation_library(original_sequence, amino_acids)

# Select the top 96 sequences based on likelihood
top_sequences = select_top_sequences(mutation_library)

# Find labels for top 96 sequences; if label not available, mark as 'NA'
top_sequence_labels = [(seq, label_dict.get(seq, 'NA')) for seq in top_sequences]

# Calculate mean of available labels, ignoring 'NA'
valid_labels = [label for _, label in top_sequence_labels if label != 'NA']
num_valid_labels = len(valid_labels)
mean_label = sum(valid_labels) / num_valid_labels if num_valid_labels > 0 else 0
max_label = max(valid_labels) if valid_labels else 'NA'

print("Number of labeled sequences in top 96:", num_valid_labels)
print("Mean label of labeled sequences:", mean_label)
print("Maximum fitness value among labeled sequences:", max_label)

# Generate 96 random mutations of the original sequence
random_mutations = generate_random_mutations(original_sequence, amino_acids, num_mutations=96)

# Find labels for random mutations; if label not available, mark as 'NA'
random_sequence_labels = [(seq, label_dict.get(seq, 'NA')) for seq in random_mutations]

# Calculate mean of available labels, ignoring 'NA'
valid_labels = [label for _, label in random_sequence_labels if label != 'NA']
num_valid_labels = len(valid_labels)
mean_label = sum(valid_labels) / num_valid_labels if num_valid_labels > 0 else 0
max_label = max(valid_labels) if valid_labels else 'NA'
print("Number of random sequences in top 96:", num_valid_labels)
print("Mean label of random sequences:", mean_label)
print("Maximum fitness value among labeled sequences:", max_label)

# Save top 96 sequences and their mutation details to a CSV file
top_sequence_info = []
for seq in top_sequences:
    label = label_dict.get(seq, 'NA')  # Get label or 'NA'
    position, aa = find_mutation_details(original_sequence, seq)
    top_sequence_info.append((seq, label, position, aa))  # Append sequence details

# Create DataFrame and save as CSV
output_df = pd.DataFrame(top_sequence_info, columns=['sequence', 'label', 'mutation_position', 'mutated_aa'])
output_df.to_csv("UBC9.csv", index=False)

# Save 96 random mutations and their details to a separate CSV file
random_sequence_info = []
for seq in random_mutations:
    label = label_dict.get(seq, 'NA')  # Get label or 'NA'
    position, aa = find_mutation_details(original_sequence, seq)
    random_sequence_info.append((seq, label, position, aa))  # Append sequence details

# Create DataFrame and save as CSV
output_df_random = pd.DataFrame(random_sequence_info, columns=['sequence', 'label', 'mutation_position', 'mutated_aa'])
output_df_random.to_csv("Random_UBC9.csv", index=False)
