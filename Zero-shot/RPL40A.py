import torch
import numpy as np
import pandas as pd
import random

# Helper function to split data into batches of a specified size
def batch_data(data, batch_size):
    # Yields data in batches, allowing for efficient processing of large data
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

# Function to calculate the likelihood of sequences using a pre-trained model
def calculate_likelihood(sequences, batch_size=8):
    all_likelihoods = []  # List to store likelihoods of each sequence
    for batch in batch_data(sequences, batch_size):
        # Convert each sequence in the batch to a tuple format expected by batch_converter
        data = [("seq", seq) for seq in batch]
        _, _, batch_tokens = batch_converter(data)
        # Send the tokens to the specified device (GPU or CPU)
        batch_tokens = batch_tokens.to(device=device, non_blocking=True)

        with torch.no_grad():
            # Perform inference to get the representation at layer 33
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]
        # Compute mean of representations across tokens and layers as likelihood score
        likelihoods = token_representations.mean(dim=1).mean(dim=1).cpu().numpy()
        all_likelihoods.extend(likelihoods)

    return all_likelihoods

# Function to select the top 'n' sequences based on calculated likelihoods
def select_top_sequences(sequences, top_n=96):
    # Calculate likelihoods for each sequence
    likelihoods = calculate_likelihood(sequences)
    # Sort indices based on likelihoods in descending order
    sorted_indices = np.argsort(likelihoods)[::-1]
    # Select top sequences based on the sorted indices
    top_sequences = [sequences[i] for i in sorted_indices[:top_n]]
    return top_sequences

# Function to generate mutated sequences based on the dataset
def generate_labeled_mutation_library(original_sequence, positions, mutated_amino_acids):
    mutation_library = []  # Store all generated mutated sequences
    for pos, mut_aa in zip(positions, mutated_amino_acids):
        # Create a mutated sequence if the amino acid differs from the original
        if original_sequence[pos - 1] != mut_aa:
            mutated_sequence = original_sequence[:pos - 1] + mut_aa + original_sequence[pos:]
            mutation_library.append(mutated_sequence)
    return mutation_library

# Function to randomly generate mutations for comparison
def generate_random_mutations(original_sequence, positions, mutated_amino_acids, num_mutations=96):
    random_mutations = []  # Store random mutations
    for _ in range(num_mutations):
        # Randomly select a mutation position and amino acid
        random_index = random.randint(0, len(positions) - 1)
        random_position = positions[random_index]
        random_index_1 = random.randint(0, len(mutated_amino_acids) - 1)
        random_aa = mutated_amino_acids[random_index_1]
        # Create mutated sequence
        mutated_sequence = original_sequence[:random_position - 1] + random_aa + original_sequence[random_position:]
        random_mutations.append(mutated_sequence)
    return random_mutations

# Set the device to GPU if available; otherwise, use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the pre-trained ESM model and alphabet for processing sequences
model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
model = model.eval().to(device)  # Set model to evaluation mode

# Obtain the batch converter from the alphabet for tokenizing sequences
batch_converter = alphabet.get_batch_converter()

# Load the dataset from an Excel file
df = pd.read_excel('./dataset/RPL40A/normalized_cleaned_NIHMS601119_supplement_02.xlsx')

# Filter out rows with missing labels
labeled_df = df.dropna(subset=['Relative E1-reactivity (avg WT=1, avg STOP=0)'])

# Extract mutations with known labels
positions = labeled_df['Position'].values
mutated_amino_acids = labeled_df['Amino Acid'].values
labels = labeled_df['Relative E1-reactivity (avg WT=1, avg STOP=0)'].values

# Define the original sequence for mutation generation
original_sequence = "MQIFVKTLTGKTITLEVESSDTIDNVKSKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGGIIEPSLKALASKYNCDKSVCRKCYARLPPRATNCRKRKCGHTNQLRPKKKLK"

# Generate a mutation library from labeled dataset
mutation_library = generate_labeled_mutation_library(original_sequence, positions, mutated_amino_acids)

# Select the top 96 sequences with the highest likelihoods
top_sequences = select_top_sequences(mutation_library, top_n=96)

# Initialize lists to store mutation data and fitness tracking
top_sequence_labels = []
mutation_positions = []
mutation_amino_acids = []
max_fitness = None

# Iterate through each top sequence to find matching labels
for seq in top_sequences:
    label_found = False
    mutation_info = []  # Store mutation position and amino acid for this sequence
    for i, (pos, mut_aa) in enumerate(zip(positions, mutated_amino_acids)):
        # Check for an actual mutation
        if seq[pos - 1] == mut_aa and original_sequence[pos - 1] != mut_aa:
            top_sequence_labels.append(labels[i])
            mutation_info.append((pos, mut_aa))
            label_found = True
            # Track maximum fitness
            if max_fitness is None or labels[i] > max_fitness:
                max_fitness = labels[i]
            break  # Stop if a matching label is found
    if not label_found:
        top_sequence_labels.append("NA")
    mutation_positions.append([info[0] for info in mutation_info])
    mutation_amino_acids.append([info[1] for info in mutation_info])

# Calculate mean fitness for available labeled mutations
labeled_top_labels = [label for label in top_sequence_labels if label != "NA"]
labeled_count = len(labeled_top_labels)
mean_label = np.mean(labeled_top_labels) if labeled_top_labels else "No labeled data"
print("Number of labeled mutations in top 96:", labeled_count)
print("Mean label value of labeled top mutations:", mean_label)
print("Maximum fitness value:", max_fitness)

# Save top sequences and mutation details to a CSV file
output_df = pd.DataFrame({
    "Sequence": top_sequences,
    "Label": top_sequence_labels,
    "Mutation_Positions": mutation_positions,
    "Mutation_Amino_Acids": mutation_amino_acids
})

output_df.to_csv("RPL40A.csv", index=False)
print("Saved top 96 mutations with labels to 'RPL40A.csv'")

# Generate 96 random mutations and calculate label sum
random_mutations = generate_random_mutations(original_sequence, positions, mutated_amino_acids, num_mutations=96)
random_mutations_labels = []
random_mutation_positions = []
random_mutation_amino_acids = []
random_max_fitness = None

# Match labels for random mutations and track fitness
for seq in random_mutations:
    label_found = False
    random_mutation_info = []  # Store mutation position and amino acid for this sequence
    for i, (pos, mut_aa) in enumerate(zip(positions, mutated_amino_acids)):
        # Check for actual mutation
        if seq[pos - 1] == mut_aa and original_sequence[pos - 1] != mut_aa:
            random_mutations_labels.append(labels[i])
            random_mutation_info.append((pos, mut_aa))
            label_found = True
            if random_max_fitness is None or labels[i] > random_max_fitness:
                random_max_fitness = labels[i]
            break
    if not label_found:
        random_mutations_labels.append("NA")
    random_mutation_positions.append([info[0] for info in random_mutation_info])
    random_mutation_amino_acids.append([info[1] for info in random_mutation_info])

# Calculate and print summary statistics
labeled_top_labels = [label for label in random_mutations_labels if label != "NA"]
labeled_count = len(labeled_top_labels)
mean_label = np.mean(labeled_top_labels) if labeled_top_labels else "No labeled data"
print("Number of labeled mutations in random 96:", labeled_count)
print("Mean label value of random mutations:", mean_label)
print("Maximum fitness value in random mutations:", random_max_fitness)

# Save random mutations and mutation details to a CSV file
output_df = pd.DataFrame({
    "Sequence": random_mutations,
    "Label": random_mutations_labels,
    "Mutation_Positions": random_mutation_positions,
    "Mutation_Amino_Acids": random_mutation_amino_acids
})

output_df.to_csv("Random_RPL40A.csv", index=False)
print("Saved random mutations with labels to 'Random_RPL40A.csv'")
