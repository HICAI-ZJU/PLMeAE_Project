# Integrating Protein Language Models and Automatic Biofoundry for Enhanced Protein Evolution

The official implementation of the paper: Integrating Protein Language Models and Automatic Biofoundry for Enhanced Protein Evolution 


## Requirements

- Python 3.7 or higher
- PyTorch
- NumPy
- copy
- random
- pickle
- itertools
- heapq
- POT


## Usage

### Module 1

We aim to analyze the impact of single-point mutations on a given protein sequence. The specific goals are:

1. **Mutation Library Generation**: Create a library of mutated sequences by introducing single-point mutations at each position in the original sequence. Given an original sequence with N amino acids and 20 possible mutations at each position, this results in a library of 20×N unique sequences.
2. **Likelihood Calculation**: Utilize the Evolutionarily Scaled Model (ESM) to calculate the likelihood of each mutated sequence. The likelihood scores serve as a proxy for evaluating the potential functional stability or desirability of each sequence.
3. **Top Sequence Selection**: Based on the calculated likelihoods, rank all mutated sequences and select the top 96 sequences with the highest likelihood scores.

Run the script from the script folder using: `python module_1.py`


### Module 2
1. **Mask Your Mutation Sites**: In the module_2.py file, modify line 14. Replace "GB1" with your protein of interest, and use the '<mask>' token to substitute the mutation sites, as shown below:
```
data = [("GB1","MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNG<mask><mask><mask>EWTYDDATKTFT<mask>TE")]
```
2. Run the script from the script folder using:
```
python module_2.py
```
This will generate the file select_96.json.


### Finetuning

1. set the parameters in 'scripts/run_fitness.sh'

2. set the path in 'tasks/fitness.py': 'path_to_train_data.csv' and 'path_to_test_data.csv'

3. Run the script from the script folder using:
```
sh run_fitness.sh
```

### Protein Mutation Analysis

`UBC9.py` and `RPL40A.py` are designed for analyzing protein mutation data. Each script provides functions to process protein sequences, generate mutations, and select optimal sequences based on likelihood scores using a pre-trained ESM model. The scripts focus on two different proteins, `UBC9` and `RPL40A`, and are designed to facilitate mutation analysis and data preparation for further study or model training.

1. Run the script:

   ```
   python ./Zero-shot/UBC9.py or./Zero-shot/RPL40A.py
   ```

2. Outputs:

   - A CSV file containing the top 96 mutations, labels, mutation positions, and amino acids.
   - A CSV file with 96 randomly generated mutations for comparative analysis.

## Contact: qiang.zhang.cs@zju.edu.cn

