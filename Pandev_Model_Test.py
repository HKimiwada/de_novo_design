import torch
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
# from sklearn.model_selection import train_test_split # Not used in the provided snippet's main path
import torch.nn as nn
import matplotlib.pyplot as plt # Not used in the provided snippet's main path
# from sklearn import preprocessing # Not used
# from sklearn.metrics import r2_score # Not used
import random # Not used
# import matplotlib as mpl # Not used
import os # Not used
import gc
import pandas as pd # Not used
# import csv # Not used
from numpy import array # Changed from `from numpy import *` to be more specific
from torch.nn import ( # Added for clarity with model definitions
    Linear,
    LSTM,
    BatchNorm1d,
    ReLU,
    Tanh,
    Dropout,
)
# from torch.utils.tensorboard import SummaryWriter # Not used
# from datetime import date # Not used
import json

# --- Configuration ---
CONFIG = {
    "device": "cuda:0" if torch.cuda.is_available() else "cpu", # Specify GPU if available, e.g., "cuda:0"
    "amino_acids": ['A', 'V', 'F', 'I', 'L','D','E','K','S','T','Y','C','N','Q', 'P','M', 'R', 'H', 'W', 'G','X'],
    "paths": {
        "b_factor_model": "/Users/hikimiwada/Documents/先端生命科学研究/de_novo_design/Pandev_Model_Files/Tensile_Strength/v1_b_factor_model.pth",
        "property_prediction_model": "/Users/hikimiwada/Documents/先端生命科学研究/de_novo_design/Pandev_Model_Files/Tensile_Strength/best.pth",
        "feature_names": "/Users/hikimiwada/Documents/先端生命科学研究/de_novo_design/Pandev_Model_Files/Tensile_Strength/tensile_features_name.npy",
        "input_sequences_json": "input_sequences.json" # Example: path to a JSON file for user_input
    },
    "hyperparameters": {
        "rc_cal_filter": 7,
        "rc_cal_stride": 1,
        "rnn_input_size": 21, # Should match len(CONFIG["amino_acids"])
        "rnn_hidden_size1": 512,
        "rnn_hidden_size2": 512, # Not used in the current RNN forward, but kept for consistency
        "rnn_num_layers": 1,
        # "rnn_seq_len": 100, # This seems like a max_seq_len, actual length is dynamic
        "property_nn_input_features": 911 # Input features for the final property prediction NN
    },
    "accepted_spidroin_types": ['MaSp1', 'MaSp2', 'MaSp3', 'MaSp','MiSp', 'Spidroin']
}

# Create a dummy input_sequences.json for demonstration if it doesn't exist
# In a real scenario, you would prepare this file with your actual sequences
default_input_sequences = {
    'MaSp1': "GQGGLGGYGQGAGAGAGAAAAAAGAAGAGQGGYGQGGRGGYGQGAGAGAAAGAAGAAGAGQGGYGQGGLGGYGAGAGASAAAAGAAGVGQGAGEGAYGYQSSSQYSLSLSAEAAGYGAGAAGGYGQGVGAGAGAGAAAAAGSGQGGQGGYGQGAGAGAGAAAGGAGAGGYGQGGYGQGAGAGAAAGASAAAGSGQGGRGGYGQGVGAGSGAGGAGAGGYGQGGYGQGAGAGAAAAAAAAAGAGQGGRGGYGQGAAAGAAGAGAGGYGQGGYGQGAGAGAAAGVAAAAGSGQGGRGGYGQGVGAGAGAAGAGAGGYGQGGYGQGAGAGAAAGAAAAATAGAGQGGRGGYGQGAAAGAAGAGAGGYGQGGYGQGAGAGAAAGAAAAAGSGQGGRGGYGQGAGAGAGAGAAAGAGAGGQGGYGQGGLGGYGSGAGAGAAAASAAGAGQAGYGGYGQGAGSGSGAAVAGAGQGGYSGYGQGAAVSAGASTTVVNSVSRMSSASTASRVSSAVSNLVSNGQVNVASLPGIISNISSSISASSPGASECEILVQVLLEVVSALLQIVSSANIGEINLNASSDYASMVGSSLQNVYG",
    'MaSp2': None,
    'MaSp3': None,
    'MaSp': None,
    'MiSp': None,
    'Spidroin': None
}
if not os.path.exists(CONFIG["paths"]["input_sequences_json"]):
    with open(CONFIG["paths"]["input_sequences_json"], 'w') as f:
        json.dump(default_input_sequences, f, indent=4)
    print(f"Created dummy input file: {CONFIG['paths']['input_sequences_json']}")

DEVICE = torch.device(CONFIG["device"])
AMINO_ACIDS = CONFIG["amino_acids"]
print(f"Using device: {DEVICE}")
print(f'Number of unique amino acids defined: {len(AMINO_ACIDS)}')

# --- Helper Functions (Data Preprocessing) ---
def onehotseq(sequence, amino_acids_list):
    seq_len = np.shape(sequence)[0]
    seq_en = np.zeros((seq_len, np.shape(amino_acids_list)[0]))
    for i in range(seq_len):
        if sequence[i] in amino_acids_list:
            pos = amino_acids_list.index(sequence[i])
            seq_en[i, pos] = 1
        else:
            pos = amino_acids_list.index('X') # Default to 'X' if not found
            seq_en[i, pos] = 1
    return seq_en

def one_hot_encoding_batch(input_sequence_batch, num_files_per_sample, seq_lengths_per_sample, amino_acids_list):
    """
    Processes a batch of input sequences.
    input_sequence_batch: (batch_size, max_files_in_batch, max_seq_len_in_batch)
    num_files_per_sample: (batch_size,) - number of actual sequences for each item in batch
    seq_lengths_per_sample: (batch_size, max_files_in_batch) - actual length of each sequence
    """
    batch_size = input_sequence_batch.shape[0]
    max_files = input_sequence_batch.shape[1]
    max_len = input_sequence_batch.shape[2]
    
    ohe = np.zeros((batch_size, max_files, max_len, len(amino_acids_list)))

    for i in range(batch_size): # Iterate over samples in batch
        for j in range(int(num_files_per_sample[i])): # Iterate over sequences for this sample
            seq_len = int(seq_lengths_per_sample[i, j])
            if seq_len > 0:
                current_sequence = input_sequence_batch[i, j, 0:seq_len]
                seq_en = onehotseq(current_sequence, amino_acids_list)
                ohe[i, j, 0:seq_len, :] = seq_en
    return ohe

# --- Model Definitions ---

# LSTM Model for B-Factor Prediction
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_layers, device_for_bn): # seq_len removed, device_for_bn added
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        # self.seq_len = seq_len # Not explicitly used in forward if using pack_padded_sequence

        self.bnn1 = nn.Linear(input_size, 32)
        self.bnn2 = nn.Linear(32, 64)
        self.bnn3 = nn.Linear(64, 64)
        self.bnn4 = nn.Linear(64, 64)

        self.lstm1 = nn.LSTM(64, hidden_size1, num_layers, batch_first=True, bidirectional=True, dropout=0.5)
        # BatchNorm1d device should match the device the model will be on.
        self.bn_lstm1 = nn.BatchNorm1d(2 * hidden_size1) # device parameter removed, will be handled by .to(device)
        
        self.nn1 = nn.Linear(2 * hidden_size1, 2 * hidden_size1)
        self.nn2 = nn.Linear(2 * hidden_size1, 512)
        self.nn3 = nn.Linear(512, 512)
        self.nn4 = nn.Linear(512, 256)
        self.nn5 = nn.Linear(256, 256)
        self.nn6 = nn.Linear(256, 128)
        self.nn7 = nn.Linear(128, 32)
        self.nn8 = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh() # Not used in forward
        self.drop = nn.Dropout(p=0.5) # Not used in forward

    def forward(self, x, array_lengths):
        # x is expected to be on the correct device already
        initial_batch_size = x.size(0)
        initial_seq_len = x.size(1) # This is max_seq_len in the padded batch

        x = torch.reshape(x, (x.size(0) * x.size(1), x.size(2))) # (batch*seq, features)

        out = self.relu(self.bnn1(x))
        out = self.relu(self.bnn2(out))
        out = self.relu(self.bnn3(out))
        out = self.relu(self.bnn4(out))

        out = torch.reshape(out, (initial_batch_size, initial_seq_len, out.size(1))) # (batch, seq, processed_features)
        
        # Pack sequence
        # Ensure array_lengths are on CPU for pack_padded_sequence
        pack = nn.utils.rnn.pack_padded_sequence(out, array_lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # Initialize hidden and cell states on the same device as the input
        h0 = torch.zeros(2 * self.num_layers, initial_batch_size, self.hidden_size1, device=x.device)
        c0 = torch.zeros(2 * self.num_layers, initial_batch_size, self.hidden_size1, device=x.device)

        out_packed, _ = self.lstm1(pack, (h0, c0))
        
        # Unpack sequence
        unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        # unpacked shape: (batch_size, actual_max_seq_len_in_batch, 2*hidden_size1)
        
        # Potentially apply BatchNorm1d here if needed, after permuting
        # out_permuted = unpacked.permute(0, 2, 1) # (batch_size, 2*hidden_size1, actual_max_seq_len_in_batch)
        # out_bn = self.bn_lstm1(out_permuted)
        # out = out_bn.permute(0, 2, 1) # (batch_size, actual_max_seq_len_in_batch, 2*hidden_size1)
        # For now, following original logic which applies NN layers directly after reshape
        out = unpacked

        this_batch_actual_max_len = unpacked.size(1)
        out = torch.reshape(out, (out.size(0) * out.size(1), out.size(2))) # (batch*actual_seq, features_lstm)

        out = self.relu(self.nn1(out))
        out = self.relu(self.nn2(out))
        out = self.relu(self.nn3(out))
        out = self.relu(self.nn4(out))
        out = self.relu(self.nn5(out))
        out = self.relu(self.nn6(out))
        out = self.relu(self.nn7(out))
        out = self.nn8(out) # Final layer, usually no ReLU for regression

        out = torch.reshape(out, (initial_batch_size, this_batch_actual_max_len, 1))
        
        gc.collect() # Consider if gc.collect() is truly necessary here
        return out

# Property Prediction Network
class network(nn.Module): # Renamed from 'network' for clarity
    def __init__(self, input_features):
        super(network, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(input_features, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 24), nn.ReLU(),
            nn.Linear(24, 12), nn.ReLU(),
            nn.Linear(12, 12), nn.ReLU(),
            nn.Linear(12, 8), nn.ReLU(),
            nn.Linear(8, 8), nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.nn(x)

# --- Model Loading Function ---
def load_model(model_class, model_path, device, *args, **kwargs):
    """Loads a PyTorch model."""
    # Ensure model class and its components are known for deserialization
    # This list might need to be extended if your models use other custom layers/activations
    safe_globals = [
        RNN,         
        Linear,
        LSTM,
        BatchNorm1d,
        ReLU,
        Tanh,
        Dropout,
    ]
    # Add custom classes to torch.serialization.add_safe_globals
    # This is an alternative to weights_only=False for more complex objects,
    # but weights_only=True is generally safer if you only need the state_dict.
    # For this refactoring, we'll stick to how the original code loaded models.
    
    # For loading full model objects as in original code:
    try:
        # The original code used torch.load(..., weights_only=False) implicitly by not setting it.
        # For security, weights_only=True is preferred if you only need the state dict.
        # However, to match the original script's behavior of loading the whole model:
        model_instance = model_class(*args, **kwargs) # Instantiate model first
        # If loading a model saved as state_dict:
        # model_instance.load_state_dict(torch.load(model_path, map_location=device))
        # If loading a full model object (as in the original script):
        model_instance = torch.load(model_path, map_location=device)
        model_instance.to(device)
        model_instance.eval()
        print(f"Model loaded successfully from {model_path} to {device}")
        return model_instance
    except Exception as e:
        print(f"Error loading the model from {model_path}: {e}")
        # Fallback for trying to load if it was saved as a full model object with specific globals
        try:
            torch.serialization.add_safe_globals(safe_globals)
            # Try loading again as a full model object
            model_instance = torch.load(model_path, map_location=device, weights_only=False) # Explicitly False
            model_instance.to(device)
            model_instance.eval()
            print(f"Model loaded successfully (with add_safe_globals) from {model_path} to {device}")
            return model_instance
        except Exception as e_safe:
            print(f"Error loading the model from {model_path} even with safe_globals: {e_safe}")
            raise


# --- Core Calculation Functions ---
def b_factor_calculation(ohe_data, num_files_info, seq_length_info, b_factor_model, device, amino_acids_list):
    b_factor_pred = np.zeros((ohe_data.shape[0], ohe_data.shape[1], ohe_data.shape[2]))
    ohe_tensor = torch.from_numpy(ohe_data).float().to(device) # Ensure float type

    max_seq_len_overall = ohe_data.shape[2] # Max seq length in the entire dataset padding

    with torch.no_grad():
        for i in range(ohe_tensor.shape[0]): # Iterate through samples in the batch
            if int(num_files_info[i]) != 0:
                # Prepare input for the current sample
                # input_x: (num_actual_files_for_sample, max_seq_len_overall, num_amino_acids)
                input_x = ohe_tensor[i, 0:int(num_files_info[i]), :, :]
                
                # Get actual sequence lengths for the files in this sample
                # array_lengths: (num_actual_files_for_sample,)
                current_seq_lengths = torch.from_numpy(seq_length_info[i, 0:int(num_files_info[i])]).long()
                
                # The RNN model expects batch_first, so input_x is (batch_of_sequences, max_len_in_batch, features)
                # Here, 'batch_of_sequences' is num_files_info[i]
                outputs = b_factor_model(input_x, current_seq_lengths) # Pass lengths to model
                
                # outputs shape: (num_actual_files_for_sample, max_actual_len_in_this_sub_batch, 1)
                outputs = outputs.squeeze(-1) # Remove last dimension: (num_files, max_actual_len)
                
                # Place predictions back, carefully considering padding
                for j in range(outputs.shape[0]): # Iterate over sequences in this sub-batch
                    actual_len = int(current_seq_lengths[j].item())
                    b_factor_pred[i, j, 0:actual_len] = outputs[j, 0:actual_len].cpu().numpy()
    return b_factor_pred


def rc_map_calculation(b_factor_data, ohe_data, input_sequences_info, num_files_info, seq_length_info, rc_filter, rc_stride, amino_acids_list):
    num_amino_acids_no_x = len(amino_acids_list) -1 # Exclude 'X' for final map
    heat_map = np.zeros((input_sequences_info.shape[0], num_amino_acids_no_x, num_amino_acids_no_x, rc_filter - 1, 2))

    for prot_idx in range(heat_map.shape[0]): # Iterate over samples in batch
        num_sequences_for_protein = int(num_files_info[prot_idx])
        if num_sequences_for_protein == 0:
            continue

        # Accumulate subs_map across all sequences for this protein
        accumulated_subs_map = np.zeros((len(amino_acids_list), len(amino_acids_list), rc_filter - 1, 2))
        total_valid_windows = np.zeros((rc_filter -1,)) # To count contributions for normalization

        for seq_idx in range(num_sequences_for_protein):
            current_seq_len = int(seq_length_info[prot_idx, seq_idx])
            if current_seq_len <= rc_filter -1 : # Need at least 'filter' residues
                continue

            # Extract data for the current sequence
            b_factor_seq = b_factor_data[prot_idx, seq_idx, 0:current_seq_len]
            ohe_seq = ohe_data[prot_idx, seq_idx, 0:current_seq_len, :]

            b_mean = np.mean(b_factor_seq)
            b_std = np.std(b_factor_seq)
            if b_std == 0: b_std = 1.0 # Avoid division by zero

            index = 0
            while (index + rc_filter) <= current_seq_len: # Ensure full window
                # conv_unit are indices of amino acids
                conv_unit_indices = np.argmax(ohe_seq[index : index + rc_filter, :], axis=1)
                b_unit_values = b_factor_seq[index : index + rc_filter]

                # Iterate for m = 1 to rc_filter-1 (i.e., k in paper m_k)
                # look corresponds to k in the original paper (distance from the first amino acid)
                for k_offset in range(1, rc_filter): # k_offset goes from 1 to FILTER-1
                    aa_idx_first = conv_unit_indices[0]
                    aa_idx_kth = conv_unit_indices[k_offset]
                    
                    # m in paper is k_offset-1 for 0-indexed array `subs_map`
                    m_idx = k_offset -1 

                    accumulated_subs_map[aa_idx_first, aa_idx_kth, m_idx, 0] += (b_unit_values[0] - b_mean) / b_std
                    accumulated_subs_map[aa_idx_first, aa_idx_kth, m_idx, 1] += (b_unit_values[k_offset] - b_mean) / b_std
                    if seq_idx == 0 : # Count valid windows only once per m for this protein
                         total_valid_windows[m_idx] +=1


                index += rc_stride
        
        # Normalize accumulated_subs_map by the number of sequences and valid windows
        for m_idx in range(rc_filter - 1):
            if num_sequences_for_protein > 0 : #and total_valid_windows[m_idx]>0:
                # Original normalization: /(seq_length[prot,fl]-(m+1)))
                # This seems like it should be averaged over windows
                # For now, mimic original: average by num_sequences_for_protein
                # A more robust normalization might be by the actual count of (first_aa, k_th_aa, m) pairs
                heat_map[prot_idx, :, :, m_idx, :] = (accumulated_subs_map[:-1, :-1, m_idx, :] / num_sequences_for_protein)


    return heat_map


def feature_engineering(heat_map_data, all_feature_names_array, current_spidroin_type_name, amino_acids_list):
    # Determine how many features match the current_spidroin_type_name prefix
    # Expected feature name format: SpidroinType_AA1_AA2_Distance_StatType
    relevant_feature_names = [name for name in all_feature_names_array if name.startswith(current_spidroin_type_name + '_')]
    num_good_features = len(relevant_feature_names)

    if num_good_features == 0:
        print(f"Warning: No features found for spidroin type {current_spidroin_type_name} in the provided feature names list.")
        return np.zeros((heat_map_data.shape[0], 0)), []


    feature_engineered_data = np.zeros((heat_map_data.shape[0], num_good_features))
    extracted_feature_names_ordered = [] # To store the names in the order they are extracted

    current_feature_idx = 0
    # heat_map_data shape: (batch, num_aa-1, num_aa-1, rc-1, 2)
    for i in range(heat_map_data.shape[1]): # iterate aa1 (index for amino_acids_list, excluding 'X')
        for j in range(heat_map_data.shape[2]): # iterate aa2 (index for amino_acids_list, excluding 'X')
            for k in range(heat_map_data.shape[3]): # iterate distance m (0 to rc-2, so k+1 is actual distance)
                for l in range(heat_map_data.shape[4]): # iterate stat type (0 or 1)
                    # Construct the feature name as it would appear in all_feature_names_array
                    # amino_acids_list[i] and [j] are used because heat_map is already sliced to exclude 'X'
                    feature_name = f"{current_spidroin_type_name}_{amino_acids_list[i]}_{amino_acids_list[j]}_{k+1}_{l+1}"
                    
                    if feature_name in relevant_feature_names:
                        if current_feature_idx < num_good_features:
                            feature_engineered_data[:, current_feature_idx] = heat_map_data[:, i, j, k, l]
                            extracted_feature_names_ordered.append(feature_name)
                            current_feature_idx += 1
                        else:
                            # This case should not be reached if num_good_features is calculated correctly
                            print(f"Warning: More features found than pre-allocated for {current_spidroin_type_name}.")


    print(f'Number of features extracted for spidroin {current_spidroin_type_name}: {current_feature_idx}')
    # Return only populated features and their names
    return feature_engineered_data[:, 0:current_feature_idx], extracted_feature_names_ordered


# --- Main Pipeline ---
def run_prediction_pipeline(config):
    device = torch.device(config["device"])
    amino_acids = config["amino_acids"]
    
    # Load models
    b_factor_model = load_model(
        RNN, # Model Class
        config["paths"]["b_factor_model"],
        device,
        input_size=config["hyperparameters"]["rnn_input_size"], # Must match len(amino_acids)
        hidden_size1=config["hyperparameters"]["rnn_hidden_size1"],
        hidden_size2=config["hyperparameters"]["rnn_hidden_size2"],
        num_layers=config["hyperparameters"]["rnn_num_layers"],
        device_for_bn=device # Pass device for BatchNorm
    )
    property_model = load_model(
        network, # Model Class
        config["paths"]["property_prediction_model"],
        device,
        input_features=config["hyperparameters"]["property_nn_input_features"] # Make sure this matches combined features
    )

    # Load feature names (used to select relevant features)
    # These are the names of ALL possible features the property_model was trained on.
    all_learned_feature_names = np.load(config["paths"]["feature_names"], allow_pickle=True)
    print(f"Loaded {len(all_learned_feature_names)} total learned feature names.")

    # Load input sequences
    with open(config["paths"]["input_sequences_json"], 'r') as f:
        user_input_sequences = json.load(f)

    # Prepare data structures for batch processing (here, batch size is 1 for simplicity of example)
    # This part needs to be adapted if you process multiple proteins/items at once.
    # For this example, we simulate a "batch" of one item, which itself can contain multiple sequences (e.g. MaSp1).
    
    # We assume a single "protein sample" for prediction, which might be composed of different spidroin types.
    # The final `all_features_for_sample` will concatenate features from these types.
    
    # Let's determine max number of sequences for any spidroin type and max length
    max_seq_per_type = 1 # Since each type has one sequence string or None
    max_len_overall = 0
    for spidroin_type in config["accepted_spidroin_types"]:
        seq = user_input_sequences.get(spidroin_type)
        if seq:
            max_len_overall = max(max_len_overall, len(seq))

    # If no sequences, set a default max_len to avoid errors, e.g., 1
    if max_len_overall == 0: max_len_overall = 1


    # Placeholder for all features from different spidroin types for THE ONE input sample
    # The size of 10000 is from original code, better to make it dynamic or sum of expected features
    # For now, let's collect features in a list and then concatenate.
    collected_features_for_sample = []


    for spidroin_type in config["accepted_spidroin_types"]:
        sequence_str = user_input_sequences.get(spidroin_type) # Get sequence for current type

        if sequence_str:
            seq_list = list(sequence_str)
            current_seq_len = len(seq_list)

            # Simulate batch of 1 item, with 1 file/sequence for that item
            # Shape: (batch_size=1, num_files=1, seq_len)
            input_sequence_arr = np.full((1, 1, max_len_overall), '', dtype=object) # Padded with empty for object array
            input_sequence_arr[0, 0, :current_seq_len] = seq_list
            
            num_files_arr = np.array([1]) # One file for this type
            seq_length_arr = np.array([[current_seq_len]]) # Actual length

            ohe = one_hot_encoding_batch(input_sequence_arr, num_files_arr, seq_length_arr, amino_acids)
            
            b_factor = b_factor_calculation(ohe, num_files_arr, seq_length_arr, b_factor_model, device, amino_acids)
            
            heat_map = rc_map_calculation(
                b_factor, ohe, input_sequence_arr, num_files_arr, seq_length_arr,
                config["hyperparameters"]["rc_cal_filter"],
                config["hyperparameters"]["rc_cal_stride"],
                amino_acids
            )
            
            # Features for this specific spidroin type
            # heat_map is (1, num_aa-1, num_aa-1, rc-1, 2)
            # `all_learned_feature_names` contains names like "MaSp1_A_C_1_1", "MaSp2_G_T_3_2" etc.
            current_type_features, _ = feature_engineering(heat_map, all_learned_feature_names, spidroin_type, amino_acids)
            # current_type_features shape: (1, num_features_for_this_type)
            if current_type_features.shape[1] > 0:
                collected_features_for_sample.append(current_type_features)

        else:
            # If sequence is None, we need to generate zero features for this spidroin type
            # based on how many features in `all_learned_feature_names` correspond to this `spidroin_type`
            
            # This mimics the original logic of creating zero features if a type is absent.
            # Count expected features for this type from the global list
            num_expected_features_for_type = 0
            for name in all_learned_feature_names:
                if name.startswith(spidroin_type + '_'):
                    num_expected_features_for_type +=1
            
            if num_expected_features_for_type > 0:
                # Shape: (1, num_features_for_this_type)
                zero_features_for_type = np.zeros((1, num_expected_features_for_type))
                collected_features_for_sample.append(zero_features_for_type)
            print(f"No sequence for {spidroin_type}, adding {num_expected_features_for_type} zero features.")


    if not collected_features_for_sample:
        print("No features could be generated. Exiting.")
        return

    # Concatenate features from all spidroin types for the single sample
    # Each element in collected_features_for_sample is (1, num_features_for_its_type)
    final_features_for_sample = np.concatenate(collected_features_for_sample, axis=1)
    # final_features_for_sample shape: (1, total_combined_features)
    
    print(f'Shape of all_features combined for the sample: {final_features_for_sample.shape}')
    
    # Check if the number of features matches the property prediction model's input
    if final_features_for_sample.shape[1] != config["hyperparameters"]["property_nn_input_features"]:
        print(f"Warning: Number of generated features ({final_features_for_sample.shape[1]}) "
              f"does not match property_model input_features ({config['hyperparameters']['property_nn_input_features']}).")
        # You might want to pad with zeros or truncate if this is intentional,
        # or raise an error if it indicates a mismatch.
        # For now, we'll let it proceed, but it will likely error out in the model.
        # Adjust property_nn_input_features in CONFIG if this is the new correct number.


    all_features_tensor = torch.from_numpy(final_features_for_sample).to(device).type(dtype=torch.float32)

    with torch.no_grad():
        predicted_prop = property_model(all_features_tensor)
        print(f'Predicted tensile strength: {predicted_prop.item():.2f}')

    return predicted_prop.item()


# --- Script Execution ---
if __name__ == "__main__":
    # To run this, ensure your model files and feature_names.npy are at the specified paths in CONFIG
    # Also, an input_sequences.json should exist (a dummy one is created if not).
    
    # Example: Modify config for an experiment (e.g., different model path)
    # CONFIG["paths"]["b_factor_model"] = "path/to/another_b_factor_model.pth"
    # CONFIG["hyperparameters"]["rc_cal_filter"] = 8 
    
    predicted_value = run_prediction_pipeline(CONFIG)
    # You can save 'predicted_value' or do further processing here