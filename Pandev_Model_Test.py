import torch
# Use a single, consistently named device variable
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Original global
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn as nn
# import matplotlib.pyplot as plt # Not used
import os # Now used for os.path.exists
import gc
# import pandas as pd # Not used
from numpy import array
from torch.nn import (
    Linear,
    LSTM,
    BatchNorm1d,
    ReLU,
    Tanh,
    Dropout,
)
import json

# --- Configuration ---
CONFIG = {
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "amino_acids": ['A', 'V', 'F', 'I', 'L','D','E','K','S','T','Y','C','N','Q', 'P','M', 'R', 'H', 'W', 'G','X'],
    "paths": {
        "b_factor_model": "/Users/hikimiwada/Documents/先端生命科学研究/de_novo_design/Pandev_Model_Files/b_factor_model.pth",
        "property_prediction_model": "/Users/hikimiwada/Documents/先端生命科学研究/de_novo_design/Pandev_Model_Files/young/best.pth",
        "feature_names": "/Users/hikimiwada/Documents/先端生命科学研究/de_novo_design/Pandev_Model_Files/young/young_features_name.npy",
        "input_sequences_json": "input_sequences.json"
    },
    "hyperparameters": {
        "rc_cal_filter": 7,
        "rc_cal_stride": 1,
        "rnn_input_size": 21,
        "rnn_hidden_size1": 512,
        "rnn_hidden_size2": 512, # Used in RNN hidden state init, even if not for a second LSTM layer
        "rnn_num_layers": 1,
        "rnn_seq_len": 100, # Added: This was 100 in the original Colab script for RNN init
        "property_nn_input_features": 825 # {tensile_strength: 911, strain_at_break: 1398, toughness: 871, young's_modulus: 825}
    },
    "accepted_spidroin_types": ['MaSp1', 'MaSp2', 'MaSp3', 'MaSp','MiSp', 'Spidroin']
}

# Consistent device usage
DEVICE = torch.device(CONFIG["device"])
AMINO_ACIDS = CONFIG["amino_acids"]

# Make the original global `device` point to our consistent `DEVICE`
# This is for the hardcoded `device` usage within the user's RNN class.
# Ideally, the RNN class would be refactored to not hardcode `device`.
device = DEVICE # Global device alias for RNN internal use

print(f"Using device: {DEVICE}")
print(f'Number of unique amino acids defined: {len(AMINO_ACIDS)}')

default_input_sequences = { # Ensure this provides sequences for testing if needed
    'MaSp1': "GQGGLGGYGQGAGAGAGAAAAAAGAAGAGQGGYGQGGRGGYGQGAGAGAAAGAAGAAGAGQGGYGQGGLGGYGAGAGASAAAAGAAGVGQGAGEGAYGYQSSSQYSLSLSAEAAGYGAGAAGGYGQGVGAGAGAGAAAAAGSGQGGQGGYGQGAGAGAGAAAGGAGAGGYGQGGYGQGAGAGAAAGASAAAGSGQGGRGGYGQGVGAGSGAGGAGAGGYGQGGYGQGAGAGAAAAAAAAAGAGQGGRGGYGQGAAAGAAGAGAGGYGQGGYGQGAGAGAAAGVAAAAGSGQGGRGGYGQGVGAGAGAAGAGAGGYGQGGYGQGAGAGAAAGAAAAATAGAGQGGRGGYGQGAAAGAAGAGAGGYGQGGYGQGAGAGAAAGAAAAAGSGQGGRGGYGQGAGAGAGAGAAAGAGAGGQGGYGQGGLGGYGSGAGAGAAAASAAGAGQAGYGGYGQGAGSGSGAAVAGAGQGGYSGYGQGAAVSAGASTTVVNSVSRMSSASTASRVSSAVSNLVSNGQVNVASLPGIISNISSSISASSPGASECEILVQVLLEVVSALLQIVSSANIGEINLNASSDYASMVGSSLQNVYG",
    'MaSp2': None, 'MaSp3': None, 'MaSp': None, 'MiSp': None, 'Spidroin': None
}
if not os.path.exists(CONFIG["paths"]["input_sequences_json"]):
    with open(CONFIG["paths"]["input_sequences_json"], 'w') as f:
        json.dump(default_input_sequences, f, indent=4)
    print(f"Created/verified input file: {CONFIG['paths']['input_sequences_json']}")


# --- Helper Functions (Data Preprocessing) ---
def onehotseq(sequence, amino_acids_list):
    seq_len = np.shape(sequence)[0]
    seq_en = np.zeros((seq_len, np.shape(amino_acids_list)[0]))
    for i in range(seq_len):
        if sequence[i] in amino_acids_list:
            pos = amino_acids_list.index(sequence[i])
            seq_en[i, pos] = 1
        else:
            pos = amino_acids_list.index('X')
            seq_en[i, pos] = 1
    return seq_en

def one_hot_encoding_batch(input_sequence_batch, num_files_per_sample, seq_lengths_per_sample, amino_acids_list):
    batch_size = input_sequence_batch.shape[0]
    max_files = input_sequence_batch.shape[1]
    max_len = input_sequence_batch.shape[2]
    ohe = np.zeros((batch_size, max_files, max_len, len(amino_acids_list)))
    for i in range(batch_size):
        for j in range(int(num_files_per_sample[i])):
            seq_len = int(seq_lengths_per_sample[i, j])
            if seq_len > 0:
                current_sequence = input_sequence_batch[i, j, 0:seq_len]
                seq_en = onehotseq(current_sequence, amino_acids_list)
                ohe[i, j, 0:seq_len, :] = seq_en
    return ohe

# --- Model Definitions ---
# LSTM Model for B-Factor Prediction (User's version with hardcoded 'device')
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_layers, seq_len,num_classes=1): # seq_len is expected
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.seq_len = seq_len # Store seq_len if model logic depends on it (original did this)

        self.bnn1 = nn.Linear(input_size, 32)
        self.bnn2 = nn.Linear(32,64)
        self.bnn3 = nn.Linear(64,64)
        self.bnn4 = nn.Linear(64,64)

        self.lstm1 = nn.LSTM(64, hidden_size1, num_layers, batch_first=True, bidirectional=True, dropout=0.5)
        # User's RNN uses global 'device' for BatchNorm1d. We aliased global 'device' to 'DEVICE'.
        self.bn_lstm1 = nn.BatchNorm1d(2*hidden_size1, device=device)
        self.nn1 = nn.Linear(2*hidden_size1, 2*hidden_size1)
        self.nn2 = nn.Linear(2*hidden_size1, 512)
        self.nn3 = nn.Linear(512, 512)
        self.nn4 = nn.Linear(512, 256)
        self.nn5 = nn.Linear(256, 256)
        self.nn6 = nn.Linear(256, 128)
        self.nn7 = nn.Linear(128, 32)
        self.nn8 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.drop = nn.Dropout(p=0.5)

    def forward(self, x, array_lengths):
        initial_batch_size = x.size(0) # Renamed from x.size(0) in original line
        inital_seq_len = x.size(1)
        # x is already on DEVICE from b_factor_calculation, but float conversion might be needed
        x = x.float() # Ensure float, .to(device) might be redundant if already on DEVICE
        # If x is not guaranteed to be on `device` (our DEVICE), then uncomment:
        # x = Variable(x.float()).to(device)


        x_reshaped = torch.reshape(x, (initial_batch_size * inital_seq_len, x.size(2)))

        out = self.relu(self.bnn1(x_reshaped))
        out = self.relu(self.bnn2(out))
        out = self.relu(self.bnn3(out))
        out = self.relu(self.bnn4(out))

        out = torch.reshape(out, (initial_batch_size, inital_seq_len, out.size(1)))
        
        # pack_padded_sequence expects lengths on CPU
        pack = nn.utils.rnn.pack_padded_sequence(out, array_lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # Hidden states use global 'device' (aliased to DEVICE)
        h0 = Variable(torch.zeros(2*self.num_layers, initial_batch_size, self.hidden_size1).to(device))
        c0 = Variable(torch.zeros(2*self.num_layers, initial_batch_size, self.hidden_size1).to(device))
        # h1, c1 were for a second LSTM layer not used in the original path, ensure sizes are consistent if re-enabled
        # h1 = Variable(torch.zeros(2*self.num_layers, initial_batch_size, self.hidden_size2).to(device)) # Corrected batch_size dim
        # c1 = Variable(torch.zeros(2*self.num_layers, initial_batch_size, self.hidden_size2).to(device)) # Corrected batch_size dim

        out_packed, _ = self.lstm1(pack, (h0,c0))
        del(h0); del(c0); gc.collect()

        unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        this_batch_actual_max_len = unpacked.size(1) # Renamed from this_batch_len
        out = unpacked
        
        out = torch.reshape(out, (out.size(0)*this_batch_actual_max_len, out.size(2))) # Corrected reshape

        out = self.relu(self.nn1(out))
        out = self.relu(self.nn2(out))
        out = self.relu(self.nn3(out))
        out = self.relu(self.nn4(out))
        out = self.relu(self.nn5(out))
        out = self.relu(self.nn6(out))
        out = self.relu(self.nn7(out))
        out = self.nn8(out)

        out = torch.reshape(out, (initial_batch_size, this_batch_actual_max_len, 1))
        return out

# Property Prediction Network (User's version)
class network(nn.Module):
    def __init__(self, property_nn_input_features): # Takes no arguments
        super(network, self).__init__()
        self.nn = nn.Sequential(nn.Linear(property_nn_input_features,128), # Input features hardcoded
                                nn.ReLU(), nn.Linear(128,64), nn.ReLU(),
                                nn.Linear(64,24), nn.ReLU(), nn.Linear(24,12), nn.ReLU(),
                                nn.Linear(12,12), nn.ReLU(), nn.Linear(12,8), nn.ReLU(),
                                nn.Linear(8,8), nn.ReLU(), nn.Linear(8,1))
    def forward(self, x):
        return self.nn(x)

# --- Model Loading Function (Corrected safe_globals) ---
def load_model(model_class, model_path, device_to_load_on, *args, **kwargs): # Renamed device arg
    # Added 'network' to safe_globals
    safe_globals = [RNN, network, Linear, LSTM, BatchNorm1d, ReLU, Tanh, Dropout]
    
    try:
        # Instantiate model first. This is where __init__ args are used.
        model_instance = model_class(*args, **kwargs)
        # Then load state_dict if model was saved that way, OR load full object.
        # Based on original code, it was full object load.
        # The error "GLOBAL __main__.RNN was not an allowed global" implies full object save.
        
        # Try direct load first, assuming it's a full model object
        # (This path might use weights_only=True by default in new PyTorch, causing issues for full objects)
        loaded_data = torch.load(model_path, map_location=device_to_load_on) # Default weights_only might be True
        
        if isinstance(loaded_data, dict): # It's a state_dict
             model_instance.load_state_dict(loaded_data)
        else: # It's a full model object
            model_instance = loaded_data

        model_instance.to(device_to_load_on)
        model_instance.eval()
        print(f"Model loaded successfully from {model_path} to {device_to_load_on}")
        return model_instance
    except Exception as e:
        print(f"Initial model load attempt failed for {model_path}: {e}")
        print("Attempting fallback load with add_safe_globals and weights_only=False...")
        try:
            torch.serialization.add_safe_globals(safe_globals)
            model_instance = torch.load(model_path, map_location=device_to_load_on, weights_only=False)
            model_instance.to(device_to_load_on)
            model_instance.eval()
            print(f"Model loaded successfully (with add_safe_globals) from {model_path} to {device_to_load_on}")
            return model_instance
        except Exception as e_safe:
            print(f"ERROR: Fallback model loading failed for {model_path}: {e_safe}")
            raise

# --- Core Calculation Functions ---
def b_factor_calculation(ohe_data, num_files_info, seq_length_info, b_factor_model, current_device, amino_acids_list):
    b_factor_pred = np.zeros((ohe_data.shape[0], ohe_data.shape[1], ohe_data.shape[2]))
    # Ensure ohe_tensor is on the correct device passed to the function
    ohe_tensor = torch.from_numpy(ohe_data).float().to(current_device)

    with torch.no_grad():
        for i in range(ohe_tensor.shape[0]):
            if int(num_files_info[i]) != 0:
                input_x = ohe_tensor[i, 0:int(num_files_info[i]), :, :]
                current_seq_lengths = torch.from_numpy(seq_length_info[i, 0:int(num_files_info[i])]).long()
                outputs = b_factor_model(input_x, current_seq_lengths)
                outputs = outputs.squeeze(-1)
                for j in range(outputs.shape[0]):
                    actual_len = int(current_seq_lengths[j].item())
                    b_factor_pred[i, j, 0:actual_len] = outputs[j, 0:actual_len].cpu().numpy()
    return b_factor_pred

# Corrected rc_map_calculation
def rc_map_calculation(b_factor_data, ohe_data, input_sequences_info, num_files_info, seq_length_info, rc_filter, rc_stride, amino_acids_list):
    num_amino_acids_no_x = len(amino_acids_list) - 1
    heat_map_shape = (input_sequences_info.shape[0], num_amino_acids_no_x, num_amino_acids_no_x, rc_filter - 1, 2)
    final_heat_map = np.zeros(heat_map_shape) # Changed variable name for clarity

    for prot_idx in range(final_heat_map.shape[0]):
        num_sequences_for_protein = int(num_files_info[prot_idx])
        if num_sequences_for_protein == 0:
            continue

        # This will store the sum of *normalized* contributions from each sequence
        accumulated_normalized_subs_map = np.zeros((len(amino_acids_list), len(amino_acids_list), rc_filter - 1, 2))

        for seq_idx in range(num_sequences_for_protein):
            current_seq_len = int(seq_length_info[prot_idx, seq_idx])
            if current_seq_len < rc_filter : # sequence must be at least as long as the filter
                continue

            b_factor_seq = b_factor_data[prot_idx, seq_idx, 0:current_seq_len]
            ohe_seq = ohe_data[prot_idx, seq_idx, 0:current_seq_len, :]
            b_mean = np.mean(b_factor_seq)
            b_std = np.std(b_factor_seq)
            if b_std == 0: b_std = 1.0

            # This subs_map is for the current sequence only
            local_subs_map_for_seq = np.zeros_like(accumulated_normalized_subs_map)
            index = 0
            while (index + rc_filter) <= current_seq_len:
                conv_unit_indices = np.argmax(ohe_seq[index : index + rc_filter, :], axis=1)
                b_unit_values = b_factor_seq[index : index + rc_filter]
                for k_offset in range(1, rc_filter): # k_offset is distance, 1 to FILTER-1
                    aa_idx_first = conv_unit_indices[0]
                    aa_idx_kth = conv_unit_indices[k_offset]
                    m_idx = k_offset - 1 # 0-indexed for array

                    local_subs_map_for_seq[aa_idx_first, aa_idx_kth, m_idx, 0] += (b_unit_values[0] - b_mean) / b_std
                    local_subs_map_for_seq[aa_idx_first, aa_idx_kth, m_idx, 1] += (b_unit_values[k_offset] - b_mean) / b_std
                index += rc_stride
            
            # Normalize this sequence's subs_map by (L - (m+1)) and add to accumulator
            for m_idx_norm in range(rc_filter - 1):
                # m_idx_norm is 0 to rc_filter-2. (m+1) in paper is (m_idx_norm+1)
                denominator = float(current_seq_len - (m_idx_norm + 1))
                if denominator > 0:
                    accumulated_normalized_subs_map[:, :, m_idx_norm, :] += local_subs_map_for_seq[:, :, m_idx_norm, :] / denominator
                # else: if denominator is 0 or negative, means seq_len is too short for this m, skip adding.
                # This case should ideally be prevented by `current_seq_len < rc_filter` or `(index + rc_filter) <= current_seq_len`

        # Average the accumulated *normalized* contributions by the number of sequences
        if num_sequences_for_protein > 0:
            final_heat_map[prot_idx, ...] = accumulated_normalized_subs_map[:-1, :-1, :, :] / num_sequences_for_protein
            
    return final_heat_map


def feature_engineering(heat_map_data, all_feature_names_array, current_spidroin_type_name, amino_acids_list):
    relevant_feature_names = [name for name in all_feature_names_array if name.startswith(current_spidroin_type_name + '_')]
    num_good_features = len(relevant_feature_names)

    if num_good_features == 0:
        # print(f"INFO: No features defined for spidroin type {current_spidroin_type_name} in the provided feature names list.")
        return np.zeros((heat_map_data.shape[0], 0)), []

    feature_engineered_data = np.zeros((heat_map_data.shape[0], num_good_features))
    extracted_feature_names_ordered = [] 
    current_feature_idx = 0

    for i in range(heat_map_data.shape[1]): 
        for j in range(heat_map_data.shape[2]): 
            for k in range(heat_map_data.shape[3]): 
                for l in range(heat_map_data.shape[4]): 
                    feature_name = f"{current_spidroin_type_name}_{amino_acids_list[i]}_{amino_acids_list[j]}_{k+1}_{l+1}"
                    if feature_name in relevant_feature_names:
                        if current_feature_idx < num_good_features: # Ensure we don't go out of bounds
                            feature_engineered_data[:, current_feature_idx] = heat_map_data[:, i, j, k, l]
                            extracted_feature_names_ordered.append(feature_name) # Keep track if needed later
                            current_feature_idx += 1
                        else: # Should not happen if num_good_features is calculated from relevant_feature_names
                            print(f"Warning: Exceeded pre-allocated feature space for {current_spidroin_type_name}. Feature: {feature_name}")
    
    if current_feature_idx < num_good_features:
        # This means some names in relevant_feature_names were not actually constructed and found.
        # This could happen if all_feature_names_array has names that this loop structure doesn't generate.
        # print(f"Warning: Only {current_feature_idx} out of {num_good_features} expected features were populated for {current_spidroin_type_name}.")
        pass # We will return the sliced array anyway.

    # print(f'Number of features extracted for spidroin {current_spidroin_type_name}: {current_feature_idx}')
    return feature_engineered_data[:, 0:current_feature_idx], extracted_feature_names_ordered


# --- Main Pipeline (Corrected model loading calls) ---
def run_prediction_pipeline(config_dict): # Renamed arg for clarity
    # Use consistent DEVICE from global scope, initialized from config_dict
    # amino_acids also from global scope

    # Load models with corrected __init__ arguments
    b_factor_model = load_model(
        RNN,
        config_dict["paths"]["b_factor_model"],
        DEVICE, # Pass the consistent DEVICE
        # Args for RNN.__init__
        input_size=config_dict["hyperparameters"]["rnn_input_size"],
        hidden_size1=config_dict["hyperparameters"]["rnn_hidden_size1"],
        hidden_size2=config_dict["hyperparameters"]["rnn_hidden_size2"],
        num_layers=config_dict["hyperparameters"]["rnn_num_layers"],
        seq_len=config_dict["hyperparameters"]["rnn_seq_len"] # Added missing seq_len
        # num_classes defaults to 1 in RNN.__init__
    )
    
    property_model = load_model(
        network, # User's class name
        config_dict["paths"]["property_prediction_model"],
        property_nn_input_features=config_dict["hyperparameters"]["property_nn_input_features"],
        # No arguments for network.__init__()
    )

    all_learned_feature_names = np.load(config_dict["paths"]["feature_names"], allow_pickle=True)
    # print(f"Loaded {len(all_learned_feature_names)} total learned feature names.")

    with open(config_dict["paths"]["input_sequences_json"], 'r') as f:
        user_input_sequences = json.load(f)

    max_len_overall = 0
    has_any_sequence = False
    for spidroin_type_iter in config_dict["accepted_spidroin_types"]: # Renamed var
        seq = user_input_sequences.get(spidroin_type_iter)
        if seq:
            has_any_sequence = True
            max_len_overall = max(max_len_overall, len(seq))
    if not has_any_sequence :
        print("INFO: No sequences provided in input_sequences.json. Result will be based on zero features if any are defined for empty types.")
    if max_len_overall == 0: max_len_overall = 1


    collected_features_for_sample = []
    total_feature_count_debug = 0

    for spidroin_type in config_dict["accepted_spidroin_types"]:
        sequence_str = user_input_sequences.get(spidroin_type)

        if sequence_str:
            seq_list = list(sequence_str)
            current_seq_len = len(seq_list)
            input_sequence_arr = np.full((1, 1, max_len_overall), '', dtype=object)
            input_sequence_arr[0, 0, :current_seq_len] = seq_list
            num_files_arr = np.array([1])
            seq_length_arr = np.array([[current_seq_len]])

            ohe = one_hot_encoding_batch(input_sequence_arr, num_files_arr, seq_length_arr, AMINO_ACIDS)
            b_factor = b_factor_calculation(ohe, num_files_arr, seq_length_arr, b_factor_model, DEVICE, AMINO_ACIDS)
            heat_map = rc_map_calculation(
                b_factor, ohe, input_sequence_arr, num_files_arr, seq_length_arr,
                config_dict["hyperparameters"]["rc_cal_filter"],
                config_dict["hyperparameters"]["rc_cal_stride"],
                AMINO_ACIDS
            )
            current_type_features, _ = feature_engineering(heat_map, all_learned_feature_names, spidroin_type, AMINO_ACIDS)
            if current_type_features.shape[1] > 0:
                collected_features_for_sample.append(current_type_features)
            total_feature_count_debug += current_type_features.shape[1]
            # print(f"Features for {spidroin_type}: {current_type_features.shape[1]}")


        else: # No sequence string for this spidroin_type
            # Count how many features in all_learned_feature_names start with this spidroin_type prefix
            num_expected_features_for_type = 0
            for name in all_learned_feature_names:
                if name.startswith(spidroin_type + '_'):
                    num_expected_features_for_type += 1
            
            if num_expected_features_for_type > 0:
                zero_features_for_type = np.zeros((1, num_expected_features_for_type))
                collected_features_for_sample.append(zero_features_for_type)
            # print(f"INFO: No sequence for {spidroin_type}, adding {num_expected_features_for_type} zero features.")
            total_feature_count_debug += num_expected_features_for_type


    if not collected_features_for_sample and total_feature_count_debug == 0 : # Check if any features AT ALL were collected/expected
        print("ERROR: No features could be generated or defined for any spidroin type. Exiting.")
        return None

    # Ensure all_features is created even if collected_features_for_sample is empty (e.g. all types are None but have defined zero features)
    if collected_features_for_sample:
        final_features_for_sample = np.concatenate(collected_features_for_sample, axis=1)
    elif total_feature_count_debug > 0 : # All were zero feature blocks for None sequences
        final_features_for_sample = np.zeros((1, total_feature_count_debug))
    else: # No sequences and no zero features defined for None types
        final_features_for_sample = np.zeros((1,0)) # or handle as error above


    print(f'Shape of all_features combined for the sample: {final_features_for_sample.shape}')
    
    if final_features_for_sample.shape[1] != config_dict["hyperparameters"]["property_nn_input_features"]:
        print(f"WARNING: Number of generated features ({final_features_for_sample.shape[1]}) "
              f"does not match property_model expected input_features ({config_dict['hyperparameters']['property_nn_input_features']}).")

    if final_features_for_sample.shape[1] == 0 and config_dict["hyperparameters"]["property_nn_input_features"] > 0:
        print("ERROR: No features to predict on, but model expects features. Prediction will fail or be meaningless.")
        return None
    elif final_features_for_sample.shape[1] == 0 and config_dict["hyperparameters"]["property_nn_input_features"] == 0:
         print("INFO: No features and model expects 0 features. This might be a special case.") # Or an error
         # Fall through, but model(empty_tensor) might error.
    
    all_features_tensor = torch.from_numpy(final_features_for_sample).to(DEVICE).type(dtype=torch.float32)

    with torch.no_grad():
        predicted_prop = property_model(all_features_tensor)
        print(f'Predicted tensile strength: {predicted_prop.item():.2f}')

    return predicted_prop.item()

# --- Script Execution ---
if __name__ == "__main__":
    predicted_value = run_prediction_pipeline(CONFIG)
    if predicted_value is not None:
        print(f"Final predicted value: {predicted_value}")
    else:
        print("Pipeline did not produce a prediction.")