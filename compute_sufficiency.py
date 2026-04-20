import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1' #for annoying `NotImplementedError: The operator 'aten::.....` mps error
import torch
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle

parser = argparse.ArgumentParser(description="Compute default sufficiency")
parser.add_argument('--finetuned_model_name', type=str, help="Name of my finetuned model")
args = parser.parse_args()
finetuned_model_name = args.finetuned_model_name

if "modernbert" in finetuned_model_name:
    whichmodel = "modernbert"
elif "bert" in finetuned_model_name:
    whichmodel = "bert"
elif "llama2" in finetuned_model_name:
    whichmodel = "llama2"
else:
    whichmodel = None

assert whichmodel in {"bert", "modernbert", "llama2"}

overview_file = "sufficiency_scores.json"

if not os.path.exists(overview_file):
    dict_models_sufficiency = dict()
else:
    with open(overview_file, 'r') as f:
        dict_models_sufficiency = json.load(f)

################################################################
################################################################
try:
    if torch.has_mps:
        device = 'mps'  #m1 mac
    elif torch.cuda.is_available():
        device = 'cuda'  #linux/vm
    else:
        device = 'cpu'
except AttributeError:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
# try:
#     device = 'mps' if torch.has_mps else 'cpu' #m1 mac
# except AttributeError:
#     device = 'cuda' if torch.cuda.is_available() else 'cpu' #linux

print("Model:", whichmodel)
print("Device set to:", device)

"""
Loading the model
"""
if whichmodel == "bert":
    from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
    model = BertForSequenceClassification.from_pretrained("runs/results/"+finetuned_model_name)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
elif whichmodel == "modernbert":
    from transformers import AutoTokenizer, ModernBertForSequenceClassification, Trainer, TrainingArguments
    model = ModernBertForSequenceClassification.from_pretrained("runs/results/"+finetuned_model_name)
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
elif whichmodel == "llama2":
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig, Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True)

    model = AutoModelForSequenceClassification.from_pretrained("runs/results/"+finetuned_model_name,
                                                               quantization_config=quantization_config,
                                                               device_map="auto")
    lora_config = LoraConfig(
        r=16,  #rank of adaptation matrix
        lora_alpha=32,  #scaling
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  #applied to specific attention layers
    )

    model = get_peft_model(model, lora_config)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    # ADD MASK TOKEN NEEDED FOR LIME
    tokenizer.add_special_tokens({'mask_token': '[MASK]'})
    if device == "cuda":
        model.resize_token_embeddings(len(tokenizer))
    model.config.mask_token_id = tokenizer.mask_token_id

model = model.to(device)
model.eval()

# function to extract the topk mask
def create_topk_mask(attribs, tokens, topk=-1, rm_special_tokens=True, dynamic=False, dyn_threshold="mean_pos", peaks=True):
    """
    :param attribs:             list of token attributions
    :param tokens:              list of tokens where the attribs were computed for
    :param topk:                defaults to -1. If dynamic==False, topk should be set to a positive integer
    :param ranked:              boolean
    :param rm_special_tokens:   defaults to True
    :param dynamic:             defaults to False; set to True if mask is to be computed for dyn. top-k based on loc max
    :return:                    boolean mask based on `attribs` list, with 1s indexed at topk highest attribution values
        e.g. create_topk_mask([1,2,3,3,4,5],["a","b","c","d","[CLS]","[SEP]"],3) returns [0, 1, 1, 1]
        e.g. create_topk_mask([1,2,3,3,4,5],["a","b","c","d","[CLS]","[SEP]"],3,rm_special_tokens=False)
        --> returns [0, 0, 1, 0, 1, 1] (or [0, 0, 0, 1, 1, 1], because ties are solved by random choice)
    """
    assert len(attribs) == len(tokens)
    if dynamic == False:  # regular case where we want to measure agreement for a specific topk, i.e. a positive integer
        assert topk > 0

    if rm_special_tokens:
        attribs_2 = [a for t,a in zip(tokens, attribs) if t not in {"[CLS]","[SEP]","<s>","</s>","Ġ"}]
        tokens_2 = [t for t,a in zip(tokens, attribs) if t not in {"[CLS]","[SEP]","<s>","</s>","Ġ"}]
    else:
        attribs_2 = attribs
        tokens_2 = tokens

    if dynamic:
        assert topk == -1  # contradiction: can't compute dynamic topk when a topk integer is set -> set to -1 (default)
        attribs_2, local_maxima_indices = compute_and_plot_local_maxima(list_of_floats=attribs_2,
                                                                        plot=False,
                                                                        dyn_threshold=dyn_threshold,
                                                                        peaks=peaks)
        dynamic_topk_mask = [0 for _ in attribs_2]
        for loc_max_i in local_maxima_indices:
            dynamic_topk_mask[loc_max_i] = 1
        return dynamic_topk_mask, tokens_2

    assign_indices = list(enumerate(attribs_2))
    assign_indices.sort(key=lambda tup: (tup[1], random.random())) #if tie -> randomize
    sorted_indices = [i for i,a in assign_indices]
    topk_sorted_indices = sorted_indices[-topk:] #e.g. top-2 of [1,2,3,0] is [3,0]
    topk_mask = [0 for a in attribs_2] #initialize 0s mask

    for i in topk_sorted_indices:
        topk_mask[i] = 1 #assign 1 at topk indices; outside the topk remains 0
    return topk_mask, tokens_2

# We don't need the following two functions, but they could work with dynamic k
def get_local_maxima(list_of_floats, dyn_threshold="mean_pos", peaks=True):
    """
    Computes the local maxima of a list of floats and returns respective indices.
    Algorithm:
        Point is local maxima if greater than its strict left and right neighbor (except points at index = 0|-1,
        which should only be greater than right or left strict neighbor, respectively) and if greater or equal than a
        threshold. Threshold is mean of the distribution.
    :param list_of_floats:  e.g. list of attribution values
    :return:                array of indices of local maxima
    """
    try:
        if dyn_threshold=="mean":
            threshold = np.mean(list_of_floats)
        elif dyn_threshold=="mean_plus_1std":
            threshold = np.mean(list_of_floats) + 1 * np.std(list_of_floats)
        elif dyn_threshold=="mean_plus_2std":
            threshold = np.mean(list_of_floats) + 2 * np.std(list_of_floats)
        elif dyn_threshold == "mean_min_1std":
            threshold = np.mean(list_of_floats) - 1 * np.std(list_of_floats)
        elif dyn_threshold == "mean_min_2std":
            threshold = np.mean(list_of_floats) - 2 * np.std(list_of_floats)
        elif dyn_threshold=="median":
            threshold = np.median(list_of_floats)

        elif dyn_threshold=="mean_pos":
            list_of_floats_pos = [f for f in list_of_floats if f > 0]
            threshold = np.mean(list_of_floats_pos)
        elif dyn_threshold=="mean_plus_1std_pos":
            list_of_floats_pos = [f for f in list_of_floats if f > 0]
            threshold = np.mean(list_of_floats_pos) + 1 * np.std(list_of_floats_pos)
        elif dyn_threshold=="mean_plus_2std_pos":
            list_of_floats_pos = [f for f in list_of_floats if f > 0]
            threshold = np.mean(list_of_floats_pos) + 2 * np.std(list_of_floats_pos)
        elif dyn_threshold == "mean_min_1std_pos":
            list_of_floats_pos = [f for f in list_of_floats if f > 0]
            threshold = np.mean(list_of_floats_pos) - 1 * np.std(list_of_floats_pos)
        elif dyn_threshold == "mean_min_2std_pos":
            list_of_floats_pos = [f for f in list_of_floats if f > 0]
            threshold = np.mean(list_of_floats_pos) - 2 * np.std(list_of_floats_pos)
        elif dyn_threshold=="median_pos":
            list_of_floats_pos = [f for f in list_of_floats if f > 0]
            threshold = np.median(list_of_floats_pos)

        if peaks==False:
            indices = np.where(list_of_floats >= threshold)[0]
            indices = list(set(indices.tolist() + []))
            indices.sort()
            return np.array(indices)

        # Roll the input list to create arrays representing left and right neighbors of each element
        # e.g.
        # roll_left         = [0.1, 0.2, 0.3]
        # list_of_floats    = [0.3, 0.1, 0.2]
        # roll_right        = [0.2, 0.3, 0.1]
        # -> for each element list_of_floats[i], its strict neighbors are roll_left[i] and roll_right[i]
        roll_left = np.roll(list_of_floats, 1)
        roll_right = np.roll(list_of_floats, -1)

        # Find indices where the current element is greater than its strict left and right neighbors,
        # and the current element is greater than or equal to the threshold
        indices = \
        np.where((roll_left < list_of_floats) & (roll_right < list_of_floats) & (list_of_floats >= threshold))[0]
        # print(list_of_floats)
        # print(indices)
        # Create a list to store additional indices for special cases (first and last elements)
        additional_indices = []

        # Check if the first element is greater than the second and greater or equal to the threshold
        if list_of_floats[0] > list_of_floats[1] and list_of_floats[0] >= threshold:
            additional_indices.append(0)

        # Check if the last element is greater than the second-to-last and greater or equal to the threshold
        if list_of_floats[-1] > list_of_floats[-2] and list_of_floats[-1] >= threshold:
            additional_indices.append(len(list_of_floats) - 1)

        # Check for spikes with the middle point as a local maximum
        i = 1
        while i < len(list_of_floats) - 1:
            if list_of_floats[i] >= threshold:
                j = i
                # Continue iterating through the list while consecutive elements have the same value
                while j < len(list_of_floats) - 1 and list_of_floats[j] == list_of_floats[j + 1]:
                    j += 1
                if j > i:
                    # Check if the cluster is attached to a higher peak without lower points in between
                    try:
                        if (list_of_floats[i - 1] < list_of_floats[i]) and (list_of_floats[j + 1] < list_of_floats[i]):
                            cluster_size = j-i + 1 # j is index last element in cluster, i is index first element
                            # Find the middle point of the cluster and mark it as a local maximum, if n elements uneven
                            if cluster_size % 2 != 0:
                                middle_idx = i + (cluster_size//2)
                                additional_indices.append(middle_idx)
                            # Find middle two elements and take random choice if n elements in cluster are even
                            else:
                                # Calculate the index of the first center element
                                center1_index =  i + (cluster_size//2) - 1  # Subtract 1 because Python uses 0-based indexing
                                # Calculate the index of the second center element
                                center2_index =  i + (cluster_size//2)
                                random_middle_idx = random.choice([center1_index,center2_index])
                                additional_indices.append(random_middle_idx)
                        else:
                            # Skip the cluster if it is attached to a higher peak without lower points in between
                            pass
                    except IndexError:
                        # Handle the case where an error occurs (e.g., if the cluster is at the beginning or end)
                        # print("Skipped error")  # Skip an error for one specific case
                        pass
                    i = j
            i += 1

        # Combine the main indices and additional indices, remove duplicates, and sort them
        indices = list(set(indices.tolist() + additional_indices))
        indices.sort()

        return np.array(indices)
    except Exception as e:
        print("Error:", e)
        return np.array([])  # Return an empty array if an exception occurs

def compute_and_plot_local_maxima(list_of_floats, dyn_threshold="mean_pos", plot=True, peaks=True):
    """
    Compute local maxima of a list of floats
    :param list_of_floats:  e.g. list of attribution values
    :param plot:            defaults to True    --> plots the curves with local maxima
    :return:                tuple of 2          --> (list of floats , local maxima indices)

    #example
    d = [0.1,0.5,0.8,0.2,0.3,0.1,0.5,0.6,0.6,0.1]
    compute_and_plot_local_maxima(d)
    """
    local_maxima_indices = get_local_maxima(list_of_floats, dyn_threshold=dyn_threshold, peaks=peaks)
    if plot:
        plt.plot(list_of_floats)
        plt.plot(local_maxima_indices, [list_of_floats[i] for i in local_maxima_indices], 'ro')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Peaks in Data')
        plt.show()
    return list_of_floats, local_maxima_indices

# Function to modify the instances to mask or remove the non-rationales (and test sufficiency between orig and mod)
def modify_input(instance_scores, instance_tokens, original_or_removeCTXT_or_removeTOP):
    """
    :param instance_scores:
    :param instance_tokens:
    :param original_or_removeCTXT_or_removeTOP: return
        * the original tokens in a single string format (original)
        * remove the context tokens, i.e. that are NOT topk (CTXT)
        * remove the topk tokens (TOPK)
    :return:
    """
    topk_mask, tokens000 = create_topk_mask(attribs=instance_scores, tokens=instance_tokens, topk=1, rm_special_tokens=True, dynamic=False)

    assert original_or_removeCTXT_or_removeTOP in {"original", "removeCTXT", "removeTOP"}

    if whichmodel == "bert":
        tokens = [t.strip("##") for t in tokens000]
    elif whichmodel == "modernbert":
        tokens = [t.strip("Ġ") for t in tokens000]
    elif whichmodel == "llama2":
        tokens = [t.strip("▁") for t in tokens000]

    # if original_or_removeCTXT_or_removeTOP == "original":
    #     modified_input = " ".join(tokens)
    # elif original_or_removeCTXT_or_removeTOP == "removeCTXT":
    #     modified_input = " ".join([t for t,m in zip(tokens, topk_mask) if m == 1])
    # elif original_or_removeCTXT_or_removeTOP == "removeTOP":
    #     modified_input = " ".join([t for t, m in zip(tokens, topk_mask) if m == 1])
    if original_or_removeCTXT_or_removeTOP == "original":
        modified_input = tokens
    elif original_or_removeCTXT_or_removeTOP == "removeCTXT":
        modified_input = [t for t,m in zip(tokens, topk_mask) if m == 1]
    elif original_or_removeCTXT_or_removeTOP == "removeTOP":
        modified_input = [t for t,m in zip(tokens, topk_mask) if m == 0]
    return modified_input

# load test data from pickled explanations
target_explanation_pickle = None
for explanation_pickle in os.listdir("explanations/"):
    if finetuned_model_name in explanation_pickle:
        target_explanation_pickle = explanation_pickle
        print("explanation pickle found for", finetuned_model_name)
        break
if not target_explanation_pickle:
    raise FileNotFoundError(
        "There is no explanation pickle for the given model (at least not in this directory)")

with open("explanations/"+target_explanation_pickle, 'rb') as file:
    dataset_explanations = pickle.load(file)

def get_prediction_probability(text):
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt")
    inputs.pop('token_type_ids', None)  # Remove token_type_ids if it exists
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Send inputs to device
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs[0]
    probs = torch.softmax(logits, dim=-1)
    return probs

# # max len for truncation + some stats
# MAX_LEN=0
# larger=0
# smaller=0
# for d in test:
#     n_tokens = len(d["text"].split())
#     if n_tokens > MAX_LEN:
#         MAX_LEN=n_tokens
#     if n_tokens > 512:
#         larger+=1
#     else:
#         smaller+=1
# print("MAX_LEN", MAX_LEN)
# print(">512 tokens",larger) #MovieReviews: 1578 examples are larger than 512 tokens
# print("<512 tokens",smaller) #MovieReviews: 421 examples are smaller than 512 tokens
#
# MAX_LEN += 20 #add some room for segmentation error
# MAX_LEN = min(512, MAX_LEN)

MAX_LEN = 25

# Compute sufficiency for each example
dataset_suff_and_comp_scores = []

for instance in dataset_explanations:
    instance_suff_and_comp_scores = dict()
    for i, explanation_method in enumerate(instance):
        scores = explanation_method.scores
        tokens = explanation_method.tokens
        label = explanation_method.target
        instance_tokens_original = modify_input(instance_scores=scores, instance_tokens=tokens,
                                                original_or_removeCTXT_or_removeTOP="original")
        instance_tokens_removedCTXT = modify_input(instance_scores=scores, instance_tokens=tokens,
                                                   original_or_removeCTXT_or_removeTOP="removeCTXT")
        instance_tokens_removedTOP = modify_input(instance_scores=scores, instance_tokens=tokens,
                                                  original_or_removeCTXT_or_removeTOP="removeTOP")

        # Get the model's prediction for the full input
        input_prob_original = get_prediction_probability(" ".join(instance_tokens_original))[0][label].item()
        input_prob_removedCTXT = get_prediction_probability(" ".join(instance_tokens_removedCTXT))[0][label].item()
        input_prob_removedTOP = get_prediction_probability(" ".join(instance_tokens_removedTOP))[0][label].item()

        # Compute the sufficiency and comprehensiveness scores
        sufficiency = input_prob_original - input_prob_removedCTXT
        comprehensiveness = input_prob_original - input_prob_removedTOP
        instance_suff_and_comp_scores[i] = {"suff": sufficiency, "comp": comprehensiveness}

    dataset_suff_and_comp_scores.append(instance_suff_and_comp_scores)

# Per finetuned, create an entry in the overview file
dict_models_sufficiency[finetuned_model_name] = dataset_suff_and_comp_scores

with open(overview_file, 'w') as f: # Save the updated data back to the overview json file
    json.dump(dict_models_sufficiency, f, indent=4)
