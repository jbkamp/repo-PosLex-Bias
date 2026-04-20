import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1' #for annoying `NotImplementedError: The operator 'aten::.....` mps error
import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import random
from collections import Counter
import numpy as np
import os
import numpy as np
from scipy.spatial.distance import jensenshannon
from itertools import combinations, product
import seaborn as sns
import pandas as pd
import json

"""
* calculate (per sentence, per attribution method, excluding special tokens)
    top-1
    top-3, 
    top-5,
    dynamic-k (excl special tokens) 
* measure preference distribution for words and indices
"""
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

def find_topk_token_span_onsets(tokens, topk_mask, strip_chars=("##", "▁", "Ġ"), return_sentence=False):
    assert len(tokens) == len(topk_mask), "Tokens and mask must be the same length."

    def clean(token):
        for prefix in strip_chars:
            if token.startswith(prefix):
                token = token[len(prefix):]
        return token

    clean_tokens = [clean(tok) for tok in tokens]

    # Reconstruct sentence and track character spans for each token
    sentence = ""
    spans = []
    for tok in clean_tokens:
        if sentence and not sentence.endswith(" "):
            sentence += " "
        start = len(sentence)
        sentence += tok
        end = len(sentence)
        spans.append((start, end))

    # Select spans where mask == 1
    target_spans = [spans[i] for i, m in enumerate(topk_mask) if m == 1]

    if return_sentence:
        return target_spans, sentence
    else:
        return target_spans

def create_hits(k, dataset_explanations, return_topk_token_span_onsets=False):
    assert (type(k)==int and (k==-1 or k>0))

    if k == -1:
        is_dynamic = True
        dynamic_k_tracker_d = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}  # to compute avg dynamic k at the end
    else:
        is_dynamic = False

    if return_topk_token_span_onsets:
        hits = {0: {"words": [], "indices": [], "word_onsets": []},  # attrib method 0
                  1: {"words": [], "indices": [], "word_onsets": []},  # ...
                  2: {"words": [], "indices": [], "word_onsets": []},
                  3: {"words": [], "indices": [], "word_onsets": []},
                  4: {"words": [], "indices": [], "word_onsets": []},
                  5: {"words": [], "indices": [], "word_onsets": []},  # attrib method 5
                  }
    else:
        hits = {0: {"words": [], "indices": [],},  # attrib method 0
                1: {"words": [], "indices": []},  # ...
                2: {"words": [], "indices": []},
                3: {"words": [], "indices": []},
                4: {"words": [], "indices": []},
                5: {"words": [], "indices": []},  # attrib method 5
                }

    for example in dataset_explanations:
        for i, explanation_method in enumerate(example):
            topk_mask, tokens = create_topk_mask(attribs=explanation_method.scores,
                                                 tokens=explanation_method.tokens,
                                                 topk=k,
                                                 rm_special_tokens=True,
                                                 dynamic=is_dynamic)
            if return_topk_token_span_onsets:
                word_onsets = find_topk_token_span_onsets(tokens, topk_mask)
                hits[i]["word_onsets"].append(word_onsets)
            for idx, m in enumerate(topk_mask):
                if m == 1:  # if the mask is 1 for a specific token...
                    clean_token = tokens[idx].strip("▁") # llama2
                    clean_token = clean_token.strip("Ġ") # modernbert
                    clean_token = clean_token.strip("##") # bert
                    hits[i]["words"].append(clean_token)
                    hits[i]["indices"].append(idx)

            if is_dynamic:
                dynamic_k_tracker_d[i] += sum(topk_mask)

    if len(dataset_explanations[0]) == 5: #if PartSHAP was not computed at all: n_methods = 5
        hits_with_empty_explanations_at_index_0 = dict()
        assert len(hits[5]["words"]) == 0
        n_elements = len(hits[0]["words"])
        if return_topk_token_span_onsets:
            hits_with_empty_explanations_at_index_0[0] = {"words": [np.nan for _ in range(n_elements)],
                                                          "indices": [np.nan for _ in range(n_elements)],
                                                          "word_onsets": [[(np.nan, np.nan)] for _ in range(n_elements)]}
        else:
            hits_with_empty_explanations_at_index_0[0] = {"words": [np.nan for _ in range(n_elements)],
                                                          "indices": [np.nan for _ in range(n_elements)]}
        hits_with_empty_explanations_at_index_0[1] = hits[0]
        hits_with_empty_explanations_at_index_0[2] = hits[1]
        hits_with_empty_explanations_at_index_0[3] = hits[2]
        hits_with_empty_explanations_at_index_0[4] = hits[3]
        hits_with_empty_explanations_at_index_0[5] = hits[4]
        return hits_with_empty_explanations_at_index_0
    else:
        return hits

'''
1) Inter-model agreement 
2) Cumulative
'''
def mean_js(list_of_distributions):
    pairs = list(combinations(list_of_distributions, 2))
    js_values = [jensenshannon(p, q) for p, q in pairs]
    return np.mean(js_values)

def extract_distribution_from_hits(hits, words_or_indices_or_sentence_hits, attribution_method_idx, word_types, indices, sentence_hits):
    assert words_or_indices_or_sentence_hits in {"words", "indices", "sentence_hits"}
    assert attribution_method_idx in {0,1,2,3,4,5}
    count_dict = Counter(hits[attribution_method_idx][words_or_indices_or_sentence_hits])
    if words_or_indices_or_sentence_hits == "words":
        count_distrib = [count_dict[w] for w in word_types] # raw count distribution
    elif words_or_indices_or_sentence_hits == "indices":
        count_distrib = [count_dict[i] for i in indices] # raw count distribution
    elif words_or_indices_or_sentence_hits == "sentence_hits":
        count_distrib = [count_dict[i] for i in sentence_hits]  # raw count distribution
    total_freq = sum(count_dict.values())
    relative_distrib = [round((count / total_freq) * 100) for count in count_distrib] # relative distribution
    return relative_distrib

def determine_params_from_file_name(dataset_explanations_file_name):
    if "punct_comma_random.pickle" in dataset_explanations_file_name:
        dataset_name = "period-comma"
        dataset_word_types = [".", ","]
        dataset_indices = list(range(20))
    elif "unique_punctuation_marks_random.pickle" in dataset_explanations_file_name:
        dataset_name = "unique-punctuation"
        dataset_word_types = [".", ",", ";", ":", "!", "?", "-", "_", "(", ")",
                              "[", "]", "{", "}", "/", "*", "#", "'", '"', "`"]
        dataset_indices = list(range(20))
    elif "noun_det_punct.pickle" in dataset_explanations_file_name:
        dataset_name = "noun-det-period"
        dataset_word_types = ["table", "the", "."]
        dataset_indices = list(range(20))
    return dataset_name, dataset_word_types, dataset_indices

def determine_params_from_dataset_name(dataset_name):
    if "period-comma" in dataset_name:
        dataset_word_types = [".", ","]
        dataset_indices = list(range(20))
    elif "unique-punctuation" in dataset_name:
        dataset_word_types = [".", ",", ";", ":", "!", "?", "-", "_", "(", ")",
                              "[", "]", "{", "}", "/", "*", "#", "'", '"', "`"]
        dataset_indices = list(range(20))
    elif "noun-det-period" in dataset_name:
        dataset_word_types = ["table", "the", "."]
        dataset_indices = list(range(20))
    return dataset_word_types, dataset_indices

k = 1
methods = {0:"PartSHAP", 1:"LIME", 2:"VanGrad", 3:"GradxI", 4:"IntGrad", 5:"IntGradxI"}
range_methods_to_include = range(6) # needed for loops; if (1,6) --> we exclude PartSHAP (idx 0), otherwise it would be range(6)
dataset_names = ["period-comma", "unique-punctuation", "noun-det-period"]
##################################################
def plot_JS_heatmap(*all_js_dicts):
    num_plots = len(all_js_dicts)
    print("num plots", num_plots)
    assert 1 <= num_plots <= 3, "Can only plot between 1 and 3 heatmaps."

    fig, axes = plt.subplots(1, num_plots, figsize=(8, 3.5), squeeze=False)  # wider & taller
    axes = axes[0]

    for i, all_js_dict in enumerate(all_js_dicts):
        datasets = sorted(set([key[0] for key in all_js_dict.keys()]))
        methods_list = sorted(set([key[1] for key in all_js_dict.keys()]))

        heatmap_data = []
        for dataset in datasets:
            row = []
            for method in methods_list:
                row.append(all_js_dict.get((dataset, method), None))
            heatmap_data.append(row)

        heatmap_df = pd.DataFrame(
            heatmap_data,
            index=datasets,
            columns=[methods[m] for m in methods_list]
        )

        ax = axes[i]

        sns.heatmap(
            heatmap_df,
            annot=True,
            cmap='YlGnBu',
            cbar=False,  # Disable individual colorbars
            fmt='.2f',
            linewidths=0.5,
            vmin=0,
            vmax=1.0,
            ax=ax,
            square=False
        )

        # Remove leading zeros in annotation text (e.g., 0.34 → .34)
        for text in ax.texts:
            val = text.get_text()
            if val.startswith("0."):
                text.set_text(val[1:])
            elif val.startswith("-0."):
                text.set_text("-" + val[2:])

        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')

        if i == 0:
            # ax.set_ylabel("Datasets")
            ax.set_yticklabels(ax.get_yticklabels(), rotation=30)
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])
            ax.tick_params(left=False)

        # ax.set_xlabel("Attribution Methods")

        model_suffix = models[i] if i < len(models) else whichmodel
        model_suffix_title = {
            "_bert": "BERT",
            "_modernbert": "ModernBERT",
            "_llama2": "Llama2"
        }[model_suffix]
        ax.set_title(model_suffix_title)

    # Add global colorbar using a ScalarMappable
    norm = plt.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap='YlGnBu', norm=norm)
    sm.set_array([])

    divider = make_axes_locatable(axes[-1])
    cax = divider.append_axes("right", size="4%", pad=0.15)  # more space from plot
    cbar = fig.colorbar(sm, cax=cax)
    cbar.outline.set_visible(False)  # remove contour

    plt.tight_layout()
    outname = whichmodel + "_" + WORDS_OR_INDICES + "_JS_interseed"
    if num_plots > 1:
        outname = f"_{num_plots}plots" + "_" + WORDS_OR_INDICES + "_JS_interseed"
    plt.savefig(outname + ".pdf")
    plt.show()

list_all_hits = [] #list which will contain a `all_hits` dictionary per model
list_all_models_js_for_heatmap_WORDS = []
list_all_models_js_for_heatmap_INDICES = []
# models = ["_bert", "_modernbert", "_llama2"]
models = ["_bert", "_modernbert"]
for whichmodel in models:
    dataset_explanations_file_names = [file for file in os.listdir("explanations/") if whichmodel in file]
    # Populating `all_hits` with all top-1 hits
    all_hits = dict()
    for file_name in dataset_explanations_file_names:
        with open("explanations/" + file_name, "rb") as f:
            dataset_explanations = pickle.load(f)
        hits = create_hits(k=k, dataset_explanations=dataset_explanations)
        dataset_name, dataset_word_types, dataset_indices = determine_params_from_file_name(file_name)
        all_hits[file_name] = {"dataset_name": dataset_name,
                               "dataset_word_types": dataset_word_types,
                               "dataset_indices": dataset_indices,
                               "hits": hits}
    list_all_hits.append(all_hits)

    # Prepare Bias(cons) for inter-seed JS heatmaps
    for WORDS_OR_INDICES in ["words", "indices"]:
        # INITIALISE `all_distributions` which will contain either distributions from WORDS or INDICES (as defined above)
        all_distributions = {(dataset_name, attribution_idx): [] for dataset_name, attribution_idx in list(product(dataset_names, range_methods_to_include))}

        for (dataset_name, attribution_idx) in all_distributions.keys():
            for picklename, details in all_hits.items():
                if details["dataset_name"] == dataset_name:
                    distribution = extract_distribution_from_hits(hits=details["hits"],
                                                                  words_or_indices_or_sentence_hits=WORDS_OR_INDICES,
                                                                  attribution_method_idx=attribution_idx,
                                                                  word_types=details["dataset_word_types"],
                                                                  indices=details["dataset_indices"],
                                                                  sentence_hits=None)
                    all_distributions[(dataset_name, attribution_idx)].append(distribution)

        all_js = dict()
        for datasetXattribmethod, list_of_distributions in all_distributions.items():
            print(datasetXattribmethod, round(mean_js(list_of_distributions),2))
            all_js[datasetXattribmethod] = round(mean_js(list_of_distributions),2)

        # plot_JS_heatmap(all_js)

        if WORDS_OR_INDICES == "words":
            list_all_models_js_for_heatmap_WORDS.append(all_js)
            if len(list_all_models_js_for_heatmap_WORDS) == len(models):
                plot_JS_heatmap(*list_all_models_js_for_heatmap_WORDS)
        elif WORDS_OR_INDICES == "indices":
            list_all_models_js_for_heatmap_INDICES.append(all_js)
            if len(list_all_models_js_for_heatmap_INDICES) == len(models):
                plot_JS_heatmap(*list_all_models_js_for_heatmap_INDICES)

"""
Compute aggregate frequencies
"""
def get_aggregate_hits(all_hits):
    aggregate_hits = {details["dataset_name"]: {attrib_idx:{"words":[],"indices":[]}
                                             for attrib_idx in range_methods_to_include} for run_name, details in all_hits.items()}
    for ds_name in dataset_names:
        for run_name, details in all_hits.items():
            if details["dataset_name"] == ds_name:
                # for attribution_idx in details["hits"]:
                for attribution_idx in range_methods_to_include:
                    aggregate_hits[ds_name][attribution_idx]["words"] += details["hits"][attribution_idx]["words"]
                    aggregate_hits[ds_name][attribution_idx]["indices"] += details["hits"][attribution_idx]["indices"]
    return aggregate_hits

list_aggregate_hits = [get_aggregate_hits(all_hits=all_hits) for all_hits in list_all_hits] #one per model

"""
- plot aggregated hits (horizontal barplots)
- compute JS between aggregates and uniform distribution (to quantify the biases given by the plots); 
    --> add score to subplot title
- PRINT values corresponding to the plot, to be put in our table in the paper!
"""

def average_js_intermethod(target_distribution, other_distributions):
    distances = [
        jensenshannon(target_distribution, other)
        for other in other_distributions
    ]
    return round(np.mean(distances), 2)

def remove_zero_add_zero(float_number):
    stringed = str(float_number)
    if stringed[0:2] == "0.":
        if len(stringed) == 3:
            stringed += "0" #add zero if second decimal is missing e.g. .1 --> .10
        return stringed[1:] #remove heading zero before .decimals, e.g. 0.11 --> .11
    else:
        return float_number

def poslex_freq_plot(word_types, indices, hits, model_name, dataset_name, range_methods_to_include,
                     title=True, adapted_fig_size=None, adapted_file_name=None, adapted_yticks_size=None,
                     adapted_subplots_title=False):
    print(model_name, dataset_name)
    if not adapted_fig_size: #default size for 5 methods
        fig, axes = plt.subplots(nrows=len(range_methods_to_include), ncols=2, figsize=(11, 30))
    else:
        fig, axes = plt.subplots(nrows=len(range_methods_to_include), ncols=2, figsize=adapted_fig_size)

    word_distributions = []
    index_distributions = []

    # Precompute the word and index distributions
    for i in range_methods_to_include:
        count_dict_words = Counter(hits[i]["words"])
        word_counts = [count_dict_words[w] for w in word_types]
        total_words_freq = sum(word_counts)
        word_percentages = [((count / total_words_freq) * 100) for count in word_counts]
        word_distributions.append(word_percentages)

        count_dict_indices = Counter(hits[i]["indices"])
        index_counts = [count_dict_indices[idx] for idx in indices]
        total_indices_freq = sum(index_counts)
        index_percentages = [((count / total_indices_freq) * 100) for count in index_counts]
        index_distributions.append(index_percentages)

    # Compute average pairwise JS distances between methods (inter-method)
    word_js_avgs = [
        average_js_intermethod(word_distributions[i], word_distributions[:i] + word_distributions[i + 1:])
        for i in range(len(word_distributions))
    ]

    index_js_avgs = [
        average_js_intermethod(index_distributions[i], index_distributions[:i] + index_distributions[i + 1:])
        for i in range(len(index_distributions))
    ]

    TOPRINT_index_JS = [] #bias agg
    TOPRINT_word_JS = [] #bias agg

    assert len(word_distributions) == len(index_distributions)
    assert len(word_distributions) == len(range_methods_to_include)

    for i, method_original_i in zip(range(len(word_distributions)),range_methods_to_include):
    #for i in range_methods_to_include:
        # Indices plot
        index_percentages = index_distributions[i]
        index_baseline_threshold_chance = round(100 / len(indices), 5)
        index_uniform_distribution = [100 / len(index_percentages) for _ in index_percentages]
        index_JS = round(jensenshannon(index_percentages, index_uniform_distribution), 2)
        TOPRINT_index_JS.append(index_JS)

        axes[i, 0].barh(indices, index_percentages, color="#2F4F4F", height=0.5)
        axes[i, 0].set_xlim(0, 101)
        if not adapted_subplots_title:
            axes[i, 0].set_title(
                f'{methods[method_original_i]} - Position Bias-agg:{remove_zero_add_zero(index_JS)} | Bias-attr:{remove_zero_add_zero(index_js_avgs[i])}',
                fontsize=14)
        else:
            full_name_dict = {"VanGrad": "Vanilla Gradient", "IntGrad": "Integrated Gradient"}
            axes[i, 0].set_title(
                f'{full_name_dict[methods[method_original_i]]} - Position Bias',
                fontsize=14)
        axes[i, 0].set_xlabel('Frequency (%)', fontsize=14)
        axes[i, 0].set_yticks(range(len(indices)))
        axes[i, 0].tick_params(axis='y', labelsize=14)
        axes[i, 0].set_xticks(range(0, 101, 5))
        axes[i, 0].axvline(x=index_baseline_threshold_chance, linestyle="dotted", color="red", linewidth=2)
        axes[i, 0].spines['top'].set_visible(False)
        axes[i, 0].spines['right'].set_visible(False)

        # Words plot
        word_percentages = word_distributions[i]
        word_baseline_threshold_chance = round(100 / len(word_types), 5)
        word_uniform_distribution = [100 / len(word_percentages) for _ in word_percentages]
        word_JS = round(jensenshannon(word_percentages, word_uniform_distribution), 2)
        TOPRINT_word_JS.append(word_JS)

        axes[i, 1].barh(word_types, word_percentages, color="#A9A9A9", height=0.5)
        axes[i, 1].set_xlim(0, 101)
        if not adapted_subplots_title:
            axes[i, 1].set_title(
                f'{methods[method_original_i]} - Lexical Bias-agg:{remove_zero_add_zero(word_JS)} | Bias-attr:{remove_zero_add_zero(word_js_avgs[i])}',
                fontsize=14)
        else:
            full_name_dict = {"VanGrad": "Vanilla Gradient", "IntGrad": "Integrated Gradient"}
            axes[i, 1].set_title(
                f'{full_name_dict[methods[method_original_i]]} - Lexical Bias',
                fontsize=14)
        axes[i, 1].set_xlabel('Frequency (%)', fontsize=14)
        axes[i, 1].axvline(x=word_baseline_threshold_chance, linestyle="dotted", color="red", linewidth=2)
        axes[i, 1].set_xticks(range(0, 101, 5))
        if adapted_yticks_size:
            axes[i, 1].tick_params(axis='y', labelsize=adapted_yticks_size)
        else:
            axes[i, 1].tick_params(axis='y', labelsize=14)
        axes[i, 1].set_ylim(-0.5, len(word_types) - 0.5)
        axes[i, 1].spines['top'].set_visible(False)
        axes[i, 1].spines['right'].set_visible(False)
        axes[i, 1].set_yticks(list(range(len(word_types))))
        axes[i, 1].set_yticklabels(word_types)

    print("position bias agg", TOPRINT_index_JS)
    print("lexical bias attr", TOPRINT_word_JS)
    print("position bias attr", index_js_avgs)
    print("lexical bias attr", word_js_avgs)

    title_model_name = {"_bert": "BERT", "_modernbert": "ModernBERT", "_llama2": "Llama2"}[model_name]
    if title:
        fig.suptitle(title_model_name + "; " + dataset_name + "; top-k=" + str(k), fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    if adapted_file_name:
        plt.savefig(adapted_file_name)
    else:
        plt.savefig(model_name + "_barplot_" + dataset_name + ".pdf")
    plt.show()

for model_idx, aggregate_hits in enumerate(list_aggregate_hits): # for each model, do...
    for dataset_name, hits in aggregate_hits.items():
        dataset_word_types, dataset_indices = determine_params_from_dataset_name(dataset_name)
        poslex_freq_plot(word_types=dataset_word_types, indices=dataset_indices, hits=hits, model_name=models[model_idx],
                         dataset_name=dataset_name, range_methods_to_include=range_methods_to_include)

#figure 1
poslex_freq_plot(word_types=determine_params_from_dataset_name("period-comma")[0],
                 indices=determine_params_from_dataset_name("period-comma")[1],
                 hits=list_aggregate_hits[1]["period-comma"], #[1] here is model_idx for modernbert
                 model_name=models[1], #[1] here is model_idx for modernbert
                 dataset_name="period-comma",
                 title=False,
                 range_methods_to_include=[2, 4],
                 adapted_fig_size=(10,10),
                 adapted_file_name="figure1.pdf",
                 adapted_yticks_size="50",
                 adapted_subplots_title=True)

"""
Additional statistical tests for OOD / non-random considerations
Null hypothesis: 
    The top-1 selections across seeds are just random noise, with no consistent structure

For each seed:
    - Sample 1000 positions uniformly at random from {0... 19}
    - Convert each random sample into a distribution
    - Compute mean JS across these fake random seeds
    - Repeat many times (e.g. 100 or 1000) to obtain a null distribution

This answers:
    - How much inter-seed JS would we expect if explanations were purely random?

Procedure:   
    1) For a given Model (10 random seeds, 1000 top-1s per random seed) X Method, e.g. BERT X SHAP
        -> compute a single Bias-cons score
        -> e.g. bias-cons-observed = 0.1
    2) Generate 10 fake/null models: 1000 randomly sampled top-1s from 20 possible positions
        -> compute a single Bias-cons score
            -> e.g. bias-cons-null = 0.0001
        -> repeat 100 or 1000 times and store
            -> [bias-cons-null1, bias-cons-null2, ... bias-cons-null1000]
                = [0.0001, 0.001, 0.002, ... ]
    3) Compare bias-cons-observed vs. 1000xbias-cons-null
        -> p-value = P(null JS >= observed JS) 

"""

def compute_relative_distribution(hits_list, num_positions):
    """
    Convert a list of top-1 selections into a relative frequency distribution over positions.
    """
    count_dict = Counter(hits_list)
    # ensure all positions are included
    counts = [count_dict.get(i, 0) for i in range(num_positions)]
    total = sum(counts)
    if total == 0:
        return [0] * num_positions
    rel_distrib = [count / total for count in counts]
    return rel_distrib

def mean_js(list_of_distributions):
    """
    Compute the mean pairwise JS distance between distributions.
    """
    pairs = combinations(list_of_distributions, 2)
    js_values = [jensenshannon(p, q) for p, q in pairs]
    return np.mean(js_values)

def generate_random_hits(num_examples, num_positions):
    """
    Generate a random list of top-1 selections (with replacement).
    """
    return np.random.choice(num_positions, size=num_examples, replace=True).tolist()

def compute_null_bias_cons(num_examples, num_positions, num_seeds, num_iterations=1000):
    """
    Generate a null distribution of Bias-cons values using random selections.
    Returns a list of mean JS values under the null hypothesis.
    """
    null_values = []
    for _ in range(num_iterations):
        # generate random distributions for each seed
        random_distributions = []
        for _ in range(num_seeds):
            random_hits = generate_random_hits(num_examples, num_positions)
            random_distributions.append(compute_relative_distribution(random_hits, num_positions))
        # compute mean JS across seeds
        null_values.append(mean_js(random_distributions))
    return null_values

def compute_bias_cons_and_pvalue(list_all_hits, model_idx, dataset_name, attribution_idx, num_positions,
                                 num_iterations=1000):
    """
    Compute observed Bias-cons for a given model/dataset/method and p-value against random.
    """
    # Step 1: Extract actual per-seed distributions
    actual_distributions = []
    model_hits = list_all_hits[model_idx]  # select model
    for picklename, details in model_hits.items():
        if details["dataset_name"] == dataset_name:
            hits_for_seed = details["hits"][attribution_idx]["indices"]
            actual_distributions.append(compute_relative_distribution(hits_for_seed, num_positions))

    num_seeds = len(actual_distributions)
    num_examples = len(details["hits"][attribution_idx]["indices"])  # 1000

    # Step 2: Compute observed Bias-cons
    observed_bias_cons = mean_js(actual_distributions)

    # Step 3: Generate null distribution
    null_distribution = compute_null_bias_cons(num_examples, num_positions, num_seeds, num_iterations=num_iterations)

    # Step 4: Compute p-value
    p_value = np.mean([val >= observed_bias_cons for val in null_distribution])

    return observed_bias_cons, p_value, null_distribution

# Initialize results storage
bias_cons_results = {
    "observed": {},  # observed Bias-cons
    "p_value": {},  # p-value against random noise
    "null_distribution": {}  # optional, can store for later inspection
}

num_iterations = 1000  # number of random samples for null distribution

for model_idx, model_hits in enumerate(list_all_hits):
    model_name = models[model_idx]  # "_bert" or "_modernbert"
    bias_cons_results["observed"][model_name] = {}
    bias_cons_results["p_value"][model_name] = {}
    bias_cons_results["null_distribution"][model_name] = {}

    for picklename, details in model_hits.items():
        dataset_name = details["dataset_name"]
        if dataset_name not in bias_cons_results["observed"][model_name]:
            bias_cons_results["observed"][model_name][dataset_name] = {}
            bias_cons_results["p_value"][model_name][dataset_name] = {}
            bias_cons_results["null_distribution"][model_name][dataset_name] = {}

        num_positions = len(details["dataset_indices"])
        num_examples = len(details["hits"][0]["indices"])  # top-1 hits, 1000 examples

        for attribution_idx in range_methods_to_include:
            observed, p_val, null_dist = compute_bias_cons_and_pvalue(
                list_all_hits=list_all_hits,
                model_idx=model_idx,
                dataset_name=dataset_name,
                attribution_idx=attribution_idx,
                num_positions=num_positions,
                num_iterations=num_iterations
            )

            bias_cons_results["observed"][model_name][dataset_name][attribution_idx] = observed
            bias_cons_results["p_value"][model_name][dataset_name][attribution_idx] = p_val
            bias_cons_results["null_distribution"][model_name][dataset_name][attribution_idx] = null_dist

print("Done computing Bias-cons and p-values for all models/datasets/methods!")

# Compute effect sizes
def excess_bias_ratio(observed, null_distribution):
    """
    Effect size: how much larger observed Bias-cons is than expected under noise
    """
    null_mean = np.mean(null_distribution)
    return np.inf if null_mean == 0 else observed / null_mean

bias_cons_extra_stats = {"effect_size": {}}

total_tests = 0
for model_name in bias_cons_results["observed"]:
    for dataset_name in bias_cons_results["observed"][model_name]:
        total_tests += len(bias_cons_results["observed"][model_name][dataset_name])

for model_name in bias_cons_results["observed"]:
    bias_cons_extra_stats["effect_size"][model_name] = {}

    for dataset_name in bias_cons_results["observed"][model_name]:
        bias_cons_extra_stats["effect_size"][model_name][dataset_name] = {}

        for attribution_idx in bias_cons_results["observed"][model_name][dataset_name]:
            observed = bias_cons_results["observed"][model_name][dataset_name][attribution_idx]
            null_dist = bias_cons_results["null_distribution"][model_name][dataset_name][attribution_idx]

            #store effect size
            bias_cons_extra_stats["effect_size"][model_name][dataset_name][attribution_idx] = \
                excess_bias_ratio(observed, null_dist)

all_effect_sizes = list(bias_cons_extra_stats['effect_size']['_bert']['noun-det-period'].values()) + \
                   list(bias_cons_extra_stats['effect_size']['_bert']['period-comma'].values()) + \
                   list(bias_cons_extra_stats['effect_size']['_bert']['unique-punctuation'].values()) + \
                   list(bias_cons_extra_stats['effect_size']['_modernbert']['noun-det-period'].values()) + \
                   list(bias_cons_extra_stats['effect_size']['_modernbert']['period-comma'].values()) + \
                   list(bias_cons_extra_stats['effect_size']['_modernbert']['unique-punctuation'].values())

print(np.mean(all_effect_sizes))
print(np.std(all_effect_sizes))
print(np.min(all_effect_sizes))
print(np.max(all_effect_sizes))


"""
Causal data
"""
### SELECT OPTION by uncommenting line #######################################
# onlypositive, onlynegative, onlypositiveornegativedir = False, False, ""
onlypositive, onlynegative, onlypositiveornegativedir = True, False, "onlypositive/"
# onlypositive, onlynegative, onlypositiveornegativedir = False, True, "onlynegative/"
##############################################################################

def extract_sentence_bounds(sentence_bounds_jsonfile, only_positive_class=True, only_negative_class=False):
    with open(sentence_bounds_jsonfile, "r") as f:
        bounds = json.load(f)
    if only_positive_class:
        bounds = bounds[:300] #only take first half, which are the positive class cases
        return bounds
    elif only_negative_class:
        bounds = bounds[300:]  # only take first half, which are the positive class cases
        return bounds
    else:
        return bounds

def determine_sentence_idx_hit(word_onsets, sentence_bounds):
    assert len(word_onsets[0]) == 1  #for now, only works for topk=1
    assert len(sentence_bounds[0]) == 3 #only works for three sentences per input instance
    assert len(word_onsets) == len(sentence_bounds)
    list_sentence_idx_hits = []
    for w, bounds in zip(word_onsets, sentence_bounds):
        word_start = w[0][0]
        if word_start < bounds[0]: #first sentence end
            list_sentence_idx_hits.append(0)
            continue
        elif word_start < bounds[0] + bounds[1] + 1: #first + second sentence + whitespace
            list_sentence_idx_hits.append(1)
            continue
        elif word_start < bounds[0] + bounds[1] + bounds[2] + 2: #first + second + third sentence + 2*whitespace
            list_sentence_idx_hits.append(2)
            continue
    return list_sentence_idx_hits

dataset_names_causal = ["exp1"]

# Create all_hits_causal and plot
def plot_JS_heatmap_causal(*all_js_dicts):
    num_plots = len(all_js_dicts)
    assert 1 <= num_plots <= 3, "Can only plot between 1 and 3 heatmaps."

    fig, axes = plt.subplots(1, num_plots, figsize=(8, 2), squeeze=False)  # wider & taller
    axes = axes[0]

    for i, all_js_dict in enumerate(all_js_dicts):
        datasets = sorted(set([key[0] for key in all_js_dict.keys()]))
        print(datasets)
        methods_list = sorted(set([key[1] for key in all_js_dict.keys()]))
        print(methods_list)

        heatmap_data = []
        for dataset in datasets:
            row = []
            for method in methods_list:
                row.append(all_js_dict.get((dataset, method), None))
            heatmap_data.append(row)

        datasetname_converter = {"exp1": "causal"}

        heatmap_df = pd.DataFrame(
            heatmap_data,
            index=[datasetname_converter[d_name] for d_name in datasets],
            columns=[methods[m] for m in methods_list]
        )

        ax = axes[i]

        sns.heatmap(
            heatmap_df,
            annot=True,
            cmap='YlGnBu',
            cbar=False,  # Disable individual colorbars
            fmt='.2f',
            linewidths=0.5,
            vmin=0,
            vmax=1.0,
            ax=ax,
            square=False
        )

        # Remove leading zeros in annotation text (e.g., 0.34 → .34)
        for text in ax.texts:
            val = text.get_text()
            if val.startswith("0."):
                text.set_text(val[1:])
            elif val.startswith("-0."):
                text.set_text("-" + val[2:])

        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')

        if i == 0:
            # ax.set_ylabel("Datasets")
            ax.set_yticklabels(ax.get_yticklabels(), rotation=30)
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])
            ax.tick_params(left=False)

        # ax.set_xlabel("Attribution Methods")

        model_suffix = models[i] if i < len(models) else whichmodel
        model_suffix_title = {
            "_bert": "BERT",
            "_modernbert": "ModernBERT",
            "_llama2": "Llama2"
        }[model_suffix]
        ax.set_title(model_suffix_title)

    # Add global colorbar using a ScalarMappable
    norm = plt.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap='YlGnBu', norm=norm)
    sm.set_array([])

    divider = make_axes_locatable(axes[-1])
    cax = divider.append_axes("right", size="4%", pad=0.15)  # more space from plot
    cbar = fig.colorbar(sm, cax=cax)
    cbar.outline.set_visible(False)  # remove contour

    plt.tight_layout()

    if onlypositive and not onlynegative:
        filename_append = "POS"
    elif onlynegative and not onlypositive:
        filename_append = "NEG"
    elif not onlypositive and not onlynegative:
        filename_append = "ALL"


    outname = whichmodel + "_sentence_hits_JS_interseed_CAUSAL_" + filename_append
    if num_plots > 1:
        outname = f"_{num_plots}plots_sentence_hits_JS_interseed_CAUSAL_" + filename_append
    print("!!!",num_plots)
    plt.savefig(outname + ".pdf")
    plt.show()

list_all_hits_causal = []  # populate with one `all_hits_causal` per model
list_all_models_js_for_heatmap_INDICES_CAUS = []
for whichmodel in models:
    if onlypositiveornegativedir == "":
        dataset_explanations_file_names_positive = ["onlypositive/"+file for file in
                                                    os.listdir("explanations_causal/" + "onlypositive/") if
                                                    whichmodel in file]
        dataset_explanations_file_names_negative = ["onlynegative/"+file for file in
                                                    os.listdir("explanations_causal/" + "onlynegative/") if
                                                    whichmodel in file]
        dataset_explanations_file_names = dataset_explanations_file_names_positive + dataset_explanations_file_names_negative
    else:
        dataset_explanations_file_names = [onlypositiveornegativedir+file for file in
                                           os.listdir("explanations_causal/" + onlypositiveornegativedir) if
                                           whichmodel in file]
    # Populating `all_hits` with all top-1 hits
    all_hits_causal = dict()
    for file_name in dataset_explanations_file_names:
        print(file_name)
        if "onlypositive/" in file_name:
            sentence_bounds = extract_sentence_bounds("causal_data/exp1/test_sentence_bounds.json",
                                                      only_positive_class=True,
                                                      only_negative_class=False)
        elif "onlynegative/" in file_name:
            sentence_bounds = extract_sentence_bounds("causal_data/exp1/test_sentence_bounds.json",
                                                      only_positive_class=False,
                                                      only_negative_class=True)
        with open("explanations_causal/" + file_name, "rb") as f:
            dataset_explanations = pickle.load(f)
        hits = create_hits(k=k, dataset_explanations=dataset_explanations, return_topk_token_span_onsets=True)
        # dataset_name, dataset_word_types, dataset_indices = determine_params_from_file_name(file_name)
        for method_i, details in hits.items():
            details["sentence_hits"] = determine_sentence_idx_hit(details["word_onsets"], sentence_bounds)

        all_hits_causal[file_name] = {"dataset_name": "exp1",
                                      "hits": hits}
    list_all_hits_causal.append(all_hits_causal)

    # Prepare Bias(cons) for inter-seed JS heatmaps
    # INITIALISE `all_distributions` which will contain either distributions from INDICES (as defined above)
    all_distributions_causal = {(dataset_name, attribution_idx): [] for dataset_name, attribution_idx in
                                list(product(dataset_names_causal, range_methods_to_include))}

    for (dataset_name, attribution_idx) in all_distributions_causal.keys():
        for picklename, details in all_hits_causal.items():
            if details["dataset_name"] == dataset_name:
                distribution = extract_distribution_from_hits(hits=details["hits"],
                                                              words_or_indices_or_sentence_hits="sentence_hits",
                                                              attribution_method_idx=attribution_idx,
                                                              word_types=None,
                                                              indices=None,
                                                              sentence_hits=[0, 1, 2])
                all_distributions_causal[(dataset_name, attribution_idx)].append(distribution)

    print(all_distributions_causal)

    all_js_causal = dict()
    for datasetXattribmethod, list_of_distributions_causal in all_distributions_causal.items():
        print(datasetXattribmethod, round(mean_js(list_of_distributions_causal), 2))
        all_js_causal[datasetXattribmethod] = round(mean_js(list_of_distributions_causal), 2)

    list_all_models_js_for_heatmap_INDICES_CAUS.append(all_js_causal)
    if len(list_all_models_js_for_heatmap_INDICES_CAUS) == len(models):
        plot_JS_heatmap_causal(*list_all_models_js_for_heatmap_INDICES_CAUS)

def get_aggregate_hits_causal(all_hits_causal):
    aggregate_hits_causal = {details["dataset_name"]: {attrib_idx:{"words":[],"indices":[],"sentence_hits":[]}
                                                       for attrib_idx in range_methods_to_include}
                             for run_name, details in all_hits_causal.items()}
    for ds_name in dataset_names_causal:
        for run_name, details in all_hits_causal.items():
            if details["dataset_name"] == ds_name:
                # for attribution_idx in details["hits"]:
                for attribution_idx in range_methods_to_include:
                    aggregate_hits_causal[ds_name][attribution_idx]["words"] += details["hits"][attribution_idx]["words"]
                    aggregate_hits_causal[ds_name][attribution_idx]["indices"] += details["hits"][attribution_idx]["indices"]
                    aggregate_hits_causal[ds_name][attribution_idx]["sentence_hits"] += details["hits"][attribution_idx]["sentence_hits"]
    return aggregate_hits_causal

list_aggregate_hits_causal = [get_aggregate_hits_causal(all_hits_causal=all_hits_causal) for all_hits_causal in list_all_hits_causal] #one per model

def poslex_freq_plot_causal(sentence_hits, hits, model_name, hide_title=False):
    fig, ax = plt.subplots(figsize=(12, 3.5))

    sentence_hits_distributions = []

    # Compute word onset distributions per method
    for i in range_methods_to_include:
        count_dict_sentence_hits = Counter(hits[i]["sentence_hits"])
        sentence_hits_counts = [count_dict_sentence_hits[idx] for idx in sentence_hits]
        total = sum(sentence_hits_counts)
        percentages = [(count / total) * 100 for count in sentence_hits_counts]
        sentence_hits_distributions.append(percentages)

    # Compute inter-method JS averages
    index_js_avgs = [
        average_js_intermethod(sentence_hits_distributions[i], sentence_hits_distributions[:i] + sentence_hits_distributions[i + 1:])
        for i in range(len(sentence_hits_distributions))
    ]

    # Plot all methods in the same barplot (grouped horizontally)
    bar_height = 0.8/ len(sentence_hits_distributions)
    y_positions = list(range(len(sentence_hits)))

    legend_labels = []  # prepare legend labels

    for i, method_original_i in zip(range(len(sentence_hits_distributions)), range_methods_to_include):
        percentages = sentence_hits_distributions[i]
        js = round(jensenshannon(percentages, [100 / len(percentages)] * len(percentages)), 2)
        offset = [(y + (bar_height * (len(sentence_hits_distributions) - 1) / 2)) - i * bar_height for y in y_positions]

        color_palette = ["#2F4F4F", "#696969", "#A9A9A9", "#D3D3D3", "#F5F5F5"]

        hatch_patterns = ['//', '\\\\', '', '||', '---']
        color = color_palette[i % len(color_palette)]
        hatch = hatch_patterns[i % len(hatch_patterns)]

        ax.barh(offset, percentages, height=bar_height, color=color, hatch=hatch)

        label = f"{methods[method_original_i]:<9}  Bias-agg: {remove_zero_add_zero(js):<2}  Bias-attr: {remove_zero_add_zero(round(index_js_avgs[i], 2)):<2}"

        legend_labels.append(label)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(sentence_hits, fontsize=14)
    ax.set_xlim(0, 101)
    ax.set_ylabel('Sentence Position', fontsize=14)
    ax.set_xlabel('Frequency (%)', fontsize=14)

    if onlypositive:
        affix_title = " Positive Class"
    elif onlynegative:
        affix_title = " Negative Class"
    else:
        affix_title = ""

    model_suffix_title = {
        "_bert": "BERT",
        "_modernbert": "ModernBERT",
        "_llama2": "Llama2"
    }[model_name]

    if not hide_title:
        ax.set_title(f'{model_suffix_title} – Sentence Hits{affix_title}')

    ax.axvline(x=round(100 / len(sentence_hits), 5), linestyle="dotted", color="red", linewidth=2, label="_nolegend_")
    ax.set_xticks(range(0, 101, 10))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    from matplotlib.font_manager import FontProperties
    font_prop = FontProperties(family='monospace', size=13)
    ax.legend(legend_labels, loc='upper right', prop=font_prop)

    plt.tight_layout()

    if onlypositive:
        affix = "_onlypositive"
    elif onlynegative:
        affix = "_onlynegative"
    else:
        affix = ""

    plt.savefig(f"{model_name}_sentencehits_barplot_{dataset_name}{affix}.pdf")
    plt.show()

for model_idx, aggregate_hits_causal in enumerate(list_aggregate_hits_causal): # for each model, do...
    for dataset_name, hits in aggregate_hits_causal.items():
        # dataset_word_types, dataset_indices = determine_params_from_dataset_name(dataset_name)
        poslex_freq_plot_causal(sentence_hits=[0,1,2], hits=hits, model_name=models[model_idx], hide_title=True)

with open("causal_data/exp1/test.json", "r") as f:
    test = json.load(f)
lens = []
for x in test:
    lens.append(len((x["text"]).split()))
print(min(lens))
print(max(lens))
print(np.mean(lens))

"""
Sufficiency
"""
with open("sufficiency_scores.json", "r") as f:
    suff_comp_all = json.load(f)

def compute_avg_suff_or_comp_run(suff_comp_all, run_name, method_idx, suff_or_comp):
    suff_or_comp_scores = []
    for x in suff_comp_all[run_name]:
        suff_or_comp_scores.append(x[method_idx][suff_or_comp])
    return np.mean(suff_or_comp_scores)

run_names_bert = [run_name for run_name in suff_comp_all.keys() if "_bert" in run_name]
run_names_modernbert = [run_name for run_name in suff_comp_all.keys() if "_modernbert" in run_name]
run_names_llama2 = [run_name for run_name in suff_comp_all.keys() if "_llama2" in run_name]

def compute_avg_suff_or_comp_model(suff_comp_all, run_names_model, method_idx, suff_or_comp):
    for run_name in run_names_model:
        return compute_avg_suff_or_comp_run(suff_comp_all, run_name, method_idx, suff_or_comp)

for model, run_names_model in zip(["bert", "modernbert", "llama2"],
                                  [run_names_bert, run_names_modernbert, run_names_llama2]):
    print(model)
    # for method_idx in range(1, 6):
    for method_idx in range(6):
        suff = compute_avg_suff_or_comp_model(suff_comp_all, run_names_model, str(method_idx), "suff")
        comp = compute_avg_suff_or_comp_model(suff_comp_all, run_names_model, str(method_idx), "comp")
        print(f"suff: {round(suff,2)}, comp: {round(comp,2)}")
    print()

# """
# Check dataset properties.
# We measure the true distribution of
#     1) words
#     2) words per index
# They should be equally distributed, as per prepare script.
# """
#
# # (1) distribution of words
# dev_words = {"table":0, "the":0, ".":0, "[CLS]":0, "[SEP]":0}
# dev_words_per_idx = {idx:{"table":0, "the":0, ".":0, "[CLS]":0, "[SEP]":0} for idx in range(23)}
# for example in dataset_explanations:
#     for idx,t in enumerate(example[0].tokens):
#         dev_words[t] += 1
#         dev_words_per_idx[idx][t] += 1
# print(dev_words)
# #{'table': 6725, 'the': 6586, '.': 6689, '[CLS]': 1000, '[SEP]': 2000}
#
# # (2) distribution of words per index
# for idx,word in dev_words_per_idx.items():
#     print(idx, "\t", word)
# # 0 	 {'table': 0, 'the': 0, '.': 0, '[CLS]': 1000, '[SEP]': 0}
# # 1 	 {'table': 357, 'the': 300, '.': 343, '[CLS]': 0, '[SEP]': 0}
# # 2 	 {'table': 332, 'the': 331, '.': 337, '[CLS]': 0, '[SEP]': 0}
# # 3 	 {'table': 336, 'the': 336, '.': 328, '[CLS]': 0, '[SEP]': 0}
# # 4 	 {'table': 346, 'the': 316, '.': 338, '[CLS]': 0, '[SEP]': 0}
# # 5 	 {'table': 338, 'the': 305, '.': 357, '[CLS]': 0, '[SEP]': 0}
# # 6 	 {'table': 321, 'the': 365, '.': 314, '[CLS]': 0, '[SEP]': 0}
# # 7 	 {'table': 343, 'the': 319, '.': 338, '[CLS]': 0, '[SEP]': 0}
# # 8 	 {'table': 332, 'the': 336, '.': 332, '[CLS]': 0, '[SEP]': 0}
# # 9 	 {'table': 328, 'the': 340, '.': 332, '[CLS]': 0, '[SEP]': 0}
# # 10 	 {'table': 344, 'the': 327, '.': 329, '[CLS]': 0, '[SEP]': 0}
# # 11 	 {'table': 0, 'the': 0, '.': 0, '[CLS]': 0, '[SEP]': 1000}
# # 12 	 {'table': 324, 'the': 334, '.': 342, '[CLS]': 0, '[SEP]': 0}
# # 13 	 {'table': 313, 'the': 355, '.': 332, '[CLS]': 0, '[SEP]': 0}
# # 14 	 {'table': 350, 'the': 311, '.': 339, '[CLS]': 0, '[SEP]': 0}
# # 15 	 {'table': 329, 'the': 333, '.': 338, '[CLS]': 0, '[SEP]': 0}
# # 16 	 {'table': 361, 'the': 318, '.': 321, '[CLS]': 0, '[SEP]': 0}
# # 17 	 {'table': 339, 'the': 335, '.': 326, '[CLS]': 0, '[SEP]': 0}
# # 18 	 {'table': 347, 'the': 338, '.': 315, '[CLS]': 0, '[SEP]': 0}
# # 19 	 {'table': 336, 'the': 327, '.': 337, '[CLS]': 0, '[SEP]': 0}
# # 20 	 {'table': 322, 'the': 337, '.': 341, '[CLS]': 0, '[SEP]': 0}
# # 21 	 {'table': 327, 'the': 323, '.': 350, '[CLS]': 0, '[SEP]': 0}
# # 22 	 {'table': 0, 'the': 0, '.': 0, '[CLS]': 0, '[SEP]': 1000}




