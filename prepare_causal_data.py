import json
import os
import random
import numpy as np
from itertools import combinations, permutations

datapath = "./causal_data/explanations_causal/"

## these txt files are TAB-DELIMITED versions of .xlsx files
with open(os.path.join(datapath,"climate_change.txt"), "r", encoding="utf-8") as f:
    climate_change = f.readlines()
with open(os.path.join(datapath,"putinism_policy.txt"), "r", encoding="utf-8") as f:
    putinism_policy = f.readlines()
with open(os.path.join(datapath,"traffic_congestion.txt"), "r", encoding="utf-8") as f:
    traffic_congestion = f.readlines()

def create_text_label_pairs(list_of_raw_lines):
    text_label_pairs = []
    for line in list_of_raw_lines[1:]:
        split_elements = line.split("\t")
        text = split_elements[1].strip()
        label = split_elements[2].strip()
        if len(label) > 0: #exclude instances without annotated label
        text_label_pairs.append((text, int(label)))
    return text_label_pairs

text_label_pairs_CC = create_text_label_pairs(climate_change)
text_label_pairs_PP = create_text_label_pairs(putinism_policy)
text_label_pairs_TC = create_text_label_pairs(traffic_congestion)

text_label_pairs = text_label_pairs_CC + text_label_pairs_PP + text_label_pairs_TC

texts_1 = [t for t,l in text_label_pairs if l==1] #242
texts_0 = [t for t,l in text_label_pairs if l==0] #403

print(np.mean([len(t.split()) for t in texts_1])) #24.6
print(np.min([len(t.split()) for t in texts_1])) #6
print(np.max([len(t.split()) for t in texts_1])) #60
print(np.mean([len(t.split()) for t in texts_0])) #18.4
print(np.min([len(t.split()) for t in texts_0])) #1
print(np.max([len(t.split()) for t in texts_0])) #57

texts_0 = [t for t in texts_0 if len(t)>5] #401

"""
Create triples
Concatenate
"""
random.seed(42)
random.shuffle(texts_1)
random.shuffle(texts_0)

# populate with all unique triples as (abc,def,ghi,...)
def generate_base_triples(texts):
    triples = [tuple(texts[i:i+3]) for i in range(0, len(texts), 3)]
    if len(triples[-1]) < 3: # ensure all triples have exactly 3 elements
        needed = 3 - len(triples[-1])
        triples[-1] += tuple(texts[:needed]) #pad last triple with first elements in texts
    return triples

def generate_combination_triples(texts, n):
    base_triples = generate_base_triples(texts) # e.g. 81 for class1
    all_possible_combination_triples = random.sample(list(combinations(texts, 3)), n-len(base_triples)) # e.g. 500-81=419 for class1
    combination_triples = [set(triple) for triple in base_triples]
    for triple in all_possible_combination_triples:
        if set(triple) not in combination_triples:
            combination_triples.append(set(triple))
    return combination_triples

def generate_permutation_triples(combination_triples):
    permutation_triples = [list(permutations(triple)) for triple in combination_triples]
    return permutation_triples

combination_triples_1 = generate_combination_triples(texts_1, 500)
permutation_triples_1 = generate_permutation_triples(combination_triples_1)
combination_triples_0 = generate_combination_triples(texts_0, 500)
permutation_triples_0 = generate_permutation_triples(combination_triples_0)

train_triples = [(pt, 1) for pt in permutation_triples_1[:450]] + [(pt, 0) for pt in permutation_triples_0[:450]]
test_triples = [(pt, 1) for pt in permutation_triples_1[450:]] + [(pt, 0) for pt in permutation_triples_0[450:]]

train_concatenations = [(" ".join(permutation_tuple), lab) for (tt, lab) in train_triples for permutation_tuple in tt]
test_concatenations = [(" ".join(permutation_tuple), lab) for (tt, lab) in test_triples for permutation_tuple in tt]

# storing indices sentence bounds for analysis later
train_sentence_bounds = [tuple(len(permutation) for permutation in permutation_tuple) for (tt, lab) in train_triples for permutation_tuple in tt]
test_sentence_bounds = [tuple(len(permutation) for permutation in permutation_tuple) for (tt, lab) in test_triples for permutation_tuple in tt]
with open(os.path.join(datapath,"train_sentence_bounds.json"), "w") as file:
    json.dump(train_sentence_bounds, file, indent=3)
with open(os.path.join(datapath,"test_sentence_bounds.json"), "w") as file:
    json.dump(test_sentence_bounds, file, indent=3)

# saving to train and test files
train = [{"text": text, "label": label} for text,label in train_concatenations]
dev = [{"text": text, "label": label} for text,label in test_concatenations]
test = [{"text": text, "label": label} for text,label in test_concatenations]

with open(os.path.join(datapath,"train.json"), "w") as file:
    json.dump(train, file, indent=4)
with open(os.path.join(datapath,"dev.json"), "w") as file:
    json.dump(dev, file, indent=4)
with open(os.path.join(datapath,"test.json"), "w") as file:
    json.dump(test, file, indent=4)

print("done dumping the datasets to json")
