import json
import os
import random
from transformers import BertTokenizer

"""
Data experiment 1:
    * 20 word word sentence (originally with SEP token at position 11, now without)
    * random label between 0, 1 (originally 3 classes, now 2)
    * random words among "table", "the", "." (they can appear 0 or more times)
"""
datapath = "./toy_data/noun_det_punct/"

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# sep_token = tokenizer.sep_token

content = "table"
function = "the"
punctuation = "."

# generates a single instance of length 10 with each token at least once
def generate_single_instance():
    instance = []
    for _ in range(10):
        instance.append(random.choice([content, function, punctuation]))
    random.shuffle(instance)
    return ' '.join(instance)

# generates a combined instance with a separator token
def generate_combined_instance():
    instance1 = generate_single_instance()
    instance2 = generate_single_instance()
    # combined_instance = f"{instance1} {sep_token} {instance2}"
    combined_instance = f"{instance1} {instance2}"
    return combined_instance

# generates the total set with unique instances
total_set = set()
for _ in range(10000):
    combined_instance = generate_combined_instance()
    total_set.add(combined_instance)

# converts set to list and assigns random labels
total_set = list(total_set)
labels = [0 for _ in range(int(len(total_set)/2))] + [1 for _ in range(int(len(total_set)/2))]
random.shuffle(labels)
assert len(total_set) == len(labels)

# combines instances with their labels
total_set_with_labels = list(zip(total_set, labels))

# prints a few examples to verify
random.shuffle(total_set_with_labels) #just for visualisation we shuffle
for i in range(5):
    print(total_set_with_labels[i])

# 0% rationales
train = [{"text": text, "label": label} for text,label in total_set_with_labels[:8000]]
dev = [{"text": text, "label": label} for text,label in total_set_with_labels[8000:9000]]
test = [{"text": text, "label": label} for text,label in total_set_with_labels[9000:]]

with open(os.path.join(datapath,"train.json"), "w") as file:
    json.dump(train, file, indent=4)
with open(os.path.join(datapath,"dev.json"), "w") as file:
    json.dump(dev, file, indent=4)
with open(os.path.join(datapath,"test.json"), "w") as file:
    json.dump(test, file, indent=4)

print("done dumping the datasets to json")

"""
Data experiment 2 -- `punct_comma_random`:
    * 20 word sentence
    * 50% label 0, 50% label 1
    * random combinations of "." and ","
"""

datapath = "./toy_data/punct_comma_random/"

def generate_single_instance():
    instance = []
    for _ in range(20):
        instance.append(random.choice([".", ","]))
    return ' '.join(instance)

# generates the total set with unique instances (in this case, they are all equal)
total_set = list()
for _ in range(10000):
    instance = generate_single_instance()
    total_set.append(instance)

# converts set to list and assigns random labels
labels = [0 for _ in range(int(len(total_set)/2))] + [1 for _ in range(int(len(total_set)/2))]
assert len(total_set) == len(labels)

# combines instances with their labels
total_set_with_labels = list(zip(total_set, labels))
random.shuffle(total_set_with_labels) #just for visualisation

# prints a few examples to verify
for i in range(5):
    print(total_set_with_labels[i])

# 0% rationales
train = [{"text": text, "label": label} for text,label in total_set_with_labels[:8000]]
dev = [{"text": text, "label": label} for text,label in total_set_with_labels[8000:9000]]
test = [{"text": text, "label": label} for text,label in total_set_with_labels[9000:]]

with open(os.path.join(datapath,"train.json"), "w") as file:
    json.dump(train, file, indent=4)
with open(os.path.join(datapath,"dev.json"), "w") as file:
    json.dump(dev, file, indent=4)
with open(os.path.join(datapath,"test.json"), "w") as file:
    json.dump(test, file, indent=4)

print("done dumping the datasets to json")

"""
Data experiment 3 -- `unique_punctuation_marks_random`:
    * 20 word sentence
    * 50% label 0, 50% label 1
    * random combinations of tokens, each must occur once per instance 
            [".", ",", ";", ":", "!", "?", "-", "_", "(", ")", 
            "[", "]", "{", "}", "/", "*", "#", "'", '"', "`"]
"""

datapath = "./toy_data/unique_punctuation_marks_random/"

def generate_single_instance():
    instance = [".", ",", ";", ":", "!", "?", "-", "_", "(", ")",
                "[", "]", "{", "}", "/", "*", "#", "'", '"', "`"]
    random.shuffle(instance)
    return ' '.join(instance)

# generates the total set with unique instances (in this case, they are all equal)
total_set = list()
c = 0
while c < 10000:
    instance = generate_single_instance()
    if instance not in total_set:
        total_set.append(instance)
        c+=1
    print(c)

# converts set to list and assigns random labels
labels = [0 for _ in range(int(len(total_set)/2))] + [1 for _ in range(int(len(total_set)/2))]
assert len(total_set) == len(labels)

# combines instances with their labels
total_set_with_labels = list(zip(total_set, labels))
random.shuffle(total_set_with_labels) #just for visualisation

# prints a few examples to verify
for i in range(5):
    print(total_set_with_labels[i])

# 0% rationales
train = [{"text": text, "label": label} for text,label in total_set_with_labels[:8000]]
dev = [{"text": text, "label": label} for text,label in total_set_with_labels[8000:9000]]
test = [{"text": text, "label": label} for text,label in total_set_with_labels[9000:]]

with open(os.path.join(datapath,"train.json"), "w") as file:
    json.dump(train, file, indent=4)
with open(os.path.join(datapath,"dev.json"), "w") as file:
    json.dump(dev, file, indent=4)
with open(os.path.join(datapath,"test.json"), "w") as file:
    json.dump(test, file, indent=4)

print("done dumping the datasets to json")