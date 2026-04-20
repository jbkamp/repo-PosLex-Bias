import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1' #for annoying `NotImplementedError: The operator 'aten::.....` mps error
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" #for cuda to save memory when computing explantions llama2
import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from ferret import Benchmark
import pickle
import argparse


################################################################
parser = argparse.ArgumentParser(description="Finetune a Bert model on a given dataset")
parser.add_argument('--excludellama2', type=str, required=True,
                    help="Flag to exclude llama2 computations which are sometimes too big for a gpu. It will then only loop over bert and modernbert runs")
parser.add_argument('--excludebert', type=str, required=True)
parser.add_argument('--onlypositiveclass', type=str, required=True,
                    help="Flag to only evaluate positive examples, e.g. causal instances.")
parser.add_argument('--onlynegativeclass', type=str, required=True,
                    help="Flag to only evaluate negative examples, e.g. NOT causal instances.")
parser.add_argument('--specific_dataset', type=str, required=False, default=None,
                    help="Specify a single dataset instead of looping over all.")
args = parser.parse_args()

excludellama2 = args.excludellama2
assert excludellama2 in {"true", "false"}

excludebert = args.excludebert
assert excludebert in {"true", "false"}

onlypositiveclass = args.onlypositiveclass
assert onlypositiveclass in {"true", "false"}

onlynegativeclass = args.onlynegativeclass
assert onlynegativeclass in {"true", "false"}

assert (onlypositiveclass, onlynegativeclass) != ("true", "true")

if onlypositiveclass == "true":
    onlypositiveornegativedir = "onlypositive/"
elif onlynegativeclass == "true":
    onlypositiveornegativedir = "onlynegative/"
else:
    onlypositiveornegativedir = ""

specific_dataset = args.specific_dataset
################################################################
dev_or_test, n_labels = "test", 2
# runs = {'7voftwv9_dandy-glade-8_bert_punct_only',
#         '3rz9qjhm_scarlet-silence-9_bert_punct_comma_fixed'}
runs = [run for run in os.listdir("runs/results/") if not run.startswith("checkpoint")]
if excludellama2 == "true":
    runs = [run for run in runs if "llama2" not in run]
if excludebert == "true":
    runs = [run for run in runs if "_bert_" not in run]
################################################################

# mps is giving annoying errors. Solved it by adding
#  1) model.to(torchfloat32) AND
#  2) * torch.tensor(step_sizes, dtype=torch.float32).view(n_steps, 1).to(grad.device)
#        --> to make it work, I added the dtype=torch.float32 part to the captum part of integrated gradients script
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu':
    device = 'mps' if torch.has_mps else 'cpu' #m1 mac

print(device)

"""
run attribution methods
"""
def get_dataset_explanations(dataset, model, tokenizer, run_name):
    """
    RUN once
    :param dataset: e.g. `dev`, as [(text,label)1, (text,label)2, ... , (text,label)N]
    :param model: the pretrained huggingface model
    :param tokenizer: the pretrained huggingface tokenizer
    :return: list of ferret explanation objects, one for each instance in the dataset provided
    """
    bench = Benchmark(model=model, tokenizer=tokenizer)
    dataset_explanations = []
    for i,x in enumerate(dataset):
        X = x[0] # text
        y = x[1] # label
        instance_explanations = bench.explain(X, target=y)
        dataset_explanations.append(instance_explanations)
        print(run_name, i)
    return dataset_explanations

# pickle
for run_name in runs:
    # determine run name and paths
    print("run_name:", run_name)
    dataset_name = "_".join(run_name.split("_")[3:])
    if dataset_name == "exp1":  # causal dataset
        explanations_path = "explanations_causal/"
        dataset_path = "causal_data/" + dataset_name + "/"
    else:
        explanations_path = "explanations/"
        dataset_path = "toy_data/" + dataset_name + "/"
    # loop over runs to check various conditions
    print("Search if", "./"+explanations_path+onlypositiveornegativedir+dev_or_test+"_dataset_explanations_"+run_name+".pickle", "exists...")
    if os.path.exists("./"+explanations_path+onlypositiveornegativedir+dev_or_test+"_dataset_explanations_"+run_name+".pickle"):
        print(run_name, "already has explanations computed, continuing loop")
    else:
        if specific_dataset:
            if dataset_name != specific_dataset:
                print("Run does NOT match `specific_dataset` arg", specific_dataset, ": skip.")
                continue
            else:
                print(print("Run matches `specific_dataset` arg", specific_dataset))
        # load data
        with open(dataset_path+"train.json", "r") as file:
            train_dict = json.load(file)
            train = [(x["text"], x["label"]) for x in train_dict]  # convert to list-of-tuples format
        with open(dataset_path+"dev.json", "r") as file:
            dev_dict = json.load(file)
            dev = [(x["text"], x["label"]) for x in dev_dict]
            if onlypositiveclass == "true":
                dev = [(text, label) for (text, label) in dev if label == 1]
            elif onlynegativeclass == "true":
                dev = [(text, label) for (text, label) in dev if label == 0]
        with open(dataset_path+"test.json", "r") as file:
            test_dict = json.load(file)
            test = [(x["text"], x["label"]) for x in test_dict]
            if onlypositiveclass == "true":
                test = [(text, label) for (text, label) in test if label == 1]
            elif onlynegativeclass == "true":
                test = [(text, label) for (text, label) in test if label == 0]
        # load model
        if "modernbert" in run_name:
            from transformers import ModernBertForSequenceClassification, AutoTokenizer
            model = ModernBertForSequenceClassification.from_pretrained("./runs/results/"+run_name)
            tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        elif "bert" in run_name:
            model = BertForSequenceClassification.from_pretrained("./runs/results/"+run_name)
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        elif "llama2" in run_name:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
            # from peft import LoraConfig, get_peft_model
            # quantization_config = BitsAndBytesConfig(
            #     load_in_4bit=True,
            #     bnb_4bit_compute_dtype=torch.float16,
            #     bnb_4bit_use_double_quant=True)
            # model = AutoModelForSequenceClassification.from_pretrained("meta-llama/Llama-2-7b-hf", num_labels=n_labels,
            #                                                            quantization_config=quantization_config,
            #                                                            device_map="auto")
            # lora_config = LoraConfig(
            #     r=16,  # rank of adaptation matrix
            #     lora_alpha=32,  # scaling
            #     lora_dropout=0.05,
            #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # applied to specific attention layers
            # )
            # model = get_peft_model(model, lora_config)
            # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
            # tokenizer.pad_token = tokenizer.eos_token
            # model.config.pad_token_id = model.config.eos_token_id
            #
            # # ADD MASK TOKEN NEEDED FOR LIME
            # tokenizer.add_special_tokens({'mask_token': '[MASK]'})
            # if device == "cuda":
            #     model.resize_token_embeddings(len(tokenizer))
            # model.config.mask_token_id = tokenizer.mask_token_id
            #
            # # reduce model max length to 30; other: skip lime and shap and compute separately.
            # tokenizer.model_max_length = 30

            from peft import PeftModel

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True)

            tokenizer = AutoTokenizer.from_pretrained("./runs/results/"+run_name)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.add_special_tokens({'mask_token': '[MASK]'})

            base_model = AutoModelForSequenceClassification.from_pretrained("meta-llama/Llama-2-7b-hf",
                                                                            quantization_config=quantization_config,
                                                                            device_map="auto")

            base_model.resize_token_embeddings(len(tokenizer))

            model = PeftModel.from_pretrained(base_model, "./runs/results/"+run_name)

            model.config.pad_token_id = model.config.eos_token_id
            model.config.mask_token_id = tokenizer.mask_token_id

        model.to(torch.float32)
        model.to(device)
        model.eval()
        model.zero_grad()
        # compute explanations
        if dev_or_test=="dev":
            dataset_explanations = get_dataset_explanations(dev, model, tokenizer, run_name)
        elif dev_or_test=="test":
            dataset_explanations = get_dataset_explanations(test, model, tokenizer, run_name)
        # pickle explanations
        with open("./"+explanations_path+onlypositiveornegativedir+dev_or_test+"_dataset_explanations_"+run_name+".pickle", "wb") as file:
            pickle.dump(dataset_explanations, file)
        print(run_name + " ---------------------> done")