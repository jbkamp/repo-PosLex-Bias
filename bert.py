import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" #for cuda to save memory when computing explantions llama2
import torch, gc
gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
#MOVED DOWN in the code ---> from transformers import ModernBertTokenizer, ModernBertForSequenceClassification
from datasets import Dataset #from Huggingface
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# from sklearn.model_selection import train_test_split
import json
import wandb
import argparse

################################################################
parser = argparse.ArgumentParser(description="Finetune a Bert model on a given dataset")
parser.add_argument('--whichmodel', type=str, required=True, help="Model name, e.g. BERT or ModernBERT")
parser.add_argument('--dataset_path', type=str, required=True, help="Path of the dataset")
parser.add_argument('--n_labels', type=int, required=True, help="Number of class labels (usually 2)")
args = parser.parse_args()

whichmodel = args.whichmodel
assert whichmodel in {"bert", "modernbert", "llama2"}
n_labels = args.n_labels
dataset_path = args.dataset_path.strip("/") + "/" #make sure it has a trailing "/"
dataset_name = dataset_path.split("/")[-2]

with open(dataset_path+"train.json", "r") as file:
    train_dict = json.load(file)
    train = [(x["text"], x["label"]) for x in train_dict] #convert to list-of-tuples format
with open(dataset_path+"dev.json", "r") as file:
    dev_dict = json.load(file)
    dev = [(x["text"], x["label"]) for x in dev_dict]
with open(dataset_path+"test.json", "r") as file:
    test_dict = json.load(file)
    test = [(x["text"], x["label"]) for x in test_dict]

# max len for truncation
MAX_LEN=0
for text,label in train+dev+test:
    n_tokens = len(text.split())
    if n_tokens > MAX_LEN:
        MAX_LEN=n_tokens
print(MAX_LEN) #113
MAX_LEN += 10 #add some room for segmentation error as sent.split() tokenization is not perfect
MAX_LEN = min(512, MAX_LEN)

#sep_token = tokenizer.sep_token

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu':
    device = 'mps' if torch.has_mps else 'cpu' #m1 mac
print(device)

## Convert to Huggingface input format: raw datasets (lists) --> Dataset objects --> tokenized --> tensored
dataset = {"train":Dataset.from_list([{"text":Xi,"label":yi} for (Xi,yi) in train]),
           "dev":Dataset.from_list([{"text":Xi,"label":yi} for (Xi,yi) in dev]),
           "test":Dataset.from_list([{"text":Xi,"label":yi} for (Xi,yi) in test])}

if whichmodel == "bert":
    from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=n_labels)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
elif whichmodel == "modernbert":
    from transformers import AutoTokenizer, ModernBertForSequenceClassification, Trainer, TrainingArguments
    model = ModernBertForSequenceClassification.from_pretrained("answerdotai/ModernBERT-base",num_labels=n_labels)
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
elif whichmodel == "llama2":
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig, Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True)

    model = AutoModelForSequenceClassification.from_pretrained("meta-llama/Llama-2-7b-hf", num_labels=n_labels,
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


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LEN) # setting padding to `longest` did not work.
    # Looked for alternative solution to speed up training. The max_length argument controls the length
    # of the padding and truncation. It can be an integer or None, in which case it will default to the maximum length the
    # model can accept. If the model has no specific maximum input length, truncation or padding to max_length is deactivated.

dataset_tokenized = {"train":dataset["train"].map(tokenize_function, batched=True),
                     "dev":dataset["dev"].map(tokenize_function, batched=True),
                     "test":dataset["test"].map(tokenize_function, batched=True)}

dataset_tensored = {"train":dataset_tokenized["train"].with_format("torch", columns=["text","label","input_ids","attention_mask"], device=device),
                    "dev":dataset_tokenized["dev"].with_format("torch", columns=["text","label","input_ids","attention_mask"], device=device),
                    "test":dataset_tokenized["test"].with_format("torch", columns=["text","label","input_ids","attention_mask"], device=device)}

# define accuracy metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro") #average=None for per-class scores
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# to avoid attribution error when running on cuda; default is True and does not give problems on mps/m1/mac
pin_memory = False if device=="cuda" else True
load_best_model = False if whichmodel == "llama2" else True #eval loss is computed differently for 4bit quantization

training_args = TrainingArguments(
    output_dir = './runs/results',
    num_train_epochs = 5,
    per_device_train_batch_size = 4, #bert, modernbert: 32. default 8; roberta paper: 16/32
    #per_device_eval_batch_size = 8, #default 8
    #fp16 = True, # WORKS ON CUDA, NOT ON MPS. speed-up, especially for small batch sizes: https://huggingface.co/docs/transformers/v4.18.0/en/performance
    #gradient_accumulation_steps = 16, #default 1; good for memory, but slows down training a bit: https://huggingface.co/docs/transformers/v4.18.0/en/performance
    #gradient_checkpointing = True #default False; slows down training but frees memory; especially for big models https://huggingface.co/docs/transformers/v4.18.0/en/performance
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    #metric_for_best_model="loss", #default "loss"
    load_best_model_at_end = load_best_model, #saves model at best epoch --> needed for wandb
    warmup_steps = 189, #default 0; 1 epoch =>[837 w batchsize = 8; 837/2 = 419 w batchsize = 16]. warmupsteps in roberta paper: 6% of total steps. e.g. 10epc,bsz16: (419*10)*6/100 = 251
    weight_decay = 0.01, #default 0
    learning_rate = 5e-6, #float(sys.argv[1]), #default 5e-5; 1e-5 is used in roberta paper; 1e-5 -> 1e-6 interval works best based on online forums
    logging_strategy = "steps",
    #logging_steps = 8, #default 500
    logging_dir = './runs/logs', #logging_dir (str, optional) — TensorBoard log directory. Will default to *output_dir/runs/CURRENT_DATETIME_HOSTNAME*.
    #dataloader_num_workers = 8, #default 0
    report_to = "wandb",
    #run_name = 'roberta-classification',
    dataloader_pin_memory = pin_memory
)

trainer = Trainer(
    model = model,
    args = training_args,
    compute_metrics = compute_metrics,
    train_dataset = dataset_tensored["train"],
    eval_dataset = dataset_tensored["dev"],
)

# INITIATE WANDB PROJECT
wandb.init(project="wandb_proj_poslex_bias")

trainer.train()
trainer.evaluate()

## Saving model locally: https://pytorch.org/tutorials/beginner/saving_loading_models.html
results_path = './runs/results/'
run_name = wandb.run.name #human-readible name as given on wandb dashboard
run_id = wandb.run.id #id as given in wandb run subdirs
out_path = os.path.join(results_path, "{}_{}_{}_{}".format(run_id, run_name, whichmodel, dataset_name))
os.mkdir(out_path)
print("{} created".format(out_path))
trainer.save_model(out_path)

## Saving the data_original splits
data_path = os.path.join(out_path, "data_original/")
os.mkdir(data_path)
with open(data_path + "train.json", "w") as outfile:
    json.dump(dict(train), outfile)
with open(data_path + "dev.json", "w") as outfile:
    json.dump(dict(dev), outfile)
with open(data_path + "test.json", "w") as outfile:
    json.dump(dict(test), outfile)