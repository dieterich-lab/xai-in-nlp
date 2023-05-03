import csv

id2tag = {0: 'Anrede', 1: 'Diagnosen', 2: 'AllergienUnverträglichkeitenRisiken', 3: 'Anamnese', 4: 'Medikation', 5: 'KUBefunde', 6: 'Befunde', 7: 'EchoBefunde', 8: 'Zusammenfassung', 9: 'Mix', 10: 'Abschluss'}
tag2id = {tag: id for id, tag in id2tag.items()}

# Get list of train data with sentences and their labels:
print(" Loading Train & Dev Data..")
labels, sents = [], []

###TRAIN###
with open("./doctoral_letters/MIEdeep/data/PETsectionclass/full/full_main.tsv") as f:
    reader = csv.reader(f, delimiter="\t")
    for i, line in enumerate(reader):
        sent, label = line[0], line[1]
        sents.append(sent)
        labels.append(label)
    print(sent, label)
assert len(labels) == len(sents)

labels = [tag2id[label] for label in labels]

print(f"Number of sentences: {len(sents)}")
avg_len = sum([len(i) for i in sents]) / len(sents)
print(f"Average sentence length of sentences: {round(avg_len)}")

# Load Model and Dependencies
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-german-cased", max_len = 128)

from sklearn.model_selection import train_test_split

train_sents, dev_sents, train_lab, dev_lab  = train_test_split(sents, labels, test_size=0.1, random_state=123)
print(f"Train data splitted into Train set of length: {len(train_sents)} and Dev set of length: {len(dev_sents)}")
assert len(train_sents) == len(train_lab)
assert len(dev_sents) == len(dev_lab)


###TEST###
print(" Loading Test Data..")
sents, labels = [], []

with open("./doctoral_letters/MIEdeep/data/PETsectionclass/full/full_heldout.tsv") as f:
    reader = csv.reader(f, delimiter="\t")
    for i, line in enumerate(reader):
        sent, label = "".join(line[:1]), "".join(line[1:])
        sents.append(sent)
        labels.append(label)
    print(sent, label)
assert len(labels) == len(sents)

labels = [tag2id[label] for label in labels]

test_sents = sents
test_lab = labels

print(" Tokenizing Sentences..")
train_encodings = tokenizer(train_sents, truncation=True, padding=True)
dev_encodings = tokenizer(dev_sents, truncation=True, padding=True)
test_encodings = tokenizer(test_sents, truncation=True, padding=True)

print(f" Data Loaded:\n\n {train_sents[0]} \n Tokenized: {tokenizer.tokenize(train_sents[0])} \n with IDs: {tokenizer.convert_tokens_to_ids(tokenizer.tokenize(train_sents[0]))}")


import torch, tensorflow as tf
from torch.utils.data import DataLoader
torch.cuda.empty_cache()
print(f" Feeding Data and Loading Model..")

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# Create the DataLoader for our train, dev & test sets.

batch_size = 5
train_data = Dataset(train_encodings, train_lab)
dev_data = Dataset(dev_encodings, dev_lab)


train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
dev_dataloader = DataLoader(dev_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

print(f" Data fed:\n\n{train_data[0]}")


# Checking for the GPU
device_name = tf.test.gpu_device_name()
print(device_name)
device = torch.device("cuda")

from transformers import BertForSequenceClassification, AdamW, BertConfig

# Load BertForSequenceClassification, the pretrained BERT model with a single 
# linear classification layer on top. 
model = BertForSequenceClassification.from_pretrained("bert-base-german-cased", num_labels = 11)


# Telling the model to run on GPU
model.cuda()

from transformers import Trainer, TrainingArguments
training_args = TrainingArguments(
    output_dir="./BertSeqCA",
    metric_for_best_model='eval_loss',
    evaluation_strategy='steps',
    load_best_model_at_end = True,
    num_train_epochs=2, 
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    do_train=True, 
    do_eval=True,
    overwrite_output_dir=True)

trainer = Trainer(
    model=model,
    args=training_args, 
    train_dataset=train_data,
    eval_dataset=dev_data,
    tokenizer = tokenizer)

# Start Training:
trainer.train()

###EVAL###
model.eval()

from seqeval.metrics import f1_score
import numpy as np

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

predictions = []
true_labels = []
eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0


import tqdm

for d, batch in enumerate(tqdm.tqdm(test_dataloader)):
    if True:
        b_input_ids, b_input_mask, b_labels =  batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device) 

        with torch.no_grad():
            tmp_eval_loss = model(b_input_ids, token_type_ids=None,
                                  attention_mask=b_input_mask, labels=b_labels)
            logits = model(b_input_ids, token_type_ids=None,
                           attention_mask=b_input_mask)

        logits = logits.logits.to('cpu')
        predictions.extend([np.argmax(logits, axis=1)])
        label_ids = b_labels.to('cpu').numpy()
        true_labels.extend(label_ids)

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        eval_loss += tmp_eval_loss.logits.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += b_input_ids.size(0)
        nb_eval_steps += 1

preds = []
for pred in predictions:
    for p in pred:
        preds.append(id2tag[p.item()])

trues = []
for tr in true_labels:
    trues.append(id2tag[tr])

from sklearn.metrics import classification_report, accuracy_score

labels = [id2tag[l] for l in test_lab] # Convert back to labels
print(classification_report(trues, preds, labels=labels, zero_division=0))#, labels=list(unique_tags)))
print(accuracy_score(trues, preds))

from sklearn.metrics import confusion_matrix
import numpy as np

#Get the confusion matrix
cm = confusion_matrix(trues, preds)
print(cm)

cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm)

print(cm.diagonal())
