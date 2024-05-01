# !pip install transformers datasets

import torch, os
import pandas as pd
from transformers import pipeline, BertForSequenceClassification, BertTokenizerFast
from torch.utils.data import Dataset
from imblearn.over_sampling import SMOTE
   
# from torch import cuda
# device = 'cuda' if cuda.is_available() else 'cpu'
# device

import os
import pandas as pd

# Veri çerçevesini oluşturmak için boş bir liste
data = []

# Kategorilerin bulunduğu ana klasörün yolu

import os

# Assuming your folder is uploaded to Google Drive at 'My Drive/path/to/your/folder'
root_dir = './news'

# Ana klasördeki kategorileri dolaşma
for category in os.listdir(root_dir):
    category_dir = os.path.join(root_dir, category)

    # Her kategori klasöründeki haber dosyalarını dolaşma
    for filename in os.listdir(category_dir):
        file_path = os.path.join(category_dir, filename)
        print("Reading file:", file_path)  # Print file path

        # Dosyayı açma ve içeriğini okuma
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Veri listesine kategori ve içerik ekleme
        data.append({'category': category, 'text': content})

# Veri listesinden veri çerçevesi oluşturma
df_org = pd.DataFrame(data)

# Rastgele sıralama
df_org = df_org.sample(frac=1.0, random_state=42)

# İlk beş satırı gösterme
print(df_org.head())

labels = df_org['category'].unique().tolist()
labels = [s.strip() for s in labels ]
labels


for key, value in enumerate(labels):
    print(value)

NUM_LABELS= len(labels)

id2label={id:label for id,label in enumerate(labels)}

label2id={label:id for id,label in enumerate(labels)}

label2id

id2label

df_org.head()

# df_org['labels_num'] = pd.factorize(df_org.category)[0]
# df_org.head()

df_org["labels"]=df_org.category.map(lambda x: label2id[x.strip()])

df_org.head()

df_org.category.value_counts().plot(kind='pie', figsize=(10,10))

tokenizer = BertTokenizerFast.from_pretrained("dbmdz/bert-base-turkish-uncased", max_length=512)

model = BertForSequenceClassification.from_pretrained("dbmdz/bert-base-turkish-uncased", num_labels=NUM_LABELS, id2label=id2label, label2id=label2id)
# model.to(device)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

from imblearn.over_sampling import SMOTE

# Veri setlerini oluşturma
SIZE = df_org.shape[0]

train_texts = list(df_org.text[:SIZE//2])
val_texts = list(df_org.text[SIZE//2:(3*SIZE)//4])
test_texts = list(df_org.text[(3*SIZE)//4:])

train_labels = list(df_org.labels[:SIZE//2])
val_labels = list(df_org.labels[SIZE//2:(3*SIZE)//4])
test_labels = list(df_org.labels[(3*SIZE)//4:])

len(train_texts)

len(train_texts), len(val_texts), len(test_texts)

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings  = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

class DataLoader(Dataset):

    def __init__(self, encodings, labels):
       
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):

        # Retrieve tokenized data for the given index
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # Add the label for the given index to the item dictionary
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        """
        Returns the number of data items in the dataset.

        Returns:
            (int): The number of data items in the dataset.
        """
        return len(self.labels)

train_dataloader = DataLoader(train_encodings, train_labels)

val_dataloader = DataLoader(val_encodings, val_labels)

test_dataset = DataLoader(test_encodings, test_labels)

"""## Training with Trainer Class"""

from transformers import TrainingArguments, Trainer

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    
    # Extract true labels from the input object
    labels = pred.label_ids

    # Obtain predicted class labels by finding the column index with the maximum probability
    preds = pred.predictions.argmax(-1)

    # Compute macro precision, recall, and F1 score using sklearn's precision_recall_fscore_support function
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')

    # Calculate the accuracy score using sklearn's accuracy_score function
    acc = accuracy_score(labels, preds)

    # Return the computed metrics as a dictionary
    return {
        'Accuracy': acc,
        'F1': f1,
        'Precision': precision,
        'Recall': recall
    }


# !pip install transformers[torch]
# !pip install transformers==4.11.3 accelerate==0.21.0
# !pip install transformers==4.11.3 tokenizers==0.11.1


training_args = TrainingArguments(
    output_dir='./outputs',
    do_train=True,
    do_eval=True,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=100,
    weight_decay=0.01,
    logging_strategy='steps',
    logging_dir='./multi-class-logs',
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    fp16=False,  # Disable FP16 mixed precision training
    load_best_model_at_end=True
)


trainer = Trainer(
    # the pre-trained model that will be fine-tuned
    model=model,
     # training arguments that we defined above
    args=training_args,
    train_dataset=train_dataloader,
    eval_dataset=val_dataloader,
    compute_metrics= compute_metrics
)

# trainer.train()

# q=[trainer.evaluate(eval_dataset=df_org) for df_org in [train_dataloader, val_dataloader, test_dataset]]

# pd.DataFrame(q, index=["train","val","test"]).iloc[:,:5]


def predict(text):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs[0].softmax(1)
    pred_label_idx = probs.argmax()
    pred_label = model.config.id2label[pred_label_idx.item()]
    return probs, pred_label_idx, pred_label

"""## Save model for inference"""

# model_path = "turkish-text-classification-model"
# trainer.save_model(model_path)
# tokenizer.save_pretrained(model_path)

"""## Re-Load saved model for inference"""

model_path = "turkish-text-classification-model"


model = BertForSequenceClassification.from_pretrained(model_path).to("cuda")
tokenizer= BertTokenizerFast.from_pretrained(model_path)
nlp= pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# results=trainer.predict(test_dataset)
# print(results)

df_siniflandirma=pd.read_csv("siniflandirma.csv",sep='|')
df_siniflandirma["metin"]=df_siniflandirma["metin"].str.replace("\n"," ")

from tqdm import tqdm

for index,row in tqdm(df_siniflandirma.iterrows(), total=len(df_siniflandirma), desc="Processing Rows"):
    df_siniflandirma.loc[index,"kategori"]=predict(row["metin"])[2]


df_siniflandirma.to_csv("tahmin2.csv",sep="|",index=False)    