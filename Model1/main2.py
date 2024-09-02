import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")
ROOT = os.getcwd().split('\\LLM')[0] + '\\LLM\\data\\'
supplier_df = pd.read_pickle(os.path.join(ROOT, "supplier_df.pkl"))
category_df = pd.read_pickle(os.path.join(ROOT, "category_df.pkl"))
mapping_df = pd.read_pickle(os.path.join(ROOT, "mapping_df.pkl"))

mapping_df = mapping_df[['Артикул', 'type_id', 'description_category_id']]
mapping_df.drop_duplicates(inplace=True)

vcdf = mapping_df['Артикул'].value_counts()
vcdf = vcdf[vcdf==1]
articuls = vcdf.index.tolist()

clean_mapping_df = mapping_df[mapping_df['Артикул'].isin(articuls)]

clean_supplier_df = supplier_df[supplier_df['Код артикула'].isin(clean_mapping_df['Артикул'])]
clean_supplier_df.loc[:, 'text'] = clean_supplier_df.apply(lambda row: '/'.join(row[['Название', 'Группа товаров', 'Раздел']].tolist()), axis=1)
clean_supplier_df.rename(columns={'Код артикула':'articul'}, inplace=True)

prepare_supplier_df = clean_supplier_df[['articul', 'text']]
if prepare_supplier_df.shape[0] != clean_supplier_df.shape[0]:
  print('ERROR: не все товары были найдены у поставщика')

def get_category_text(v) -> str | None:
    type_id = v['type_id']
    description_category_id = v['description_category_id']
    filtered_category_df = category_df[((category_df['2_type_id'] == type_id) & (category_df['1_description_category_id'] == description_category_id))]
    if filtered_category_df.empty:
      print(f'NOT FOUND ERROR: {type_id}')
      return None
    elif filtered_category_df.shape[0] > 1:
      print(f'NOT UNIQUE ERROR: {type_id}')
    else:
      return '/'.join(filtered_category_df.iloc[0][['0_category_name', '1_category_name', '2_type_name']].tolist())

test_data = pd.DataFrame({
    'text': clean_mapping_df.merge(right=prepare_supplier_df, left_on='Артикул', right_on='articul', how='left')['text'].tolist(),
    'label': clean_mapping_df[['description_category_id', 'type_id']].apply(get_category_text, axis=1).tolist()
})

test_data = test_data.iloc[:500, :]

test_data['label_1'] = test_data['label'].apply(lambda row: row.split('/')[0])
test_data['label_2'] = test_data['label'].apply(lambda row: row.split('/')[1])
test_data['label_3'] = test_data['label'].apply(lambda row: row.split('/')[2])

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# Подготовка данных
df = test_data
model_name = "google/flan-t5-small"

# Преобразуем категории в числовые метки
def my_mapper(x):
    if x == "Строительство и ремонт":
        return 1
    elif x == "Дом и сад":
        return 2
    else:
        return 0
label_mapping = {label: my_mapper(label) for label in df['label_1'].unique()}
print(len(label_mapping))
df['label_id'] = df['label_1'].map(label_mapping)

# 2. Подготовка данных для тренировки
train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'].tolist(),
                                                                    df['label_id'].tolist(),
                                                                    test_size=0.2,
                                                                    random_state=42)
tokenizer = AutoTokenizer.from_pretrained(model_name)
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=256)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=256)

# Создание тензоров датасета
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

train_dataset = Dataset(train_encodings, train_labels)
val_dataset = Dataset(val_encodings, val_labels)

# 4. Модель и тренировка
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_mapping))

training_args = TrainingArguments(
    output_dir='./model_1',
    learning_rate=1e-2,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    disable_tqdm=False,  # Включаем отображение прогресса в консоли
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

# 5. Тестирование на новых данных
test_texts = test_data.iloc[:5, 0].tolist()
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=256, return_tensors="pt")

# Предсказание
model.eval()
with torch.no_grad():
    outputs = model(**test_encodings)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

# Выводим результаты
for text, pred in zip(test_texts, predictions):
    print(f"Товар: {text} -> Предсказанная категория: {list(label_mapping.keys())[pred]}")
print()

model.save_pretrained('./model_1')
tokenizer.save_pretrained('./model_1')
