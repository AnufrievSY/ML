# import pandas as pd
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
# from sklearn.model_selection import train_test_split
# import warnings
# warnings.filterwarnings("ignore")
# import logging
# logging.getLogger("transformers").setLevel(logging.ERROR)
#
# # 1. Создание простого тренировочного датасета
# data = {
#     "text": [
#         "Компьютер Apple", "Телевизор Samsung", "Дрель Bosch",
#         "Компьютер Lenovo", "Домашний кинотеатр Sony", "Шуруповерт Makita",
#         "Ноутбук HP", "Кухонный гарнитур", "Телевизор LG",
#         "Ударная дрель", "Садовый домик", "Игровой ПК",
#     ],
#     "label": [
#         "Компьютеры", "Телевизоры", "Дрели",
#         "Компьютеры", "Телевизоры", "Дрели",
#         "Компьютеры", "Дом", "Телевизоры",
#         "Дрели", "Дом", "Компьютеры",
#     ]
# }
#
# df = pd.DataFrame(data)
#
# # Преобразуем категории в числовые метки
# label_mapping = {label: idx for idx, label in enumerate(df['label'].unique())}
# df['label_id'] = df['label'].map(label_mapping)
#
# # 2. Подготовка данных для тренировки
# train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'].tolist(), df['label_id'].tolist(), test_size=0.2, random_state=42)
#
# # 3. Тестируем 3 модели
# model_names = [
#     "google/flan-t5-small",  # Flan-T5
#     "papluca/xlm-roberta-base-language-detection",  # https://huggingface.co/papluca/xlm-roberta-base-language-detection
#     "amberoad/bert-multilingual-passage-reranking-msmarco",  # https://huggingface.co/amberoad/bert-multilingual-passage-reranking-msmarco
#     "microsoft/Multilingual-MiniLM-L12-H384",  # https://huggingface.co/microsoft/Multilingual-MiniLM-L12-H384
#     "s-nlp/russian_toxicity_classifier",  # https://huggingface.co/s-nlp/russian_toxicity_classifier
#     "apanc/russian-inappropriate-messages",  # https://huggingface.co/apanc/russian-inappropriate-messages
#     "apanc/russian-sensitive-topics",  # https://huggingface.co/apanc/russian-sensitive-topics
#     "blanchefort/rubert-base-cased-sentiment",  # https://huggingface.co/blanchefort/rubert-base-cased-sentiment
# ]
#
# # Запускаем классификацию для каждой модели
# for model_name in model_names:
#   print(f"Results for model: {model_name}")
#   try:
#       tokenizer = AutoTokenizer.from_pretrained(model_name)
#
#       # Токенизация данных
#       train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
#       val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)
#
#       # Создание тензоров датасета
#       class Dataset(torch.utils.data.Dataset):
#           def __init__(self, encodings, labels):
#               self.encodings = encodings
#               self.labels = labels
#
#           def __getitem__(self, idx):
#               item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#               item['labels'] = torch.tensor(self.labels[idx])
#               return item
#
#           def __len__(self):
#               return len(self.labels)
#
#       train_dataset = Dataset(train_encodings, train_labels)
#       val_dataset = Dataset(val_encodings, val_labels)
#
#       # 4. Модель и тренировка
#       model = AutoModelForSequenceClassification.from_pretrained(model_name,
#                                                                  num_labels=len(label_mapping),
#                                                                  ignore_mismatched_sizes=True)
#
#       training_args = TrainingArguments(
#           output_dir='./results',
#           learning_rate=2e-5,
#           per_device_train_batch_size=4,
#           per_device_eval_batch_size=4,
#           num_train_epochs=3,
#           weight_decay=0.01,
#           eval_strategy="epoch",
#       )
#
#       trainer = Trainer(
#           model=model,
#           args=training_args,
#           train_dataset=train_dataset,
#           eval_dataset=val_dataset
#       )
#
#       trainer.train()
#
#       # 5. Тестирование на новых данных
#       test_texts = ["Игровой компьютер", "Мощная дрель", "Большой телевизор", "Загородный дом"]
#       test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
#
#       # Предсказание
#       model.eval()
#       with torch.no_grad():
#           outputs = model(**test_encodings)
#           logits = outputs.logits
#           predictions = torch.argmax(logits, dim=-1)
#
#       # Выводим результаты
#       for text, pred in zip(test_texts, predictions):
#           print(f"Товар: {text} -> Предсказанная категория: {list(label_mapping.keys())[pred]}")
#       print()
#   except Exception as e:
#       print(e)
#       print()
#
#
# """
# Results for model: google/flan-t5-small
# {'eval_loss': 1.423122525215149, 'eval_runtime': 0.079, 'eval_samples_per_second': 37.958, 'eval_steps_per_second': 12.653, 'epoch': 1.0}
# {'eval_loss': 1.4228242635726929, 'eval_runtime': 0.079, 'eval_samples_per_second': 37.968, 'eval_steps_per_second': 12.656, 'epoch': 2.0}
# {'eval_loss': 1.4238773584365845, 'eval_runtime': 0.081, 'eval_samples_per_second': 37.027, 'eval_steps_per_second': 12.342, 'epoch': 3.0}
# {'train_runtime': 7.9243, 'train_samples_per_second': 3.407, 'train_steps_per_second': 1.136, 'train_loss': 1.450564702351888, 'epoch': 3.0}
# Товар: Игровой компьютер -> Предсказанная категория: Дом
# Товар: Мощная дрель -> Предсказанная категория: Дом
# Товар: Большой телевизор -> Предсказанная категория: Дом
# Товар: Загородный дом -> Предсказанная категория: Телевизоры
#
# Results for model: papluca/xlm-roberta-base-language-detection
# {'eval_loss': 1.3947914838790894, 'eval_runtime': 0.132, 'eval_samples_per_second': 22.722, 'eval_steps_per_second': 7.574, 'epoch': 1.0}
# {'eval_loss': 1.403877854347229, 'eval_runtime': 0.113, 'eval_samples_per_second': 26.539, 'eval_steps_per_second': 8.846, 'epoch': 2.0}
# {'eval_loss': 1.4193035364151, 'eval_runtime': 0.151, 'eval_samples_per_second': 19.862, 'eval_steps_per_second': 6.621, 'epoch': 3.0}
# {'train_runtime': 39.4406, 'train_samples_per_second': 0.685, 'train_steps_per_second': 0.228, 'train_loss': 1.4146892759535048, 'epoch': 3.0}
# Товар: Игровой компьютер -> Предсказанная категория: Компьютеры
# Товар: Мощная дрель -> Предсказанная категория: Компьютеры
# Товар: Большой телевизор -> Предсказанная категория: Компьютеры
# Товар: Загородный дом -> Предсказанная категория: Компьютеры
#
# Results for model: amberoad/bert-multilingual-passage-reranking-msmarco
# {'eval_loss': 1.432491660118103, 'eval_runtime': 0.097, 'eval_samples_per_second': 30.922, 'eval_steps_per_second': 10.307, 'epoch': 1.0}
# {'eval_loss': 1.4290947914123535, 'eval_runtime': 0.093, 'eval_samples_per_second': 32.251, 'eval_steps_per_second': 10.75, 'epoch': 2.0}
# {'eval_loss': 1.4399651288986206, 'eval_runtime': 0.159, 'eval_samples_per_second': 18.865, 'eval_steps_per_second': 6.288, 'epoch': 3.0}
# {'train_runtime': 18.362, 'train_samples_per_second': 1.47, 'train_steps_per_second': 0.49, 'train_loss': 1.3690284093221028, 'epoch': 3.0}
# Товар: Игровой компьютер -> Предсказанная категория: Телевизоры
# Товар: Мощная дрель -> Предсказанная категория: Телевизоры
# Товар: Большой телевизор -> Предсказанная категория: Телевизоры
# Товар: Загородный дом -> Предсказанная категория: Телевизоры
#
# Results for model: microsoft/Multilingual-MiniLM-L12-H384
# Unable to load vocabulary from file. Please check that the provided vocabulary is accessible and not corrupted.
#
# Results for model: s-nlp/russian_toxicity_classifier
# {'eval_loss': 1.4930938482284546, 'eval_runtime': 0.0939, 'eval_samples_per_second': 31.963, 'eval_steps_per_second': 10.654, 'epoch': 1.0}
# {'eval_loss': 1.54498291015625, 'eval_runtime': 0.101, 'eval_samples_per_second': 29.692, 'eval_steps_per_second': 9.897, 'epoch': 2.0}
# {'eval_loss': 1.5699114799499512, 'eval_runtime': 0.116, 'eval_samples_per_second': 25.855, 'eval_steps_per_second': 8.618, 'epoch': 3.0}
# {'train_runtime': 25.1317, 'train_samples_per_second': 1.074, 'train_steps_per_second': 0.358, 'train_loss': 1.2558274798923068, 'epoch': 3.0}
# Товар: Игровой компьютер -> Предсказанная категория: Компьютеры
# Товар: Мощная дрель -> Предсказанная категория: Телевизоры
# Товар: Большой телевизор -> Предсказанная категория: Компьютеры
# Товар: Загородный дом -> Предсказанная категория: Компьютеры
#
# Results for model: apanc/russian-inappropriate-messages
# {'eval_loss': 1.4334074258804321, 'eval_runtime': 0.096, 'eval_samples_per_second': 31.241, 'eval_steps_per_second': 10.414, 'epoch': 1.0}
# {'eval_loss': 1.4225648641586304, 'eval_runtime': 0.11, 'eval_samples_per_second': 27.266, 'eval_steps_per_second': 9.089, 'epoch': 2.0}
# {'eval_loss': 1.4253922700881958, 'eval_runtime': 0.098, 'eval_samples_per_second': 30.602, 'eval_steps_per_second': 10.201, 'epoch': 3.0}
# {'train_runtime': 23.9615, 'train_samples_per_second': 1.127, 'train_steps_per_second': 0.376, 'train_loss': 1.3011903762817383, 'epoch': 3.0}
# Товар: Игровой компьютер -> Предсказанная категория: Компьютеры
# Товар: Мощная дрель -> Предсказанная категория: Компьютеры
# Товар: Большой телевизор -> Предсказанная категория: Телевизоры
# Товар: Загородный дом -> Предсказанная категория: Компьютеры
#
# Results for model: apanc/russian-sensitive-topics
# {'eval_loss': 1.4901646375656128, 'eval_runtime': 0.097, 'eval_samples_per_second': 30.92, 'eval_steps_per_second': 10.307, 'epoch': 1.0}
# {'eval_loss': 1.4301685094833374, 'eval_runtime': 0.094, 'eval_samples_per_second': 31.907, 'eval_steps_per_second': 10.636, 'epoch': 2.0}
# {'eval_loss': 1.4181976318359375, 'eval_runtime': 0.114, 'eval_samples_per_second': 26.311, 'eval_steps_per_second': 8.77, 'epoch': 3.0}
# {'train_runtime': 22.6926, 'train_samples_per_second': 1.19, 'train_steps_per_second': 0.397, 'train_loss': 1.2587325837877061, 'epoch': 3.0}
# Товар: Игровой компьютер -> Предсказанная категория: Дом
# Товар: Мощная дрель -> Предсказанная категория: Телевизоры
# Товар: Большой телевизор -> Предсказанная категория: Телевизоры
# Товар: Загородный дом -> Предсказанная категория: Телевизоры
#
# Results for model: blanchefort/rubert-base-cased-sentiment
# {'eval_loss': 1.4155100584030151, 'eval_runtime': 0.095, 'eval_samples_per_second': 31.566, 'eval_steps_per_second': 10.522, 'epoch': 1.0}
# {'eval_loss': 1.4204503297805786, 'eval_runtime': 0.096, 'eval_samples_per_second': 31.243, 'eval_steps_per_second': 10.414, 'epoch': 2.0}
# {'eval_loss': 1.4224344491958618, 'eval_runtime': 0.11, 'eval_samples_per_second': 27.267, 'eval_steps_per_second': 9.089, 'epoch': 3.0}
# {'train_runtime': 24.7959, 'train_samples_per_second': 1.089, 'train_steps_per_second': 0.363, 'train_loss': 1.4362160364786785, 'epoch': 3.0}
# Товар: Игровой компьютер -> Предсказанная категория: Дрели
# Товар: Мощная дрель -> Предсказанная категория: Компьютеры
# Товар: Большой телевизор -> Предсказанная категория: Дрели
# Товар: Загородный дом -> Предсказанная категория: Дрели
# """

import torch
print(torch.cuda.is_available())