# Данные

В проекте используется датасет Kaggle **H&M Personalized Fashion Recommendations**.

Исходные данные не хранятся в Git-репозитории из-за большого объёма.

## Ожидаемая структура

После загрузки датасета исходные файлы должны находиться в:

```text
data/raw/
├── articles.csv
├── customers.csv
├── transactions_train.csv
├── sample_submission.csv
└── images/
```

## Предобработанные данные

Ноутбук:

```text
notebooks/02_preprocessing.ipynb
```

создаёт необходимые промежуточные файлы в:

```text
data/processed/
```

В том числе:

```text
transactions.parquet
transactions_mapped.parquet
customer_mapping.parquet
article_mapping.parquet
train.parquet
validation.parquet
test.parquet
validation_ground_truth.parquet
test_ground_truth.parquet
```

## Kaggle

Датасет:

**H&M Personalized Fashion Recommendations**

Страница соревнования:

https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations

Сгенерированные данные, изображения, embeddings, checkpoints и submissions намеренно исключены из Git через `.gitignore`.
