import re
import pandas as pd
from transformers import BertTokenizerFast
from sklearn.model_selection import train_test_split
import emoji
import os
import matplotlib.pyplot as plt
import numpy as np
import ast

# Очистка строки твита
def clean_text(text):
    """Функция очистки одного текста"""
    if not isinstance(text, str):
        return ""
    
    # Привести к нижнему регистру
    text = text.lower()
    # Удалить ссылки
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Удалить упоминания (@username)
    text = re.sub(r'@\w+', '', text)
    # Удалить хештеги (но оставить текст)
    text = re.sub(r'#(\w+)', r'\1', text)
    # Удалить эмодзи
    text = emoji.replace_emoji(text, replace='')
    # Удалить специальные символы и цифры (но оставить буквы и базовую пунктуацию)
    text = re.sub(r'[^a-zA-Z\s\.\,\!\?]', '', text)
    # Удалить лишние пробелы
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Управление очисткой большого датасета   
def preprocess_clean_dataset(input_file, output_file):
    """Очистка и сохранение датасета"""
    columns = ['target', 'ids', 'date', 'topic', 'user', 'text']
    df = pd.read_csv(input_file, encoding='latin-1', names=columns)
    
    # Очистка текста
    df['cleaned_text'] = df['text'].apply(clean_text)
    # Удаление пустых строк
    df = df[df['cleaned_text'].str.len() > 0]
    
    # Сохраняем только очищенный текст
    result = df[['cleaned_text']]
    result.to_csv(output_file, index=False)
    print(f"Очищенные данные сохранены в {output_file}")
    os.remove(input_file)
    print(f"Файл удален: {input_file}")


# Добавление токенизации
def save_tokenized_dataset(cleaned_file, output_file):
    """Токенизация с BERT и сохранение итогового общего файла train.csv"""
    # Загружаем очищенные данные
    df = pd.read_csv(cleaned_file)
    
    # Инициализируем BERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    
    def tokenize_function(text):
        # Токенизируем с добавлением специальных токенов [CLS] и [SEP]
        tokens = tokenizer.encode(
            text,
            add_special_tokens=True,  # Добавляет [CLS] в начало и [SEP] в конец
            max_length=128,
            truncation=True,
            padding=False
        )
        return tokens
    
    # Токенизируем текст
    df['token_ids'] = df['cleaned_text'].apply(tokenize_function)
    
    df.to_csv(output_file, index=False)
    print(f"Токенизированные данные сохранены в {output_file}")
    print(f"Пример токенов: {df['token_ids'].iloc[0][:10]}...")


# Разделение датасета на train / val
def split_dataset(tokenized_file, train_file, val_file, val_size=0.1):
    """Разделение данных на train/val"""
    df = pd.read_csv(tokenized_file)
    
    # Преобразуем строки обратно в списки
    df['token_ids'] = df['token_ids'].apply(
        lambda x: [int(i) for i in x.strip('[]').split(',')] if isinstance(x, str) else []
    )
    
    # Разделяем на train и val
    train_df, val_df = train_test_split(df, test_size=val_size, random_state=42)
    
    # Сохраняем
    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    
    print(f"Train данных: {len(train_df)}")
    print(f"Val данных: {len(val_df)}")
    print(f"Train сохранен в {train_file}")
    print(f"Val сохранен в {val_file}")


# анализируем для подбора оптимальной длины последовательности для torch.Dataset
def analyze_for_sequence(data_path: str) -> None:
    
    csv_file = "./data/train.csv"
    # Загружаем данные
    df = pd.read_csv(csv_file)

    # Преобразуем строки с token_ids в списки
    df['token_list'] = df['token_ids'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    # Анализируем длины последовательностей
    sequence_lengths = df['token_list'].apply(len)

    print("=== СТАТИСТИКА ДЛИН ПОСЛЕДОВАТЕЛЬНОСТЕЙ ===")
    print(f"Общее количество последовательностей: {len(sequence_lengths)}")
    print(f"Средняя длина: {sequence_lengths.mean():.1f}")
    print(f"Медианная длина: {sequence_lengths.median():.1f}")
    print(f"Минимальная длина: {sequence_lengths.min()}")
    print(f"Максимальная длина: {sequence_lengths.max()}")
    print(f"Стандартное отклонение: {sequence_lengths.std():.1f}")

    # Процентили
    percentiles = [25, 50, 75, 90, 95, 99]
    for p in percentiles:
        print(f"{p}% перцентиль: {sequence_lengths.quantile(p/100):.1f}")

    # Визуализация
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Гистограмма длин
    axes[0, 0].hist(sequence_lengths, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(sequence_lengths.mean(), color='red', linestyle='--', label=f'Среднее: {sequence_lengths.mean():.1f}')
    axes[0, 0].axvline(sequence_lengths.median(), color='green', linestyle='--', label=f'Медиана: {sequence_lengths.median():.1f}')
    axes[0, 0].set_xlabel('Длина последовательности')
    axes[0, 0].set_ylabel('Количество')
    axes[0, 0].set_title('Распределение длин последовательностей')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Box plot
    axes[0, 1].boxplot(sequence_lengths)
    axes[0, 1].set_ylabel('Длина последовательности')
    axes[0, 1].set_title('Box plot длин последовательностей')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Кумулятивное распределение
    sorted_lengths = np.sort(sequence_lengths)
    y_vals = np.arange(len(sorted_lengths)) / float(len(sorted_lengths))
    axes[1, 0].plot(sorted_lengths, y_vals, linewidth=2)
    axes[1, 0].set_xlabel('Длина последовательности')
    axes[1, 0].set_ylabel('Доля последовательностей')
    axes[1, 0].set_title('Кумулятивное распределение длин')
    axes[1, 0].grid(True, alpha=0.3)

    # Добавляем линии для процентилей
    for p in [50, 75, 90, 95]:
        percentile_val = sequence_lengths.quantile(p/100)
        axes[1, 0].axvline(percentile_val, color='red', linestyle='--', alpha=0.7)
        axes[1, 0].text(percentile_val, 0.5, f'{p}%', rotation=90, va='center')

    # 4. Топ самых частых длин
    length_counts = sequence_lengths.value_counts().head(20)
    axes[1, 1].bar(length_counts.index, length_counts.values, alpha=0.7, color='lightcoral')
    axes[1, 1].set_xlabel('Длина последовательности')
    axes[1, 1].set_ylabel('Количество')
    axes[1, 1].set_title('Топ-20 самых частых длин')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

    # Анализ для выбора оптимальной sequence_length
    print("\n=== РЕКОМЕНДАЦИИ ПО ВЫБОРУ SEQUENCE_LENGTH ===")

    # Рассчитываем покрытие для разных длин
    lengths_to_test = [5, 10, 15, 20, 25, 30, 40, 50, 64, 128]
    print("Покрытие данных для разных sequence_length:")
    for length in lengths_to_test:
        coverage = (sequence_lengths >= length).sum() / len(sequence_lengths) * 100
        usable_sequences = (sequence_lengths >= length).sum()
        print(f"sequence_length = {length:3d}: {coverage:5.1f}% данных ({usable_sequences:5d} последовательностей)")

    # Рекомендация
    recommended_length = sequence_lengths.quantile(0.2)  # 80% перцентиль
    print(f"\nРекомендуемая sequence_length: {int(recommended_length)}")
    print(f"Это покроет {((sequence_lengths >= recommended_length).sum() / len(sequence_lengths) * 100):.1f}% данных")

    # Анализ потерь при обрезке
    print(f"\nПри sequence_length = {int(recommended_length)}:")
    print(f"Будет потеряно {((sequence_lengths < recommended_length).sum() / len(sequence_lengths) * 100):.1f}% данных")
    print(f"Останется {(sequence_lengths >= recommended_length).sum()} последовательностей для обучения")