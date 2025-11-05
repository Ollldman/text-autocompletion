import re
import pandas as pd
from transformers import BertTokenizerFast
import emoji


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

    
def preprocess_clean_dataset(input_file, output_file):
    """Очистка и сохранение датасета"""
    columns = ['target', 'ids', 'date', 'topic', 'user', 'text']
    df = pd.read_csv(input_file, encoding='latin-1', names=columns)
    
    # Очистка текста
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Сохраняем только очищенный текст
    result = df[['cleaned_text']]
    result.to_csv(output_file, index=False)
    print(f"Очищенные данные сохранены в {output_file}")



def save_tokenized_dataset(cleaned_file, output_file):
    """Токенизация с BERT и сохранение"""
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
    
    # Сохраняем токенизированные данные
    result = df[['token_ids']]
    result.to_csv(output_file, index=False)
    print(f"Токенизированные данные сохранены в {output_file}")
    print(f"Пример токенов: {df['token_ids'].iloc[0][:10]}...")
