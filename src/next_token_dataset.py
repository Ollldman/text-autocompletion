import pandas as pd
from torch.utils.data import Dataset
import ast
import torch

class NextTokenDataset(Dataset):
    def __init__(self, csv_file, sequence_length=5, min_sequence_length=8):
        self.sequence_length = sequence_length
        self.min_sequence_length = min_sequence_length
        
        # Загружаем данные
        self.df = pd.read_csv(csv_file)
        self.df['token_list'] = self.df['token_ids'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        
        # ФИЛЬТРУЕМ
        self.df = self.df[self.df['token_list'].apply(len) >= self.min_sequence_length]
        
        # СОХРАНЯЕМ СВЯЗЬ примеров с исходными текстами
        self.samples = []
        self.sample_to_text_map = []  # Храним индекс исходного текста для каждого примера
        
        self._create_meaningful_samples()
        
        print(f"Создано {len(self.samples)} осмысленных примеров")
    
    def _create_meaningful_samples(self):
        """Создает примеры и сохраняет связь с исходными текстами"""
        for text_idx, (_, row) in enumerate(self.df.iterrows()):
            token_list = row['token_list']
            cleaned_text = row['cleaned_text']
            
            if len(token_list) <= self.sequence_length:
                continue
                
            for i in range(len(token_list) - self.sequence_length):
                input_seq = token_list[i:i + self.sequence_length]
                target_token = token_list[i + self.sequence_length]
                
                self.samples.append((input_seq, target_token))
                self.sample_to_text_map.append((text_idx, i))  # Сохраняем связь
    

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        input_seq, target_token = self.samples[idx]
        return (
            torch.tensor(input_seq, dtype=torch.long),
            torch.tensor(target_token, dtype=torch.long)
        )
    
    def get_sample_info(self, idx):
        """Правильное получение информации о примере"""
        input_seq, target_token = self.samples[idx]
        text_idx, position = self.sample_to_text_map[idx]  # Получаем исходный текст
        
        original_row = self.df.iloc[text_idx]
        original_text = original_row['cleaned_text']
        full_tokens = original_row['token_list']
        
        # Показываем контекст в исходном тексте
        start_pos = max(0, position - 3)
        end_pos = min(len(full_tokens), position + self.sequence_length + 3)
        context_tokens = full_tokens[start_pos:end_pos]
        
        return {
            'input_sequence': input_seq,
            'target_token': target_token,
            'original_text': original_text,
            'context_tokens': context_tokens,
            'position_in_text': position
        }