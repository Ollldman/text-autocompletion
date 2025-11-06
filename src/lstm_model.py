import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMAutocomplete(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, num_layers=3, dropout=0.2):
        """
        LSTM модель для автодополнения текста
        
        Args:
            vocab_size: размер словаря
            embedding_dim: размер эмбеддингов
            hidden_dim: размер скрытого состояния LSTM
            num_layers: количество LSTM слоев
            dropout: dropout для регуляризации
        """
        super(LSTMAutocomplete, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Слой эмбеддингов
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM слои
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout для регуляризации
        self.dropout = nn.Dropout(dropout)
        
        # Выходной слой - предсказывает распределение по словарю
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        print(f"Модель инициализирована:")
        print(f"  Словарь: {vocab_size}")
        print(f"  Эмбеддинги: {embedding_dim}")
        print(f"  LSTM: {hidden_dim} hidden, {num_layers} layers")
        
    def forward(self, x, hidden=None):
        """
        Forward pass модели
        
        Args:
            x: входные токены [batch_size, sequence_length]
            hidden: скрытое состояние LSTM
            
        Returns:
            logits: предсказания [batch_size, vocab_size]
            hidden: новое скрытое состояние
        """
        # Эмбеддинги: [batch_size, seq_len] -> [batch_size, seq_len, embedding_dim]
        x = self.embedding(x)
        
        # LSTM: [batch_size, seq_len, embedding_dim] -> [batch_size, seq_len, hidden_dim]
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Берем только последний выход LSTM для предсказания следующего токена
        # [batch_size, seq_len, hidden_dim] -> [batch_size, hidden_dim]
        last_hidden = lstm_out[:, -1, :]
        
        # Dropout
        last_hidden = self.dropout(last_hidden)
        
        # Выходной слой: [batch_size, hidden_dim] -> [batch_size, vocab_size]
        logits = self.fc(last_hidden)
        
        return logits, hidden
    
    def init_hidden(self, batch_size, device):
        """Инициализация скрытого состояния LSTM"""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)
    

    def predict_next_token(self, input_sequence):
        """
        Предсказывает следующий токен для автодополнения
        
        Args:
            input_sequence: входная последовательность токенов [seq_len]
            top_k: количество лучших кандидатов для возврата
            
        Returns:
            top_tokens: список кортежей (token_id, вероятность)
        """
        self.eval()
        device = next(self.parameters()).device
        
        with torch.no_grad():
            # Добавляем batch dimension: [seq_len] -> [1, seq_len]
            input_batch = input_sequence.unsqueeze(0).to(device)
            
            # Forward pass
            logits, _ = self.forward(input_batch)
            
            # Softmax для получения вероятностей
            probs = torch.softmax(logits[0], dim=0)  # [vocab_size]
            
            # Реализация для top-K вариантов
            # # Берем top-k наиболее вероятных токенов
            # top_probs, top_indices = torch.topk(probs, top_k)
            
            # # Возвращаем список кандидатов
            # top_tokens = [
            #     (token_id.item(), prob.item()) 
            #     for token_id, prob in zip(top_indices, top_probs)
            # ]
            
            # return top_tokens
            top_prob, top_token = torch.max(probs, dim=0)
            return top_token.item(), top_prob.item()
    
    def suggest_completion(self, text, tokenizer):
        """
        Удобный метод для предложения автодополнения
        
        Args:
            text: входной текст для дополнения
            tokenizer: BERT tokenizer для кодирования/декодирования
            top_k: количество вариантов дополнения
            
        Returns:
            suggestions: список вариантов дополнения
        """
        # Кодируем текст
        input_ids = tokenizer.encode(text, add_special_tokens=False)
        input_tensor = torch.tensor(input_ids, dtype=torch.long)
        
        # Предсказываем следующие токены
        top_token, probability = self.predict_next_token(input_tensor)
        
        # Декодируем предсказанный токен
        predicted_token = tokenizer.decode([top_token])
            
        # Формируем полное предложение
        full_suggestion = text + " " + predicted_token
            
        suggestions = {
            'completion': predicted_token,
            'full_text': full_suggestion,
            'probability': probability
        }
        
        return suggestions
    