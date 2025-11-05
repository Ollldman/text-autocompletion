import torch
import torch.nn as nn

class SequenceLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, dropout=0.2):
        super(SequenceLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        # Слой для векторного представления токенов
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # LSTM слой
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        
        # Регуляризация
        self.dropout = nn.Dropout(dropout)
        
        # Выходной слой - предсказывает распределение по всему словарю
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, hidden=None):
        # x: [batch_size, sequence_length]
        
        # Векторные представления
        x = self.embedding(x)  # [batch_size, sequence_length, hidden_size]
        
        # Пропускаем через LSTM
        lstm_out, hidden = self.lstm(x, hidden) 
        # lstm_out: [batch_size, sequence_length, hidden_size]
        
        # Применяем dropout
        lstm_out = self.dropout(lstm_out)
        
        # Прогнозируем для каждого временного шага
        output = self.fc(lstm_out)  # [batch_size, sequence_length, vocab_size]
        
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return (h0, c0)