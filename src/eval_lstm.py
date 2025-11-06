import torch
import torch.nn as nn
from tqdm import tqdm
from evaluate import load


def evaluate_with_rouge(model, test_loader, tokenizer, device, num_examples=5):
    """
    Оценка модели с ROUGE метрикой и примерами генерации
    """
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    rouge = load('rouge')
    # Для ROUGE метрики
    all_predictions = []
    all_references = []
    
    criterion = nn.CrossEntropyLoss()
    
    print("Starting evaluation with ROUGE...")
    
    with torch.no_grad():
        for x_batch, y_batch in tqdm(test_loader, desc="Evaluating", ascii=True):
            x = x_batch.to(device)
            y = y_batch.to(device)
            
            # Forward pass
            logits, _ = model(x)
            loss = criterion(logits, y)
            
            total_loss += loss.item()
            
            # Accuracy
            predictions = torch.argmax(logits, dim=1)
            batch_correct = (predictions == y).sum().item()
            total_correct += batch_correct
            total_samples += y.size(0)
            
            # Генерируем автодополнения для ROUGE (только первый в батче)
            input_tokens = x_batch[0]
            target_token = y_batch[0]
            
            # Генерируем предсказание
            predicted_token_id, _ = model.predict_next_token(input_tokens)
            
            # Декодируем в текст
            input_text = tokenizer.decode(input_tokens.tolist(), skip_special_tokens=True)
            predicted_text = tokenizer.decode([predicted_token_id], skip_special_tokens=True)
            target_text = tokenizer.decode([target_token.item()], skip_special_tokens=True)
            
            # Формируем полные тексты для ROUGE
            full_prediction = input_text + " " + predicted_text
            full_reference = input_text + " " + target_text
            
            all_predictions.append(full_prediction)
            all_references.append(full_reference)
    
    # Вычисляем ROUGE метрики
    if all_predictions and all_references:
        rouge_results = rouge.compute(
            predictions=all_predictions,
            references=all_references,
            use_stemmer=True
        )
    else:
        rouge_results = {
            'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0,
            'rougeLsum': 0.0
        }
    
    # Вычисляем accuracy
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else float('inf')
    
    # Генерируем примеры
    examples = generate_examples(model, test_loader.dataset, tokenizer, num_examples)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'rouge': rouge_results,
        'examples': examples
    }

def generate_examples(model, dataset, tokenizer, num_examples=3):
    """
    Генерирует примеры автодополнения
    """
    examples = []
    
    # Берем случайные примеры
    indices = torch.randperm(100)[:num_examples]
    
    for idx in indices:
        try:
            input_tokens, target_token = dataset[idx]
            
            # Получаем предсказание
            predicted_token_id, probability = model.predict_next_token(input_tokens)
            
            # Формируем пример
            input_text = tokenizer.decode(input_tokens.tolist(), skip_special_tokens=True)
            predicted_text = tokenizer.decode([predicted_token_id], skip_special_tokens=True)
            target_text = tokenizer.decode([target_token.item()], skip_special_tokens=True)
            
            examples.append({
                'input_text': input_text,
                'predicted_word': predicted_text,
                'true_next_word': target_text,
                'probability': probability,
                'is_correct': predicted_token_id == target_token.item()
            })
        except Exception as e:
            continue
    
    return examples