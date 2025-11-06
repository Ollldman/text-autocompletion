import matplotlib.pyplot as plt
import torch
import json
from datetime import datetime
import os


from src import evaluate_with_rouge, train_epoch

def plot_training_results(train_losses, val_metrics, save_path=None):
    """
    Визуализация результатов обучения
    
    Args:
        train_losses: список train loss по эпохам
        val_metrics: список словарей с валидационными метриками по эпохам
        save_path: путь для сохранения графиков
    """
    epochs = range(1, len(train_losses) + 1)
    
    # Создаем subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Результаты обучения модели автодополнения', fontsize=16, fontweight='bold')
    
    # 1. График потерь
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    val_losses = [metric['loss'] for metric in val_metrics]
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_title('Функция потерь')
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. График точности
    val_accuracies = [metric['accuracy'] for metric in val_metrics]
    ax2.plot(epochs, val_accuracies, 'g-', label='Val Accuracy', linewidth=2)
    ax2.set_title('Точность (Accuracy)')
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. График ROUGE-1 и ROUGE-2
    rouge1_scores = [metric['rouge']['rouge1'] for metric in val_metrics]
    rouge2_scores = [metric['rouge']['rouge2'] for metric in val_metrics]
    
    ax3.plot(epochs, rouge1_scores, 'purple', label='ROUGE-1', linewidth=2)
    ax3.plot(epochs, rouge2_scores, 'orange', label='ROUGE-2', linewidth=2)
    ax3.set_title('ROUGE метрики')
    ax3.set_xlabel('Эпоха')
    ax3.set_ylabel('ROUGE Score')
    ax3.set_ylim(0, 1)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. График ROUGE-L
    rougeL_scores = [metric['rouge']['rougeL'] for metric in val_metrics]
    ax4.plot(epochs, rougeL_scores, 'brown', label='ROUGE-L', linewidth=2)
    ax4.set_title('ROUGE-L метрика')
    ax4.set_xlabel('Эпоха')
    ax4.set_ylabel('ROUGE-L Score')
    ax4.set_ylim(0, 1)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Графики сохранены в {save_path}")
    
    plt.show()

def print_final_metrics(train_losses, val_metrics):
    """
    Вывод финальных метрик обучения
    """
    print("\n" + "="*80)
    print("ИТОГОВЫЕ МЕТРИКИ ОБУЧЕНИЯ")
    print("="*80)
    
    final_epoch = len(train_losses)
    print(f"Количество эпох: {final_epoch}")
    print(f"Финальный Train Loss: {train_losses[-1]:.4f}")
    print(f"Финальный Val Loss: {val_metrics[-1]['loss']:.4f}")
    print(f"Финальная Accuracy: {val_metrics[-1]['accuracy']:.4f}")
    print(f"Финальный ROUGE-1: {val_metrics[-1]['rouge']['rouge1']:.4f}")
    print(f"Финальный ROUGE-2: {val_metrics[-1]['rouge']['rouge2']:.4f}")
    print(f"Финальный ROUGE-L: {val_metrics[-1]['rouge']['rougeL']:.4f}")
    
    # Статистика улучшений
    if final_epoch > 1:
        loss_improvement = train_losses[0] - train_losses[-1]
        accuracy_improvement = val_metrics[-1]['accuracy'] - val_metrics[0]['accuracy']
        rouge1_improvement = val_metrics[-1]['rouge']['rouge1'] - val_metrics[0]['rouge']['rouge1']
        
        print(f"\nУлучшение за обучение:")
        print(f"  Loss: +{loss_improvement:.4f}")
        print(f"  Accuracy: +{accuracy_improvement:.4f}")
        print(f"  ROUGE-1: +{rouge1_improvement:.4f}")

def plot_comparison_chart(val_metrics):
    """
    Сравнительная диаграмма финальных метрик
    """
    final_metrics = val_metrics[-1]
    
    metrics_names = ['Accuracy', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L']
    metrics_values = [
        final_metrics['accuracy'],
        final_metrics['rouge']['rouge1'],
        final_metrics['rouge']['rouge2'],
        final_metrics['rouge']['rougeL']
    ]
    
    colors = ['#2E8B57', '#4169E1', '#FF6347', '#FFD700']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics_names, metrics_values, color=colors, alpha=0.7, edgecolor='black')
    
    # Добавляем значения на столбцы
    for bar, value in zip(bars, metrics_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Финальные метрики качества модели', fontsize=14, fontweight='bold')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

def show_examples_from_epoch(val_metrics, epoch_num=None):
    """
    Показывает примеры генерации из конкретной эпохи
    
    Args:
        val_metrics: список метрик по эпохам
        epoch_num: номер эпохи (если None, берет последнюю)
    """
    if epoch_num is None:
        epoch_num = len(val_metrics) - 1
    
    examples = val_metrics[epoch_num]['examples']
    
    print(f"\n" + "="*80)
    print(f"ПРИМЕРЫ ГЕНЕРАЦИИ - ЭПОХА {epoch_num + 1}")
    print("="*80)
    
    for i, example in enumerate(examples, 1):
        status = "✅ ПРАВИЛЬНО" if example['is_correct'] else "❌ НЕПРАВИЛЬНО"
        print(f"\nПример {i}: {status}")
        print(f"Вход: '{example['input_text']}'")
        print(f"Предсказано: '{example['predicted_word']}' (вероятность: {example['probability']:.3f})")
        print(f"Ожидалось: '{example['true_next_word']}'")
        print(f"Совпадение: {example['is_correct']}")

# Использования в основном цикле обучения
def train_complete_with_plots(
        model, 
        train_loader, 
        val_loader, 
        tokenizer, 
        optimizer,
        criterion, 
        device,
        epochs=10, 
        ):
    """
    Полный цикл обучения с сохранением метрик и построением графиков
    """
    model.to(device)
    train_losses = []
    val_metrics = []
    
    print(f"Starting training for {epochs} epochs")
    
    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*60}")
        
        # Обучение
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        # Валидация
        val_results = evaluate_with_rouge(model, val_loader, tokenizer, device)
        val_metrics.append(val_results)
        
        # Вывод текущих результатов
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_results['loss']:.4f}")
        print(f"Val Accuracy: {val_results['accuracy']:.4f}")
        print(f"ROUGE-1: {val_results['rouge']['rouge1']:.4f}")
        print(f"ROUGE-2: {val_results['rouge']['rouge2']:.4f}")
        print(f"ROUGE-L: {val_results['rouge']['rougeL']:.4f}")
    
    # После завершения обучения
    print("\n" + "="*80)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("="*80)
    
    # Строим графики
    plot_training_results(train_losses, val_metrics, save_path="training_results.png")
    
    # Выводим финальные метрики
    print_final_metrics(train_losses, val_metrics)
    
    # Сравнительная диаграмма
    plot_comparison_chart(val_metrics)
    
    # Показываем примеры из последней эпохи
    show_examples_from_epoch(val_metrics)
    
    return {
        'train_losses': train_losses,
        'val_metrics': val_metrics,
        'model': model
    }


def save_model_with_metadata(
        model, 
        results, 
        tokenizer, 
        train_loader, 
        val_loader,
        model_name="lstm_autocomplete",
        save_dir="./models"):
    """
    Сохраняет модель с подробными метаданными
    
    Args:
        model: обученная модель
        results: результаты обучения (train_losses, val_metrics)
        tokenizer: токенизатор
        train_loader: train DataLoader
        val_loader: val DataLoader
        model_name: название модели
        save_dir: директория для сохранения
    """
    
    # Создаем директорию если не существует
    os.makedirs(save_dir, exist_ok=True)
    
    # Генерируем timestamp для уникальности
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{model_name}_{timestamp}"
    
    # 1. Сохраняем веса модели
    model_path = os.path.join(save_dir, f"{model_filename}.pth")
    
    # Собираем метаданные
    metadata = {
        # Информация о модели
        'model_info': {
            'name': model_name,
            'vocab_size': model.vocab_size,
            'embedding_dim': model.hidden_dim,
            'num_layers': model.num_layers,
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        },
        
        # Информация о данных
        'data_info': {
            'train_samples': len(train_loader.dataset),
            'val_samples': len(val_loader.dataset),
            'train_batches': len(train_loader),
            'val_batches': len(val_loader),
            'batch_size': train_loader.batch_size,
            'sequence_length': train_loader.dataset.sequence_length
        },
        
        # Гиперпараметры обучения
        'training_info': {
            'epochs': len(results['train_losses']),
            'learning_rate': 0.001,  # Можно передавать как параметр
            'timestamp': timestamp,
            'training_time': 'N/A'  # Можно добавить расчет времени
        },
        
        # Финальные метрики
        'final_metrics': {
            'final_train_loss': results['train_losses'][-1],
            'final_val_loss': results['val_metrics'][-1]['loss'],
            'final_accuracy': results['val_metrics'][-1]['accuracy'],
            'final_rouge1': results['val_metrics'][-1]['rouge']['rouge1'],
            'final_rouge2': results['val_metrics'][-1]['rouge']['rouge2'],
            'final_rougeL': results['val_metrics'][-1]['rouge']['rougeL']
        },
        
        # История обучения (только последние значения для экономии места)
        'training_history': {
            'train_losses': results['train_losses'],
            'val_losses': [metric['loss'] for metric in results['val_metrics']],
            'val_accuracies': [metric['accuracy'] for metric in results['val_metrics']],
            'val_rouge1': [metric['rouge']['rouge1'] for metric in results['val_metrics']],
            'val_rouge2': [metric['rouge']['rouge2'] for metric in results['val_metrics']]
        },
        
        # Информация о токенизаторе
        'tokenizer_info': {
            'name': 'bert-base-uncased',
            'vocab_size': tokenizer.vocab_size,
            'special_tokens': {
                'pad_token': tokenizer.pad_token,
                'unk_token': tokenizer.unk_token,
                'cls_token': tokenizer.cls_token,
                'sep_token': tokenizer.sep_token
            }
        }
    }
    
    # Сохраняем модель с метаданными
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'metadata': metadata,
        'model_class': model.__class__.__name__
    }
    
    torch.save(checkpoint, model_path)
    
    # 2. Сохраняем метаданные в отдельный JSON файл
    metadata_path = os.path.join(save_dir, f"{model_filename}_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # 3. Сохраняем примеры генерации
    examples = results['val_metrics'][-1]['examples']
    examples_data = {
        'generation_examples': examples,
        'timestamp': timestamp
    }
    
    examples_path = os.path.join(save_dir, f"{model_filename}_examples.json")
    with open(examples_path, 'w', encoding='utf-8') as f:
        json.dump(examples_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nМодель и метаданные сохранены:")
    print(f"   Модель: {model_path}")
    print(f"   Метаданные: {metadata_path}")
    print(f"   Примеры: {examples_path}")
    
    return model_path

def load_model_with_metadata(model_path, model_class, device='cpu'):
    """
    Загружает модель с метаданными
    
    Args:
        model_path: путь к файлу модели
        model_class: класс модели
        device: устройство для загрузки
    
    Returns:
        model: загруженная модель
        metadata: метаданные
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    # Создаем модель на основе метаданных
    model_info = checkpoint['metadata']['model_info']
    model = model_class(
        vocab_size=model_info['vocab_size'],
        embedding_dim=model_info['embedding_dim'],
        hidden_dim=model_info['hidden_dim'],
        num_layers=model_info['num_layers']
    )
    
    # Загружаем веса
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"    Модель загружена: {model_path}")
    print(f"   Параметры: {model_info['total_parameters']:,}")
    print(f"   Точность: {checkpoint['metadata']['final_metrics']['final_accuracy']:.4f}")
    
    return model, checkpoint['metadata']

# Пример использования после обучения

# # Сохраняем модель
# model_path = save_model_with_metadata(
#     model=results['model'],
#     results=results,
#     tokenizer=tokenizer,
#     train_loader=train_loader,
#     val_loader=val_loader,
#     model_name="my_trained_lstm"
# )

# # Загрузка модели для проверки
# loaded_model, metadata = load_model_with_metadata(model_path, LSTMAutocomplete, device)
# print_model_summary(metadata)