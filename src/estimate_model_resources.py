import torch
import psutil
import os
import time
from thop import profile
from memory_profiler import memory_usage
import GPUtil

def evaluate_model_resources(model, tokenizer, device, sequence_length=20, num_tests=100):
    """
    Полная оценка ресурсов модели
    
    Args:
        model: модель для оценки
        tokenizer: токенизатор
        device: устройство
        sequence_length: длина последовательности для теста
        num_tests: количество тестов для усреднения
    """
    print("\n" + "="*80)
    print("ОЦЕНКА РЕСУРСОВ МОДЕЛИ")
    print("="*80)
    
    model.eval()
    
    # 1. Оценка памяти модели
    memory_stats = evaluate_memory_usage(model, device)
    
    # 2. Оценка вычислительной сложности
    flops_stats = evaluate_computational_complexity(model, sequence_length, device)
    
    # 3. Оценка скорости inference
    speed_stats = evaluate_inference_speed(model, tokenizer, sequence_length, num_tests, device)
    
    # 4. Оценка энергоэффективности (если GPU)
    if device.type == 'cuda':
        gpu_stats = evaluate_gpu_usage(model, tokenizer, sequence_length, device)
        memory_stats.update(gpu_stats)
    
    # Сводный отчет
    print_summary_report(memory_stats, flops_stats, speed_stats)
    
    return {
        'memory': memory_stats,
        'computation': flops_stats,
        'speed': speed_stats
    }

def evaluate_memory_usage(model, device):
    """Оценка использования памяти"""
    print("\n ОЦЕНКА ПАМЯТИ:")
    print("-" * 40)
    
    memory_stats = {}
    
    # Память параметров модели
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size = param_size + buffer_size
    total_size_mb = total_size / (1024 ** 2)
    
    memory_stats['parameters_memory_mb'] = total_size_mb
    memory_stats['parameters_count'] = sum(p.numel() for p in model.parameters())
    
    print(f"Параметры модели: {memory_stats['parameters_count']:,}")
    print(f"Память параметров: {total_size_mb:.2f} MB")
    
    # Память во время inference
    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() / (1024 ** 2)
        
        # Тестовый forward pass для оценки пиковой памяти
        test_input = torch.randint(0, model.vocab_size, (1, 20)).to(device)
        with torch.no_grad():
            _ = model(test_input)
        
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
        memory_stats['gpu_peak_memory_mb'] = peak_memory
        memory_stats['gpu_initial_memory_mb'] = initial_memory
        
        print(f"Пиковая память GPU: {peak_memory:.2f} MB")
        print(f"Использование GPU: {(peak_memory - initial_memory):.2f} MB")
    
    # ОЗУ процесса
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / (1024 ** 2)
    memory_stats['ram_usage_mb'] = ram_usage
    print(f"Использование ОЗУ: {ram_usage:.2f} MB")
    
    return memory_stats

def evaluate_computational_complexity(model, sequence_length, device):
    """Оценка вычислительной сложности"""
    print("\n ВЫЧИСЛИТЕЛЬНАЯ СЛОЖНОСТЬ:")
    print("-" * 40)
    
    # Создаем тестовый вход
    dummy_input = torch.randint(0, model.vocab_size, (1, sequence_length)).to(device)
    
    # Используем thop для подсчета FLOPs
    try:
        flops, params = profile(model, inputs=(dummy_input,), verbose=False) #type: ignore
        flops_g = flops / 1e9  # В гигафлопсах
        
        complexity_stats = {
            'flops': flops,
            'flops_giga': flops_g,
            'parameters': params
        }
        
        print(f"FLOPs на запрос: {flops_g:.2f} G")
        print(f"Параметры: {params:,}")
        
    except Exception as e:
        print(f"Ошибка подсчета FLOPs: {e}")
        # Альтернативный ручной расчет для LSTM
        complexity_stats = estimate_lstm_complexity(model, sequence_length)
    
    return complexity_stats

def estimate_lstm_complexity(model, sequence_length):
    """Ручная оценка сложности LSTM"""
    embedding_flops = model.vocab_size * model.embedding_dim
    
    # LSTM FLOPs = 8 * n * m * k (где n=input_size, m=hidden_size, k=sequence_length)
    lstm_flops = 8 * model.embedding_dim * model.hidden_dim * sequence_length * model.num_layers
    
    # Linear layer FLOPs
    linear_flops = model.hidden_dim * model.vocab_size
    
    total_flops = embedding_flops + lstm_flops + linear_flops
    total_flops_g = total_flops / 1e9
    
    print(f"FLOPs (оценка): {total_flops_g:.2f} G")
    print(f"  - Эмбеддинги: {embedding_flops/1e6:.2f} M")
    print(f"  - LSTM: {lstm_flops/1e6:.2f} M")
    print(f"  - Linear: {linear_flops/1e6:.2f} M")
    
    return {
        'flops': total_flops,
        'flops_giga': total_flops_g,
        'estimation_method': 'manual'
    }

def evaluate_inference_speed(model, tokenizer, sequence_length, num_tests, device):
    """Оценка скорости inference"""
    print("\n  СКОРОСТЬ INFERENCE:")
    print("-" * 40)
    
    model.eval()
    
    # Подготовка тестовых данных
    test_inputs = []
    for i in range(num_tests):
        # Случайная последовательность
        input_ids = torch.randint(0, model.vocab_size, (1, sequence_length)).to(device)
        test_inputs.append(input_ids)
    
    # Тест скорости
    start_time = time.time()
    
    with torch.no_grad():
        for input_tensor in test_inputs:
            _ = model(input_tensor)
    
    # Синхронизация для GPU
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    total_time = time.time() - start_time
    avg_time_per_batch = total_time / num_tests
    tokens_per_second = sequence_length / avg_time_per_batch
    
    speed_stats = {
        'total_inference_time': total_time,
        'avg_time_per_batch': avg_time_per_batch,
        'tokens_per_second': tokens_per_second,
        'batches_per_second': 1 / avg_time_per_batch,
        'num_tests': num_tests
    }
    
    print(f"Среднее время inference: {avg_time_per_batch * 1000:.2f} ms")
    print(f"Токенов в секунду: {tokens_per_second:.0f}")
    print(f"Запросов в секунду: {1/avg_time_per_batch:.1f}")
    print(f"Всего тестов: {num_tests}")
    
    return speed_stats

def evaluate_gpu_usage(model, tokenizer, sequence_length, device):
    """Оценка использования GPU"""
    print("\n ИСПОЛЬЗОВАНИЕ GPU:")
    print("-" * 40)
    
    gpu_stats = {}
    
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]  # Первая GPU
            gpu_stats.update({
                'gpu_name': gpu.name,
                'gpu_memory_total': gpu.memoryTotal,
                'gpu_memory_used': gpu.memoryUsed,
                'gpu_load': gpu.load * 100
            })
            
            print(f"GPU: {gpu.name}")
            print(f"Память GPU: {gpu.memoryUsed}/{gpu.memoryTotal} MB")
            print(f"Загрузка GPU: {gpu.load * 100:.1f}%")
            
    except Exception as e:
        print(f"Информация о GPU недоступна: {e}")
    
    return gpu_stats

def print_summary_report(memory_stats, flops_stats, speed_stats):
    """Сводный отчет по ресурсам"""
    print("\n" + "="*80)
    print(" СВОДНЫЙ ОТЧЕТ ПО РЕСУРСАМ")
    print("="*80)

    print(" ПРОИЗВОДИТЕЛЬНОСТЬ:")
    print(f"  • Запросов в секунду: {speed_stats['batches_per_second']:.1f}")
    print(f"  • Токенов в секунду: {speed_stats['tokens_per_second']:.0f}")
    print(f"  • Время inference: {speed_stats['avg_time_per_batch'] * 1000:.2f} ms")
    
    print("\n ПАМЯТЬ:")
    print(f"  • Параметры модели: {memory_stats['parameters_count']:,}")
    print(f"  • Память модели: {memory_stats.get('parameters_memory_mb', 0):.2f} MB")
    
    if 'gpu_peak_memory_mb' in memory_stats:
        print(f"  • Пиковая память GPU: {memory_stats['gpu_peak_memory_mb']:.2f} MB")
    
    print(f"  • ОЗУ процесса: {memory_stats['ram_usage_mb']:.2f} MB")
    
    print("\n ВЫЧИСЛЕНИЯ:")
    print(f"  • FLOPs на запрос: {flops_stats.get('flops_giga', 0):.2f} G")
    
    # Оценка пригодности для мобильных устройств
    print("\n ПРИГОДНОСТЬ ДЛЯ МОБИЛЬНЫХ УСТРОЙСТВ:")
    
    model_size_mb = memory_stats.get('parameters_memory_mb', 0)
    inference_time_ms = speed_stats['avg_time_per_batch'] * 1000
    
    mobile_rating = "ОТЛИЧНО"
    if model_size_mb > 100:
        mobile_rating = "ХОРОШО"
    if model_size_mb > 200:
        mobile_rating = "УДОВЛЕТВОРИТЕЛЬНО"
    if model_size_mb > 500:
        mobile_rating = "НЕ РЕКОМЕНДУЕТСЯ"
    
    speed_rating = "ОТЛИЧНО"
    if inference_time_ms > 50:
        speed_rating = "ХОРОШО"
    if inference_time_ms > 100:
        speed_rating = "УДОВЛЕТВОРИТЕЛЬНО"
    if inference_time_ms > 200:
        speed_rating = "НЕ РЕКОМЕНДУЕТСЯ"
    
    print(f"  • Размер модели: {mobile_rating} ({model_size_mb:.1f} MB)")
    print(f"  • Скорость: {speed_rating} ({inference_time_ms:.1f} ms)")
    
    overall_rating = " ПРИГОДНА" if model_size_mb < 200 and inference_time_ms < 100 else " ТРЕБУЕТ ОПТИМИЗАЦИИ"
    print(f"  • Общая оценка: {overall_rating}")