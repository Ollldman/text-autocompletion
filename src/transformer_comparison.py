from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.pipelines import pipeline
import torch
from evaluate import load

class ModelComparator:
    def __init__(self,
                lstm_model,
                lstm_tokenizer,
                transformer_model_name="distilgpt2"):
        """
        Простой компаратор для сравнения LSTM и Transformer моделей
        
        Args:
            lstm_model: ваша обученная LSTM модель
            lstm_tokenizer: токенизатор для LSTM (BERT)
            transformer_model_name: название трансформер модели
        """
        self.lstm_model = lstm_model
        self.lstm_tokenizer = lstm_tokenizer
        self.transformer_model_name = transformer_model_name
        
        # Загружаем трансформер
        print("Загрузка DistilGPT2...")
        self.transformer_tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)
        self.transformer_model = AutoModelForCausalLM.from_pretrained(transformer_model_name)
        
        # Устанавливаем pad_token
        if self.transformer_tokenizer.pad_token is None:
            self.transformer_tokenizer.pad_token = self.transformer_tokenizer.eos_token

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transformer_model.to(self.device)
        self.transformer_model.eval()
        
        # Загружаем ROUGE метрику
        self.rouge = load("rouge")
        
        print("Модели загружены!")
    
    def lstm_predict(self, input_tokens):
        """Предсказание следующего слова с помощью LSTM"""
        self.lstm_model.eval()
        
        with torch.no_grad():
            token_id, probability = self.lstm_model.predict_next_token(input_tokens)
            predicted_word = self.lstm_tokenizer.decode([token_id])
            return predicted_word, probability
    
    def transformer_predict(self, text):
        """Предсказание с помощью трансформера"""
        """DistilGPT2 предсказывает только одно следующее слово"""
        self.transformer_model.eval()
        
        # Токенизируем входной текст
        inputs = self.transformer_tokenizer.encode(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            # Получаем логиты для следующего токена
            outputs = self.transformer_model(inputs)
            next_token_logits = outputs.logits[0, -1, :]
            
            # Применяем softmax для вероятностей
            probs = torch.softmax(next_token_logits, dim=0)
            
            # Берем самый вероятный токен
            next_token_id = torch.argmax(probs).item()
            probability = probs[next_token_id].item() # type: ignore
            
            # Декодируем только один токен
            next_word = self.transformer_tokenizer.decode([next_token_id])
            
            return next_word, probability
    
    def evaluate_on_test_loader(self, test_loader, num_batches=10):
        """
        Честная оценка на test_loader
        """
        print("\n" + "="*80)
        print("ЧЕСТНОЕ СРАВНЕНИЕ НА TEST_LOADER")
        print("="*80)
        
        lstm_correct = 0
        transformer_correct = 0
        total_samples = 0
        
        lstm_predictions = []
        transformer_predictions = []
        true_targets = []
        
        print(f"Оценка на {num_batches} батчах...")
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                if batch_idx >= num_batches:
                    break
                
                for i in range(min(5, len(inputs))):  # Берем по 5 примеров из каждого батча
                    input_tokens = inputs[i]
                    target_token = targets[i]
                    
                    # Декодируем входной текст
                    input_text = self.lstm_tokenizer.decode(input_tokens.tolist(), skip_special_tokens=True)
                    true_next_word = self.lstm_tokenizer.decode([target_token.item()])
                    
                    # LSTM предсказание
                    lstm_word, lstm_prob = self.lstm_predict(input_tokens)
                    
                    # Transformer предсказание
                    transformer_word, transformer_prob = self.transformer_predict(input_text)
                    
                    # Сравниваем с истинным значением
                    lstm_is_correct = (lstm_word.strip() == true_next_word.strip())
                    transformer_is_correct = (transformer_word.strip() == true_next_word.strip())
                    
                    lstm_correct += lstm_is_correct
                    transformer_correct += transformer_is_correct
                    total_samples += 1
                    
                    # Сохраняем для ROUGE
                    lstm_full = input_text + " " + lstm_word
                    transformer_full = input_text + " " + transformer_word
                    true_full = input_text + " " + true_next_word
                    
                    lstm_predictions.append(lstm_full)
                    transformer_predictions.append(transformer_full)
                    true_targets.append(true_full)
                    
                    # Выводим первые несколько примеров
                    if total_samples <= 5:
                        print(f"\n Пример {total_samples}:")
                        print(f"   Вход: '{input_text}'")
                        print(f"   Истинное следующее слово: '{true_next_word}'")
                        print(f"   LSTM: '{lstm_word}' ({lstm_prob:.3f}) {'✅' if lstm_is_correct else '❌'}")
                        print(f"   DistilGPT2: '{transformer_word}' ({transformer_prob:.3f}) {'✅' if transformer_is_correct else '❌'}")
        
        # Вычисляем метрики
        lstm_accuracy = lstm_correct / total_samples
        transformer_accuracy = transformer_correct / total_samples
        
        # ROUGE метрики
        lstm_rouge = self.rouge.compute(
            predictions=lstm_predictions,
            references=true_targets,
            use_stemmer=True
        )
        
        transformer_rouge = self.rouge.compute(
            predictions=transformer_predictions,
            references=true_targets,
            use_stemmer=True
        )
        
        print(f"\n📊 РЕЗУЛЬТАТЫ НА {total_samples} ПРИМЕРАХ:")
        print(f"   LSTM Accuracy: {lstm_accuracy:.4f}")
        print(f"   DistilGPT2 Accuracy: {transformer_accuracy:.4f}")
        
        print(f"\n🎯 ROUGE МЕТРИКИ (чем выше - тем лучше):")
        print(f"   LSTM ROUGE-1: {lstm_rouge['rouge1']:.4f}")
        print(f"   DistilGPT2 ROUGE-1: {transformer_rouge['rouge1']:.4f}")
        print(f"   LSTM ROUGE-2: {lstm_rouge['rouge2']:.4f}")
        print(f"   DistilGPT2 ROUGE-2: {transformer_rouge['rouge2']:.4f}")
        print(f"   LSTM ROUGE-L: {lstm_rouge['rougeL']:.4f}")
        print(f"   DistilGPT2 ROUGE-L: {transformer_rouge['rougeL']:.4f}")
        
        # Определяем победителя
        if lstm_accuracy > transformer_accuracy:
            print(f"\n🏆 ПОБЕДИТЕЛЬ: LSTM модель!")
        elif transformer_accuracy > lstm_accuracy:
            print(f"\n🏆 ПОБЕДИТЕЛЬ: DistilGPT2 модель!")
        else:
            print(f"\n🏆 НИЧЬЯ!")
        
        return {
            'lstm_accuracy': lstm_accuracy,
            'transformer_accuracy': transformer_accuracy,
            'lstm_rouge': lstm_rouge,
            'transformer_rouge': transformer_rouge,
            'total_samples': total_samples
        }

# Простая версия использования
def quick_comparison(lstm_model, lstm_tokenizer, test_loader):
    """Быстрое сравнение без классов"""
    
    # Загружаем DistilGPT2
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    print("🔄 Загрузка DistilGPT2...")
    transformer_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    transformer_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer_model.to(device)
    transformer_model.eval()
    
    if transformer_tokenizer.pad_token is None:
        transformer_tokenizer.pad_token = transformer_tokenizer.eos_token
    
    # Тестируем на нескольких примерах
    print("\n БЫСТРОЕ СРАВНЕНИЕ:")
    
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if batch_idx >= 2:  # Только 2 батча
            break
            
        for i in range(min(3, len(inputs))):  # По 3 примера из батча
            input_tokens = inputs[i]
            target_token = targets[i]
            
            # Декодируем
            input_text = lstm_tokenizer.decode(input_tokens.tolist(), skip_special_tokens=True)
            true_word = lstm_tokenizer.decode([target_token.item()])
            
            # LSTM предсказание
            lstm_word, lstm_prob = lstm_model.predict_next_token(input_tokens)
            lstm_word = lstm_tokenizer.decode([lstm_word])
            
            # DistilGPT2 предсказание (только одно слово)
            inputs_transformer = transformer_tokenizer.encode(input_text, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = transformer_model(inputs_transformer)
                next_token_logits = outputs.logits[0, -1, :]
                next_token_id = torch.argmax(next_token_logits).item()
                transformer_word = transformer_tokenizer.decode([next_token_id])
            
            print(f"\n '{input_text}'")
            print(f"   Истина: '{true_word}'")
            print(f"   LSTM: '{lstm_word}' ({lstm_prob:.3f})")
            print(f"   GPT2: '{transformer_word}'")
