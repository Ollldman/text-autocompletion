import requests
import zipfile
import os


def download_and_extract(URL:str, path_to_extract: str)-> None:
    """Скачивание и распаковка архива"""
    
    # Скачивание файла
    print("Скачивание архива...")
    response = requests.get(URL, stream=True)
    
    # Проверка успешности запроса
    if response.status_code == 200:
        # Сохранение архива (бинарный режим!)
        with open("dataset.zip", "wb") as f:
            f.write(response.content)
        print("Архив успешно скачан")
        
        # Распаковка
        print("Распаковка архива...")
        with zipfile.ZipFile("dataset.zip", 'r') as zip_ref:
            zip_ref.extractall(path_to_extract)
        print(f"Архив успешно распакован в папку {path_to_extract}")

        os.remove("dataset.zip")
        print("Временный архив удален")
        
    else:
        print(f"Ошибка скачивания: {response.status_code}")
