import ai_engine
import github_utils  #  Импортируем github_utils
import getpass
import time
import os #  Добавляем os
from threading import Event

def main():
    #  Получаем токен GitHub (запрашиваем у пользователя)
    token = getpass.getpass("Enter your GitHub token: ")
    os.environ["AI_PROGRESS_TOKEN"] = token

    #  Настройки AI
    ai_settings = {
        'fantasyType': 'normal',  #  Или 'progressive'
        'aiType': 'mccfr',
        'training_mode': True  #  Обязательно True для обучения
    }

    #  Создаем агента
    agent = ai_engine.CFRAgent(iterations=500000, batch_size=100, stop_threshold=0.001, max_nodes=100000)

    #  Загружаем прогресс (если есть)
    agent.load_progress()

    #  Настраиваем тайм-аут (если нужен)
    timeout_event = Event()
    # timeout_event.set()  #  Для немедленного завершения (для отладки)

    #  Запускаем обучение
    result = {}  #  Пустой словарь (для совместимости с кодом)
    start_time = time.time()
    agent.train(timeout_event, result)
    end_time = time.time()

    print(f"Training completed in {end_time - start_time:.2f} seconds.")

    #  Сохраняем прогресс (после обучения)
    agent.save_progress()


if __name__ == "__main__":
    main()
