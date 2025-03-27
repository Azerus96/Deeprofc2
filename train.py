import ai_engine
# Убедимся, что github_utils импортируется ПЕРЕД использованием в ai_engine
# (хотя в данном случае ai_engine импортирует его сам)
import github_utils
import getpass
import time
import os
from threading import Event
import logging # Добавим импорт logging

# Настройка логирования (можно настроить уровень и формат)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("--- Starting Training Script ---")

    # 1. Получаем токен GitHub
    try:
        token = getpass.getpass("Enter your GitHub token (for saving/loading progress): ")
        if not token:
            logger.warning("No GitHub token provided. Saving/Loading progress will be disabled.")
            os.environ["AI_PROGRESS_TOKEN"] = ""
        else:
            os.environ["AI_PROGRESS_TOKEN"] = token
            logger.info("GitHub token received.")
    except Exception as e:
        logger.error(f"Error getting GitHub token: {e}. Proceeding without GitHub integration.")
        os.environ["AI_PROGRESS_TOKEN"] = ""


    # 2. Настройки AI (общие)
    ai_settings = {
        'fantasyType': 'normal',  # 'normal' или 'progressive'
        'aiType': 'mccfr',        # Тип агента
        'training_mode': True,    # Обязательно True для обучения
        # Можно добавить лимиты для генерации действий, если нужно переопределить значения по умолчанию
        # 'normal_placement_limit': 500,
        # 'fantasy_placement_limit': 2000,
    }
    logger.info(f"AI Settings: {ai_settings}")

    # 3. Параметры обучения CFR (ЯВНО ЗАДАНЫ ЗДЕСЬ)
    training_iterations = 1000000  # Желаемое количество симуляций игр
    max_cfr_nodes = 1000000      # Максимальное количество узлов для хранения
    batch_processing_size = 64   # Количество игр, симулируемых параллельно в батче
    convergence_threshold = 0.001 # Порог сходимости (среднее абсолютное сожаление)
    num_parallel_workers = None  # Количество параллельных процессов (None = os.cpu_count())

    logger.info(f"Training Parameters: Iterations={training_iterations}, Max Nodes={max_cfr_nodes}, Batch Size={batch_processing_size}, Convergence Threshold={convergence_threshold}, Workers={num_parallel_workers or 'Default (CPU Count)'}")


    # 4. Создаем агента CFR с ЯВНО УКАЗАННЫМИ ПАРАМЕТРАМИ
    try:
        agent = ai_engine.CFRAgent(
            iterations=training_iterations,
            stop_threshold=convergence_threshold,
            batch_size=batch_processing_size,
            max_nodes=max_cfr_nodes,
            ai_settings=ai_settings,
            num_workers=num_parallel_workers
        )
        logger.info("CFRAgent created successfully with specified parameters.")
    except Exception as e:
        logger.exception("Failed to create CFRAgent. Exiting.")
        return # Выход, если агент не создан

    # 5. Загружаем прогресс (если есть и токен предоставлен)
    agent.load_progress()

    # 6. Настраиваем событие для возможной остановки
    timeout_event = Event()

    # 7. Запускаем обучение
    result = {} # Пустой словарь для совместимости сигнатуры train
    logger.info("Starting agent training...")
    start_time = time.time()
    try:
        agent.train(timeout_event, result) # Обучение происходит здесь
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user (KeyboardInterrupt).")
        # Агент должен сохраняться периодически и в конце train
    except Exception as e:
        logger.exception("An error occurred during training.")
        # Попытка сохранить прогресс даже при ошибке
        logger.info("Attempting to save progress after error...")
        agent.save_progress(0) # Передаем 0, т.к. точное число неизвестно

    end_time = time.time()
    logger.info(f"Training process finished or interrupted in {end_time - start_time:.2f} seconds.")

    logger.info("--- Training Script Finished ---")


if __name__ == "__main__":
    main()
