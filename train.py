# train_tpu_compatible.py
# Версия, адаптированная для запуска на TPU в Colab.
# - Убран блок mp.set_start_method('spawn'), т.к. он не нужен для ThreadPoolExecutor
#   и мог конфликтовать с инициализацией TPU.
# - Добавлена проверка jax.devices() в начале main().
# - Убран один из дублирующихся вызовов agent.load_progress().

import ai_engine as ai_engine # Используем последнюю исправленную версию
import github_utils
import getpass
import time
import os
from threading import Event
import logging
# import multiprocessing as mp # <-- Больше не импортируем multiprocessing
import jax # Импортируем JAX здесь для ранней проверки

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("--- Starting Training Script (TPU Compatible Version) ---")

    # --- Ранняя проверка доступности устройств JAX ---
    # Это поможет убедиться, что JAX видит TPU до создания агента.
    try:
        devices = jax.devices()
        logger.info(f"JAX Devices Detected: {devices}")
        # Проверяем, есть ли TPU в списке
        is_tpu_available = any(d.platform.lower() == 'tpu' for d in devices)
        if is_tpu_available:
            logger.info("✅ TPU detected by JAX.")
        else:
            logger.warning("⚠️ TPU not detected by JAX. Check Colab Runtime settings and JAX installation.")
            # Можно либо выйти, либо продолжить на CPU/GPU, если они есть
            # return # Раскомментировать, если нужно прервать выполнение без TPU
    except Exception as e:
        logger.error(f"Error checking JAX devices: {e}", exc_info=True)
        logger.error("Cannot proceed without confirming JAX device availability.")
        return

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


    # 2. Настройки AI (убедимся, что fantasyType='standard')
    ai_settings = {
        'fantasyType': 'standard', # Установлено согласно нашему решению
        'aiType': 'mccfr',
        'training_mode': True,
        # 'normal_placement_limit': 500, # Можно раскомментировать для настройки
        # 'fantasy_placement_limit': 2000,
        # 'street1_placement_limit': 10000,
    }
    logger.info(f"AI Settings: {ai_settings}")

    # 3. Параметры обучения CFR
    training_iterations = 1000000
    max_cfr_nodes = 1000000 # Важно для TPU с большим объемом памяти! Можно увеличить.
    batch_processing_size = 3
    convergence_threshold = 0.001
    num_parallel_workers = None # None = os.cpu_count() - для ThreadPoolExecutor это нормально

    logger.info(f"Training Parameters: Iterations={training_iterations}, Max Nodes={max_cfr_nodes}, Batch Size={batch_processing_size}, Convergence Threshold={convergence_threshold}, Workers={num_parallel_workers or 'Default (CPU Count)'}")


    # 4. Создаем агента CFR
    try:
        # Используем имя импортированного модуля
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
        # Ловим ошибки, которые могут возникнуть здесь, включая возможные проблемы JAX
        logger.exception("Failed to create CFRAgent. Exiting.")
        return

    # 5. Загружаем прогресс (ОДИН РАЗ, лучше внутри agent.train)
    # agent.load_progress() # <-- Убрал отсюда, т.к. вызывается внутри train()

    # 6. Настраиваем событие для возможной остановки
    timeout_event = Event()

    # 7. Запускаем обучение
    result = {}
    logger.info("Starting agent training...")
    start_time = time.time()
    try:
        # agent.train() вызовет load_progress() внутри себя
        agent.train(timeout_event, result)
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user (KeyboardInterrupt).")
        logger.info("Attempting to save progress after interruption...")
        # Передаем 0, т.к. точное число обработанных игр неизвестно здесь
        # Лучше, если train() сам сохранит прогресс перед выходом из-за KeyboardInterrupt
        agent.save_progress(result.get("games_processed", 0)) # Попытка сохранить актуальное число
    except Exception as e:
        logger.exception("An critical error occurred during training.")
        logger.info("Attempting to save progress after error...")
        agent.save_progress(result.get("games_processed", 0)) # Попытка сохранить актуальное число

    end_time = time.time()
    logger.info(f"Training process finished or interrupted in {end_time - start_time:.2f} seconds.")
    if "nodes_count" in result:
        logger.info(f"Final Nodes Count: {result['nodes_count']}")
    if "games_processed" in result:
        logger.info(f"Total Games Processed: {result['games_processed']}")

    logger.info("--- Training Script Finished ---")


if __name__ == "__main__":
    # --- Блок mp.set_start_method УДАЛЕН ---
    # Он не нужен для ThreadPoolExecutor и мог конфликтовать с TPU.

    # Запускаем основную функцию
    main()
