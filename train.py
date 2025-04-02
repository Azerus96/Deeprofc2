import ai_engine
import github_utils
import getpass
import time
import os
from threading import Event
import logging
import multiprocessing as mp # <--- Импортируем multiprocessing

# Настройка логирования
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
        'fantasyType': 'standard',
        'aiType': 'mccfr',
        'training_mode': True,
        # 'normal_placement_limit': 500,
        # 'fantasy_placement_limit': 2000,
    }
    logger.info(f"AI Settings: {ai_settings}")

    # 3. Параметры обучения CFR
    training_iterations = 1000000
    max_cfr_nodes = 1000000
    batch_processing_size = 64
    convergence_threshold = 0.001
    num_parallel_workers = None # None = os.cpu_count()

    logger.info(f"Training Parameters: Iterations={training_iterations}, Max Nodes={max_cfr_nodes}, Batch Size={batch_processing_size}, Convergence Threshold={convergence_threshold}, Workers={num_parallel_workers or 'Default (CPU Count)'}")


    # 4. Создаем агента CFR
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
        return

    # 5. Загружаем прогресс
    agent.load_progress()

    # 6. Настраиваем событие для возможной остановки
    timeout_event = Event()

    # 7. Запускаем обучение
    result = {}
    logger.info("Starting agent training...")
    start_time = time.time()
    try:
        agent.train(timeout_event, result)
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user (KeyboardInterrupt).")
    except Exception as e:
        logger.exception("An error occurred during training.")
        logger.info("Attempting to save progress after error...")
        agent.save_progress(0)

    end_time = time.time()
    logger.info(f"Training process finished or interrupted in {end_time - start_time:.2f} seconds.")

    logger.info("--- Training Script Finished ---")


if __name__ == "__main__":
    # --- ВАЖНО: Устанавливаем метод старта multiprocessing ---
    # Делаем это *перед* любым созданием пула процессов (внутри agent.train)
    # и внутри блока if __name__ == "__main__"
    try:
        # 'spawn' безопаснее для JAX и других библиотек с потоками
        mp.set_start_method('spawn', force=True)
        logger.info("Multiprocessing start method set to 'spawn'.")
    except RuntimeError as e:
        # Контекст может быть уже установлен (например, в некоторых средах)
        logger.warning(f"Could not set multiprocessing start method (possibly already set): {e}")
        current_method = mp.get_start_method()
        logger.info(f"Current multiprocessing start method: '{current_method}'")
        if current_method != 'spawn':
             logger.warning("Current start method is not 'spawn', potential issues with JAX might still occur.")
    except Exception as e:
        logger.exception("An unexpected error occurred while setting multiprocessing start method.")

    # Запускаем основную функцию
    main()
