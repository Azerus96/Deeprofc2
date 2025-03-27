from github import Github, GithubException, RateLimitExceededException
import os
import base64
import logging
import time
import pickle
from typing import Optional, Any

# Настройка логирования для этого модуля
logger = logging.getLogger(__name__)

# --- Настройки GitHub ---
# Имя пользователя и репозиторий GitHub для сохранения/загрузки прогресса.
# ЗАМЕНИТЕ 'YourGitHubUsername' и 'YourRepositoryName' на ваши значения!
# Или установите переменные окружения GITHUB_USERNAME и GITHUB_REPOSITORY.
DEFAULT_GITHUB_USERNAME = "Azerus96"
DEFAULT_GITHUB_REPOSITORY = "Deeprofc2"
# Имя файла для сохранения данных в репозитории
DEFAULT_AI_PROGRESS_FILENAME = "cfr_data_ofc.pkl"

# Получаем настройки из переменных окружения или используем значения по умолчанию
GITHUB_USERNAME = os.environ.get("GITHUB_USERNAME") or DEFAULT_GITHUB_USERNAME
GITHUB_REPOSITORY = os.environ.get("GITHUB_REPOSITORY") or DEFAULT_GITHUB_REPOSITORY


def save_ai_progress_to_github(data: Any, filename: str = DEFAULT_AI_PROGRESS_FILENAME) -> bool:
    """
    Сохраняет данные прогресса ИИ (сериализованные с помощью pickle) в указанный файл
    в репозитории GitHub.

    Args:
        data: Данные для сохранения (любой объект, поддерживаемый pickle).
        filename (str): Имя файла для сохранения в репозитории GitHub.

    Returns:
        bool: True, если сохранение прошло успешно, False в противном случае.
    """
    # Получаем токен из переменной окружения, установленной в train.py
    token = os.environ.get("AI_PROGRESS_TOKEN")
    if not token:
        logger.warning("GitHub token (AI_PROGRESS_TOKEN) not found or empty. Saving to GitHub is disabled.")
        return False

    if GITHUB_USERNAME == DEFAULT_GITHUB_USERNAME or GITHUB_REPOSITORY == DEFAULT_GITHUB_REPOSITORY:
         logger.warning(f"Using default GitHub username/repository ({DEFAULT_GITHUB_USERNAME}/{DEFAULT_GITHUB_REPOSITORY}). Please update github_utils.py or set environment variables.")

    logger.info(f"Attempting to save AI progress to GitHub: {GITHUB_USERNAME}/{GITHUB_REPOSITORY}/{filename}")

    try:
        # Сериализуем данные перед подключением к GitHub
        serialized_data = pickle.dumps(data)
        logger.debug(f"Data serialized successfully ({len(serialized_data)} bytes).")

        # Подключаемся к GitHub
        g = Github(token)
        # Получаем объект пользователя
        user = g.get_user(GITHUB_USERNAME)
        # Получаем объект репозитория
        repo = user.get_repo(GITHUB_REPOSITORY)
        logger.debug("Connected to GitHub repository.")

        commit_message = f"Update AI progress ({time.strftime('%Y-%m-%d %H:%M:%S')})"
        branch_name = "main" # Или другая основная ветка

        try:
            # Пытаемся получить текущее содержимое файла, чтобы узнать его SHA
            contents = repo.get_contents(filename, ref=branch_name)
            logger.debug(f"File '{filename}' found. Updating existing file.")
            # Обновляем существующий файл
            repo.update_file(
                path=contents.path,
                message=commit_message,
                content=serialized_data,
                sha=contents.sha,
                branch=branch_name,
            )
            logger.info(f"✅ AI progress successfully updated on GitHub: {GITHUB_USERNAME}/{GITHUB_REPOSITORY}/{filename}")
            return True

        except GithubException as e:
            if e.status == 404:
                # Файл не найден, создаем новый
                logger.info(f"File '{filename}' not found. Creating new file.")
                repo.create_file(
                    path=filename,
                    message=commit_message,
                    content=serialized_data,
                    branch=branch_name,
                )
                logger.info(f"✅ New AI progress file created on GitHub: {GITHUB_USERNAME}/{GITHUB_REPOSITORY}/{filename}")
                return True
            elif e.status == 409: # Conflict - может возникнуть при параллельных сохранениях (маловероятно здесь)
                 logger.error(f"GitHub API conflict (409) while saving '{filename}'. Retrying might be needed.")
                 return False
            else:
                # Другая ошибка GitHub API
                logger.error(f"GitHub API error while saving '{filename}': Status={e.status}, Data={e.data}", exc_info=True)
                return False
        except RateLimitExceededException:
             logger.error("GitHub API rate limit exceeded. Please wait before trying again.")
             return False

    except pickle.PicklingError as pe:
        logger.error(f"Error serializing data with pickle: {pe}", exc_info=True)
        return False
    except GithubException as ge:
         # Ошибка при получении пользователя или репозитория
         logger.error(f"GitHub API error during connection/setup: Status={ge.status}, Data={ge.data}", exc_info=True)
         return False
    except Exception as e:
        # Ловим другие неожиданные ошибки
        logger.exception(f"Unexpected error during save_ai_progress_to_github: {e}")
        return False


def load_ai_progress_from_github(filename: str = DEFAULT_AI_PROGRESS_FILENAME) -> Optional[Any]:
    """
    Загружает и десериализует данные прогресса ИИ из указанного файла
    в репозитории GitHub.

    Args:
        filename (str): Имя файла для загрузки из репозитория GitHub.

    Returns:
        Optional[Any]: Загруженные и десериализованные данные, если успешно,
                       None в противном случае.
    """
    token = os.environ.get("AI_PROGRESS_TOKEN")
    if not token:
        logger.warning("GitHub token (AI_PROGRESS_TOKEN) not found or empty. Loading from GitHub is disabled.")
        return None

    if GITHUB_USERNAME == DEFAULT_GITHUB_USERNAME or GITHUB_REPOSITORY == DEFAULT_GITHUB_REPOSITORY:
         logger.warning(f"Using default GitHub username/repository ({DEFAULT_GITHUB_USERNAME}/{DEFAULT_GITHUB_REPOSITORY}). Please update github_utils.py or set environment variables.")

    logger.info(f"Attempting to load AI progress from GitHub: {GITHUB_USERNAME}/{GITHUB_REPOSITORY}/{filename}")

    try:
        # Подключаемся к GitHub
        g = Github(token)
        user = g.get_user(GITHUB_USERNAME)
        repo = user.get_repo(GITHUB_REPOSITORY)
        logger.debug("Connected to GitHub repository.")

        branch_name = "main"

        try:
            # Получаем содержимое файла
            contents = repo.get_contents(filename, ref=branch_name)
            logger.debug(f"File '{filename}' found. Decoding content.")

            # Декодируем содержимое из base64
            file_content_bytes = base64.b64decode(contents.content)

            # Проверяем, не пустой ли файл
            if not file_content_bytes:
                logger.warning(f"Loaded file '{filename}' from GitHub is empty. Returning None.")
                return None

            # Десериализуем данные с помощью pickle
            logger.debug("Deserializing data using pickle...")
            data = pickle.loads(file_content_bytes)

            logger.info(f"✅ AI progress successfully loaded from GitHub: {GITHUB_USERNAME}/{GITHUB_REPOSITORY}/{filename}")
            return data

        except GithubException as e:
            if e.status == 404:
                # Файл не найден - это нормальная ситуация при первом запуске
                logger.warning(f"File '{filename}' not found in the GitHub repository. No progress loaded.")
                return None
            else:
                # Другая ошибка GitHub API
                logger.error(f"GitHub API error while loading '{filename}': Status={e.status}, Data={e.data}", exc_info=True)
                return None
        except RateLimitExceededException:
             logger.error("GitHub API rate limit exceeded. Please wait before trying again.")
             return None

    except pickle.UnpicklingError as ue:
        logger.error(f"Error deserializing data from '{filename}' with pickle: {ue}. The file might be corrupted or incompatible.", exc_info=True)
        return None
    except GithubException as ge:
         # Ошибка при получении пользователя или репозитория
         logger.error(f"GitHub API error during connection/setup: Status={ge.status}, Data={ge.data}", exc_info=True)
         return None
    except Exception as e:
        # Ловим другие неожиданные ошибки
        logger.exception(f"Unexpected error during load_ai_progress_from_github: {e}")
        return None
