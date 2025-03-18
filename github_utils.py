from github import Github, GithubException
import os
import base64
import logging
import time
import pickle
from typing import Optional

# Настройка логирования
logger = logging.getLogger(__name__)

# GitHub repository settings (can be overridden by environment variables)
GITHUB_USERNAME = os.environ.get("GITHUB_USERNAME") or "Azerus96"  # ЗАМЕНИТЕ на ваше имя пользователя
GITHUB_REPOSITORY = os.environ.get("GITHUB_REPOSITORY") or "deeprofc2"  # ЗАМЕНИТЕ на имя вашего репозитория
AI_PROGRESS_FILENAME = "cfr_data.pkl"


def save_ai_progress_to_github(data, filename: str = AI_PROGRESS_FILENAME) -> bool:
    """
    Сохранение прогресса ИИ в GitHub.

    Args:
        data: Данные для сохранения (словарь).
        filename (str): Имя файла для сохранения. По умолчанию AI_PROGRESS_FILENAME.

    Returns:
        bool: True если сохранение успешно, False иначе.
    """
    token = os.environ.get("AI_PROGRESS_TOKEN")
    if not token:
        logger.warning("AI_PROGRESS_TOKEN не установлен. Сохранение отключено.")
        return False

    logger.info(f"Сохранение прогресса ИИ на GitHub.")

    try:
        g = Github(token)
        repo = g.get_user(GITHUB_USERNAME).get_repo(GITHUB_REPOSITORY)

        try:
            # Получаем текущую версию с GitHub (если есть)
            contents = repo.get_contents(filename, ref="main")
            # Сериализуем данные в байты
            serialized_data = pickle.dumps(data)
            # Всегда обновляем файл
            repo.update_file(
                contents.path,
                f"Обновление прогресса ИИ ({time.strftime('%Y-%m-%d %H:%M:%S')})",
                serialized_data,
                contents.sha,
                branch="main",
            )
            logger.info(f"✅ Прогресс ИИ успешно сохранен на GitHub: {GITHUB_REPOSITORY}/{filename}")
            return True

        except GithubException as e:
            if e.status == 404:
                # Создаем новый файл, если не существует
                serialized_data = pickle.dumps(data)
                repo.create_file(
                    filename,
                    f"Начальный прогресс ИИ ({time.strftime('%Y-%m-%d %H:%M:%S')})",
                    serialized_data,
                    branch="main",
                )
                logger.info(f"✅ Создан новый файл прогресса на GitHub: {GITHUB_REPOSITORY}/{filename}")
                return True
            else:
                logger.error(f"Ошибка GitHub API: {e.status}, {e.data}")
                return False
    except Exception as e:
        logger.exception(f"Неожиданная ошибка при сохранении: {e}")
        return False


def load_ai_progress_from_github(filename: str = AI_PROGRESS_FILENAME) -> Optional[dict]:
    """
    Загрузка прогресса ИИ из GitHub.

    Args:
        filename (str): Имя файла для загрузки. По умолчанию AI_PROGRESS_FILENAME.

    Returns:
        Optional[dict]:  Словарь с данными, если загрузка успешна, None иначе.
    """
    token = os.environ.get("AI_PROGRESS_TOKEN")
    if not token:
        logger.warning("AI_PROGRESS_TOKEN не установлен. Загрузка отключена.")
        return None

    try:
        g = Github(token)
        repo = g.get_user(GITHUB_USERNAME).get_repo(GITHUB_REPOSITORY)

        try:
            logger.info(f"Попытка загрузки прогресса ИИ с GitHub: {GITHUB_REPOSITORY}/{filename}")
            contents = repo.get_contents(filename, ref="main")
            file_content = base64.b64decode(contents.content)

            # Проверяем, не пустой ли файл
            if len(file_content) == 0:
                logger.warning("GitHub файл пуст. Отмена загрузки.")
                return None

            # Десериализуем данные
            data = pickle.loads(file_content)

            logger.info(f"✅ Прогресс ИИ успешно загружен с GitHub: {GITHUB_REPOSITORY}/{filename}")
            return data

        except GithubException as e:
            if e.status == 404:
                logger.warning(f"Файл {filename} не найден в репозитории GitHub.")
                return None
            else:
                logger.error(f"Ошибка GitHub API при загрузке: status={e.status}, data={e.data}")
                return None
    except Exception as e:
        logger.exception(f"Неожиданная ошибка при загрузке с GitHub: {e}")
        return None
