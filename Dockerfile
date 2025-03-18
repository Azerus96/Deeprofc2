# Используем официальный образ Python
FROM python:3.9

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файлы проекта в контейнер
COPY . /app

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Запускаем приложение с помощью Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]
