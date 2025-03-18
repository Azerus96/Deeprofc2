from flask import Flask, render_template, jsonify, session, request
import os
import ai_engine
from ai_engine import CFRAgent, RandomAgent, Card, GameState  # Добавляем GameState
import github_utils  # utils УДАЛЯЕМ, github_utils ДОБАВЛЯЕМ
import time
import json
from threading import Thread, Event
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Секретный ключ для сессий Flask

# Глобальные экземпляры AI
cfr_agent = None  # Инициализируем, но создаем, только если нужен MCCFR
random_agent = RandomAgent()  # Для случайных ходов


def initialize_ai_agent(ai_settings):
    """Инициализирует AI агента (CFR или Random) на основе настроек."""
    global cfr_agent
    logger.info(f"Инициализация AI агента с настройками: {ai_settings}")
    try:
        iterations = int(ai_settings.get("iterations", 500000))  # Увеличиваем default
        stop_threshold = float(ai_settings.get("stopThreshold", 0.001))
        batch_size = int(ai_settings.get("batch_size", 1))  #  Добавляем batch_size
    except ValueError as e:
        logger.error(
            f"Неверные значения iterations, stopThreshold или batch_size: {e}. Используются значения по умолчанию."
        )
        iterations = 500000  # Увеличиваем default
        stop_threshold = 0.001
        batch_size = 1

    if ai_settings.get("aiType") == "mccfr":  # Создаем CFRAgent только если нужен MCCFR
        cfr_agent = CFRAgent(iterations=iterations, stop_threshold=stop_threshold, batch_size=batch_size)
        logger.info(f"AI агент MCCFR инициализирован: {cfr_agent}")

        if os.environ.get("AI_PROGRESS_TOKEN"):  # Если задан токен GitHub
            try:
                # Загрузка с GitHub
                logger.info("Попытка загрузить прогресс AI с GitHub...")
                if github_utils.load_ai_progress_from_github():
                    data = github_utils.load_ai_progress_from_github() #  Загружаем с GitHub!
                    if data:
                        cfr_agent.nodes_mask = data["nodes_mask"]
                        cfr_agent.regret_sums = data["regret_sums"]
                        cfr_agent.strategy_sums = data["strategy_sums"]
                        cfr_agent.num_actions_arr = data["num_actions_arr"]
                        cfr_agent.node_counter = data["node_counter"]
                        cfr_agent.nodes_map = data["nodes_map"]  #  Загружаем nodes_map
                        cfr_agent.iterations = data["iterations"]
                        cfr_agent.stop_threshold = data.get("stop_threshold", 0.001)  #  Используем значение по умолчанию
                        cfr_agent.batch_size = data.get("batch_size", 1)
                        cfr_agent.max_nodes = data.get("max_nodes", 100000)
                        logger.info("Прогресс AI успешно загружен и применен к агенту.")
                    else:
                        logger.warning("Прогресс AI с GitHub загружен, но данные повреждены или пусты.")
                else:
                    logger.warning("Не удалось загрузить прогресс AI с GitHub.")

            except Exception as e:
                logger.error(f"Ошибка загрузки прогресса AI: {e}")
        else:
            logger.info("AI_PROGRESS_TOKEN не установлен. Загрузка прогресса отключена.")
    else:
        cfr_agent = None  # Если не MCCFR, то cfr_agent не нужен
        logger.info("Используется случайный AI агент.")


# ... (serialize_card - без изменений)

def serialize_move(move):
    """Преобразует действие (move) в словарь (для JSON)."""
    logger.debug(f"Сериализация хода: {move}")
    if move is None:  # Добавляем обработку None
        return None
    #  Преобразуем JAX-массив в список списков словарей
    serialized = {}
    for line, cards in move.items():
        if isinstance(cards, jnp.ndarray):
            serialized[line] = []
            for card_array in cards:
                if not jnp.array_equal(card_array, jnp.array([-1, -1])):
                    serialized[line].append(ai_engine.array_to_card(card_array).to_dict())
        elif isinstance(cards, list):
            serialized[line] = [card.to_dict() if isinstance(card, Card) else None for card in cards]
        else:
            serialized[line] = cards #  Если там что-то другое (например, None)
    logger.debug(f"Сериализованный ход: {serialized}")
    return serialized

def action_from_array(action_array: jnp.ndarray) -> Dict[str, List[Card]]:
    """Преобразует JAX-массив действия обратно в словарь."""
    action_dict = {
        "top": [],
        "middle": [],
        "bottom": [],
        "discarded": []
    }
    for i in range(3):
        card = array_to_card(action_array[i])
        if card:
            action_dict["top"].append(card)
    for i in range(3, 8):
        card = array_to_card(action_array[i])
        if card:
            action_dict["middle"].append(card)
    for i in range(8, 13):
        card = array_to_card(action_array[i])
        if card:
            action_dict["bottom"].append(card)
    for i in range(13, min(17, len(action_array))):  #  Учитываем, что discarded может быть разной длины
        card = array_to_card(action_array[i])
        if card:
            action_dict["discarded"].append(card)
    return action_dict

@app.route("/")
def home():
    """Главная страница (не используется в данном приложении)."""
    logger.debug("Обработка запроса главной страницы")
    return render_template("index.html")  # Предполагается, что есть index.html


@app.route("/training")
def training():
    """Страница тренировки."""
    logger.debug("Обработка запроса страницы тренировки")

    # Инициализация состояния сессии
    session["game_state"] = {
        "selected_cards": [],
        "board": {
            "top": [None] * 3,
            "middle": [None] * 5,
            "bottom": [None] * 5,
        },
        "discarded_cards": [],
        "ai_settings": {  # Устанавливаем правильные значения по умолчанию
            "fantasyType": "normal",
            "fantasyMode": False,
            "aiTime": "60",
            "iterations": "500000",  # Увеличиваем default
            "stopThreshold": "0.0001",
            "aiType": "mccfr",
            "placementMode": "standard",  # Убрал, т.к. это внутренняя логика
            "training_mode": True,  # ДОБАВЛЯЕМ training_mode
            "batch_size": "1"
        },
    }
    logger.info(f"Инициализировано состояние игры: {session['game_state']}")

    # Проверка необходимости реинициализации AI (только если изменились настройки)
    if session.get("previous_ai_settings") != session["game_state"]["ai_settings"]:
        initialize_ai_agent(session["game_state"]["ai_settings"])
        session["previous_ai_settings"] = session["game_state"]["ai_settings"].copy()  # Важно: копируем!
        logger.info(f"Реинициализирован AI агент с настройками: {session['game_state']['ai_settings']}")

    logger.info(f"Текущее состояние игры в сессии: {session['game_state']}")
    return render_template("training.html", game_state=session["game_state"])


@app.route("/update_state", methods=["POST"])
def update_state():
    """Обновление состояния игры (вызывается из JavaScript)."""
    logger.debug("Обработка запроса обновления состояния - START")
    if not request.is_json:
        logger.error("Ошибка: Запрос не в формате JSON")
        return jsonify({"error": "Content type must be application/json"}), 400

    try:
        game_state = request.get_json()
        logger.debug(f"Получено обновление состояния игры: {game_state}")

        if not isinstance(game_state, dict):
            logger.error("Ошибка: Неверный формат состояния игры (не словарь)")
            return jsonify({"error": "Invalid game state format"}), 400

        # Инициализация состояния игры в сессии, если его нет
        if "game_state" not in session:
            session["game_state"] = {}  # Пустой словарь, если не было
            logger.info("Инициализировано новое состояние сессии при обновлении.")

        logger.debug(f"Состояние сессии ДО обновления: {session['game_state']}")

        # Обновление доски - сохраняем существующие карты
        if "board" in game_state:
            current_board = session["game_state"].get(
                "board",
                {  # Используем .get()
                    "top": [None] * 3,
                    "middle": [None] * 5,
                    "bottom": [None] * 5,
                },
            )

            # Обновляем только новые карты, сохраняя существующие
            for line in ["top", "middle", "bottom"]:
                if line in game_state["board"]:
                    new_line = game_state["board"][line]
                    current_line = current_board.get(line, [])  # Используем .get()
                    for i, new_card in enumerate(new_line):
                        if i < len(current_line):  # Проверяем индекс
                            if new_card is not None:
                                # Важно: преобразуем словарь в объект Card
                                current_line[i] = (
                                    Card.from_dict(new_card) if isinstance(new_card, dict) else None
                                )
                    current_board[line] = current_line

            session["game_state"]["board"] = current_board
            logger.debug(f"Обновленная доска: {session['game_state']['board']}")

        # Обновление других ключей. Преобразуем словари в объекты Card.
        for key in ["selected_cards", "discarded_cards"]:
            if key in game_state:
                session["game_state"][key] = [
                    Card.from_dict(card) if isinstance(card, dict) else None
                    for card in game_state[key]
                ]
                logger.debug(f"Обновлены {key} в сессии: {session['game_state'][key]}")

        # Добавляем карты, удаленные через "-", в discarded_cards
        if "removed_cards" in game_state:
            removed_cards = [
                Card.from_dict(card) if isinstance(card, dict) else card
                for card in game_state["removed_cards"]
            ]

            # Исправление: преобразуем карты в строковое представление для сравнения
            discarded_cards_str = [str(card) for card in session["game_state"].get("discarded_cards", [])]

            # Добавляем только новые карты
            for card in removed_cards:
                if str(card) not in discarded_cards_str:
                    session["game_state"].setdefault("discarded_cards", []).append(card)

            logger.debug(
                f"removed_cards добавлены в discarded_cards сессии: {session['game_state']['discarded_cards']}"
            )

            # Удаляем удаленные карты из selected_cards (если они там есть)
            session["game_state"]["selected_cards"] = [
                card
                for card in session["game_state"].get("selected_cards", [])
                if str(card) not in [str(removed_card) for removed_card in removed_cards]
            ]

        if "ai_settings" in game_state:
            #  ДОБАВЛЯЕМ training_mode
            ai_settings = game_state["ai_settings"]
            session["game_state"]["ai_settings"] = ai_settings
            # Реинициализация AI агента при изменении настроек
            if ai_settings != session.get("previous_ai_settings"):
                logger.info("Настройки AI изменились, реинициализация агента")
                initialize_ai_agent(ai_settings)
                session["previous_ai_settings"] = ai_settings.copy()  # Важно: копируем!

        logger.debug(f"Состояние сессии ПОСЛЕ обновления: {session['game_state']}")
        logger.debug("Обработка запроса обновления состояния - END")
        return jsonify({"status": "success"})

    except Exception as e:
        logger.exception(f"Ошибка в update_state: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/ai_move", methods=["POST"])
def ai_move():
    global cfr_agent
    global random_agent

    logger.debug("Обработка запроса хода AI - START")
    game_state_data = request.get_json()
    logger.debug(f"Получены данные состояния игры для хода AI: {game_state_data}")

    if not isinstance(game_state_data, dict):
        logger.error("Ошибка: game_state_data не является словарем")
        return jsonify({"error": "Invalid game state data format"}), 400

    num_cards = len(game_state_data.get("selected_cards", []))
    ai_settings = game_state_data.get("ai_settings", {})
    ai_type = ai_settings.get("aiType", "mccfr")  # 'mccfr' по умолчанию

    try:
        # Обработка и валидация данных
        selected_cards = [Card.from_dict(card) for card in game_state_data.get("selected_cards", [])]
        discarded_cards = [Card.from_dict(card) for card in game_state_data.get("discarded_cards", [])]

        board_data = game_state_data.get("board", {})
        board = ai_engine.Board()
        for line in ["top", "middle", "bottom"]:
            line_data = board_data.get(line, [])
            for card_data in line_data:
                if card_data:  # Проверяем, что card_data не None
                    board.place_card(line, Card.from_dict(card_data))
        logger.debug(f"Обработанная доска: {board}")

        # Создание состояния игры
        game_state = ai_engine.GameState(
            selected_cards=selected_cards,
            board=board,
            discarded_cards=discarded_cards,
            ai_settings=ai_settings,
            deck=ai_engine.Card.get_all_cards(),  # Передаем полную колоду
        )
        logger.debug(f"Создано состояние игры: {game_state}")

        #  УДАЛЯЕМ проверку на терминальное состояние ДО хода.
        #  Теперь терминальное состояние определяется ПОСЛЕ применения действия.

        #  УДАЛЯЕМ проверку на отсутствие доступных ходов (это делает get_placement)

        # Если игра не завершена, получаем ход AI
        timeout_event = Event()
        result = {"move": None}

        # Выбор AI агента
        if ai_type == "mccfr":
            if cfr_agent is None:  # Проверяем, инициализирован ли агент
                logger.error("Ошибка: MCCFR агент не инициализирован")
                return jsonify({"error": "MCCFR agent not initialized"}), 500
            ai_thread = Thread(target=cfr_agent.get_move, args=(game_state, timeout_event, result))
        elif ai_type == "random":
            ai_thread = Thread(target=random_agent.get_move, args=(game_state, timeout_event, result))
        else:
            logger.error(f"Неизвестный тип AI агента: {ai_type}")
            return jsonify({"error": f"Unknown AI agent type: {ai_type}"}), 400

        ai_thread.start()
        ai_thread.join(timeout=int(ai_settings.get("aiTime", 60)))  # Таймаут из настроек, default = 60

        if ai_thread.is_alive():
            timeout_event.set()
            ai_thread.join()  # Ожидаем завершения потока
            logger.warning("Время ожидания хода AI истекло")
            return jsonify({"error": "AI move timed out"}), 504

        move = result.get("move")
        logger.debug(f"Получен ход AI: {move}")

        #  Добавляем проверку на наличие ошибки
        if move is None or "error" in move:
            error_message = "Неизвестная ошибка при получении хода AI"
            if move and "error" in move:
                error_message = move["error"]
            logger.error(f"Ошибка хода AI: {error_message}")
            return jsonify({"error": error_message}), 500


        #  БОЛЬШЕ НЕ ПРИМЕНЯЕМ ход к копии состояния игры.
        #  Вместо этого, возвращаем ход (move) в формате, пригодном для фронтенда.

        #  Возвращаем ход и информацию о роялти (если игра завершена)
        #  (расчет роялти и проверку на завершение игры делаем на стороне клиента)
        return jsonify({
            "move": serialize_move(move),  #  Сериализуем ход
            "royalties": {},  #  Пока что пустой словарь (роялти считаются на клиенте)
            "total_royalty": 0  #  Пока что 0
        }), 200

    except Exception as e:
        logger.exception(f"Ошибка в ai_move: {e}")
        return jsonify({"error": str(e)}), 500
