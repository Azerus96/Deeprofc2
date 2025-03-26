# ai_engine.py

import itertools
from collections import defaultdict, Counter
from threading import Event, Thread
import time
import math
import logging
from typing import List, Dict, Tuple, Optional, Union
from github_utils import save_ai_progress_to_github, load_ai_progress_from_github # Предполагается, что этот модуль доступен

import jax.numpy as jnp
import jax
from jax import random
from jax import jit
import numpy as np # Используем numpy для некоторых операций, не требующих JIT

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Класс Card ---
class Card:
    RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
    SUITS = ["♥", "♦", "♣", "♠"]
    RANK_MAP = {rank: i for i, rank in enumerate(RANKS)}
    SUIT_MAP = {suit: i for i, suit in enumerate(SUITS)}

    def __init__(self, rank: str, suit: str):
        if rank not in self.RANKS:
            raise ValueError(f"Invalid rank: {rank}. Rank must be one of: {self.RANKS}")
        if suit not in self.SUITS:
            raise ValueError(f"Invalid suit: {suit}. Suit must be one of: {self.SUITS}")
        self.rank = rank
        self.suit = suit

    def __repr__(self) -> str:
        return f"{self.rank}{self.suit}"

    def __eq__(self, other: Union["Card", Dict]) -> bool:
        if isinstance(other, dict):
            # Сравнение с словарем (например, из JSON)
            return self.rank == other.get("rank") and self.suit == other.get("suit")
        # Сравнение с другой картой
        return isinstance(other, Card) and self.rank == other.rank and self.suit == other.suit

    def __hash__(self) -> int:
        # Хеш на основе неизменяемых атрибутов
        return hash((self.rank, self.suit))

    def to_dict(self) -> Dict[str, str]:
        """Преобразует карту в словарь."""
        return {"rank": self.rank, "suit": self.suit}

    @staticmethod
    def from_dict(card_dict: Dict[str, str]) -> "Card":
        """Создает карту из словаря."""
        return Card(card_dict["rank"], card_dict["suit"])

    @staticmethod
    def get_all_cards() -> List["Card"]:
        """Возвращает стандартную колоду из 52 карт."""
        return [Card(rank, suit) for rank in Card.RANKS for suit in Card.SUITS]

# --- Класс Hand ---
class Hand:
    """Представляет руку игрока (карты, которые еще не размещены/сброшены)."""
    def __init__(self, cards: Optional[List[Card]] = None):
        # Обработка инициализации из JAX массива (если нужно)
        if isinstance(cards, jnp.ndarray):
             cards_list = [array_to_card(c) for c in cards if not jnp.array_equal(c, jnp.array([-1, -1]))]
             # Убираем None значения, которые могли появиться
             self.cards = [c for c in cards_list if c is not None]
        else:
             self.cards = cards if cards is not None else []

    def add_card(self, card: Card) -> None:
        """Добавляет карту в руку."""
        if not isinstance(card, Card):
            raise TypeError("card must be an instance of Card")
        self.cards.append(card)

    def add_cards(self, cards_to_add: List[Card]) -> None:
        """Добавляет несколько карт в руку."""
        for card in cards_to_add:
            self.add_card(card)

    def remove_card(self, card: Card) -> None:
        """Удаляет одну карту из руки."""
        if not isinstance(card, Card):
            raise TypeError("card must be an instance of Card")
        try:
            self.cards.remove(card)
        except ValueError:
            # Не страшно, если карты нет, но логируем предупреждение
            logger.warning(f"Card {card} not found in hand to remove: {self.cards}")

    def remove_cards(self, cards_to_remove: List[Card]) -> None:
        """Удаляет несколько карт из руки (более надежный способ)."""
        temp_hand = self.cards[:]
        removed_count = 0
        for card_to_remove in cards_to_remove:
            try:
                temp_hand.remove(card_to_remove)
                removed_count += 1
            except ValueError:
                logger.warning(f"Card {card_to_remove} not found in hand during multi-remove.")
        # Проверка, что удалили ожидаемое количество
        if removed_count != len(cards_to_remove):
             logger.warning(f"Expected to remove {len(cards_to_remove)} cards, but removed {removed_count}.")
        self.cards = temp_hand

    def __repr__(self) -> str:
        # Сортируем для единообразия вывода
        st = lambda cards: ", ".join(map(str, sorted(cards, key=lambda c: (Card.RANK_MAP[c.rank], Card.SUIT_MAP[c.suit]))))
        return f"Hand: [{st(self.cards)}]"

    def __len__(self) -> int:
        return len(self.cards)

    def __iter__(self):
        return iter(self.cards)

    def __getitem__(self, index: int) -> Card:
        return self.cards[index]

    def to_jax(self) -> jnp.ndarray:
         """Преобразует руку в JAX массив."""
         if not self.cards:
             return jnp.empty((0, 2), dtype=jnp.int32)
         return jnp.array([card_to_array(card) for card in self.cards], dtype=jnp.int32)

# --- Класс Board ---
class Board:
    """Представляет доску игрока с тремя линиями."""
    def __init__(self):
        self.top: List[Card] = []
        self.middle: List[Card] = []
        self.bottom: List[Card] = []

    def get_placed_count(self) -> int:
        """Возвращает общее количество карт на доске."""
        return len(self.top) + len(self.middle) + len(self.bottom)

    def place_card(self, line: str, card: Card) -> None:
        """Размещает карту на указанную линию."""
        target_line = getattr(self, line, None)
        if target_line is None:
             raise ValueError(f"Invalid line: {line}. Line must be one of: 'top', 'middle', 'bottom'")

        max_len = 3 if line == "top" else 5
        if len(target_line) >= max_len:
             raise ValueError(f"{line.capitalize()} line is full ({len(target_line)}/{max_len})")
        target_line.append(card)

    def is_full(self) -> bool:
        """Проверяет, заполнена ли доска полностью (13 карт)."""
        return len(self.top) == 3 and len(self.middle) == 5 and len(self.bottom) == 5

    def clear(self) -> None:
        """Очищает доску."""
        self.top = []
        self.middle = []
        self.bottom = []

    def __repr__(self) -> str:
        # Сортируем для единообразия вывода
        st = lambda cards: ", ".join(map(str, sorted(cards, key=lambda c: (Card.RANK_MAP[c.rank], Card.SUIT_MAP[c.suit]))))
        return f"Top: [{st(self.top)}]\nMiddle: [{st(self.middle)}]\nBottom: [{st(self.bottom)}]"

    def get_cards(self, line: str) -> List[Card]:
        """Возвращает список карт на указанной линии."""
        if line == "top": return self.top
        elif line == "middle": return self.middle
        elif line == "bottom": return self.bottom
        else: raise ValueError("Invalid line specified")

    def get_all_cards(self) -> List[Card]:
        """Возвращает все карты на доске в виде одного списка."""
        return self.top + self.middle + self.bottom

    def get_line_jax(self, line: str) -> jnp.ndarray:
        """Возвращает JAX массив карт на указанной линии."""
        cards = self.get_cards(line)
        if not cards:
            return jnp.empty((0, 2), dtype=jnp.int32)
        return jnp.array([card_to_array(card) for card in cards], dtype=jnp.int32)

    def to_jax_placement(self) -> jnp.ndarray:
        """Преобразует всю доску в JAX массив [13, 2] для оценки."""
        placement = jnp.full((13, 2), -1, dtype=jnp.int32) # 13 слотов для доски
        idx = 0
        for card in self.top:
             if idx < 3: placement = placement.at[idx].set(card_to_array(card)); idx += 1
        idx = 3
        for card in self.middle:
             if idx < 8: placement = placement.at[idx].set(card_to_array(card)); idx += 1
        idx = 8
        for card in self.bottom:
             if idx < 13: placement = placement.at[idx].set(card_to_array(card)); idx += 1
        return placement

# --- Вспомогательные функции для преобразования Card <-> JAX array ---
# Не используем JIT для функций, создающих/преобразующих объекты Python
def card_to_array(card: Optional[Card]) -> jnp.ndarray:
    """Преобразует Card в JAX-массив [rank_idx, suit_idx]."""
    if card is None:
        return jnp.array([-1, -1], dtype=jnp.int32)
    # Используем RANK_MAP и SUIT_MAP для получения индексов
    return jnp.array([Card.RANK_MAP.get(card.rank, -1), Card.SUIT_MAP.get(card.suit, -1)], dtype=jnp.int32)

def array_to_card(card_array: jnp.ndarray) -> Optional[Card]:
    """Преобразует JAX-массив [rank_idx, suit_idx] обратно в Card."""
    if card_array is None or card_array.shape != (2,) or jnp.array_equal(card_array, jnp.array([-1, -1])):
        return None
    try:
        rank_idx = int(card_array[0])
        suit_idx = int(card_array[1])
        # Проверка валидности индексов
        if 0 <= rank_idx < len(Card.RANKS) and 0 <= suit_idx < len(Card.SUITS):
             return Card(Card.RANKS[rank_idx], Card.SUITS[suit_idx])
        else:
             # Логируем ошибку, если индексы некорректны
             # logger.error(f"Invalid card array indices: rank={rank_idx}, suit={suit_idx}")
             return None
    except (IndexError, ValueError):
        # Логируем ошибку при преобразовании
        # logger.error(f"Error converting array to card: {card_array}")
        return None

def action_to_jax(action_dict: Dict[str, List[Card]]) -> jnp.ndarray:
    """Преобразует словарь действия в JAX-массив [17, 2] (13 доска + 4 сброс)."""
    action_array = jnp.full((17, 2), -1, dtype=jnp.int32)
    # Заполняем слоты доски (0-12)
    idx = 0
    for card in action_dict.get("top", []):
        if idx < 3: action_array = action_array.at[idx].set(card_to_array(card)); idx += 1
    idx = 3
    for card in action_dict.get("middle", []):
        if idx < 8: action_array = action_array.at[idx].set(card_to_array(card)); idx += 1
    idx = 8
    for card in action_dict.get("bottom", []):
        if idx < 13: action_array = action_array.at[idx].set(card_to_array(card)); idx += 1
    # Заполняем слоты сброса (13-16)
    idx = 13
    for card in action_dict.get("discarded", []):
        if idx < 17: action_array = action_array.at[idx].set(card_to_array(card)); idx += 1
    return action_array

def action_from_array(action_array: jnp.ndarray) -> Dict[str, List[Card]]:
    """Преобразует JAX-массив действия [17, 2] обратно в словарь."""
    if action_array is None or action_array.shape != (17, 2):
        logger.error(f"Invalid shape for action_array in action_from_array: {action_array.shape if action_array is not None else 'None'}")
        return {} # Возвращаем пустой словарь при ошибке

    action_dict = {"top": [], "middle": [], "bottom": [], "discarded": []}
    # Извлекаем карты для каждой линии и сброса
    for i in range(3): # Top
        card = array_to_card(action_array[i])
        if card: action_dict["top"].append(card)
    for i in range(3, 8): # Middle
        card = array_to_card(action_array[i])
        if card: action_dict["middle"].append(card)
    for i in range(8, 13): # Bottom
        card = array_to_card(action_array[i])
        if card: action_dict["bottom"].append(card)
    for i in range(13, 17): # Discarded
        card = array_to_card(action_array[i])
        if card: action_dict["discarded"].append(card)
    # Убираем пустые списки из результата для чистоты
    return {k: v for k, v in action_dict.items() if v}

# --- Класс GameState ---
class GameState:
    """Представляет полное состояние игры для одного игрока в определенный момент."""
    def __init__(
        self,
        selected_cards: Optional[Union[List[Card], Hand]] = None,
        board: Optional[Board] = None,
        discarded_cards: Optional[List[Card]] = None,
        ai_settings: Optional[Dict] = None,
        deck: Optional[List[Card]] = None, # Общая колода для игры
        current_player: int = 0,
        # Информация об оппоненте (обновляется извне)
        opponent_board: Optional[Board] = None,
        opponent_discarded: Optional[List[Card]] = None,
    ):
        # Инициализация руки
        if isinstance(selected_cards, Hand):
             self.selected_cards: Hand = selected_cards
        else:
             self.selected_cards: Hand = Hand(selected_cards) if selected_cards is not None else Hand()

        # Инициализация доски и сброса
        self.board: Board = board if board is not None else Board()
        self.discarded_cards: List[Card] = discarded_cards if discarded_cards is not None else []

        # Настройки и метаданные
        self.ai_settings: Dict = ai_settings if ai_settings is not None else {}
        self.current_player: int = current_player
        self.deck: List[Card] = deck if deck is not None else [] # Колода передается извне

        # Состояние оппонента (важно для info_set и payoff)
        self.opponent_board: Board = opponent_board if opponent_board is not None else Board()
        self.opponent_discarded: List[Card] = opponent_discarded if opponent_discarded is not None else []

    def get_current_player(self) -> int:
        """Возвращает индекс текущего игрока (0 или 1)."""
        return self.current_player

    def is_terminal(self) -> bool:
        """Проверяет, заполнена ли доска ТЕКУЩЕГО игрока."""
        return self.board.is_full()

    def get_street(self) -> int:
        """Определяет текущую улицу по количеству карт на доске текущего игрока."""
        placed_count = self.board.get_placed_count()
        if placed_count == 0: return 1 # Начало, перед первым ходом
        if placed_count == 5: return 2 # После первого хода
        if placed_count == 7: return 3 # После второго хода (5+2)
        if placed_count == 9: return 4 # После третьего хода (7+2)
        if placed_count == 11: return 5 # После четвертого хода (9+2)
        if placed_count == 13: return 6 # Доска заполнена (11+2)
        # Промежуточные состояния (1-4 карты) возможны только на 1й улице
        if placed_count < 5: return 1
        # Другие значения не должны возникать между улицами
        logger.warning(f"Unexpected number of placed cards ({placed_count}) for street calculation.")
        return 0 # Индикатор ошибки или неизвестного состояния

    def apply_action(self, action: Dict[str, List[Card]]) -> "GameState":
        """
        Применяет действие (размещение карт и сброс) и возвращает НОВОЕ состояние игры
        для ТЕКУЩЕГО игрока. Не меняет игрока и не обрабатывает добор карт.
        """
        # Создаем копии изменяемых объектов
        new_board = Board()
        new_board.top = self.board.top[:]
        new_board.middle = self.board.middle[:]
        new_board.bottom = self.board.bottom[:]
        new_discarded = self.discarded_cards[:]

        placed_in_action = []
        discarded_in_action = action.get("discarded", [])

        # Размещаем карты на новую доску
        for line in ["top", "middle", "bottom"]:
            cards_to_place = action.get(line, [])
            placed_in_action.extend(cards_to_place)
            for card in cards_to_place:
                try:
                    new_board.place_card(line, card)
                except ValueError as e:
                    logger.error(f"Error placing card {card} on {line} in apply_action: {e}")
                    raise # Передаем ошибку выше, т.к. это невалидное действие

        # Добавляем сброшенные карты
        new_discarded.extend(discarded_in_action)

        # Карты, которые были в руке и были разыграны (placed + discarded)
        played_cards = placed_in_action + discarded_in_action

        # Создаем новую руку, удаляя разыгранные карты
        new_hand = Hand(self.selected_cards.cards[:]) # Копируем текущую руку
        new_hand.remove_cards(played_cards) # Используем надежный метод удаления

        # Создаем новый объект GameState
        new_state = GameState(
            selected_cards=new_hand, # Обновленная рука
            board=new_board,
            discarded_cards=new_discarded,
            ai_settings=self.ai_settings.copy(), # Копируем настройки
            deck=self.deck, # Колода остается той же
            current_player=self.current_player, # Игрок тот же
            # Состояние оппонента копируется из текущего состояния
            opponent_board=self.opponent_board,
            opponent_discarded=self.opponent_discarded
        )
        return new_state

    def get_information_set(self) -> str:
        """
        Возвращает строку, представляющую информацию, известную текущему игроку в момент принятия решения.
        Включает: номер улицы, свою доску, свой сброс, доску оппонента, сброс оппонента.
        НЕ включает: свою руку (т.к. она определяет доступные действия из этого info_set).
        """
        # Функция для сортировки и форматирования списка карт
        st = lambda cards: ",".join(map(str, sorted(cards, key=lambda c: (Card.RANK_MAP.get(c.rank, -1), Card.SUIT_MAP.get(c.suit, -1)))))

        street_str = f"St:{self.get_street()}"
        my_board_str = f"T:{st(self.board.top)}|M:{st(self.board.middle)}|B:{st(self.board.bottom)}"
        my_discard_str = f"D:{st(self.discarded_cards)}"

        # Информация об оппоненте (его публичное состояние)
        opp_board_str = f"OT:{st(self.opponent_board.top)}|OM:{st(self.opponent_board.middle)}|OB:{st(self.opponent_board.bottom)}"
        opp_discard_str = f"OD:{st(self.opponent_discarded)}"

        # Собираем строку, порядок важен для консистентности хеша
        return f"{street_str}|{my_board_str}|{my_discard_str}|{opp_board_str}|{opp_discard_str}"

    def _calculate_pairwise_score(self, opponent_board: Board) -> int:
        """
        Рассчитывает очки по правилам 1-6 + scoop против одного оппонента.
        Возвращает очки с точки зрения ТЕКУЩЕГО игрока.
        Вызывается только для полных и не мертвых рук (проверки делаются в get_payoff).
        """
        line_score = 0
        # Сравнение линий с помощью _compare_hands_py
        comparison_bottom = _compare_hands_py(self.board.get_line_jax("bottom"), opponent_board.get_line_jax("bottom"))
        comparison_middle = _compare_hands_py(self.board.get_line_jax("middle"), opponent_board.get_line_jax("middle"))
        comparison_top = _compare_hands_py(self.board.get_line_jax("top"), opponent_board.get_line_jax("top"))

        line_score += comparison_bottom # +1 за победу, -1 за проигрыш
        line_score += comparison_middle
        line_score += comparison_top

        # Бонус за скуп (scoop)
        scoop_bonus = 0
        # Выиграл все 3 линии
        if comparison_bottom == 1 and comparison_middle == 1 and comparison_top == 1:
            scoop_bonus = 3
        # Проиграл все 3 линии
        elif comparison_bottom == -1 and comparison_middle == -1 and comparison_top == -1:
            scoop_bonus = -3

        return line_score + scoop_bonus

    def get_payoff(self) -> int:
        """
        Рассчитывает итоговый payoff для ТЕКУЩЕГО игрока против оппонента
        (чье состояние передано в конструкторе).
        Включает очки 1-6, скуп и разницу роялти.
        Вызывается только в конце игры, когда обе доски полны.
        """
        if not self.is_terminal() or not self.opponent_board.is_full():
            logger.warning("get_payoff called on non-terminal game state(s). Returning 0.")
            return 0

        my_placement_jax = self.board.to_jax_placement()
        opp_placement_jax = self.opponent_board.to_jax_placement()

        i_am_dead = is_dead_hand_jax(my_placement_jax, self.ai_settings)
        opponent_is_dead = is_dead_hand_jax(opp_placement_jax, self.ai_settings)

        # Обработка мертвых рук
        if i_am_dead and opponent_is_dead: return 0
        if i_am_dead:
            # Я мертвый, оппонент нет. Я проигрываю 6 очков + роялти оппонента.
            opp_royalties = calculate_royalties_jax(self.opponent_board, self.ai_settings)
            opp_royalties_sum = int(jnp.sum(opp_royalties))
            return -6 - opp_royalties_sum
        if opponent_is_dead:
            # Я не мертвый, оппонент мертвый. Я выигрываю 6 очков + свои роялти.
            my_royalties = calculate_royalties_jax(self.board, self.ai_settings)
            my_royalties_sum = int(jnp.sum(my_royalties))
            return 6 + my_royalties_sum

        # --- Обе руки не мертвые ---
        # 1. Считаем очки за сравнение линий (1-6 + scoop)
        pairwise_score = self._calculate_pairwise_score(self.opponent_board)

        # 2. Считаем роялти
        my_royalties = calculate_royalties_jax(self.board, self.ai_settings)
        my_royalties_sum = int(jnp.sum(my_royalties))
        opp_royalties = calculate_royalties_jax(self.opponent_board, self.ai_settings)
        opp_royalties_sum = int(jnp.sum(opp_royalties))

        # 3. Итоговый payoff = очки за линии + свои роялти - роялти оппонента
        total_payoff = pairwise_score + my_royalties_sum - opp_royalties_sum

        return total_payoff

    # --- Функции для проверки Фантазии ---
    def is_valid_fantasy_entry(self) -> bool:
        """Проверяет, квалифицируется ли текущая ПОЛНАЯ доска на Фантазию."""
        if not self.board.is_full(): return False # Нужна полная доска
        placement_jax = self.board.to_jax_placement()
        # Рука не должна быть мертвой
        if is_dead_hand_jax(placement_jax, self.ai_settings): return False
        # Вызываем JAX-хелпер для проверки условия QQ+
        return is_valid_fantasy_entry_jax(placement_jax, self.ai_settings)

    def is_valid_fantasy_repeat(self) -> bool:
        """Проверяет, квалифицируется ли текущая ПОЛНАЯ доска на ПОВТОР Фантазии."""
        if not self.board.is_full(): return False # Нужна полная доска
        placement_jax = self.board.to_jax_placement()
        # Рука не должна быть мертвой
        if is_dead_hand_jax(placement_jax, self.ai_settings): return False
        # Вызываем JAX-хелпер для проверки условий повтора
        return is_valid_fantasy_repeat_jax(placement_jax, self.ai_settings)

    def get_fantasy_cards_count(self) -> int:
        """
        Определяет количество карт для ВХОДА в Фантазию (0 если не входит).
        Учитывает тип фантазии (normal/progressive).
        Вызывается только если is_valid_fantasy_entry() вернул True.
        """
        # Условие входа уже проверено (QQ+ на верху, не мертвая)
        placement_jax = self.board.to_jax_placement()
        top_cards = placement_jax[0:3][jnp.any(placement_jax[0:3] != -1, axis=1)]
        top_rank, _ = evaluate_hand_jax(top_cards)

        # Прогрессивная фантазия
        if self.ai_settings.get('fantasyType') == 'progressive':
            if top_rank == 6: # Сет
                return 17
            if top_rank == 8: # Пара
                pair_rank_idx = jnp.where(jnp.bincount(top_cards[:, 0], length=13) == 2)[0][0]
                if pair_rank_idx == Card.RANK_MAP['A']: return 16
                if pair_rank_idx == Card.RANK_MAP['K']: return 15
                if pair_rank_idx == Card.RANK_MAP['Q']: return 14 # QQ дает 14
        # Обычная фантазия (или QQ в прогрессивной)
        return 14


# --- Вспомогательные JAX функции (оценка рук, роялти, фантазия) ---
@jit
def _get_rank_counts_jax(cards_jax: jnp.ndarray) -> jnp.ndarray:
    """Подсчитывает количество карт каждого ранга (JAX)."""
    if cards_jax.shape[0] == 0: return jnp.zeros(13, dtype=jnp.int32)
    ranks = cards_jax[:, 0]
    # Убедимся, что ранги в допустимом диапазоне [0, 12] перед bincount
    ranks = jnp.clip(ranks, 0, 12)
    return jnp.bincount(ranks, length=13)

@jit
def _get_suit_counts_jax(cards_jax: jnp.ndarray) -> jnp.ndarray:
    """Подсчитывает количество карт каждой масти (JAX)."""
    if cards_jax.shape[0] == 0: return jnp.zeros(4, dtype=jnp.int32)
    suits = cards_jax[:, 1]
    suits = jnp.clip(suits, 0, 3)
    return jnp.bincount(suits, length=4)

@jit
def _is_flush_jax(cards_jax: jnp.ndarray) -> bool:
    """Проверяет, является ли набор из 5 карт флешем (JAX)."""
    if cards_jax.shape[0] != 5: return False # Флеш только из 5 карт
    suits = cards_jax[:, 1]
    return jnp.all(suits == suits[0])

@jit
def _is_straight_jax(cards_jax: jnp.ndarray) -> bool:
    """Проверяет, является ли набор из 5 карт стритом (JAX)."""
    if cards_jax.shape[0] != 5: return False
    ranks = jnp.sort(cards_jax[:, 0])
    # Проверяем на уникальность рангов (важно для стрита)
    if jnp.unique(ranks).shape[0] != 5: return False
    # Особый случай: A-5 стрит (0, 1, 2, 3, 12)
    is_a5 = jnp.array_equal(ranks, jnp.array([0, 1, 2, 3, 12]))
    # Обычный стрит (разница между макс и мин = 4 И все уникальны)
    is_normal = (ranks[4] - ranks[0]) == 4
    return jnp.logical_or(is_a5, is_normal)

@jit
def _is_straight_flush_jax(cards_jax: jnp.ndarray) -> bool:
    """Проверяет стрит-флеш (JAX)."""
    if cards_jax.shape[0] != 5: return False
    # Оптимизация: сначала проверяем флеш, потом стрит
    return _is_flush_jax(cards_jax) and _is_straight_jax(cards_jax)

@jit
def _is_royal_flush_jax(cards_jax: jnp.ndarray) -> bool:
    """Проверяет роял-флеш (JAX)."""
    if cards_jax.shape[0] != 5: return False
    # Должен быть стрит-флеш с рангами T,J,Q,K,A
    if not _is_straight_flush_jax(cards_jax): return False
    ranks = jnp.sort(cards_jax[:, 0])
    return jnp.array_equal(ranks, jnp.array([8, 9, 10, 11, 12])) # Индексы T,J,Q,K,A

@jit
def _is_four_of_a_kind_jax(cards_jax: jnp.ndarray) -> bool:
    """Проверяет каре (JAX)."""
    # В OFC каре возможно только на 5 картах
    if cards_jax.shape[0] != 5: return False
    return jnp.any(_get_rank_counts_jax(cards_jax) == 4)

@jit
def _is_full_house_jax(cards_jax: jnp.ndarray) -> bool:
    """Проверяет фулл-хаус (JAX)."""
    if cards_jax.shape[0] != 5: return False
    counts = _get_rank_counts_jax(cards_jax)
    return jnp.any(counts == 3) and jnp.any(counts == 2)

@jit
def _is_three_of_a_kind_jax(cards_jax: jnp.ndarray) -> bool:
    """Проверяет сет/тройку (JAX)."""
    n = cards_jax.shape[0]
    if n < 3: return False
    counts = _get_rank_counts_jax(cards_jax)
    has_three = jnp.sum(counts == 3) == 1
    # Для 5 карт: должна быть тройка, но не пара (не фулл-хаус)
    # Для 3 карт: должна быть тройка
    if n == 5:
        has_pair = jnp.sum(counts == 2) == 1
        return has_three and not has_pair
    elif n == 3:
        return has_three
    else: # Не стандартные размеры
        return False

@jit
def _is_two_pair_jax(cards_jax: jnp.ndarray) -> bool:
    """Проверяет две пары (JAX)."""
    if cards_jax.shape[0] != 5: return False # Две пары только из 5 карт
    counts = _get_rank_counts_jax(cards_jax)
    return jnp.sum(counts == 2) == 2

@jit
def _is_one_pair_jax(cards_jax: jnp.ndarray) -> bool:
    """Проверяет одну пару (JAX)."""
    n = cards_jax.shape[0]
    if n < 2: return False
    counts = _get_rank_counts_jax(cards_jax)
    has_one_pair = jnp.sum(counts == 2) == 1
    has_no_three_or_four = jnp.sum(counts >= 3) == 0
    # Для 5 карт: одна пара, нет троек/каре
    # Для 3 карт: одна пара
    if n == 5:
        return has_one_pair and has_no_three_or_four
    elif n == 3:
        return has_one_pair
    else: # Для n=2, 4 - тоже одна пара
        return has_one_pair and has_no_three_or_four

@jit
def _identify_combination_jax(cards_jax: jnp.ndarray) -> int:
    """
    Определяет ранг комбинации (JAX). Меньший ранг лучше.
    0: RF, 1: SF, 2: 4K, 3: FH, 4: Fl, 5: St, 6: 3K, 7: 2P, 8: 1P, 9: HC, 10: Invalid/Empty
    """
    n = cards_jax.shape[0]
    if n == 0: return 10

    # --- Комбинации из 5 карт ---
    if n == 5:
        # Оптимизация: проверяем от сильных к слабым
        if _is_royal_flush_jax(cards_jax): return 0
        if _is_straight_flush_jax(cards_jax): return 1
        if _is_four_of_a_kind_jax(cards_jax): return 2
        if _is_full_house_jax(cards_jax): return 3
        if _is_flush_jax(cards_jax): return 4
        if _is_straight_jax(cards_jax): return 5
        if _is_three_of_a_kind_jax(cards_jax): return 6
        if _is_two_pair_jax(cards_jax): return 7
        if _is_one_pair_jax(cards_jax): return 8
        return 9 # High Card

    # --- Комбинации из 3 карт (Top линия) ---
    elif n == 3:
        if _is_three_of_a_kind_jax(cards_jax): return 6 # Сет = ранг тройки
        if _is_one_pair_jax(cards_jax): return 8
        return 9 # High Card

    # Другие размеры руки (1, 2, 4) - оцениваем только пару или старшую карту
    else:
        if n >= 2 and _is_one_pair_jax(cards_jax): return 8
        if n >= 1: return 9 # High card если есть хоть одна карта
        return 10 # Если n=0 (уже обработано) или что-то странное

@jit
def evaluate_hand_jax(cards_jax: jnp.ndarray) -> Tuple[int, jnp.ndarray]:
    """
    Оценка покерной комбинации (JAX-версия).
    Возвращает (ранг_комбинации, кикеры), где меньший ранг = лучшая комбинация.
    Кикеры - отсортированные ранги карт [5], важные для сравнения.
    """
    n = cards_jax.shape[0]
    default_kickers = jnp.array([-1]*5, dtype=jnp.int32)
    if n == 0: return 10, default_kickers # Ранг 10 для пустой руки

    combination_rank = _identify_combination_jax(cards_jax)
    # Если комбинация не определена (например, n=1,2,4 и не пара), вернем ранг 10
    if combination_rank == 10: return 10, default_kickers

    ranks = cards_jax[:, 0]
    rank_counts = _get_rank_counts_jax(cards_jax)
    # Сортируем все ранги по убыванию для базовых кикеров
    sorted_ranks_desc = jnp.sort(ranks)[::-1]

    # Инициализируем кикеры 5-ю старшими картами (или меньше, если карт < 5)
    kickers = jnp.full(5, -1, dtype=jnp.int32)
    num_to_fill = min(n, 5)
    kickers = kickers.at[:num_to_fill].set(sorted_ranks_desc[:num_to_fill])

    # --- Уточнение кикеров для специфичных комбинаций ---
    # (Важно для корректного сравнения рук)
    if combination_rank == 2: # 4K (n=5): [ранг каре, кикер, -1, -1, -1]
        four_rank = jnp.where(rank_counts == 4)[0][0]
        kicker = jnp.where(rank_counts == 1)[0][0] # Единственная карта не из каре
        kickers = jnp.array([four_rank, kicker, -1, -1, -1], dtype=jnp.int32)
    elif combination_rank == 3: # FH (n=5): [ранг тройки, ранг пары, -1, -1, -1]
        three_rank = jnp.where(rank_counts == 3)[0][0]
        pair_rank = jnp.where(rank_counts == 2)[0][0]
        kickers = jnp.array([three_rank, pair_rank, -1, -1, -1], dtype=jnp.int32)
    elif combination_rank == 6: # 3K
        three_rank = jnp.where(rank_counts == 3)[0][0]
        other_kickers = jnp.sort(ranks[ranks != three_rank])[::-1] # Старшие из оставшихся
        kickers = kickers.at[0].set(three_rank)
        if n == 5: # [ранг тройки, кикер1, кикер2, -1, -1]
             kickers = kickers.at[1:3].set(other_kickers[:2])
             kickers = kickers.at[3:].set(-1)
        elif n == 3: # [ранг тройки, -1, -1, -1, -1]
             kickers = kickers.at[1:].set(-1)
    elif combination_rank == 7: # 2P (n=5): [ст_пара, мл_пара, кикер, -1, -1]
        pair_ranks = jnp.sort(jnp.where(rank_counts == 2)[0])[::-1] # Сортируем ранги пар по убыванию
        kicker = jnp.where(rank_counts == 1)[0][0] # Единственная карта не из пар
        kickers = jnp.array([pair_ranks[0], pair_ranks[1], kicker, -1, -1], dtype=jnp.int32)
    elif combination_rank == 8: # 1P
        pair_rank = jnp.where(rank_counts == 2)[0][0]
        other_kickers = jnp.sort(ranks[ranks != pair_rank])[::-1] # Старшие из оставшихся
        kickers = kickers.at[0].set(pair_rank)
        if n == 5: # [ранг пары, кикер1, кикер2, кикер3, -1]
             kickers = kickers.at[1:4].set(other_kickers[:3])
             kickers = kickers.at[4].set(-1)
        elif n == 3: # [ранг пары, кикер, -1, -1, -1]
             kickers = kickers.at[1].set(other_kickers[0])
             kickers = kickers.at[2:].set(-1)
        # Для n=2,4 - кикеры уже заполнены 5-ю старшими картами
    elif combination_rank == 5 or combination_rank == 1: # Straight/SF (n=5)
        # Кикер - старшая карта стрита (A-5 стрит -> кикер 5)
        is_a5 = jnp.array_equal(jnp.sort(ranks), jnp.array([0, 1, 2, 3, 12]))
        # Если A-5, старшая карта 5 (индекс 3), иначе - старшая карта в руке
        main_kicker = jnp.where(is_a5, 3, sorted_ranks_desc[0])
        kickers = jnp.array([main_kicker, -1, -1, -1, -1], dtype=jnp.int32)
    # Для RF (0), Flush (4), HighCard (9) - кикеры уже заполнены 5-ю старшими картами

    return combination_rank, kickers

@jit
def is_dead_hand_jax(placement: jnp.ndarray, ai_settings: Dict) -> bool:
    """
    Проверяет, является ли ПОЛНОЕ размещение (13 карт) мертвой рукой (JAX-версия).
    placement: JAX-массив [13, 2], представляющий доску.
    """
    # Извлекаем карты линий (предполагаем, что placement содержит ровно 13 карт)
    top_cards = placement[0:3]
    middle_cards = placement[3:8]
    bottom_cards = placement[8:13]

    # Проверяем, что все слоты заполнены (для полной уверенности)
    # if jnp.any(top_cards == -1) or jnp.any(middle_cards == -1) or jnp.any(bottom_cards == -1):
    #     return False # Не полная доска не может быть мертвой

    # Сравниваем силу рук с помощью _compare_hands_py (которая использует evaluate_hand_jax)
    # _compare_hands_py возвращает 1 если hand1 > hand2
    cmp_top_mid = _compare_hands_py(top_cards, middle_cards)
    cmp_mid_bot = _compare_hands_py(middle_cards, bottom_cards)

    # Мертвая, если top > mid ИЛИ mid > bot
    is_dead = (cmp_top_mid > 0) or (cmp_mid_bot > 0)
    return is_dead

@jit
def calculate_royalties_jax(board: Board, ai_settings: Dict) -> jnp.ndarray:
    """
    Расчет роялти по американским правилам (JAX-версия).
    Принимает объект Board.
    Возвращает JAX-массив [top_royalty, middle_royalty, bottom_royalty].
    Роялти начисляются только если доска полна и не мертвая.
    """
    # Роялти начисляются только для полной доски
    if not board.is_full():
        return jnp.array([0, 0, 0], dtype=jnp.int32)

    placement_jax = board.to_jax_placement()
    # Проверка на мертвую руку
    if is_dead_hand_jax(placement_jax, ai_settings):
        return jnp.array([0, 0, 0], dtype=jnp.int32)

    # Извлекаем карты линий (теперь точно полные)
    top_cards = placement_jax[0:3]
    middle_cards = placement_jax[3:8]
    bottom_cards = placement_jax[8:13]

    # --- Роялти для Top линии (3 карты) ---
    top_royalty = 0
    top_rank, _ = evaluate_hand_jax(top_cards)
    if top_rank == 6: # Сет
        set_rank_idx = top_cards[0, 0] # Ранг любой карты сета
        top_royalty = 10 + set_rank_idx # AAA=12 -> 22, 222=0 -> 10
    elif top_rank == 8: # Пара
        pair_rank_idx = jnp.where(jnp.bincount(top_cards[:, 0], length=13) == 2)[0][0]
        # Бонус для 66 (индекс 4) и выше
        top_royalty = jnp.maximum(0, pair_rank_idx - Card.RANK_MAP['5']) # 66=1, 77=2, ..., AA=9

    # --- Роялти для Middle линии (5 карт) ---
    middle_rank, _ = evaluate_hand_jax(middle_cards)
    # Индексы: 0:RF, 1:SF, 2:4K, 3:FH, 4:Fl, 5:St, 6:3K
    middle_royalties_map = jnp.array([50, 30, 20, 12, 8, 4, 2, 0, 0, 0, 0], dtype=jnp.int32)
    middle_royalty = middle_royalties_map[middle_rank] # rank > 6 дает 0

    # --- Роялти для Bottom линии (5 карт) ---
    bottom_rank, _ = evaluate_hand_jax(bottom_cards)
    # Индексы: 0:RF, 1:SF, 2:4K, 3:FH, 4:Fl, 5:St
    bottom_royalties_map = jnp.array([25, 15, 10, 6, 4, 2, 0, 0, 0, 0, 0], dtype=jnp.int32)
    bottom_royalty = bottom_royalties_map[bottom_rank] # rank > 5 дает 0

    return jnp.array([top_royalty, middle_royalty, bottom_royalty], dtype=jnp.int32)

@jit
def is_valid_fantasy_entry_jax(placement: jnp.ndarray, ai_settings: Dict) -> bool:
    """Проверяет квалификацию на Фантазию (JAX). Вызывается для полной немертвой руки."""
    # Проверяем только верхнюю линию
    top_cards = placement[0:3]
    top_rank, _ = evaluate_hand_jax(top_cards)
    # Условие: QQ+ на верху (Сет тоже подходит)
    if top_rank == 8: # Пара
        pair_rank_idx = jnp.where(jnp.bincount(top_cards[:, 0], length=13) == 2)[0][0]
        return pair_rank_idx >= Card.RANK_MAP['Q'] # QQ, KK, AA
    elif top_rank == 6: # Сет
         return True
    return False

@jit
def is_valid_fantasy_repeat_jax(placement: jnp.ndarray, ai_settings: Dict) -> bool:
    """Проверяет повтор Фантазии (JAX). Вызывается для полной немертвой руки."""
    top_cards = placement[0:3]
    bottom_cards = placement[8:13]
    top_rank, _ = evaluate_hand_jax(top_cards)
    bottom_rank, _ = evaluate_hand_jax(bottom_cards)
    # Условия повтора: Сет на верху ИЛИ Каре+ на низу
    repeat = (top_rank == 6) or (bottom_rank <= 2) # rank 2 = Каре
    return repeat

# --- Функции генерации действий ---
def _generate_placements_recursive(
    cards_to_place: List[Card],
    current_board_jax: jnp.ndarray, # Текущее состояние доски (13, 2)
    ai_settings: Dict,
    card_idx: int,
    valid_placements: List[jnp.ndarray],
    max_placements: Optional[int] = 1000
):
    """
    Рекурсивный генератор допустимых размещений карт на доске.
    cards_to_place: Карты, которые нужно разместить.
    current_board_jax: JAX-массив [13, 2], представляющий текущее состояние доски.
    card_idx: Индекс текущей карты для размещения.
    valid_placements: Список для сбора валидных JAX-массивов [13, 2].
    max_placements: Ограничение на количество генерируемых размещений.
    Возвращает True, если лимит достигнут, иначе False.
    """
    if max_placements is not None and len(valid_placements) >= max_placements:
        return True # Сигнал о достижении лимита

    # Базовый случай: все карты размещены
    if card_idx == len(cards_to_place):
        # Проверяем на мертвую руку ТОЛЬКО если доска стала полной
        placed_count = jnp.sum(jnp.any(current_board_jax != -1, axis=1))
        if placed_count == 13:
            if not is_dead_hand_jax(current_board_jax, ai_settings):
                valid_placements.append(current_board_jax.copy())
        else:
            # Если доска не полна, размещение по определению не мертвое
            valid_placements.append(current_board_jax.copy())
        return False # Продолжаем поиск других вариантов

    card = cards_to_place[card_idx]
    card_arr = card_to_array(card)
    limit_reached = False

    # --- Попробовать разместить на верх ---
    # Находим первый свободный слот в top (индексы 0, 1, 2)
    top_indices = jnp.arange(3)
    top_occupied = jnp.any(current_board_jax[top_indices] != -1, axis=1)
    first_free_top = jnp.where(~top_occupied, top_indices, 3)[0] # Индекс первого свободного или 3
    if first_free_top < 3:
        next_placement = current_board_jax.at[first_free_top].set(card_arr)
        # Проверяем промежуточную валидность (опционально, может замедлить)
        # if not is_dead_hand_partial(next_placement, ai_settings): # Нужна функция для частичной проверки
        limit_reached = _generate_placements_recursive(cards_to_place, next_placement, ai_settings, card_idx + 1, valid_placements, max_placements)
        if limit_reached: return True

    # --- Попробовать разместить на середину ---
    mid_indices = jnp.arange(3, 8)
    mid_occupied = jnp.any(current_board_jax[mid_indices] != -1, axis=1)
    first_free_mid = jnp.where(~mid_occupied, mid_indices, 8)[0] # Индекс первого свободного или 8
    if first_free_mid < 8:
        next_placement = current_board_jax.at[first_free_mid].set(card_arr)
        limit_reached = _generate_placements_recursive(cards_to_place, next_placement, ai_settings, card_idx + 1, valid_placements, max_placements)
        if limit_reached: return True

    # --- Попробовать разместить на низ ---
    bot_indices = jnp.arange(8, 13)
    bot_occupied = jnp.any(current_board_jax[bot_indices] != -1, axis=1)
    first_free_bot = jnp.where(~bot_occupied, bot_indices, 13)[0] # Индекс первого свободного или 13
    if first_free_bot < 13:
        next_placement = current_board_jax.at[first_free_bot].set(card_arr)
        limit_reached = _generate_placements_recursive(cards_to_place, next_placement, ai_settings, card_idx + 1, valid_placements, max_placements)
        if limit_reached: return True

    return False # Лимит не достигнут на этом уровне

def get_actions(game_state: GameState) -> jnp.ndarray:
    """
    Генерирует JAX-массив ВСЕХ возможных допустимых действий для данного состояния игры.
    Формат действия: JAX массив [17, 2] (13 доска + 4 сброс).
    """
    logger.debug(f"get_actions - START | Player: {game_state.current_player} | Street: {game_state.get_street()}")
    # Не генерируем действия, если доска игрока уже полна
    if game_state.is_terminal():
        logger.debug("get_actions - Board is full, returning empty actions")
        return jnp.empty((0, 17, 2), dtype=jnp.int32)

    hand_cards = game_state.selected_cards.cards
    num_cards_in_hand = len(hand_cards)
    if num_cards_in_hand == 0:
        logger.debug("get_actions - No cards in hand, returning empty actions")
        return jnp.empty((0, 17, 2), dtype=jnp.int32)

    possible_actions_list = []
    street = game_state.get_street()
    # Флаг текущего хода фантазии берем из настроек (предполагаем, что он там есть)
    is_fantasy_turn = game_state.ai_settings.get("in_fantasy_turn", False)

    # --- Определение количества карт для размещения/сброса ---
    num_to_place, num_to_discard = 0, 0
    if is_fantasy_turn:
        # В фантазии размещаем 13, сбрасываем N-13 (но не более 4)
        num_to_place = 13
        num_to_discard = num_cards_in_hand - num_to_place
        if num_to_discard < 0:
             logger.error(f"Fantasy turn error: Not enough cards ({num_cards_in_hand}) to place 13.")
             num_to_place = num_cards_in_hand # Размещаем все, что есть
             num_to_discard = 0
        # Ограничиваем сброс 4 картами (максимум слотов)
        num_to_discard = min(num_to_discard, 4)
        # Корректируем num_to_place, если сброс был ограничен
        num_to_place = num_cards_in_hand - num_to_discard
        logger.debug(f"Fantasy Action: Hand={num_cards_in_hand}, Place={num_to_place}, Discard={num_to_discard}")
        # Лимит генерации для фантазии (может быть очень много вариантов)
        placement_limit = game_state.ai_settings.get("fantasy_placement_limit", 2000)
    else:
        # Обычные улицы
        if street == 1: # 5 карт -> разместить 5, сбросить 0
            if num_cards_in_hand == 5: num_to_place, num_to_discard = 5, 0
            else: logger.error(f"Street 1 error: Expected 5 cards, got {num_cards_in_hand}"); return jnp.empty((0, 17, 2), dtype=jnp.int32)
        elif 2 <= street <= 5: # 3 карты -> разместить 2, сбросить 1
            if num_cards_in_hand == 3: num_to_place, num_to_discard = 2, 1
            else: logger.error(f"Street {street} error: Expected 3 cards, got {num_cards_in_hand}"); return jnp.empty((0, 17, 2), dtype=jnp.int32)
        else:
             logger.error(f"get_actions called on invalid street {street}"); return jnp.empty((0, 17, 2), dtype=jnp.int32)
        logger.debug(f"Street {street} Action: Hand={num_cards_in_hand}, Place={num_to_place}, Discard={num_to_discard}")
        placement_limit = game_state.ai_settings.get("normal_placement_limit", 500)

    # --- Генерация действий ---
    # Перебираем все комбинации карт для РАЗМЕЩЕНИЯ
    for cards_to_place_tuple in itertools.combinations(hand_cards, num_to_place):
        cards_to_place = list(cards_to_place_tuple)
        # Определяем карты для сброса
        cards_to_discard = [card for card in hand_cards if card not in cards_to_place]
        # Проверка, что количество карт для сброса верное
        if len(cards_to_discard) != num_to_discard:
            logger.error("Internal error: Mismatch in discard count calculation.")
            continue

        # Генерируем допустимые размещения для выбранных карт
        initial_placement_jax = game_state.board.to_jax_placement() # Текущая доска [13, 2]
        valid_placements_for_combo = [] # Список для JAX массивов [13, 2]

        # Используем рекурсивный генератор
        limit_was_reached = _generate_placements_recursive(
            cards_to_place,
            initial_placement_jax,
            game_state.ai_settings,
            0, # Начинаем с первой карты из cards_to_place
            valid_placements_for_combo,
            max_placements=placement_limit
        )
        if limit_was_reached:
             logger.warning(f"Placement generation limit ({placement_limit}) reached for combo: {cards_to_place}")

        # Создаем JAX массив для сброшенных карт [4, 2]
        discard_jax = jnp.full((4, 2), -1, dtype=jnp.int32)
        for i, card in enumerate(cards_to_discard):
            if i < 4: discard_jax = discard_jax.at[i].set(card_to_array(card))

        # Объединяем каждое валидное размещение [13, 2] со сбросом [4, 2] -> [17, 2]
        for placement_13 in valid_placements_for_combo:
            action_17 = jnp.concatenate((placement_13, discard_jax), axis=0)
            possible_actions_list.append(action_17)

        # Если лимит генерации был достигнут хотя бы раз, можем прервать внешний цикл
        # чтобы не генерировать слишком много действий в целом.
        # if limit_was_reached and len(possible_actions_list) > placement_limit * 2: # Примерный порог
        #      logger.warning("Stopping action generation early due to placement limits.")
        #      break

    logger.debug(f"Generated {len(possible_actions_list)} raw actions")

    # --- Фильтрация и возврат результата ---
    if not possible_actions_list:
        logger.warning(f"No valid actions generated for Player {game_state.current_player}!")
        return jnp.empty((0, 17, 2), dtype=jnp.int32)
    else:
        # Проверка формы перед stack (на всякий случай)
        if not all(a.shape == (17, 2) for a in possible_actions_list):
             logger.error("Inconsistent action shapes generated!")
             correct_shape_actions = [a for a in possible_actions_list if a.shape == (17, 2)]
             if not correct_shape_actions: return jnp.empty((0, 17, 2), dtype=jnp.int32)
             return jnp.stack(correct_shape_actions)
        else:
             return jnp.stack(possible_actions_list) # Возвращает массив формы [N, 17, 2]

# --- Вспомогательные функции для эвристической оценки (Python) ---
def _evaluate_partial_combination_py(cards: List[Card], row_type: str) -> float:
    """Оценка потенциала неполной комбинации (Python-версия)."""
    if not cards: return 0.0
    score = 0.0; n = len(cards)
    ranks = [card.rank for card in cards]; suits = [card.suit for card in cards]
    # Используем get с -1 для отсутствующих карт
    rank_indices = sorted([Card.RANK_MAP.get(r, -1) for r in ranks])
    # Убираем -1, если они есть
    rank_indices = [r for r in rank_indices if r != -1]
    if not rank_indices: return 0.0 # Если все карты были невалидны

    # --- Оценка потенциала на Флеш ---
    if row_type in ["middle", "bottom"] and n >= 2 and n < 5:
        suit_counts = Counter(suits); max_suit_count = max(suit_counts.values()) if suit_counts else 0
        if max_suit_count >= 3: score += 5.0 * (max_suit_count - 2) # Бонус за 3, 4 карты

    # --- Оценка потенциала на Стрит ---
    if row_type in ["middle", "bottom"] and n >= 2 and n < 5:
        unique_ranks = sorted(list(set(rank_indices))); un = len(unique_ranks)
        if un >= 2:
            is_connector = all(unique_ranks[i+1] - unique_ranks[i] == 1 for i in range(un - 1))
            gaps = sum(unique_ranks[i+1] - unique_ranks[i] - 1 for i in range(un - 1))
            span = unique_ranks[-1] - unique_ranks[0] if un > 0 else 0
            if is_connector and span == un - 1: score += 4.0 * un # Коннекторы
            elif gaps == 1 and span <= 4: score += 2.0 * un # Одногэпперы
            # A-low стрит дро
            if set(unique_ranks).issuperset({0, 1, 2}): score += 3.0 # A,2,3
            if set(unique_ranks).issuperset({0, 1, 2, 3}): score += 5.0 # A,2,3,4
            # Бродвей дро
            if set(unique_ranks).issuperset({9, 10, 11, 12}): score += 4.0 # T,J,Q,K

    # --- Оценка Пар/Сетов ---
    rank_counts = Counter(ranks)
    for rank, count in rank_counts.items():
        rank_value = Card.RANK_MAP.get(rank, -1)
        if rank_value == -1: continue
        if count == 2: score += 5.0 + rank_value * 0.5 # Бонус за пару
        elif count == 3: score += 15.0 + rank_value * 1.0 # Бонус за сет

    # Небольшой бонус за высокие карты
    score += sum(r for r in rank_indices) * 0.1

    return score

def _compare_hands_py(hand1_jax: jnp.ndarray, hand2_jax: jnp.ndarray) -> int:
    """Сравнивает две руки (JAX массивы). Возвращает 1 если hand1 > hand2, -1 если hand1 < hand2, 0 если равны."""
    # Обработка пустых рук
    n1 = hand1_jax.shape[0]; n2 = hand2_jax.shape[0]
    if n1 == 0 and n2 == 0: return 0
    if n1 == 0: return -1 # Пустая рука всегда слабее
    if n2 == 0: return 1  # Непустая рука всегда сильнее

    rank1, kickers1 = evaluate_hand_jax(hand1_jax)
    rank2, kickers2 = evaluate_hand_jax(hand2_jax)

    # Сравниваем ранги комбинаций (меньше = лучше)
    if rank1 < rank2: return 1
    if rank1 > rank2: return -1

    # Если ранги равны, сравниваем кикеры
    # Преобразуем кикеры в списки Python для безопасного сравнения
    kickers1_list = kickers1.tolist()
    kickers2_list = kickers2.tolist()
    for k1, k2 in zip(kickers1_list, kickers2_list):
        if k1 > k2: return 1
        if k1 < k2: return -1
    return 0 # Полностью равны

def _is_bottom_stronger_or_equal_py(board: Board) -> bool:
    """Проверяет, сильнее ли нижний ряд среднего или равен ему (Python)."""
    bottom_jax = board.get_line_jax("bottom"); middle_jax = board.get_line_jax("middle")
    # Считаем равенство допустимым
    return _compare_hands_py(bottom_jax, middle_jax) >= 0

def _is_middle_stronger_or_equal_py(board: Board) -> bool:
    """Проверяет, сильнее ли средний ряд верхнего или равен ему (Python)."""
    middle_jax = board.get_line_jax("middle"); top_jax = board.get_line_jax("top")
    # Считаем равенство допустимым
    return _compare_hands_py(middle_jax, top_jax) >= 0

# --- Новая эвристическая оценка ---
def heuristic_baseline_evaluation(state: GameState, ai_settings: Dict) -> float:
    """Улучшенная эвристическая оценка состояния игры (не JIT)."""
    # 1. Проверка на мертвую руку (только если доска полна)
    is_full = state.board.is_full()
    if is_full:
        placement_jax = state.board.to_jax_placement()
        if is_dead_hand_jax(placement_jax, ai_settings):
            return -1000.0 # Большой штраф

    # 2. Веса и множители
    COMBINATION_WEIGHTS = jnp.array([100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 5.0, 0.0], dtype=jnp.float32)
    ROW_MULTIPLIERS = {"top": 1.0, "middle": 1.2, "bottom": 1.5}
    total_score = 0.0

    # 3. Оценка каждого ряда (готовая комбинация + потенциал)
    rows_data = {"top": state.board.top, "middle": state.board.middle, "bottom": state.board.bottom}
    max_cards_in_row = {"top": 3, "middle": 5, "bottom": 5}
    for row_name, cards in rows_data.items():
        row_score = 0.0
        cards_jax = state.board.get_line_jax(row_name)
        num_cards_in_row = cards_jax.shape[0]

        if num_cards_in_row > 0:
            # Оценка готовой комбинации
            rank, kickers = evaluate_hand_jax(cards_jax)
            if rank < len(COMBINATION_WEIGHTS):
                 row_score += COMBINATION_WEIGHTS[rank]
                 # Небольшой бонус за кикеры
                 row_score += float(jnp.sum(kickers[kickers != -1])) * 0.01

            # Оценка потенциала (если ряд не полон)
            if num_cards_in_row < max_cards_in_row[row_name]:
                 potential_score = _evaluate_partial_combination_py(cards, row_name)
                 row_score += potential_score

        # Применяем множитель ряда
        row_score *= ROW_MULTIPLIERS[row_name]
        total_score += row_score

    # 4. Бонусы за правильный порядок (даже для неполных рядов)
    # Даем бонус, только если обе сравниваемые линии не пусты
    if state.board.bottom and state.board.middle and _is_bottom_stronger_or_equal_py(state.board):
        total_score += 15.0
    if state.board.middle and state.board.top and _is_middle_stronger_or_equal_py(state.board):
        total_score += 10.0

    # 5. Штраф за сброшенные карты (небольшой)
    discard_penalty = 0.0
    for card in state.discarded_cards:
        rank_value = Card.RANK_MAP.get(card.rank, -1)
        if rank_value != -1: discard_penalty += (rank_value + 1) * 0.1
    total_score -= discard_penalty

    # 6. Добавляем текущие роялти (с весом)
    # Роялти считаются даже для неполной доски (если есть комбинации)
    current_royalties = calculate_royalties_jax(state.board, ai_settings)
    total_score += float(jnp.sum(current_royalties)) * 0.5 # Вес 0.5 для роялти

    return total_score

# --- Класс CFRNode ---
class CFRNode:
    """Узел в дереве CFR, хранящий сожаления и сумму стратегий."""
    def __init__(self, num_actions: int):
        # Инициализируем JAX массивами нужной длины
        self.regret_sum = jnp.zeros(num_actions, dtype=jnp.float32)
        self.strategy_sum = jnp.zeros(num_actions, dtype=jnp.float32)
        self.num_actions = num_actions # Сохраняем количество действий

    def get_strategy(self, realization_weight: float) -> jnp.ndarray:
        """
        Получает текущую стратегию на основе сожалений (Regret Matching)
        и обновляет сумму стратегий для расчета средней стратегии.
        """
        # Работаем только с действиями, существующими в этом узле
        if self.num_actions == 0:
            return jnp.array([], dtype=jnp.float32)

        current_regret_sum = self.regret_sum[:self.num_actions]
        # Берем только положительные сожаления
        positive_regret_sum = jnp.maximum(current_regret_sum, 0)
        normalizing_sum = jnp.sum(positive_regret_sum)

        # Если сумма положительных сожалений > 0, нормализуем их
        # Иначе используем равномерную стратегию
        uniform_strategy = jnp.ones(self.num_actions, dtype=jnp.float32) / self.num_actions
        strategy = jnp.where(normalizing_sum > 0,
                             positive_regret_sum / normalizing_sum,
                             uniform_strategy)

        # Обновляем сумму стратегий, взвешенную на вероятность достижения узла
        # Добавляем проверку на вес, чтобы избежать NaN на первых итерациях или при нулевой вероятности
        if realization_weight > 1e-9: # Малое положительное число для избежания ошибок округления
             self.strategy_sum = self.strategy_sum.at[:self.num_actions].add(realization_weight * strategy)

        return strategy

    def get_average_strategy(self) -> jnp.ndarray:
        """ Получает среднюю стратегию за все итерации. """
        if self.num_actions == 0:
            return jnp.array([], dtype=jnp.float32)

        current_strategy_sum = self.strategy_sum[:self.num_actions]
        normalizing_sum = jnp.sum(current_strategy_sum)

        # Если сумма стратегий > 0, нормализуем
        # Иначе возвращаем равномерную стратегию
        uniform_strategy = jnp.ones(self.num_actions, dtype=jnp.float32) / self.num_actions
        avg_strategy = jnp.where(normalizing_sum > 0,
                                 current_strategy_sum / normalizing_sum,
                                 uniform_strategy)
        return avg_strategy

# --- Класс CFRAgent ---
class CFRAgent:
    """Агент, использующий Counterfactual Regret Minimization (MCCFR вариант)."""
    def __init__(self, iterations: int = 500000, stop_threshold: float = 0.001, batch_size: int = 1, max_nodes: int = 100000, ai_settings: Optional[Dict] = None):
        self.iterations = iterations
        self.stop_threshold = stop_threshold # Порог для проверки сходимости
        self.save_interval = 2000 # Как часто сохранять прогресс
        self.key = random.PRNGKey(int(time.time())) # Генератор случайных чисел JAX
        self.batch_size = batch_size # Пока не используется для vmap, но может пригодиться
        self.max_nodes = max_nodes # Максимальное количество узлов CFR для хранения
        # Словарь для хранения узлов CFR: {хеш_инфосета: CFRNode}
        self.nodes_map: Dict[int, CFRNode] = {}
        # Сохраняем настройки ИИ
        self.ai_settings = ai_settings if ai_settings is not None else {}
        logger.info(f"CFRAgent initialized. Iterations={iterations}, Max Nodes={max_nodes}")
        logger.info(f"AI Settings: {self.ai_settings}")

    def get_node(self, info_set: str, num_actions: int) -> Optional[CFRNode]:
        """
        Получает или создает узел CFR для данного information set.
        Возвращает None, если достигнут лимит узлов или есть несоответствие действий.
        """
        info_hash = hash(info_set)
        node = self.nodes_map.get(info_hash)

        if node is None:
            # Проверка лимита узлов перед созданием нового
            if len(self.nodes_map) >= self.max_nodes:
                logger.warning(f"Max nodes ({self.max_nodes}) reached. Cannot create node for: {info_set[:100]}...")
                return None # Не можем создать узел
            # Создаем новый узел
            node = CFRNode(num_actions)
            self.nodes_map[info_hash] = node
            # Логируем создание нового узла (можно закомментировать для производительности)
            # if len(self.nodes_map) % (self.max_nodes // 10) == 0: # Лог каждые 10% заполнения
            #      logger.info(f"Node count: {len(self.nodes_map)} / {self.max_nodes}")

        # Проверка на несоответствие количества действий (критическая ошибка)
        elif node.num_actions != num_actions:
             logger.error(f"CRITICAL: Action count mismatch for info_set hash {info_hash}. "
                          f"Node has {node.num_actions}, requested {num_actions}. InfoSet: {info_set[:100]}...")
             # Это указывает на проблему либо в генерации info_set, либо в get_actions.
             # Возвращаем None, чтобы использовать baseline и избежать ошибок дальше.
             return None
        return node

    def get_move(self, game_state: GameState, timeout_event: Event, result: Dict) -> None:
        """ Выбирает ход на основе средней стратегии CFR или baseline оценки. """
        logger.debug("Inside CFRAgent get_move")
        # 1. Получаем доступные действия
        actions_jax = get_actions(game_state) # Shape: [N, 17, 2]
        num_available_actions = actions_jax.shape[0]

        # Если нет действий, возвращаем ошибку
        if num_available_actions == 0:
            result["move"] = {"error": "Нет доступных ходов"}
            logger.warning("No actions available in get_move, returning error.")
            return

        # 2. Получаем information set
        info_set = game_state.get_information_set()

        # 3. Получаем узел CFR
        node = self.get_node(info_set, num_available_actions)

        best_action_index = -1
        # 4. Если узел найден и валиден, используем среднюю стратегию
        if node is not None:
            avg_strategy = node.get_average_strategy()
            # Дополнительная проверка на совпадение размерности (на всякий случай)
            if avg_strategy.shape[0] == num_available_actions:
                 # Выбираем действие с максимальной вероятностью в средней стратегии
                 best_action_index = int(jnp.argmax(avg_strategy))
                 logger.debug(f"Move selected using CFR strategy. Node hash {hash(info_set)}. Best action index: {best_action_index}")
                 # logger.debug(f"Strategy: {avg_strategy}") # Отладка стратегии
            else:
                 # Это не должно происходить, если get_node отработал корректно
                 logger.error(f"Strategy length mismatch after get_node! Node={avg_strategy.shape[0]}, Actions={num_available_actions}. Using baseline.")
                 best_action_index = self._get_best_action_baseline(game_state, actions_jax)
        # 5. Если узел не найден (лимит или ошибка), используем baseline
        else:
            logger.debug(f"Node not found or error for hash {hash(info_set)}. Using baseline evaluation.")
            best_action_index = self._get_best_action_baseline(game_state, actions_jax)

        # 6. Обработка случая, если baseline тоже не сработал
        if best_action_index == -1:
             if num_available_actions > 0:
                 logger.warning("Baseline failed or node error, choosing random action.")
                 self.key, subkey = random.split(self.key)
                 best_action_index = int(random.choice(subkey, jnp.arange(num_available_actions)))
             else:
                 # Это не должно произойти из-за проверки в начале
                 result["move"] = {"error": "Нет доступных ходов после всех проверок"}
                 logger.error("Critical error: No actions available in get_move.")
                 return

        # 7. Преобразуем выбранное действие в словарь и возвращаем
        best_action_array = actions_jax[best_action_index]
        move = action_from_array(best_action_array)
        result["move"] = move
        logger.debug(f"Final selected move (index {best_action_index}): {move}")

    def _get_best_action_baseline(self, game_state: GameState, actions_jax: jnp.ndarray) -> int:
        """ Выбирает лучший ход, используя heuristic_baseline_evaluation. """
        best_score = -float('inf')
        best_action_index = -1
        key = self.key # Используем ключ состояния агента

        for i, action_array in enumerate(actions_jax):
            action_dict = action_from_array(action_array)
            try:
                # Применяем действие, чтобы получить следующее состояние
                next_state = game_state.apply_action(action_dict)
                # Передаем состояние оппонента в next_state для оценки
                # (apply_action уже копирует их из game_state)

                # Оцениваем это следующее состояние с помощью эвристики
                score = heuristic_baseline_evaluation(next_state, self.ai_settings)

                # Добавляем небольшой шум для случайного выбора среди равных
                key, subkey = random.split(key)
                noise = random.uniform(subkey, minval=-0.01, maxval=0.01)
                score += noise

                if score > best_score:
                    best_score = score
                    best_action_index = i
            except Exception as e:
                 # Логируем ошибку при оценке конкретного действия
                 logger.exception(f"Error evaluating baseline for action {i}. Action: {action_dict}")
                 continue # Пропускаем это действие

        self.key = key # Обновляем ключ состояния генератора

        # Если индекс не выбран (все действия привели к ошибке?)
        if best_action_index == -1 and actions_jax.shape[0] > 0:
             logger.warning("All baseline evaluations failed, choosing random action.")
             key, subkey = random.split(self.key)
             best_action_index = int(random.choice(subkey, jnp.arange(actions_jax.shape[0])))
             self.key = key

        return best_action_index

    def baseline_evaluation(self, state: GameState) -> float:
        """ Обертка для вызова эвристической оценки состояния. """
        return heuristic_baseline_evaluation(state, self.ai_settings)

    # --- Основной метод обучения CFR ---
    def train(self, timeout_event: Event, result: Dict) -> None:
        """ Запускает процесс обучения MCCFR на заданное количество итераций. """
        logger.info(f"Starting CFR training for {self.iterations} iterations...")
        start_time = time.time()
        iterations_completed = 0 # Счетчик успешно завершенных итераций

        for i in range(self.iterations):
            if timeout_event.is_set():
                logger.info(f"Training interrupted by timeout after {i} iterations.")
                break

            # --- Розыгрыш одной партии для MCCFR ---
            self.key, game_key = random.split(self.key)
            # Создаем полную колоду один раз для каждой итерации
            all_cards = Card.get_all_cards()
            # Разыгрываем партию и получаем траекторию
            trajectory = self._play_one_game_for_cfr(game_key, all_cards)

            # Обновляем стратегию на основе этой траектории, если она валидна
            if trajectory:
                 self._update_strategy_from_trajectory(trajectory)
                 iterations_completed += 1 # Считаем только успешные итерации
            else:
                 logger.warning(f"Skipping strategy update for iteration {i+1} due to game simulation error.")


            # --- Логирование и сохранение прогресса ---
            current_iter = i + 1
            if current_iter % 100 == 0: # Логируем каждые 100 итераций
                 elapsed_time = time.time() - start_time
                 nodes_count = len(self.nodes_map)
                 logger.info(f"Iteration {current_iter}/{self.iterations} | Nodes: {nodes_count}/{self.max_nodes} | Time: {elapsed_time:.2f}s")

            # Сохраняем прогресс через заданный интервал
            if current_iter % self.save_interval == 0:
                self.save_progress(iterations_completed) # Передаем количество пройденных итераций
                logger.info(f"Progress saved at iteration {current_iter}")

            # Опциональная проверка сходимости (может быть дорогой)
            # if current_iter % 50000 == 0 and self.check_convergence():
            #     logger.info(f"CFR agent potentially converged after {current_iter} iterations.")
            #     # break # Можно прервать обучение, если сошлось

        # --- Завершение обучения ---
        total_time = time.time() - start_time
        logger.info(f"Training finished after {iterations_completed} successful iterations in {total_time:.2f} seconds.")
        logger.info(f"Total nodes created: {len(self.nodes_map)}")
        # Финальное сохранение прогресса
        self.save_progress(iterations_completed)
        logger.info("Final progress saved.")


    def _play_one_game_for_cfr(self, key: jax.random.PRNGKey, deck: List[Card]) -> Optional[Dict]:
        """
        Разыгрывает одну партию OFC Pineapple для сбора траектории для MCCFR.
        Возвращает словарь с информацией о траектории или None при ошибке.
        """
        try: # Обернем всю симуляцию в try-except для отлова неожиданных ошибок
            key, subkey = random.split(key)
            shuffled_indices = random.permutation(subkey, jnp.arange(52))
            shuffled_deck = [deck[i] for i in shuffled_indices]
            deck_iter = iter(shuffled_deck) # Итератор для удобной раздачи

            # Инициализация состояний игроков
            game_states = {
                0: GameState(ai_settings=self.ai_settings, current_player=0, deck=deck), # Передаем полную колоду
                1: GameState(ai_settings=self.ai_settings, current_player=1, deck=deck)
            }
            # Связываем состояния оппонентов (важно для get_information_set и get_payoff)
            game_states[0].opponent_board = game_states[1].board
            game_states[0].opponent_discarded = game_states[1].discarded_cards
            game_states[1].opponent_board = game_states[0].board
            game_states[1].opponent_discarded = game_states[0].discarded_cards

            # Отслеживание состояния фантазии и завершения игры
            player_fantasies = {0: False, 1: False} # Находится ли игрок в фантазии сейчас
            fantasy_cards_count = {0: 0, 1: 0} # Сколько карт раздать для следующего хода фантазии
            player_finished = {0: False, 1: False} # Завершил ли игрок свою доску (включая фантазию)

            # Структура для хранения траектории
            trajectory = {
                'states': [], # (info_set_hash, player, num_actions)
                'actions': [], # index выбранного действия
                'reach_probs': [], # (reach_p0, reach_p1)
                'sampling_probs': [], # sigma(a*) - вероятность выбора действия
                'final_payoff': 0 # итоговый payoff для игрока 0
            }
            reach_p0 = 1.0 # Начальные вероятности достижения = 1
            reach_p1 = 1.0

            current_player = 0 # Игрок 0 ходит первым
            turn_count = 0
            max_turns = 60 # Увеличим лимит ходов (5 улиц * 2 игрока + фантазии)

            # --- Основной цикл игры ---
            while not (player_finished[0] and player_finished[1]):
                turn_count += 1
                if turn_count > max_turns:
                     logger.error(f"Max turns ({max_turns}) reached in game simulation. Aborting.")
                     return None

                # Если текущий игрок уже закончил, передаем ход
                if player_finished[current_player]:
                     current_player = 1 - current_player
                     continue

                state = game_states[current_player]
                opponent = 1 - current_player
                is_fantasy_turn = player_fantasies[current_player] and fantasy_cards_count[current_player] > 0

                # --- 1. Добор карт (если рука пуста) ---
                if len(state.selected_cards) == 0:
                    num_to_draw = 0
                    if is_fantasy_turn:
                        # Раздаем карты для фантазии
                        num_to_draw = fantasy_cards_count[current_player]
                        # Сразу сбрасываем флаг, т.к. карты раздаются один раз за ход фантазии
                        fantasy_cards_count[current_player] = 0
                        logger.info(f"Player {current_player} starting Fantasy turn with {num_to_draw} cards.")
                    else:
                        # Обычный добор по улицам
                        street = state.get_street()
                        if street == 1: num_to_draw = 5
                        elif 2 <= street <= 5: num_to_draw = 3
                        else: # Улица 6 (доска полна) или ошибка
                             if not state.is_terminal():
                                 logger.error(f"Cannot draw cards on street {street} for player {current_player} (board not full).")
                                 return None # Ошибка симуляции
                             else:
                                 # Доска полна, игрок закончил (проверка фантазии будет ниже)
                                 logger.info(f"Player {current_player} board is full. Finishing.")
                                 player_finished[current_player] = True
                                 continue # Переходим к проверке фантазии и смене игрока

                    # Раздаем карты из итератора
                    try:
                        drawn_cards = [next(deck_iter) for _ in range(num_to_draw)]
                        state.selected_cards = Hand(drawn_cards)
                        logger.debug(f"Player {current_player} draws {num_to_draw} cards. Hand: {state.selected_cards}")
                    except StopIteration:
                        # Колода закончилась раньше времени?
                        logger.error("Deck ran out of cards unexpectedly during simulation.")
                        # Считаем, что игрок не может ходить и заканчивает
                        player_finished[current_player] = True
                        continue

                # --- 2. Ход игрока ---
                # Получаем information set
                info_set = state.get_information_set()

                # Получаем доступные действия
                # Передаем флаг фантазии в ai_settings для get_actions
                state.ai_settings["in_fantasy_turn"] = is_fantasy_turn
                actions_jax = get_actions(state) # Shape (N, 17, 2)
                num_actions = actions_jax.shape[0]
                state.ai_settings["in_fantasy_turn"] = False # Сбрасываем флаг после использования

                # Если нет доступных ходов
                if num_actions == 0:
                    # Это возможно, только если доска полна или произошла ошибка в get_actions
                    if state.is_terminal():
                         logger.info(f"Player {current_player} has no actions and board is full. Finishing turn.")
                         player_finished[current_player] = True
                         # Проверка фантазии происходит здесь, после завершения доски
                    else:
                         # Если доска не полна, а ходов нет - это ошибка
                         logger.error(f"No actions generated for Player {current_player} in non-terminal state! InfoSet: {info_set[:100]}...")
                         logger.error(f"Hand: {state.selected_cards}")
                         logger.error(f"Board:\n{state.board}")
                         return None # Ошибка симуляции

                # Если игрок закончил на этом шаге (нет действий и доска полна)
                if player_finished[current_player]:
                     # Проверяем вход/повтор фантазии ПЕРЕД сменой игрока
                     is_now_in_fantasy = False
                     if not is_fantasy_turn: # Проверяем вход, если не были в фантазии
                          if state.is_valid_fantasy_entry():
                               f_count = state.get_fantasy_cards_count()
                               if f_count > 0:
                                    player_fantasies[current_player] = True
                                    fantasy_cards_count[current_player] = f_count
                                    player_finished[current_player] = False # Нужно будет сыграть фантазию
                                    is_now_in_fantasy = True
                                    logger.info(f"Player {current_player} triggers Fantasy with {f_count} cards!")
                     else: # Проверяем повтор, если были в фантазии
                          if state.is_valid_fantasy_repeat():
                               fantasy_cards_count[current_player] = 14 # Повтор всегда 14 карт
                               player_finished[current_player] = False # Нужно будет сыграть еще фантазию
                               is_now_in_fantasy = True
                               logger.info(f"Player {current_player} repeats Fantasy (14 cards)!")

                     # Если не вошли/не повторили фантазию, выходим из нее
                     if not is_now_in_fantasy:
                          player_fantasies[current_player] = False
                          fantasy_cards_count[current_player] = 0

                     current_player = opponent # Передаем ход
                     continue # Переходим к следующему игроку

                # --- Выбор действия (CFR или Baseline) ---
                node = self.get_node(info_set, num_actions)
                action_index = -1
                sampling_prob = 1.0 / num_actions # По умолчанию для baseline/ошибки

                if node is not None:
                    # Получаем стратегию и сэмплируем действие
                    current_reach = reach_p0 if current_player == 0 else reach_p1
                    strategy = node.get_strategy(current_reach) # Обновляем strategy_sum
                    if strategy.shape[0] == num_actions:
                         key, subkey = random.split(key)
                         # Используем numpy.random.choice для JAX массива вероятностей
                         action_index = int(np.random.choice(np.arange(num_actions), p=np.array(strategy)))
                         sampling_prob = strategy[action_index]
                    else:
                         logger.error(f"Strategy shape mismatch in simulation! Node={node.num_actions}, Actions={num_actions}. Using baseline.")
                         action_index = self._get_best_action_baseline(state, actions_jax)
                else:
                    # Узел не найден или ошибка -> используем baseline
                    logger.warning("Node error/limit in simulation. Choosing baseline action.")
                    action_index = self._get_best_action_baseline(state, actions_jax)

                # Если выбор действия не удался
                if action_index == -1:
                     logger.error("Failed to select action even with baseline. Aborting game.")
                     return None

                chosen_action_jax = actions_jax[action_index]
                action_dict = action_from_array(chosen_action_jax)

                # --- Сохранение шага траектории ---
                trajectory['states'].append((hash(info_set), current_player, num_actions))
                trajectory['actions'].append(action_index)
                trajectory['reach_probs'].append((reach_p0, reach_p1))
                # Убедимся, что sampling_prob не слишком мал
                trajectory['sampling_probs'].append(max(sampling_prob, 1e-9))

                # Обновляем reach probabilities для следующего состояния
                # Делаем это *после* сохранения текущих вероятностей
                if current_player == 0: reach_p0 *= sampling_prob
                else: reach_p1 *= sampling_prob

                # --- 3. Применение действия ---
                game_states[current_player] = state.apply_action(action_dict)
                # Обновляем ссылки на оппонента в обоих состояниях
                game_states[current_player].opponent_board = game_states[opponent].board
                game_states[current_player].opponent_discarded = game_states[opponent].discarded_cards
                game_states[opponent].opponent_board = game_states[current_player].board
                game_states[opponent].opponent_discarded = game_states[current_player].discarded_cards

                logger.debug(f"Player {current_player} applied action {action_index}. New board:\n{game_states[current_player].board}")

                # --- 4. Проверка завершения хода/игры/фантазии ---
                # Если это был ход фантазии, он завершается сразу
                if is_fantasy_turn:
                     logger.info(f"Player {current_player} finished Fantasy turn.")
                     player_finished[current_player] = True
                     # Проверяем повтор фантазии
                     if game_states[current_player].is_valid_fantasy_repeat():
                          fantasy_cards_count[current_player] = 14 # Повтор всегда 14 карт
                          player_finished[current_player] = False # Нужно будет сыграть еще
                          logger.info(f"Player {current_player} repeats Fantasy (14 cards)!")
                     else:
                          player_fantasies[current_player] = False # Выход из фантазии
                          fantasy_cards_count[current_player] = 0
                # Если обычный ход и доска заполнилась
                elif game_states[current_player].is_terminal():
                     logger.info(f"Player {current_player} finished board.")
                     player_finished[current_player] = True
                     # Проверяем вход в фантазию
                     if game_states[current_player].is_valid_fantasy_entry():
                          f_count = game_states[current_player].get_fantasy_cards_count()
                          if f_count > 0:
                               player_fantasies[current_player] = True
                               fantasy_cards_count[current_player] = f_count
                               player_finished[current_player] = False # Нужно будет сыграть фантазию
                               logger.info(f"Player {current_player} triggers Fantasy with {f_count} cards!")

                # --- 5. Переход хода ---
                current_player = opponent

            # --- Конец игры (выход из while) ---
            logger.info("Game simulation finished. Calculating payoff.")
            # Убедимся, что оба состояния имеют финальные ссылки друг на друга
            game_states[0].opponent_board = game_states[1].board
            game_states[0].opponent_discarded = game_states[1].discarded_cards
            game_states[1].opponent_board = game_states[0].board
            game_states[1].opponent_discarded = game_states[0].discarded_cards

            # Рассчитываем payoff для игрока 0
            final_payoff_p0 = game_states[0].get_payoff()
            trajectory['final_payoff'] = final_payoff_p0

            logger.debug(f"Final Payoff for Player 0: {final_payoff_p0}")
            logger.debug(f"Player 0 Final Board:\n{game_states[0].board}")
            logger.debug(f"Player 1 Final Board:\n{game_states[1].board}")

            return trajectory

        except Exception as e:
            # Ловим любые неожиданные ошибки во время симуляции
            logger.exception(f"Error during game simulation: {e}")
            return None


    def _update_strategy_from_trajectory(self, trajectory: Dict):
        """
        Обновляет сожаления узлов CFR на основе данных одной разыгранной партии
        по методу Outcome Sampling MCCFR.
        """
        final_payoff_p0 = trajectory['final_payoff']
        num_steps = len(trajectory['states'])

        # Проходим по траектории с начала до конца
        for t in range(num_steps):
            info_hash, player, num_actions = trajectory['states'][t]
            action_taken_index = trajectory['actions'][t]
            reach_p0, reach_p1 = trajectory['reach_probs'][t]
            sampling_prob = trajectory['sampling_probs'][t] # sigma(a*)

            node = self.nodes_map.get(info_hash)
            # Пропускаем обновление, если узел не найден, не совпадает или sampling_prob слишком мал
            if node is None or node.num_actions != num_actions or sampling_prob < 1e-9:
                 # Логируем пропуск только если узел должен был быть найден
                 if node is not None and node.num_actions != num_actions:
                      logger.warning(f"Skipping update for node {info_hash} due to action mismatch.")
                 elif node is None and info_hash in self.nodes_map: # Странная ситуация
                      logger.error(f"Node {info_hash} exists but get_node returned None?")
                 # Если sampling_prob мал, это может быть нормально при сходимости
                 # else: logger.debug(f"Skipping update for node {info_hash} (not found or low prob)")
                 continue

            # Выигрыш для игрока, который ходил в этом состоянии
            payoff_for_player = final_payoff_p0 if player == 0 else -final_payoff_p0

            # Вероятность достижения узла оппонентом
            reach_opponent = reach_p1 if player == 0 else reach_p0

            # --- Обновление сожалений по Outcome Sampling ---
            # regret_sum[a] += reach_opponent * ( payoff_outcome / sampling_prob_outcome ) * ( I(a=a*) - strategy[a] )

            # Получаем текущую стратегию узла (без обновления strategy_sum здесь!)
            current_regret_sum = node.regret_sum[:num_actions]
            positive_regret_sum = jnp.maximum(current_regret_sum, 0)
            normalizing_sum = jnp.sum(positive_regret_sum)
            uniform_strategy = jnp.ones(num_actions, dtype=jnp.float32) / num_actions
            current_strategy = jnp.where(normalizing_sum > 0, positive_regret_sum / normalizing_sum, uniform_strategy)

            # Рассчитываем множитель обновления
            # Деление на sampling_prob взвешивает важность этого исхода
            update_multiplier = reach_opponent * (payoff_for_player / sampling_prob)

            # Создаем вектор индикатора I(a=a*)
            indicator = jnp.zeros(num_actions, dtype=jnp.float32).at[action_taken_index].set(1.0)

            # Вычисляем обновление для каждого действия
            # (I(a=a*) - strategy[a]) показывает, насколько выбор действия a*
            # был лучше или хуже, чем ожидалось по текущей стратегии
            regret_update = update_multiplier * (indicator - current_strategy)

            # Применяем обновление к сумме сожалений
            node.regret_sum = node.regret_sum.at[:num_actions].add(regret_update)

            # Отладка (можно закомментировать)
            # if i % 1000 == 0 and t == num_steps // 2: # Лог для середины траектории иногда
            #      logger.debug(f"Iter {i+1} Step {t}: Update node {info_hash} P{player} N={num_actions}")
            #      logger.debug(f"  Payoff={payoff_for_player:.2f}, ReachOpp={reach_opponent:.4f}, SampProb={sampling_prob:.4f}")
            #      logger.debug(f"  Multiplier={update_multiplier:.2f}")
            #      logger.debug(f"  Strategy={current_strategy}")
            #      logger.debug(f"  RegretUpd={regret_update}")
            #      logger.debug(f"  NewRegret={node.regret_sum[:num_actions]}")


    # --- Сохранение и загрузка ---
    def save_progress(self, iterations_completed: int) -> None:
        """Сохраняет прогресс CFR (карту узлов и счетчик итераций) через GitHub."""
        try:
            # Преобразуем узлы в сериализуемый формат (словари со списками Python)
            serializable_nodes = {}
            for info_hash, node in self.nodes_map.items():
                serializable_nodes[info_hash] = {
                    "regret_sum": node.regret_sum.tolist(), # Преобразуем JAX массив в список
                    "strategy_sum": node.strategy_sum.tolist(),
                    "num_actions": node.num_actions
                }

            data_to_save = {
                "nodes_map_serialized": serializable_nodes,
                "iterations_completed": iterations_completed, # Сохраняем количество пройденных итераций
                "ai_settings": self.ai_settings,
                "timestamp": time.time() # Добавляем временную метку
            }
            logger.info(f"Preparing to save {len(serializable_nodes)} nodes. Iterations completed: {iterations_completed}")
            # Вызываем функцию сохранения из github_utils
            if not save_ai_progress_to_github(data_to_save):
                logger.error("Error saving progress to GitHub!")
            else:
                logger.info("Data for saving passed to github_utils.")
        except Exception as e:
            logger.exception(f"Unexpected error during save_progress: {e}")


    def load_progress(self) -> None:
        """Загружает прогресс CFR (карту узлов) через GitHub."""
        logger.info("Attempting to load AI progress from GitHub...")
        try:
            loaded_data = load_ai_progress_from_github() # Функция из github_utils
            if loaded_data and "nodes_map_serialized" in loaded_data:
                self.nodes_map.clear() # Очищаем текущие узлы перед загрузкой
                loaded_nodes_map_serialized = loaded_data["nodes_map_serialized"]
                loaded_ai_settings = loaded_data.get("ai_settings", {})
                iterations_completed = loaded_data.get("iterations_completed", 0)
                timestamp = loaded_data.get("timestamp", 0)

                logger.info(f"Loaded data: {len(loaded_nodes_map_serialized)} nodes, {iterations_completed} iterations completed.")
                if timestamp > 0:
                     logger.info(f"Data timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))}")

                # Сравним загруженные настройки с текущими
                if loaded_ai_settings != self.ai_settings:
                     logger.warning("Loaded AI settings differ from current settings!")
                     logger.warning(f"Loaded: {loaded_ai_settings}")
                     logger.warning(f"Current: {self.ai_settings}")
                     # Продолжаем загрузку, но с предупреждением

                num_loaded = 0
                num_errors = 0
                # Восстанавливаем узлы из словаря
                for info_hash_str, node_data in loaded_nodes_map_serialized.items():
                    try:
                        info_hash = int(info_hash_str) # Ключи могли сохраниться как строки
                        num_actions = node_data["num_actions"]

                        # Проверяем лимит узлов при загрузке
                        if len(self.nodes_map) >= self.max_nodes:
                             logger.warning(f"Max nodes ({self.max_nodes}) reached during loading. Stopping load.")
                             break

                        node = CFRNode(num_actions)
                        # Преобразуем списки обратно в JAX массивы
                        regret_sum_list = node_data["regret_sum"]
                        strategy_sum_list = node_data["strategy_sum"]

                        # Проверяем тип и длину перед созданием JAX массива
                        if isinstance(regret_sum_list, list) and isinstance(strategy_sum_list, list) and \
                           len(regret_sum_list) >= num_actions and len(strategy_sum_list) >= num_actions:
                             # Обрезаем до num_actions и преобразуем
                             node.regret_sum = node.regret_sum.at[:num_actions].set(jnp.array(regret_sum_list[:num_actions], dtype=jnp.float32))
                             node.strategy_sum = node.strategy_sum.at[:num_actions].set(jnp.array(strategy_sum_list[:num_actions], dtype=jnp.float32))
                             self.nodes_map[info_hash] = node
                             num_loaded += 1
                        else:
                             logger.warning(f"Data type/length mismatch for node hash {info_hash}. Skipping.")
                             num_errors += 1

                    except KeyError as e:
                        logger.warning(f"Missing key {e} in loaded node data for hash {info_hash}. Skipping node.")
                        num_errors += 1
                    except ValueError as e:
                        logger.warning(f"Value error (e.g., converting hash) for node hash {info_hash_str}: {e}. Skipping node.")
                        num_errors += 1
                    except Exception as e:
                        # Ловим другие возможные ошибки при обработке узла
                        logger.exception(f"Error processing loaded node data for hash {info_hash_str}: {e}. Skipping node.")
                        num_errors += 1

                logger.info(f"Successfully loaded {num_loaded} nodes from GitHub.")
                if num_errors > 0:
                     logger.warning(f"Skipped {num_errors} nodes due to errors during loading.")
                # Можно решить, нужно ли продолжать обучение с iterations_completed или начать с 0
                # self.iterations_completed = iterations_completed # Если хотим продолжить нумерацию

            else:
                logger.warning("Failed to load progress from GitHub or data is invalid/empty.")
        except Exception as e:
            logger.exception(f"Unexpected error during load_progress: {e}")


    # --- Проверка сходимости ---
    def check_convergence(self) -> bool:
        """
        Проверяет, сошлось ли обучение (упрощенная проверка).
        Считаем сошедшимся, если среднее абсолютное значение сожалений мало.
        """
        if not self.nodes_map:
            return False # Нечего проверять

        total_abs_regret = 0.0
        total_actions = 0
        num_nodes_checked = 0

        # Итерируем по значениям словаря узлов
        for node in self.nodes_map.values():
            if node.num_actions > 0:
                # Берем только действительные сожаления
                current_regrets = node.regret_sum[:node.num_actions]
                total_abs_regret += float(jnp.sum(jnp.abs(current_regrets))) # Преобразуем в float
                total_actions += node.num_actions
                num_nodes_checked += 1

        if total_actions == 0:
            return False # Нет действий для проверки

        # Вычисляем среднее абсолютное сожаление
        avg_abs_regret = total_abs_regret / total_actions
        logger.info(f"Convergence check: Avg absolute regret = {avg_abs_regret:.6f} (threshold: {self.stop_threshold})")

        # Считаем сошедшимся, если среднее сожаление меньше порога
        return avg_abs_regret < self.stop_threshold

# --- Класс RandomAgent ---
class RandomAgent:
    """Агент, выбирающий случайное допустимое действие."""
    def __init__(self):
        self.key = random.PRNGKey(int(time.time()) + 1) # Отдельный ключ для случайного агента
        self.ai_settings = {} # Настройки по умолчанию

    def get_move(self, game_state: GameState, timeout_event: Event, result: Dict) -> None:
        """Выбирает случайное допустимое действие."""
        logger.debug("Inside RandomAgent get_move")
        # Получаем доступные действия
        actions_jax = get_actions(game_state)
        num_actions = actions_jax.shape[0]

        if num_actions == 0:
            result["move"] = {"error": "Нет доступных ходов"}
            logger.warning("RandomAgent: No actions available.")
            return

        # Выбираем случайный индекс
        self.key, subkey = random.split(self.key)
        random_index = int(random.choice(subkey, jnp.arange(num_actions)))

        # Преобразуем выбранное действие в словарь
        move = action_from_array(actions_jax[random_index])
        result["move"] = move
        logger.debug(f"RandomAgent selected action index {random_index}: {move}")

    def baseline_evaluation(self, state: GameState) -> float:
        """Базовая оценка состояния для RandomAgent (можно использовать ту же эвристику)."""
        # Убедимся, что передаем ai_settings
        current_ai_settings = getattr(self, 'ai_settings', {})
        return heuristic_baseline_evaluation(state, current_ai_settings)

    # --- Остальные методы RandomAgent остаются заглушками ---
    def evaluate_move(self, game_state: GameState, action: Dict[str, List[Card]], timeout_event: Event) -> float: pass
    def shallow_search(self, state: GameState, depth: int, timeout_event: Event) -> float: pass
    def get_action_value(self, state: GameState, action: Dict[str, List[Card]], timeout_event: Event) -> float: pass
    def calculate_potential(self, cards: List[Card], line: str, board: Board, available_cards: List[Card]) -> float: pass
    def is_flush_potential(self, cards: List[Card], available_cards: List[Card]) -> bool: pass
    def is_straight_potential(self, cards: List[Card], available_cards: List[Card]) -> bool: pass
    def is_pair_potential(self, cards: List[Card], available_cards: List[Card]) -> bool: pass
    def evaluate_line_strength(self, cards: List[Card], line: str) -> float: pass
    def identify_combination(self, cards: List[Card]) -> None: pass
    def is_bottom_stronger_than_middle(self, state: GameState) -> None: pass
    def is_middle_stronger_than_top(self, state: GameState) -> None: pass
    def check_row_strength_rule(self, state: GameState) -> None: pass
    def save_progress(self) -> None: pass # RandomAgent не сохраняет прогресс
    def load_progress(self) -> None: pass # RandomAgent не загружает прогресс
