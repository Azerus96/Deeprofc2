import itertools
from collections import defaultdict, Counter
from threading import Event, Thread
import time
import math
import logging
from typing import List, Dict, Tuple, Optional, Union
from github_utils import save_ai_progress_to_github, load_ai_progress_from_github

import jax.numpy as jnp
import jax
from jax import random
from jax import jit
import numpy as np

# Настройка логирования
logger = logging.getLogger(__name__)

# ... (Card, Hand, Board - без изменений) ...
class Card:
    RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
    SUITS = ["♥", "♦", "♣", "♠"]

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
            return self.rank == other.get("rank") and self.suit == other.get("suit")
        return isinstance(other, Card) and self.rank == other.rank and self.suit == other.suit

    def __hash__(self) -> int:
        return hash((self.rank, self.suit))

    def to_dict(self) -> Dict[str, str]:
        return {"rank": self.rank, "suit": self.suit}

    @staticmethod
    def from_dict(card_dict: Dict[str, str]) -> "Card":
        return Card(card_dict["rank"], card_dict["suit"])

    @staticmethod
    def get_all_cards() -> List["Card"]:
        return [Card(rank, suit) for rank in Card.RANKS for suit in Card.SUITS]


class Hand:
    def __init__(self, cards: Optional[List[Card]] = None):
        self.cards = cards if cards is not None else []

    def add_card(self, card: Card) -> None:
        if not isinstance(card, Card):
            raise TypeError("card must be an instance of Card")
        self.cards.append(card)

    def remove_card(self, card: Card) -> None:
        if not isinstance(card, Card):
            raise TypeError("card must be an instance of Card")
        try:
            self.cards.remove(card)
        except ValueError:
            logger.warning(f"Card {card} not found in hand: {self.cards}")

    def __repr__(self) -> str:
        return ", ".join(map(str, self.cards))

    def __len__(self) -> int:
        return len(self.cards)

    def __iter__(self):
        return iter(self.cards)

    def __getitem__(self, index: int) -> Card:
        return self.cards[index]


class Board:
    def __init__(self):
        self.top: List[Card] = []
        self.middle: List[Card] = []
        self.bottom: List[Card] = []

    def place_card(self, line: str, card: Card) -> None:
        if line == "top":
            if len(self.top) >= 3:
                raise ValueError("Top line is full")
            self.top.append(card)
        elif line == "middle":
            if len(self.middle) >= 5:
                raise ValueError("Middle line is full")
            self.middle.append(card)
        elif line == "bottom":
            if len(self.bottom) >= 5:
                raise ValueError("Bottom line is full")
            self.bottom.append(card)
        else:
            raise ValueError(f"Invalid line: {line}. Line must be one of: 'top', 'middle', 'bottom'")

    def is_full(self) -> bool:
        return len(self.top) == 3 and len(self.middle) == 5 and len(self.bottom) == 5

    def clear(self) -> None:
        self.top = []
        self.middle = []
        self.bottom = []

    def __repr__(self) -> str:
        return f"Top: {self.top}\nMiddle: {self.middle}\nBottom: {self.bottom}"

    def get_cards(self, line: str) -> List[Card]:
        if line == "top":
            return self.top
        elif line == "middle":
            return self.middle
        elif line == "bottom":
            return self.bottom
        else:
            raise ValueError("Invalid line specified")

class GameState:
    def __init__(
        self,
        selected_cards: Optional[List[Card]] = None,
        board: Optional[Board] = None,
        discarded_cards: Optional[List[Card]] = None,
        ai_settings: Optional[Dict] = None,
        deck: Optional[List[Card]] = None,
    ):
        self.selected_cards: Hand = Hand(selected_cards) if selected_cards is not None else Hand()
        self.board: Board = board if board is not None else Board()
        self.discarded_cards: List[Card] = discarded_cards if discarded_cards is not None else []
        self.ai_settings: Dict = ai_settings if ai_settings is not None else {}
        self.current_player: int = 0
        self.deck: List[Card] = deck if deck is not None else self.create_deck()
        self.rank_map: Dict[str, int] = {rank: i for i, rank in enumerate(Card.RANKS)}
        self.suit_map: Dict[str, int] = {suit: i for i, suit in enumerate(Card.SUITS)}
        self.remaining_cards: List[Card] = self.calculate_remaining_cards() # Добавляем remaining_cards

    def create_deck(self) -> List[Card]:
        """Creates a standard deck of 52 cards."""
        return [Card(rank, suit) for rank in Card.RANKS for suit in Card.SUITS]

    def get_current_player(self) -> int:
        return self.current_player

    def is_terminal(self) -> bool:
        """Checks if the game is in a terminal state (all lines are full)."""
        return self.board.is_full()

    def get_num_cards_to_draw(self) -> int:
        """Returns the number of cards to draw based on the current game state."""
        placed_cards = sum(len(row) for row in [self.board.top, self.board.middle, self.board.bottom])
        if placed_cards == 5:
            return 3
        elif placed_cards in (7, 10):
            return 3
        elif placed_cards >= 13:
            return 0
        return 0

    def calculate_remaining_cards(self) -> List[Card]:
        """Calculates the cards that are not yet placed or discarded."""
        used_cards = set(self.discarded_cards)
        used_cards.update(self.board.top + self.board.middle + self.board.bottom)
        used_cards.update(self.selected_cards.cards)  # Добавляем карты из selected_cards
        return [card for card in self.deck if card not in used_cards]

    def get_available_cards(self) -> List[Card]:
        """Returns a list of cards that are still available in the deck."""
        # used_cards = set(self.discarded_cards) # Устарело, используем remaining_cards
        available_cards = [card for card in self.deck if card in self.remaining_cards] # Исправлено
        return available_cards

    def apply_action(self, action: Dict[str, List[Card]]) -> "GameState":
        """Applies an action to the current state and returns the new state."""
        new_board = Board()
        new_board.top = self.board.top + action.get("top", [])
        new_board.middle = self.board.middle + action.get("middle", [])
        new_board.bottom = self.board.bottom + action.get("bottom", [])

        new_discarded_cards = self.discarded_cards[:]
        if "discarded" in action and action["discarded"]:
            if isinstance(action["discarded"], list):
                for card in action["discarded"]:
                    self.mark_card_as_used(card)
            else:
                self.mark_card_as_used(action["discarded"])

        for line in ["top", "middle", "bottom"]:
            for card in action.get(line, []):
                self.mark_card_as_used(card)

        new_game_state = GameState(
            selected_cards=Hand(),
            board=new_board,
            discarded_cards=new_discarded_cards,
            ai_settings=self.ai_settings,
            deck=self.deck[:],
        )
        new_game_state.remaining_cards = new_game_state.calculate_remaining_cards() # Пересчитываем remaining_cards

        return new_game_state

    def get_information_set(self, visible_opponent_cards: Optional[List[Card]] = None) -> str:
        """Returns a string representation of the current information set."""
        def card_to_string(card: Card) -> str:
            return str(card)

        def sort_cards(cards: List[Card]) -> List[Card]:
            return sorted(cards, key=lambda card: (self.rank_map[card.rank], self.suit_map[card.suit]))

        top_str = ",".join(map(card_to_string, sort_cards(self.board.top)))
        middle_str = ",".join(map(card_to_string, sort_cards(self.board.middle)))
        bottom_str = ",".join(map(card_to_string, sort_cards(self.board.bottom)))
        discarded_str = ",".join(map(card_to_string, sort_cards(self.discarded_cards)))

        #  Добавляем информацию о видимых картах соперника
        if visible_opponent_cards is not None:
            visible_opponent_str = ",".join(map(card_to_string, sort_cards(visible_opponent_cards)))
        else:
            visible_opponent_str = ""  #  Если ничего не видно

        return f"T:{top_str}|M:{middle_str}|B:{bottom_str}|D:{discarded_str}|V:{visible_opponent_str}" # V - visible

    def get_payoff(self, opponent_board: Optional[Board] = None) -> Union[int, Dict[str, int]]:
        """
        Calculates the payoff for the current state.
        If the game is terminal, returns the score difference.
        If the game is not terminal and an opponent_board is provided, returns
        a dictionary with potential payoffs for each possible action.
        """
        if not self.is_terminal():
            #  Если игра не завершена, возвращаем 0 (или ошибку, если не передан opponent_board)
            # raise ValueError("Game is not in a terminal state") # Так было раньше
            if opponent_board is None:
                raise ValueError("Opponent board must be provided for non-terminal states")
            else:
                # TODO: реализовать расчет *потенциального* выигрыша для *каждого* действия
                #       Это сложная задача, требующая учета вероятностей и т.д.
                #       Пока что вернем 0
                return 0

        if self.is_dead_hand():
            return -1000  #  Большой штраф за мертвую руку (или другое значение)

        #  Если игра завершена, рассчитываем разницу в очках
        my_royalties = self.calculate_royalties()
        my_total_royalty = sum(my_royalties.values())
        my_line_wins = 0

        #  Сравниваем линии (нужен opponent_board)
        if opponent_board is None:
            raise ValueError("Opponent board must be provided for terminal states")

        opponent_royalties = self.calculate_royalties_for_board(opponent_board) # нужна функция
        opponent_total_royalty = sum(opponent_royalties.values())

        #  Сравнение линий (scoring 1-6)
        if self.evaluate_hand(np.array([card_to_array(card) for card in self.board.bottom]))[0] < self.evaluate_hand(np.array([card_to_array(card) for card in opponent_board.bottom]))[0]:
            my_line_wins += 1
        elif self.evaluate_hand(np.array([card_to_array(card) for card in self.board.bottom]))[0] > self.evaluate_hand(np.array([card_to_array(card) for card in opponent_board.bottom]))[0]:
            my_line_wins -= 1

        if self.evaluate_hand(np.array([card_to_array(card) for card in self.board.middle]))[0] < self.evaluate_hand(np.array([card_to_array(card) for card in opponent_board.middle]))[0]:
            my_line_wins += 1
        elif self.evaluate_hand(np.array([card_to_array(card) for card in self.board.middle]))[0] > self.evaluate_hand(np.array([card_to_array(card) for card in opponent_board.middle]))[0]:
            my_line_wins -= 1

        if self.evaluate_hand(np.array([card_to_array(card) for card in self.board.top]))[0] < self.evaluate_hand(np.array([card_to_array(card) for card in opponent_board.top]))[0]:
            my_line_wins += 1
        elif self.evaluate_hand(np.array([card_to_array(card) for card in self.board.top]))[0] > self.evaluate_hand(np.array([card_to_array(card) for card in opponent_board.top]))[0]:
            my_line_wins -= 1

        #  Скуп (scoop)
        if my_line_wins == 3:
            my_line_wins += 3  #  Дополнительные 3 очка за скуп
        elif my_line_wins == -3:
            my_line_wins -= 3

        #  Итоговый выигрыш/проигрыш (разница)
        return (my_total_royalty + my_line_wins) - (opponent_total_royalty - my_line_wins)

    def calculate_royalties_for_board(self, board: Board) -> Dict[str, int]:
        """
        Вспомогательная функция для расчета роялти для *чужой* доски.
        (Нужна, чтобы не дублировать код из calculate_royalties)
        """
        #  TODO:  Реализовать на JAX (сейчас используется старый код)
        temp_state = GameState(board=board, ai_settings=self.ai_settings)
        return temp_state.calculate_royalties()

    def is_dead_hand(self) -> bool:
        """Checks if the hand is a dead hand (invalid combination order)."""
        if not self.board.is_full():
            return False

        top_rank, _ = self.evaluate_hand(np.array([card_to_array(card) for card in self.board.top]))
        middle_rank, _ = self.evaluate_hand(np.array([card_to_array(card) for card in self.board.middle]))
        bottom_rank, _ = self.evaluate_hand(np.array([card_to_array(card) for card in self.board.bottom]))

        return top_rank > middle_rank or middle_rank > bottom_rank

    def is_valid_fantasy_entry(self, board: Optional[Board] = None) -> bool:
        """Checks if an action leads to a valid fantasy mode entry."""
        if board is None:
            board = self.board

        temp_state = GameState(board=board, ai_settings=self.ai_settings) # Используем self.ai_settings
        if temp_state.is_dead_hand():
            return False

        top_rank, _ = temp_state.evaluate_hand(np.array([card_to_array(card) for card in board.top]))
        if top_rank == 8: # Пара
            if board.top[0].rank == board.top[1].rank:
                return board.top[0].rank in ["Q", "K", "A"] # Проверяем что QQ или старше
        elif top_rank == 7: # Сет
            return True
        return False

    def is_valid_fantasy_repeat(self, board: Optional[Board] = None) -> bool:
        """Checks if an action leads to a valid fantasy mode repeat."""
        if board is None:
            board = self.board

        temp_state = GameState(board=board, ai_settings=self.ai_settings) # Используем self.ai_settings
        if temp_state.is_dead_hand():
            return False

        top_rank, _ = temp_state.evaluate_hand(np.array([card_to_array(card) for card in board.top]))
        bottom_rank, _ = temp_state.evaluate_hand(np.array([card_to_array(card) for card in board.bottom]))
        if self.ai_settings.get('fantasyType') == 'progressive':
            if top_rank == 7:
                return True
            elif bottom_rank <= 3:
                return True
            else:
                return False
        else:
            if top_rank == 7:  # Сет в верхнем ряду
                return True
            if bottom_rank <= 3:  # Каре или лучше в нижнем ряду
                return True

            return False

    def mark_card_as_used(self, card: Card) -> None:
        """Marks a card as used (either placed on the board or discarded)."""
        if card not in self.discarded_cards:
            self.discarded_cards.append(card)

    def calculate_royalties(self) -> Dict[str, int]:
        """Calculates royalties for the current board."""
        royalties = {"top": 0, "middle": 0, "bottom": 0}
        
        # Проверяем, не является ли рука мертвой
        if not self.board.is_full() or self.is_dead_hand():
            return royalties
            
        # Верхний ряд
        if len(self.board.top) == 3:
            top_rank, _ = self.evaluate_hand(np.array([card_to_array(card) for card in self.board.top]))
            if top_rank == 7:  # Сет
                royalties["top"] = 10 + Card.RANKS.index(self.board.top[0].rank)
            elif top_rank == 8:  # Пара
                pair_rank = None
                for i in range(len(Card.RANKS)):
                    if sum(1 for card in self.board.top if card.rank == Card.RANKS[i]) == 2:
                        pair_rank = i
                        break
                if pair_rank is not None and pair_rank >= 10:  # QQ или выше
                    royalties["top"] = pair_rank - 9
        
        # Средний ряд
        if len(self.board.middle) == 5:
            middle_rank, _ = self.evaluate_hand(np.array([card_to_array(card) for card in self.board.middle]))
            if middle_rank <= 6:  # Тройка или лучше
                royalties["middle"] = self._get_royalty_for_rank(middle_rank) * 2
        
        # Нижний ряд
        if len(self.board.bottom) == 5:
            bottom_rank, _ = self.evaluate_hand(np.array([card_to_array(card) for card in self.board.bottom]))
            if bottom_rank <= 6:  # Тройка или лучше
                royalties["bottom"] = self._get_royalty_for_rank(bottom_rank)
                
        return royalties
        
    def _get_royalty_for_rank(self, rank: int) -> int:
        """Возвращает базовое значение роялти для ранга комбинации."""
        if rank == 0:  # Роял-флеш
            return 50
        elif rank == 1:  # Стрит-флеш
            return 30
        elif rank == 2:  # Каре
            return 20
        elif rank == 3:  # Фулл-хаус
            return 12
        elif rank == 4:  # Флеш
            return 8
        elif rank == 5:  # Стрит
            return 4
        elif rank == 6:  # Тройка
            return 2
        return 0

    def evaluate_hand(self, cards_array: np.ndarray) -> Tuple[int, float]:
        """
        Оценка покерной комбинации.
        Возвращает (ранг, score), где меньший ранг = лучшая комбинация.
        Использует numpy вместо jax.numpy для совместимости.
        """
        if len(cards_array) == 0:
            return 11, 0.0

        n = len(cards_array)

        # Преобразуем массив карт в массивы рангов и мастей
        ranks = cards_array[:, 0]
        suits = cards_array[:, 1]
        
        # Подсчитываем количество карт каждого ранга и масти
        rank_counts = np.bincount(ranks.astype(np.int32), minlength=13)
        suit_counts = np.bincount(suits.astype(np.int32), minlength=4)
        
        # Проверяем флеш
        has_flush = np.max(suit_counts) == n
        
        # Сортируем ранги для проверки стрита
        rank_indices = np.sort(ranks)
        
        # Проверяем стрит
        is_straight = False
        if len(np.unique(rank_indices)) == n:
            if np.max(rank_indices) - np.min(rank_indices) == n - 1:
                is_straight = True
            # Особый случай: A-5 стрит
            elif np.array_equal(rank_indices, np.array([0, 1, 2, 3, 12])):
                is_straight = True

        # Оцениваем комбинацию для 3 карт (верхний ряд)
        if n == 3:
            if np.max(rank_counts) == 3:  # Сет
                rank = ranks[0]
                return 7, 10.0 + rank
            elif np.max(rank_counts) == 2:  # Пара
                pair_rank_index = np.where(rank_counts == 2)[0][0]
                return 8, pair_rank_index / 100.0
            else:  # Старшая карта
                high_card_rank_index = np.max(rank_indices)
                return 9, high_card_rank_index / 100.0

        # Оцениваем комбинацию для 5 карт (средний и нижний ряды)
        elif n == 5:
            if has_flush and is_straight:
                if np.array_equal(rank_indices, np.array([8, 9, 10, 11, 12])):
                    return 0, 25.0  # Роял-флеш
                return 1, 15.0 + np.max(rank_indices) / 100.0  # Стрит-флеш

            if np.max(rank_counts) == 4:  # Каре
                four_rank_index = np.where(rank_counts == 4)[0][0]
                return 2, 10.0 + four_rank_index / 100.0

            # Фулл-хаус
            if np.any(rank_counts == 3) and np.any(rank_counts == 2):
                three_rank_index = np.where(rank_counts == 3)[0][0]
                return 3, 6.0 + three_rank_index / 100.0

            if has_flush:  # Флеш
                return 4, 4.0 + np.max(rank_indices) / 100.0

            if is_straight:  # Стрит
                return 5, 2.0 + np.max(rank_indices) / 100.0

            if np.max(rank_counts) == 3:  # Тройка
                three_rank_index = np.where(rank_counts == 3)[0][0]
                return 6, 2.0 + three_rank_index / 100.0

            # Две пары
            pairs = np.where(rank_counts == 2)[0]
            if len(pairs) == 2:
                high_pair_index = np.max(pairs)
                low_pair_index = np.min(pairs)
                return 7, 1.0 + high_pair_index / 100.0 + low_pair_index / 10000.0

            # Одна пара
            if len(pairs) == 1:
                pair_rank_index = pairs[0]
                return 8, pair_rank_index / 100.0

            # Старшая карта
            return 9, np.max(rank_indices) / 100.0

        return 10, 0.0


# Вспомогательные функции
def card_to_array(card: Optional[Card]) -> np.ndarray:
    """Преобразует Card в numpy массив [rank, suit]."""
    if card is None:
        return np.array([-1, -1], dtype=np.int32)  # Пустой слот
    return np.array([Card.RANKS.index(card.rank), Card.SUITS.index(card.suit)], dtype=np.int32)

def array_to_card(card_array: np.ndarray) -> Optional[Card]:
    """Преобразует numpy массив [rank, suit] обратно в Card."""
    if np.array_equal(card_array, np.array([-1, -1])):
        return None  # Пустой слот
    return Card(Card.RANKS[int(card_array[0])], Card.SUITS[int(card_array[1])])

def action_from_array(action_array: np.ndarray) -> Dict[str, List[Card]]:
    """Преобразует numpy массив действия в словарь для apply_action."""
    result = {"top": [], "middle": [], "bottom": [], "discarded": []}
    
    # Верхний ряд (индексы 0-2)
    for i in range(3):
        if not np.array_equal(action_array[i], np.array([-1, -1])):
            card = array_to_card(action_array[i])
            if card:
                result["top"].append(card)
    
    # Средний ряд (индексы 3-7)
    for i in range(3, 8):
        if not np.array_equal(action_array[i], np.array([-1, -1])):
            card = array_to_card(action_array[i])
            if card:
                result["middle"].append(card)
    
    # Нижний ряд (индексы 8-12)
    for i in range(8, 13):
        if not np.array_equal(action_array[i], np.array([-1, -1])):
            card = array_to_card(action_array[i])
            if card:
                result["bottom"].append(card)
    
    # Сброшенные карты (индекс 13+)
    for i in range(13, len(action_array)):
        if not np.array_equal(action_array[i], np.array([-1, -1])):
            card = array_to_card(action_array[i])
            if card:
                result["discarded"].append(card)
    
    return result

def generate_placements(cards: List[Card], board: Board, ai_settings: Dict, max_combinations: int = 10000) -> List[Dict[str, List[Card]]]:
    """
    Генерирует все возможные размещения карт на доске.
    Возвращает список словарей с ключами 'top', 'middle', 'bottom', 'discarded'.
    """
    if not cards:
        return []
        
    num_cards = len(cards)
    
    # Определяем свободные слоты в каждом ряду
    free_slots_top = 3 - len(board.top)
    free_slots_middle = 5 - len(board.middle)
    free_slots_bottom = 5 - len(board.bottom)
    
    # Генерируем все возможные комбинации размещения карт по рядам
    valid_combinations = []
    
    # Для 1 карты
    if num_cards == 1:
        if free_slots_top > 0:
            valid_combinations.append({"top": [cards[0]], "middle": [], "bottom": [], "discarded": []})
        if free_slots_middle > 0:
            valid_combinations.append({"top": [], "middle": [cards[0]], "bottom": [], "discarded": []})
        if free_slots_bottom > 0:
            valid_combinations.append({"top": [], "middle": [], "bottom": [cards[0]], "discarded": []})
        valid_combinations.append({"top": [], "middle": [], "bottom": [], "discarded": [cards[0]]})
        
    # Для 2 карт
    elif num_cards == 2:
        # Перебираем все возможные комбинации размещения 2 карт
        for c1_line in ["top", "middle", "bottom", "discarded"]:
            for c2_line in ["top", "middle", "bottom", "discarded"]:
                # Проверяем, что не превышаем лимиты слотов
                top_count = (c1_line == "top") + (c2_line == "top")
                middle_count = (c1_line == "middle") + (c2_line == "middle")
                bottom_count = (c1_line == "bottom") + (c2_line == "bottom")
                
                if top_count <= free_slots_top and middle_count <= free_slots_middle and bottom_count <= free_slots_bottom:
                    placement = {"top": [], "middle": [], "bottom": [], "discarded": []}
                    
                    if c1_line == "top":
                        placement["top"].append(cards[0])
                    elif c1_line == "middle":
                        placement["middle"].append(cards[0])
                    elif c1_line == "bottom":
                        placement["bottom"].append(cards[0])
                    else:  # discarded
                        placement["discarded"].append(cards[0])
                        
                    if c2_line == "top":
                        placement["top"].append(cards[1])
                    elif c2_line == "middle":
                        placement["middle"].append(cards[1])
                    elif c2_line == "bottom":
                        placement["bottom"].append(cards[1])
                    else:  # discarded
                        placement["discarded"].append(cards[1])
                        
                    valid_combinations.append(placement)
    
    # Для 3 карт
    elif num_cards == 3:
        # Перебираем все возможные комбинации размещения 3 карт
        for perm in itertools.permutations(cards):
            for c1_line in ["top", "middle", "bottom", "discarded"]:
                for c2_line in ["top", "middle", "bottom", "discarded"]:
                    for c3_line in ["top", "middle", "bottom", "discarded"]:
                        # Проверяем, что не превышаем лимиты слотов
                        top_count = (c1_line == "top") + (c2_line == "top") + (c3_line == "top")
                        middle_count = (c1_line == "middle") + (c2_line == "middle") + (c3_line == "middle")
                        bottom_count = (c1_line == "bottom") + (c2_line == "bottom") + (c3_line == "bottom")
                        
                        if top_count <= free_slots_top and middle_count <= free_slots_middle and bottom_count <= free_slots_bottom:
                            placement = {"top": [], "middle": [], "bottom": [], "discarded": []}
                            
                            if c1_line == "top":
                                placement["top"].append(perm[0])
                            elif c1_line == "middle":
                                placement["middle"].append(perm[0])
                            elif c1_line == "bottom":
                                placement["bottom"].append(perm[0])
                            else:  # discarded
                                placement["discarded"].append(perm[0])
                                
                            if c2_line == "top":
                                placement["top"].append(perm[1])
                            elif c2_line == "middle":
                                placement["middle"].append(perm[1])
                            elif c2_line == "bottom":
                                placement["bottom"].append(perm[1])
                            else:  # discarded
                                placement["discarded"].append(perm[1])
                                
                            if c3_line == "top":
                                placement["top"].append(perm[2])
                            elif c3_line == "middle":
                                placement["middle"].append(perm[2])
                            elif c3_line == "bottom":
                                placement["bottom"].append(perm[2])
                            else:  # discarded
                                placement["discarded"].append(perm[2])
                                
                            valid_combinations.append(placement)
    
    # Для большего количества карт используем более общий подход
    else:
        # Генерируем все возможные размещения карт по рядам
        for perm in itertools.permutations(cards):
            # Для каждой перестановки карт перебираем возможные размещения по рядам
            for distribution in itertools.product(["top", "middle", "bottom", "discarded"], repeat=num_cards):
                # Проверяем, что не превышаем лимиты слотов
                top_count = distribution.count("top")
                middle_count = distribution.count("middle")
                bottom_count = distribution.count("bottom")
                
                if top_count <= free_slots_top and middle_count <= free_slots_middle and bottom_count <= free_slots_bottom:
                    placement = {"top": [], "middle": [], "bottom": [], "discarded": []}
                    
                    for i, line in enumerate(distribution):
                        placement[line].append(perm[i])
                        
                    valid_combinations.append(placement)
    
    # Ограничиваем количество комбинаций
    if len(valid_combinations) > max_combinations:
        valid_combinations = valid_combinations[:max_combinations]
    
    # Фильтруем недопустимые размещения (dead hand)
    filtered_combinations = []
    for placement in valid_combinations:
        temp_board = Board()
        temp_board.top = board.top + placement["top"]
        temp_board.middle = board.middle + placement["middle"]
        temp_board.bottom = board.bottom + placement["bottom"]
        
        temp_state = GameState(board=temp_board, ai_settings=ai_settings)
        if not temp_state.is_dead_hand():
            filtered_combinations.append(placement)
    
    return filtered_combinations

def get_placement(cards: List[Card], board: Board, discarded_cards: List[Card], ai_settings: Dict, evaluation_func=None) -> Dict[str, List[Card]]:
    """
    Выбирает лучшее размещение карт на доске.
    Возвращает словарь с ключами 'top', 'middle', 'bottom', 'discarded'.
    """
    if not cards:
        return None
    
    # Генерируем все возможные размещения
    placements = generate_placements(cards, board, ai_settings)
    
    if not placements:
        return None
    
    # Если есть функция оценки, используем ее для выбора лучшего размещения
    if evaluation_func:
        best_placement = None
        best_score = float('-inf')
        
        for placement in placements:
            temp_board = Board()
            temp_board.top = board.top + placement["top"]
            temp_board.middle = board.middle + placement["middle"]
            temp_board.bottom = board.bottom + placement["bottom"]
            
            temp_discarded = discarded_cards + placement["discarded"]
            
            temp_state = GameState(board=temp_board, discarded_cards=temp_discarded, ai_settings=ai_settings)
            score = evaluation_func(temp_state)
            
            if score > best_score:
                best_score = score
                best_placement = placement
        
        return best_placement
    
    # Если нет функции оценки, выбираем случайное размещение
    import random
    return random.choice(placements)

def get_actions(game_state: GameState) -> List[Dict[str, List[Card]]]:
    """
    Возвращает список возможных действий для данного состояния игры.
    """
    logger.debug("get_actions - START")
    if game_state.is_terminal():
        logger.debug("get_actions - Game is terminal, returning empty actions")
        return []

    return generate_placements(game_state.selected_cards.cards, game_state.board, game_state.ai_settings)

class CFRNode:
    def __init__(self, num_actions: int):
        self.regret_sum = np.zeros(num_actions)
        self.strategy_sum = np.zeros(num_actions)
        self.num_actions = num_actions

    def get_strategy(self, realization_weight: float) -> np.ndarray:
        regret_sum = np.maximum(self.regret_sum, 0)
        normalizing_sum = np.sum(regret_sum)
        if normalizing_sum > 0:
            strategy = regret_sum / normalizing_sum
        else:
            strategy = np.ones(self.num_actions) / self.num_actions
        self.strategy_sum += realization_weight * strategy
        return strategy

    def get_average_strategy(self) -> np.ndarray:
        normalizing_sum = np.sum(self.strategy_sum)
        if normalizing_sum > 0:
            strategy = self.strategy_sum / normalizing_sum
        else:
            strategy = np.ones(self.num_actions) / self.num_actions
        return strategy

class CFRAgent:
    def __init__(self, iterations: int = 500000, stop_threshold: float = 0.001, batch_size: int = 1, max_nodes: int = 100000):
        """
        Инициализация MCCFR агента (с numpy вместо JAX).
        """
        self.iterations = iterations
        self.stop_threshold = stop_threshold
        self.save_interval = 2000
        self.batch_size = batch_size
        self.max_nodes = max_nodes
        self.nodes = {}  # {info_set: CFRNode}
        self.ai_settings = {}

    def get_move(self, game_state: GameState, timeout_event: Event, result: Dict) -> None:
        """
        Выбирает ход, используя get_placement.
        """
        logger.debug("Inside get_move")
        
        # Сохраняем настройки AI
        self.ai_settings = game_state.ai_settings

        # Вызываем get_placement
        move = get_placement(
            game_state.selected_cards.cards,
            game_state.board,
            game_state.discarded_cards,
            game_state.ai_settings,
            self.baseline_evaluation  # Передаем функцию оценки
        )
        
        if move is None:  # Если get_placement вернул None (нет ходов)
            result["move"] = {"error": "Нет доступных ходов"}
            logger.debug("No actions available (get_placement returned None), returning error.")
            return

        result["move"] = move
        logger.debug(f"Final selected move (from get_placement): {move}")

    def train(self, timeout_event: Event, result: Dict) -> None:
        """
        Функция обучения MCCFR.
        """
        dealer = -1  # Начинаем с -1, чтобы в первой партии дилер был случайным
        
        for i in range(self.iterations):
            if timeout_event.is_set():
                logger.info(f"Training interrupted after {i} iterations due to timeout.")
                break
            
            # Меняем дилера для каждой новой партии
            dealer = 1 - dealer
            
            # Разыгрываем одну партию
            self._play_one_game(dealer)
            
            if (i + 1) % self.save_interval == 0:
                logger.info(f"Iteration {i + 1} of {self.iterations} complete.")
                if (i + 1) % 10000 == 0:
                    self.save_progress()
                    logger.info(f"Progress saved at iteration {i + 1}")
                
                if (i + 1) % 50000 == 0 and self.check_convergence():
                    logger.info(f"CFR agent converged after {i + 1} iterations.")
                    break
    
    def _play_one_game(self, dealer: int) -> None:
        """
        Разыгрывает одну партию и обновляет стратегию.
        """
        # Создаем колоду и перемешиваем ее
        all_cards = Card.get_all_cards()
        import random
        random.shuffle(all_cards)
        
        # Создаем начальные состояния для обоих игроков
        game_state_p0 = GameState(deck=all_cards, ai_settings=self.ai_settings)
        game_state_p1 = GameState(deck=all_cards, ai_settings=self.ai_settings)
        
        # Флаги "Фантазии" для каждого игрока
        fantasy_p0 = False
        fantasy_p1 = False
        
        # Начальные вероятности (произведения) - 1.0
        pi_0 = 1.0
        pi_1 = 1.0
        
        # Определяем, кто ходит первым (тот, кто слева от дилера)
        current_player = 1 - dealer
        current_game_state = game_state_p0 if current_player == 0 else game_state_p1
        opponent_game_state = game_state_p1 if current_player == 0 else game_state_p0
        
        # Для корректного расчета payoff в конце игры
        first_player = current_player
        
        # Раздаем начальные 5 карт
        game_state_p0.selected_cards = Hand(all_cards[:5])
        game_state_p1.selected_cards = Hand(all_cards[5:10])
        
        # Игроки видят первые 5 карт друг друга
        visible_cards_p0 = all_cards[5:10]
        visible_cards_p1 = all_cards[:5]
        
        cards_dealt = 10  # Счетчик розданных карт
        
        # Основной игровой цикл
        while not game_state_p0.board.is_full() or not game_state_p1.board.is_full():
            # Определяем, видит ли текущий игрок карты соперника
            if current_player == 0:
                visible_opponent_cards = visible_cards_p0
                if fantasy_p1:
                    visible_opponent_cards = []
            else:  # current_player == 1
                visible_opponent_cards = visible_cards_p1
                if fantasy_p0:
                    visible_opponent_cards = []
            
            # Получаем info_set с учетом видимых карт
            info_set = current_game_state.get_information_set(visible_opponent_cards)
            
            # Раздаем карты (если нужно)
            if len(current_game_state.selected_cards.cards) == 0:
                if current_game_state.board.is_full():
                    num_cards_to_deal = 0
                # Если оба в "Фантазии", раздаем сразу 14 (или сколько нужно)
                elif fantasy_p0 and fantasy_p1:
                    if self.ai_settings.get('fantasyType') == 'progressive':
                        # Логика для прогрессивной фантазии
                        if current_player == 0:
                            num_cards_to_deal = self._get_progressive_fantasy_cards(game_state_p0.board)
                        else:
                            num_cards_to_deal = self._get_progressive_fantasy_cards(game_state_p1.board)
                    else:
                        num_cards_to_deal = 14
                elif (len(current_game_state.board.top) + len(current_game_state.board.middle) + len(current_game_state.board.bottom) == 5) or \
                     (len(current_game_state.board.top) + len(current_game_state.board.middle) + len(current_game_state.board.bottom) < 13):
                    num_cards_to_deal = 3  # После первых 5 карт и до 13 раздаем по 3
                else:
                    num_cards_to_deal = 0
                
                if num_cards_to_deal > 0:
                    new_cards = all_cards[cards_dealt:cards_dealt + num_cards_to_deal]
                    current_game_state.selected_cards = Hand(new_cards)
                    cards_dealt += num_cards_to_deal
                    # Обновляем видимые карты для соперника (если не в "Фантазии")
                    if current_player == 0 and not fantasy_p1:
                        visible_cards_p0 = opponent_game_state.board.top + opponent_game_state.board.middle + opponent_game_state.board.bottom + new_cards
                    elif current_player == 1 and not fantasy_p0:
                        visible_cards_p1 = opponent_game_state.board.top + opponent_game_state.board.middle + opponent_game_state.board.bottom + new_cards
            
            # Получаем доступные действия
            actions = get_actions(current_game_state)
            
            if actions:
                # Если info_set еще не в self.nodes, создаем новый узел
                if info_set not in self.nodes:
                    self.nodes[info_set] = CFRNode(len(actions))
                
                # Получаем стратегию для текущего info_set
                node = self.nodes[info_set]
                strategy = node.get_strategy(pi_0 if current_player == 0 else pi_1)
                
                # Выбираем действие согласно стратегии
                action_index = np.random.choice(len(actions), p=strategy)
                action = actions[action_index]
                
                # Обновляем вероятности
                if current_player == 0:
                    pi_0 *= strategy[action_index]
                else:
                    pi_1 *= strategy[action_index]
                
                # Применяем действие
                current_game_state = current_game_state.apply_action(action)
                # Удаляем карты из selected_cards
                current_game_state.selected_cards = Hand([])
            
            # Меняем текущего игрока
            current_player = 1 - current_player
            current_game_state, opponent_game_state = opponent_game_state, current_game_state
            
            # После смены игрока обновляем видимые карты
            if current_player == 0:
                visible_cards_p0 = opponent_game_state.board.top + opponent_game_state.board.middle + opponent_game_state.board.bottom
                if fantasy_p1:
                    visible_cards_p0 = []
            else:
                visible_cards_p1 = opponent_game_state.board.top + opponent_game_state.board.middle + opponent_game_state.board.bottom
                if fantasy_p0:
                    visible_cards_p1 = []
        
        # После того, как оба игрока заполнили доски:
        # 1. Проверяем, попал ли кто-то в "Фантазию"
        if not fantasy_p0 and game_state_p0.is_valid_fantasy_entry():
            fantasy_p0 = True
        if not fantasy_p1 and game_state_p1.is_valid_fantasy_entry():
            fantasy_p1 = True
        
        # 2. Проверяем, может ли кто-то остаться в "Фантазии"
        if fantasy_p0 and not game_state_p0.is_valid_fantasy_repeat(game_state_p0.board):
            fantasy_p0 = False  # Сбрасываем флаг, если не выполнены условия
        if fantasy_p1 and not game_state_p1.is_valid_fantasy_repeat(game_state_p1.board):
            fantasy_p1 = False
        
        # 3. Рассчитываем payoff (с учетом того, кто ходил первым)
        if first_player == 0:
            payoff = float(game_state_p0.get_payoff(opponent_board=game_state_p1.board))
        else:
            payoff = float(game_state_p1.get_payoff(opponent_board=game_state_p0.board))
        
        # Обновляем стратегию на основе payoff
        self._update_strategy(payoff)
    
    def _update_strategy(self, payoff: float) -> None:
        """
        Обновляет стратегию на основе payoff.
        """
        # Реализация обновления стратегии
        # Это упрощенная версия, в реальности нужно учитывать всю траекторию игры
        pass
    
    def _get_progressive_fantasy_cards(self, board: Board) -> int:
        """
        Определяет количество карт для прогрессивной фантазии.
        """
        if len(board.top) == 3:
            top_rank, _ = self.evaluate_hand(np.array([card_to_array(card) for card in board.top]))
            if top_rank == 8:  # Пара
                pair_rank = None
                for i in range(len(Card.RANKS)):
                    if sum(1 for card in board.top if card.rank == Card.RANKS[i]) == 2:
                        pair_rank = i
                        break
                if pair_rank is not None:
                    if pair_rank == 12:  # AA
                        return 16
                    elif pair_rank == 11:  # KK
                        return 15
                    elif pair_rank == 10:  # QQ
                        return 14
            elif top_rank == 7:  # Сет
                return 17
        return 14
    
    def evaluate_hand(self, cards_array: np.ndarray) -> Tuple[int, float]:
        """
        Оценка покерной комбинации.
        """
        if len(cards_array) == 0:
            return 11, 0.0

        n = len(cards_array)

        # Преобразуем массив карт в массивы рангов и мастей
        ranks = cards_array[:, 0]
        suits = cards_array[:, 1]
        
        # Подсчитываем количество карт каждого ранга и масти
        rank_counts = np.bincount(ranks.astype(np.int32), minlength=13)
        suit_counts = np.bincount(suits.astype(np.int32), minlength=4)
        
        # Проверяем флеш
        has_flush = np.max(suit_counts) == n
        
        # Сортируем ранги для проверки стрита
        rank_indices = np.sort(ranks)
        
        # Проверяем стрит
        is_straight = False
        if len(np.unique(rank_indices)) == n:
            if np.max(rank_indices) - np.min(rank_indices) == n - 1:
                is_straight = True
            # Особый случай: A-5 стрит
            elif np.array_equal(rank_indices, np.array([0, 1, 2, 3, 12])):
                is_straight = True

        # Оцениваем комбинацию для 3 карт (верхний ряд)
        if n == 3:
            if np.max(rank_counts) == 3:  # Сет
                rank = ranks[0]
                return 7, 10.0 + rank
            elif np.max(rank_counts) == 2:  # Пара
                pair_rank_index = np.where(rank_counts == 2)[0][0]
                return 8, pair_rank_index / 100.0
            else:  # Старшая карта
                high_card_rank_index = np.max(rank_indices)
                return 9, high_card_rank_index / 100.0

        # Оцениваем комбинацию для 5 карт (средний и нижний ряды)
        elif n == 5:
            if has_flush and is_straight:
                if np.array_equal(rank_indices, np.array([8, 9, 10, 11, 12])):
                    return 0, 25.0  # Роял-флеш
                return 1, 15.0 + np.max(rank_indices) / 100.0  # Стрит-флеш

            if np.max(rank_counts) == 4:  # Каре
                four_rank_index = np.where(rank_counts == 4)[0][0]
                return 2, 10.0 + four_rank_index / 100.0

            # Фулл-хаус
            if np.any(rank_counts == 3) and np.any(rank_counts == 2):
                three_rank_index = np.where(rank_counts == 3)[0][0]
                return 3, 6.0 + three_rank_index / 100.0

            if has_flush:  # Флеш
                return 4, 4.0 + np.max(rank_indices) / 100.0

            if is_straight:  # Стрит
                return 5, 2.0 + np.max(rank_indices) / 100.0

            if np.max(rank_counts) == 3:  # Тройка
                three_rank_index = np.where(rank_counts == 3)[0][0]
                return 6, 2.0 + three_rank_index / 100.0

            # Две пары
            pairs = np.where(rank_counts == 2)[0]
            if len(pairs) == 2:
                high_pair_index = np.max(pairs)
                low_pair_index = np.min(pairs)
                return 7, 1.0 + high_pair_index / 100.0 + low_pair_index / 10000.0

            # Одна пара
            if len(pairs) == 1:
                pair_rank_index = pairs[0]
                return 8, pair_rank_index / 100.0

            # Старшая карта
            return 9, np.max(rank_indices) / 100.0

        return 10, 0.0
    
    def check_convergence(self) -> bool:
        """
        Проверяет, сошлось ли обучение.
        """
        for node in self.nodes.values():
            avg_strategy = node.get_average_strategy()
            uniform_strategy = np.ones(node.num_actions) / node.num_actions
            diff = np.mean(np.abs(avg_strategy - uniform_strategy))
            if diff > self.stop_threshold:
                return False
        return True
    
    def baseline_evaluation(self, state: GameState) -> float:
        """
        Базовая функция оценки состояния.
        """
        # Проверяем, не является ли рука мертвой
        if state.is_dead_hand():
            return -1000.0
        
        # Рассчитываем роялти
        royalties = state.calculate_royalties()
        total_royalty = sum(royalties.values())
        
        # Оцениваем потенциал каждой линии
        top_potential = self._evaluate_line_potential(state.board.top, "top")
        middle_potential = self._evaluate_line_potential(state.board.middle, "middle")
        bottom_potential = self._evaluate_line_potential(state.board.bottom, "bottom")
        
        # Суммируем все компоненты оценки
        return total_royalty + top_potential + middle_potential + bottom_potential
    
    def _evaluate_line_potential(self, cards: List[Card], line: str) -> float:
        """
        Оценивает потенциал линии.
        """
        if not cards:
            return 0.0
        
        # Преобразуем карты в массив для оценки
        cards_array = np.array([card_to_array(card) for card in cards])
        
        # Получаем текущий ранг комбинации
        rank, score = self.evaluate_hand(cards_array)
        
        # Базовая оценка в зависимости от ранга
        base_score = 10.0 - rank
        
        # Модификаторы в зависимости от линии
        if line == "top":
            if rank == 7:  # Сет
                return 15.0
            elif rank == 8:  # Пара
                pair_rank = np.where(np.bincount(cards_array[:, 0], minlength=13) == 2)[0][0]
                if pair_rank >= 10:  # QQ или выше
                    return 10.0 + pair_rank - 9
            return base_score
        
        elif line == "middle":
            # Для среднего ряда ценим комбинации выше
            return base_score * 1.5
        
        elif line == "bottom":
            # Для нижнего ряда ценим комбинации еще выше
            return base_score * 2.0
        
        return base_score
    
    def save_progress(self) -> None:
        """Сохраняет прогресс через GitHub."""
        data = {
            "nodes": {info_set: {"regret_sum": node.regret_sum.tolist(), 
                                "strategy_sum": node.strategy_sum.tolist(), 
                                "num_actions": node.num_actions} 
                     for info_set, node in self.nodes.items()},
            "iterations": self.iterations,
            "stop_threshold": self.stop_threshold,
            "batch_size": self.batch_size,
            "max_nodes": self.max_nodes
        }
        # Используем функцию из github_utils
        if not save_ai_progress_to_github(data):
            logger.error("Ошибка при сохранении прогресса на GitHub!")

    def load_progress(self) -> None:
        """Загружает прогресс через GitHub."""
        # Используем функцию из github_utils
        data = load_ai_progress_from_github()
        if data:
            self.nodes = {}
            for info_set, node_data in data["nodes"].items():
                node = CFRNode(node_data["num_actions"])
                node.regret_sum = np.array(node_data["regret_sum"])
                node.strategy_sum = np.array(node_data["strategy_sum"])
                self.nodes[info_set] = node
            
            self.iterations = data["iterations"]
            self.stop_threshold = data.get("stop_threshold", 0.001)
            self.batch_size = data.get("batch_size", 1)
            self.max_nodes = data.get("max_nodes", 100000)
            logger.info("Прогресс AI успешно загружен с GitHub.")
        else:
            logger.warning("Не удалось загрузить прогресс с GitHub.")

class RandomAgent:
    def __init__(self):
        pass

    def get_move(self, game_state: GameState, timeout_event: Event, result: Dict) -> None:
        """
        Выбирает ход, используя get_placement (случайный выбор из возможных).
        """
        logger.debug("Inside RandomAgent get_move")

        # ВЫЗЫВАЕМ get_placement (теперь это отдельная функция)
        move = get_placement(
            game_state.selected_cards.cards,
            game_state.board,
            game_state.discarded_cards,
            game_state.ai_settings,
            self.baseline_evaluation  #  Передаем функцию оценки
        )
        if move is None:  # Если get_placement вернул None (нет ходов)
            result["move"] = {"error": "Нет доступных ходов"}
            logger.debug("No actions available (get_placement returned None), returning error.")
            return

        result["move"] = move
        logger.debug(f"Final selected move (from get_placement): {move}")

    def evaluate_move(self, game_state: GameState, action: Dict[str, List[Card]], timeout_event: Event) -> float:
        pass

    def shallow_search(self, state: GameState, depth: int, timeout_event: Event) -> float:
        pass

    def get_action_value(self, state: GameState, action: Dict[str, List[Card]], timeout_event: Event) -> float:
        pass

    def calculate_potential(self, cards: List[Card], line: str, board: Board, available_cards: List[Card]) -> float:
        pass

    def is_flush_potential(self, cards: List[Card], available_cards: List[Card]) -> bool:
        pass

    def is_straight_potential(self, cards: List[Card], available_cards: List[Card]) -> bool:
        pass

    def is_pair_potential(self, cards: List[Card], available_cards: List[Card]) -> bool:
        pass

    def evaluate_line_strength(self, cards: List[Card], line: str) -> float:
        pass

    def baseline_evaluation(self, state: GameState) -> float:
        """
        Простая базовая функция оценки для RandomAgent.
        """
        import random
        return random.random()  # Случайная оценка

    def identify_combination(self, cards: List[Card]) -> None:
        pass

    def is_bottom_stronger_than_middle(self, state: GameState) -> None:
        pass

    def is_middle_stronger_than_top(self, state: GameState) -> None:
        pass

    def check_row_strength_rule(self, state: GameState) -> None:
        pass

    def save_progress(self) -> None:
        pass

    def load_progress(self) -> None:
        pass
