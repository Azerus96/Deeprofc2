import itertools
from collections import defaultdict
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

# Настройка логирования
logger = logging.getLogger(__name__)

class Card:
    RANKS = jnp.array(["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"])
    SUITS = jnp.array(["♥", "♦", "♣", "♠"])

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
        deck: Optional[jnp.ndarray] = None,  # Изменен тип
    ):
        self.selected_cards: Hand = Hand(selected_cards) if selected_cards is not None else Hand()
        self.board: Board = board if board is not None else Board()
        self.discarded_cards: List[Card] = discarded_cards if discarded_cards is not None else []
        self.ai_settings: Dict = ai_settings if ai_settings is not None else {}
        self.current_player: int = 0
        self.deck: jnp.ndarray = deck if deck is not None else self.create_deck_jax() # Используем create_deck_jax
        self.rank_map: Dict[str, int] = {rank: i for i, rank in enumerate(Card.RANKS)}
        self.suit_map: Dict[str, int] = {suit: i for i, suit in enumerate(Card.SUITS)}
        self.remaining_cards: jnp.ndarray = jnp.array([]) # Инициализируем пустым массивом

    def initialize_remaining_cards(self):
        self.remaining_cards = GameState.calculate_remaining_cards_jax(
            self.deck,
            jnp.array([card_to_array(card) for card in self.discarded_cards]),
            jnp.array([card_to_array(card) for card in self.board.top]),
            jnp.array([card_to_array(card) for card in self.board.middle]),
            jnp.array([card_to_array(card) for card in self.board.bottom]),
            jnp.array([card_to_array(card) for card in self.selected_cards.cards])
        )

    def create_deck_jax(self) -> jnp.ndarray:
        """Creates a standard deck of 52 cards as a JAX array."""
        all_cards = [card_to_array(Card(rank, suit)) for rank in Card.RANKS for suit in Card.SUITS]
        return jnp.array(all_cards)

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

    @jit
    def calculate_remaining_cards_jax(
        deck: jnp.ndarray,
        discarded_cards: jnp.ndarray,
        top: jnp.ndarray,
        middle: jnp.ndarray,
        bottom: jnp.ndarray,
        selected_cards: jnp.ndarray
    ) -> jnp.ndarray:
        """Calculates the cards that are not yet placed or discarded (JAX version)."""

        #  Объединяем все использованные карты
        used_cards = jnp.concatenate([discarded_cards, top, middle, bottom, selected_cards])

        #  Используем jnp.isin для поиска использованных карт в колоде
        mask = jnp.isin(deck, used_cards, invert=True)
        return deck[mask]

    @jit
    def get_available_cards_jax(self) -> jnp.ndarray:
        """Returns a JAX array of cards that are still available in the deck."""
        return self.remaining_cards

    def apply_action(self, action: jnp.ndarray) -> "GameState":
        """Applies an action to the current state and returns the new state (JAX version)."""
        new_board = Board()
        new_discarded_cards = self.discarded_cards[:]

        #  Разделяем JAX-массив action на части для каждой линии
        top_cards = action[:3]
        middle_cards = action[3:8]
        bottom_cards = action[8:13]
        discarded_cards = action[13:]

        #  Добавляем карты на доску, пропуская пустые слоты (-1, -1)
        for card_array in top_cards:
            try:
                new_board.place_card("top", array_to_card(card_array))
            except (IndexError, TypeError):
                pass
        for card_array in middle_cards:
            try:
                new_board.place_card("middle", array_to_card(card_array))
            except (IndexError, TypeError):
                pass
        for card_array in bottom_cards:
            try:
                new_board.place_card("bottom", array_to_card(card_array))
            except (IndexError, TypeError):
                pass

        #  Добавляем сброшенные карты
        for card_array in discarded_cards:
            try:
                card = array_to_card(card_array)
                if card:
                    new_discarded_cards.append(card)
            except (IndexError, TypeError):
                pass

        new_game_state = GameState(
            selected_cards=[],  #  Очищаем selected_cards
            board=new_board,
            discarded_cards=new_discarded_cards,
            ai_settings=self.ai_settings,
            deck=self.deck,  #  Используем ту же колоду
        )
        new_game_state.initialize_remaining_cards() # Пересчитываем remaining_cards
        return new_game_state

    def get_information_set(self, visible_opponent_cards: Optional[jnp.ndarray] = None) -> str:
        """Returns a string representation of the current information set (JAX-friendly)."""

        def cards_to_string(cards_jax: jnp.ndarray) -> str:
            """Converts a JAX array of cards to a sorted string representation."""
            #  Удаляем пустые слоты (-1, -1)
            cards_jax = cards_jax[jnp.any(cards_jax != -1, axis=1)]
            if cards_jax.size == 0:
                return ""
            #  Сортируем карты по рангу и масти
            sorted_indices = jnp.lexsort((cards_jax[:, 1], cards_jax[:, 0]))
            sorted_cards = cards_jax[sorted_indices]
            #  Преобразуем в строки
            return ",".join([str(array_to_card(c)) for c in sorted_cards])

        top_str = cards_to_string(jnp.array([card_to_array(c) for c in self.board.top]))
        middle_str = cards_to_string(jnp.array([card_to_array(c) for c in self.board.middle]))
        bottom_str = cards_to_string(jnp.array([card_to_array(c) for c in self.board.bottom]))
        discarded_str = cards_to_string(jnp.array([card_to_array(c) for c in self.discarded_cards]))

        #  Добавляем информацию о видимых картах соперника
        if visible_opponent_cards is not None:
            visible_opponent_str = cards_to_string(visible_opponent_cards)
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
        my_total_royalty = jnp.sum(my_royalties)
        my_line_wins = 0

        #  Сравниваем линии (нужен opponent_board)
        if opponent_board is None:
            raise ValueError("Opponent board must be provided for terminal states")

        opponent_royalties = self.calculate_royalties_for_board(opponent_board) # нужна функция
        opponent_total_royalty = jnp.sum(opponent_royalties)

        #  Сравнение линий (scoring 1-6)
        if self.evaluate_hand(jnp.array([card_to_array(card) for card in self.board.bottom]))[0] < self.evaluate_hand(jnp.array([card_to_array(card) for card in opponent_board.bottom]))[0]:
            my_line_wins += 1
        elif self.evaluate_hand(jnp.array([card_to_array(card) for card in self.board.bottom]))[0] > self.evaluate_hand(jnp.array([card_to_array(card) for card in opponent_board.bottom]))[0]:
            my_line_wins -= 1

        if self.evaluate_hand(jnp.array([card_to_array(card) for card in self.board.middle]))[0] < self.evaluate_hand(jnp.array([card_to_array(card) for card in opponent_board.middle]))[0]:
            my_line_wins += 1
        elif self.evaluate_hand(jnp.array([card_to_array(card) for card in self.board.middle]))[0] > self.evaluate_hand(jnp.array([card_to_array(card) for card in opponent_board.middle]))[0]:
            my_line_wins -= 1

        if self.evaluate_hand(jnp.array([card_to_array(card) for card in self.board.top]))[0] < self.evaluate_hand(jnp.array([card_to_array(card) for card in opponent_board.top]))[0]:
            my_line_wins += 1
        elif self.evaluate_hand(jnp.array([card_to_array(card) for card in self.board.top]))[0] > self.evaluate_hand(jnp.array([card_to_array(card) for card in opponent_board.top]))[0]:
            my_line_wins -= 1

        #  Скуп (scoop)
        if my_line_wins == 3:
            my_line_wins += 3  #  Дополнительные 3 очка за скуп
        elif my_line_wins == -3:
            my_line_wins -= 3

        #  Итоговый выигрыш/проигрыш (разница)
        return (my_total_royalty + my_line_wins) - (opponent_total_royalty - my_line_wins)

    def calculate_royalties_for_board(self, board: Board) -> jnp.ndarray:
        """
        Вспомогательная функция для расчета роялти для *чужой* доски (JAX версия).
        """
        #  Создаем JAX-массивы для top, middle, bottom
        top_cards_jax = jnp.array([card_to_array(card) for card in board.top])
        middle_cards_jax = jnp.array([card_to_array(card) for card in board.middle])
        bottom_cards_jax = jnp.array([card_to_array(card) for card in board.bottom])

        #  Проверяем, не является ли рука мертвой
        if len(top_cards_jax) < 3 or len(middle_cards_jax) < 5 or len(bottom_cards_jax) < 5 or self.is_dead_hand_for_board(board):
            return jnp.array([0, 0, 0])

        top_rank, _ = self.evaluate_hand(top_cards_jax)
        middle_rank, _ = self.evaluate_hand(middle_cards_jax)
        bottom_rank, _ = self.evaluate_hand(bottom_cards_jax)

        top_rank_index = None
        if top_rank == 7:  # Сет
            top_rank_index = top_cards_jax[0, 0]  #  Индекс ранга сета
        elif top_rank == 8: # Пара
            top_rank_index = jnp.where(jnp.bincount(top_cards_jax[:, 0], minlength=13) == 2)[0][0]

        royalties = jnp.array([
            self.get_royalty(0, top_rank, top_rank_index),
            self.get_royalty(1, middle_rank),
            self.get_royalty(2, bottom_rank)
        ])

        return royalties

    def is_dead_hand(self) -> bool:
        """Checks if the hand is a dead hand (invalid combination order)."""
        if not self.board.is_full():
            return False

        top_rank, _ = self.evaluate_hand(jnp.array([card_to_array(card) for card in self.board.top]))
        middle_rank, _ = self.evaluate_hand(jnp.array([card_to_array(card) for card in self.board.middle]))
        bottom_rank, _ = self.evaluate_hand(jnp.array([card_to_array(card) for card in self.board.bottom]))

        return top_rank > middle_rank or middle_rank > bottom_rank

    def is_dead_hand_for_board(self, board) -> bool:
        """Checks if the hand is a dead hand (invalid combination order)."""
        top_rank, _ = self.evaluate_hand(jnp.array([card_to_array(card) for card in board.top]))
        middle_rank, _ = self.evaluate_hand(jnp.array([card_to_array(card) for card in board.middle]))
        bottom_rank, _ = self.evaluate_hand(jnp.array([card_to_array(card) for card in board.bottom]))

        return top_rank > middle_rank or middle_rank > bottom_rank

    @jit
    def is_valid_fantasy_entry_jax(self, board: jnp.ndarray) -> bool:
        """Checks if an action leads to a valid fantasy mode entry (JAX version)."""
        top_cards = board[:3]
        top_cards = top_cards[jnp.any(top_cards != -1, axis=1)]

        if len(top_cards) < 3:
            return False

        if self.is_dead_hand_for_placement(board):
            return False

        top_rank, _ = self.evaluate_hand(top_cards)
        if top_rank == 8:  # Пара
            #  Проверяем, что это QQ, KK или AA
            return jnp.any(top_cards[:, 0] >= 10)  # Индексы Q, K, A: 10, 11, 12
        elif top_rank == 7:  # Сет
            return True
        return False

    @jit
    def is_valid_fantasy_repeat_jax(self, board: jnp.ndarray) -> bool:
        """Checks if an action leads to a valid fantasy mode repeat (JAX version)."""
        top_cards = board[:3]
        middle_cards = board[3:8]
        bottom_cards = board[8:13]
        top_cards = top_cards[jnp.any(top_cards != -1, axis=1)]
        middle_cards = middle_cards[jnp.any(middle_cards != -1, axis=1)]
        bottom_cards = bottom_cards[jnp.any(bottom_cards != -1, axis=1)]

        if len(top_cards) < 3 or len(middle_cards) < 5 or len(bottom_cards) < 5:
            return False

        if self.is_dead_hand_for_placement(board):
            return False

        top_rank, _ = self.evaluate_hand(top_cards)
        bottom_rank, _ = self.evaluate_hand(bottom_cards)

        if self.ai_settings['fantasyType'] == 'progressive':
            if top_rank == 7:  # Сет
                return True
            elif bottom_rank <= 3:  # Каре или лучше
                return True
            else:
                return False
        else:  #  Обычный fantasyType
            if top_rank == 7:  # Сет в верхнем ряду
                return True
            if bottom_rank <= 3:  # Каре или лучше в нижнем ряду
                return True
            return False

    def mark_card_as_used(self, card: Card) -> None:
        """Marks a card as used (either placed on the board or discarded)."""
        if card not in self.discarded_cards:
            self.discarded_cards.append(card)

    @staticmethod
    @jit
    def get_royalty(line: jnp.int32, rank: jnp.int32, rank_index: Optional[jnp.int32] = None) -> jnp.int32:
        # line: 0 - top, 1 - middle, 2 - bottom
        # rank: индекс комбинации (0-10)
        # rank_index: индекс ранга (для сетов, пар)

        top_royalties = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        top_royalties = top_royalties.at[7].set(jnp.where(rank_index is not None, 10 + rank_index, 0))
        top_royalties = top_royalties.at[8].set(jnp.where((rank_index is not None) & (rank_index >= 4), rank_index - 3, 0))

        middle_royalties = jnp.array([0, 50, 30, 20, 12, 8, 4, 2, 0, 0, 0])
        bottom_royalties = jnp.array([0, 25, 15, 10, 6, 4, 2, 0, 0, 0, 0])

        return jnp.where(line == 0, top_royalties[rank],
                        jnp.where(line == 1, middle_royalties[rank], bottom_royalties[rank]))

    @jit
    def calculate_royalties(self) -> jnp.ndarray:
        """
        Корректный расчет роялти по американским правилам (JAX-версия).
        """

        #  Создаем JAX-массивы для top, middle, bottom
        top_cards_jax = jnp.array([card_to_array(card) for card in self.board.top])
        middle_cards_jax = jnp.array([card_to_array(card) for card in self.board.middle])
        bottom_cards_jax = jnp.array([card_to_array(card) for card in self.board.bottom])

        #  Проверяем, не является ли рука мертвой
        if len(top_cards_jax) < 3 or len(middle_cards_jax) < 5 or len(bottom_cards_jax) < 5 or self.is_dead_hand():
            return jnp.array([0, 0, 0])

        top_rank, _ = self.evaluate_hand(top_cards_jax)
        middle_rank, _ = self.evaluate_hand(middle_cards_jax)
        bottom_rank, _ = self.evaluate_hand(bottom_cards_jax)

        top_rank_index = None
        if top_rank == 7:  # Сет
            top_rank_index = top_cards_jax[0, 0]  #  Индекс ранга сета
        elif top_rank == 8: # Пара
            top_rank_index = jnp.where(jnp.bincount(top_cards_jax[:, 0], minlength=13) == 2)[0][0]

        royalties = jnp.array([
            self.get_royalty(0, top_rank, top_rank_index),
            self.get_royalty(1, middle_rank),
            self.get_royalty(2, bottom_rank)
        ])

        return royalties

    @jit
    def get_line_royalties(self, cards_jax: jnp.ndarray, line: str) -> int:
        """Calculates royalties for a specific line."""
        if cards_jax.shape[0] == 0:
            return 0

        rank, _ = self.evaluate_hand(cards_jax)
        if line == "top":
            if rank == 7:
                return 10 + cards_jax[0, 0]
            elif rank == 8:
                return self.get_pair_bonus(cards_jax)
            elif rank == 9:
                return self.get_high_card_bonus(cards_jax)
        elif line == "middle":
            if rank <= 6:
                return self.get_royalties_for_hand(rank) * 2
        elif line == "bottom":
            if rank <= 6:
                return self.get_royalties_for_hand(rank)
        return 0

    @jit
    def get_royalties_for_hand(self, hand_rank: int) -> int:
        return jnp.where(hand_rank == 1, 25,
               jnp.where(hand_rank == 2, 15,
               jnp.where(hand_rank == 3, 10,
               jnp.where(hand_rank == 4, 6,
               jnp.where(hand_rank == 5, 4,
               jnp.where(hand_rank == 6, 2, 0))))))

    @jit
    def get_pair_bonus(self, cards_jax: jnp.ndarray) -> int:
        """Calculates the bonus for a pair in the top line."""
        if cards_jax.shape[0] != 3:
            return 0
        ranks = cards_jax[:, 0]
        pair_rank_index = jnp.where(jnp.bincount(ranks, minlength=13) == 2)[0]
        return jnp.where(pair_rank_index.size > 0, jnp.maximum(0, pair_rank_index[0] - 4), 0)

    @jit
    def get_high_card_bonus(self, cards_jax: jnp.ndarray) -> int:
        """Calculates the bonus for a high card in the top line."""
        if cards_jax.shape[0] != 3:
            return 0
        ranks = cards_jax[:, 0]
        if jnp.unique(ranks).shape[0] == 3:
            high_card_index = jnp.max(ranks)
            return jnp.where(high_card_index == 12, 1, 0)
        return 0

    @jit
    def get_progressive_fantasy_cards(self, board: jnp.ndarray) -> int:
        top_cards = board[:3]
        top_cards = top_cards[jnp.any(top_cards != -1, axis=1)]
        top_rank, _ = self.evaluate_hand(top_cards)
        if top_rank == 8:  # Пара
            rank = top_cards[0, 0]  # Ранг первой карты
            return jnp.where(rank == 12, 16,  # A
                    jnp.where(rank == 11, 15,  # K
                    jnp.where(rank == 10, 14,  # Q
                    14)))  # По умолчанию 14
        elif top_rank == 7:  # Сет
            return 17
        return 14

# ... (Остальные классы и функции)

def card_to_array(card: Optional[Card]) -> jnp.ndarray:
    """Преобразует Card в JAX-массив [rank, suit]."""
    if card is None:
        return jnp.array([-1, -1], dtype=jnp.int32)  #  Пустой слот
    return jnp.array([Card.RANKS.index(card.rank), Card.SUITS.index(card.suit)], dtype=jnp.int32)

def array_to_card(card_array: jnp.ndarray) -> Optional[Card]:
    """Преобразует JAX-массив [rank, suit] обратно в Card."""
    try:
        return Card(jnp.take(Card.RANKS, card_array[0]), jnp.take(Card.SUITS, card_array[1]))
    except IndexError:
        return None

@jit
def generate_placements(cards_jax: jnp.ndarray, board: jnp.ndarray, ai_settings: Dict, max_combinations: int = 10000) -> jnp.ndarray:
    """
    Генерирует все возможные *допустимые* размещения карт на доске (JAX-версия).
    Принимает и возвращает JAX-массивы.
    """
    num_cards = cards_jax.shape[0]

    #  Определяем количество свободных слотов в каждой линии
    free_slots_top = 3 - jnp.sum(jnp.any(board[:3] != -1, axis=1))
    free_slots_middle = 5 - jnp.sum(jnp.any(board[3:8] != -1, axis=1))
    free_slots_bottom = 5 - jnp.sum(jnp.any(board[8:13] != -1, axis=1))

    #  Генерируем допустимые комбинации линий (используя JAX)
    valid_combinations = []

    if num_cards == 1:
        if free_slots_top > 0:
            valid_combinations.append(jnp.array([0]))
        if free_slots_middle > 0:
            valid_combinations.append(jnp.array([1]))
        if free_slots_bottom > 0:
            valid_combinations.append(jnp.array([2]))

    elif num_cards == 2:
        for c1_line in range(3):
            if (c1_line == 0 and free_slots_top > 0) or (c1_line == 1 and free_slots_middle > 0) or (c1_line == 2 and free_slots_bottom > 0):
                for c2_line in range(3):
                    if c1_line == c2_line:
                        if (c1_line == 0 and free_slots_top > 1) or (c1_line == 1 and free_slots_middle > 1) or (c1_line == 2 and free_slots_bottom > 1):
                            valid_combinations.append(jnp.array([c1_line, c2_line]))
                    elif (c2_line == 0 and free_slots_top > 0) or (c2_line == 1 and free_slots_middle > 0) or (c2_line == 2 and free_slots_bottom > 0):
                        valid_combinations.append(jnp.array([c1_line, c2_line]))

    elif num_cards == 3:
        for c1_line in range(3):
            if (c1_line == 0 and free_slots_top > 0) or (c1_line == 1 and free_slots_middle > 0) or (c1_line == 2 and free_slots_bottom > 0):
                for c2_line in range(3):
                    if c1_line == c2_line:
                        if (c1_line == 0 and free_slots_top > 1) or (c1_line == 1 and free_slots_middle > 1) or (c1_line == 2 and free_slots_bottom > 1):
                            for c3_line in range(3):
                                if c2_line == c3_line:
                                    if (c1_line == 0 and free_slots_top > 2) or (c1_line == 1 and free_slots_middle > 2) or (c1_line == 2 and free_slots_bottom > 2):
                                        valid_combinations.append(jnp.array([c1_line, c2_line, c3_line]))
                                elif (c3_line == 0 and free_slots_top > 0) or (c3_line == 1 and free_slots_middle > 0) or (c3_line == 2 and free_slots_bottom > 0):
                                    valid_combinations.append(jnp.array([c1_line, c2_line, c3_line]))
                    elif (c2_line == 0 and free_slots_top > 0) or (c2_line == 1 and free_slots_middle > 0) or (c2_line == 2 and free_slots_bottom > 0):
                        for c3_line in range(3):
                            if c2_line == c3_line:
                                if (c2_line == 0 and free_slots_top > 1) or (c2_line == 1 and free_slots_middle > 1) or (c2_line == 2 and free_slots_bottom > 1):
                                    valid_combinations.append(jnp.array([c1_line, c2_line, c3_line]))
                            elif (c3_line == 0 and free_slots_top > 0) or (c3_line == 1 and free_slots_middle > 0) or (c3_line == 2 and free_slots_bottom > 0):
                                valid_combinations.append(jnp.array([c1_line, c2_line, c3_line]))

    #  TODO: Добавить логику для 4, 5 карт (JAX-версия)
    else:
        #  Временное решение для num_cards > 3 (неоптимальное)
        line_combinations = jnp.array(list(itertools.product([0, 1, 2], repeat=num_cards)))
        for comb in line_combinations:
            counts = jnp.bincount(comb, minlength=3)
            if counts[0] <= free_slots_top and counts[1] <= free_slots_middle and counts[2] <= free_slots_bottom:
                valid_combinations.append(comb)


    #  Ограничиваем количество комбинаций и преобразуем в JAX-массив
    if len(valid_combinations) > max_combinations:
        valid_combinations = valid_combinations[:max_combinations]
    valid_combinations = jnp.array(valid_combinations)

    #  Генерируем размещения для каждой допустимой комбинации линий
    all_placements = []
    for comb in valid_combinations:
        #  Создаем перестановки карт (JAX-версия)
        permutations = jnp.array(list(itertools.permutations(cards_jax)))

        for perm in permutations:
            placement = jnp.full((14, 2), -1, dtype=jnp.int32)  #  Пустое размещение
            top_indices = jnp.where(comb == 0)[0]
            middle_indices = jnp.where(comb == 1)[0]
            bottom_indices = jnp.where(comb == 2)[0]

            #  Заполняем размещение
            placement = placement.at[top_indices].set(perm[:len(top_indices)])
            placement = placement.at[jnp.array(middle_indices) + 3].set(perm[len(top_indices):len(top_indices) + len(middle_indices)])
            placement = placement.at[jnp.array(bottom_indices) + 8].set(perm[len(top_indices) + len(middle_indices):])
            all_placements.append(placement)

    all_placements = jnp.array(all_placements)

    #  Фильтруем недопустимые размещения (dead hand) - JAX-версия
    is_dead_hand_vmap = jax.vmap(is_dead_hand_for_placement)  #  Векторизуем функцию проверки
    dead_hands = is_dead_hand_vmap(all_placements, ai_settings)
    valid_placements = all_placements[~dead_hands] #  Используем маску для фильтрации

    return valid_placements

@jit
def is_dead_hand_for_placement(placement: jnp.ndarray, ai_settings: Dict) -> bool:
    """
    Проверяет, является ли размещение мертвой рукой (JAX-версия).
    Принимает JAX-массив placement (14, 2).
    """
    top_cards = placement[:3]
    middle_cards = placement[3:8]
    bottom_cards = placement[8:13]

    #  Удаляем пустые слоты (-1, -1)
    top_cards = top_cards[jnp.any(top_cards != -1, axis=1)]
    middle_cards = middle_cards[jnp.any(middle_cards != -1, axis=1)]
    bottom_cards = bottom_cards[jnp.any(bottom_cards != -1, axis=1)]

    #  Если каких-то линий нет (например, в начале игры), считаем, что рука не мертвая
    if len(top_cards) == 0 or len(middle_cards) == 0 or len(bottom_cards) == 0:
        return False

    top_rank = _identify_combination(top_cards)
    middle_rank = _identify_combination(middle_cards)
    bottom_rank = _identify_combination(bottom_cards)

    return (top_rank > middle_rank) or (middle_rank > bottom_rank)

@jit
def generate_actions_jax(game_state: GameState) -> jnp.ndarray:
    """
    Возвращает JAX-массив возможных действий для данного состояния игры.
    """
    logger.debug("generate_actions_jax - START")
    if game_state.is_terminal():
        logger.debug("generate_actions_jax - Game is terminal, returning empty actions")
        return jnp.array([])

    num_cards = len(game_state.selected_cards.cards)
    actions = []

    if num_cards > 0:
        selected_cards_jax = jnp.array([card_to_array(card) for card in game_state.selected_cards.cards])

        # Режим фантазии
        if game_state.ai_settings.get("fantasyMode", False):
            #  1.  Сначала проверяем, можем ли мы ОСТАТЬСЯ в "Фантазии"
            can_repeat = False
            if game_state.ai_settings.get("fantasyType") == "progressive":
                #  Для progressive fantasy repeat - сет вверху или каре (и лучше) внизу
                permutations = jnp.array(list(itertools.permutations(selected_cards_jax)))
                for perm in permutations:
                    action = jnp.full((14, 2), -1, dtype=jnp.int32)
                    action = action.at[:3].set(perm[:3])  #  Top
                    action = action.at[3:8].set(perm[3:8])  #  Middle
                    action = action.at[8:13].set(perm[8:13])  # Bottom

                    if game_state.is_valid_fantasy_repeat_jax(action):
                        can_repeat = True
                        actions.append(action)
                        break  #  Если нашли хоть одно, дальше не ищем
            else:
                #  Для обычной "Фантазии" - сет вверху или каре (и лучше) внизу
                permutations = jnp.array(list(itertools.permutations(selected_cards_jax)))
                for perm in permutations:
                    action = jnp.full((14, 2), -1, dtype=jnp.int32)
                    action = action.at[:3].set(perm[:3])  #  Top
                    action = action.at[3:8].set(perm[3:8])  #  Middle
                    action = action.at[8:13].set(perm[8:13])  # Bottom

                    if game_state.is_valid_fantasy_repeat_jax(action):
                        can_repeat = True
                        actions.append(action)
                        break

            #  2.  Если остаться в "Фантазии" нельзя (или не были в ней),
            #      генерируем все допустимые действия и выбираем лучшее по роялти
            if not can_repeat:
                possible_actions = []
                permutations = jnp.array(list(itertools.permutations(selected_cards_jax)))
                for perm in permutations:
                    action = jnp.full((14, 2), -1, dtype=jnp.int32)
                    action = action.at[:3].set(perm[:3])  #  Top
                    action = action.at[3:8].set(perm[3:8])  #  Middle
                    action = action.at[8:13].set(perm[8:13])  # Bottom

                    if not is_dead_hand_for_placement(action, game_state.ai_settings):
                        possible_actions.append(action)

                #  Выбираем действие с максимальным роялти
                if possible_actions:
                    #  Создаем временный Board для каждого действия
                    temp_boards = []
                    for action in possible_actions:
                        temp_board = Board()
                        for i in range(3):
                            try:
                                temp_board.place_card("top", array_to_card(action[i]))
                            except (IndexError, TypeError):
                                pass
                        for i in range(3, 8):
                            try:
                                temp_board.place_card("middle", array_to_card(action[i]))
                            except (IndexError, TypeError):
                                pass
                        for i in range(8, 13):
                            try:
                                temp_board.place_card("bottom", array_to_card(action[i]))
                            except (IndexError, TypeError):
                                pass
                        temp_boards.append(temp_board)

                    #  Векторизуем расчет роялти
                    calculate_royalties_vmap = jax.vmap(game_state.calculate_royalties_for_board)
                    royalties = calculate_royalties_vmap(jnp.array(temp_boards))  #  Передаем массив Board
                    total_royalties = jnp.sum(royalties, axis=1)  #  Суммируем по каждой линии

                    #  Находим индекс лучшего действия
                    best_action_index = jnp.argmax(total_royalties)
                    actions.append(possible_actions[best_action_index])

        # Особый случай: ровно 3 карты
        elif num_cards == 3:
            for discarded_index in range(3):
                indices_to_place = jnp.array([j for j in range(3) if j != discarded_index])
                cards_to_place_jax = selected_cards_jax[indices_to_place]
                discarded_card_jax = selected_cards_jax[discarded_index]

                #  Преобразуем текущее состояние доски в JAX-массив
                current_board_jax = jnp.full((14, 2), -1, dtype=jnp.int32)
                for i, card in enumerate(game_state.board.top):
                    current_board_jax = current_board_jax.at[i].set(card_to_array(card))
                for i, card in enumerate(game_state.board.middle):
                    current_board_jax = current_board_jax.at[i + 3].set(card_to_array(card))
                for i, card in enumerate(game_state.board.bottom):
                    current_board_jax = current_board_jax.at[i + 8].set(card_to_array(card))

                placements = generate_placements(cards_to_place_jax, current_board_jax, game_state.ai_settings)
                for placement in placements:
                    action = placement.at[13].set(discarded_card_jax)
                    actions.append(action)

        # Общий случай
        else:
            #  Преобразуем текущее состояние доски в JAX-массив
            current_board_jax = jnp.full((14, 2), -1, dtype=jnp.int32)
            for i, card in enumerate(game_state.board.top):
                current_board_jax = current_board_jax.at[i].set(card_to_array(card))
            for i, card in enumerate(game_state.board.middle):
                current_board_jax = current_board_jax.at[i + 3].set(card_to_array(card))
            for i, card in enumerate(game_state.board.bottom):
                current_board_jax = current_board_jax.at[i + 8].set(card_to_array(card))

            placements = generate_placements(selected_cards_jax, current_board_jax, game_state.ai_settings)
            for placement in placements:
                #  Заполняем discarded
                placed_indices = []
                for i in range(13):  #  Первые 13 слотов
                    if not jnp.array_equal(placement[i], jnp.array([-1, -1])):
                        for j in range(selected_cards_jax.shape[0]):
                            if jnp.array_equal(placement[i], selected_cards_jax[j]):
                                placed_indices.append(j)
                                break
                discarded_indices = [i for i in range(selected_cards_jax.shape[0]) if i not in placed_indices]
                discarded_cards_jax = selected_cards_jax[jnp.array(discarded_indices)]

                #  Добавляем все возможные варианты discarded карт (JAX-версия)
                for i in range(discarded_cards_jax.shape[0] + 1):
                    for discarded_combination in itertools.combinations(discarded_cards_jax, i):
                        action = placement.copy()
                        for j, card_array in enumerate(discarded_combination):
                            action = action.at[13 + j].set(card_array)
                        actions.append(action)

    logger.debug(f"Generated {len(actions)} actions")
    logger.debug("generate_actions_jax - END")
    return jnp.array(actions)

class CFRNode:
    def __init__(self, num_actions: int):
        self.regret_sum = jnp.zeros(num_actions)
        self.strategy_sum = jnp.zeros(num_actions)
        self.num_actions = num_actions

    @jit
    def get_strategy(self, realization_weight: float) -> jnp.ndarray:
        regret_sum = jnp.maximum(self.regret_sum, 0)
        normalizing_sum = jnp.sum(regret_sum)
        strategy = jnp.where(normalizing_sum > 0, regret_sum / normalizing_sum, jnp.ones(self.num_actions) / self.num_actions)
        self.strategy_sum += realization_weight * strategy
        return strategy

    @jit
    def get_average_strategy(self) -> jnp.ndarray:
        normalizing_sum = jnp.sum(self.strategy_sum)
        strategy = jnp.where(normalizing_sum > 0, self.strategy_sum / normalizing_sum, jnp.ones(self.num_actions) / self.num_actions)
        return strategy

class CFRAgent:
    def __init__(self, iterations: int = 500000, stop_threshold: float = 0.001, batch_size: int = 1, max_nodes: int = 100000, ai_settings: Optional[Dict] = None):
        """
        Инициализация оптимизированного MCCFR агента (с JAX).
        """
        self.iterations = iterations
        self.stop_threshold = stop_threshold
        self.save_interval = 2000
        self.key = random.PRNGKey(0)
        self.batch_size = batch_size
        self.max_nodes = max_nodes
        self.nodes_mask = jnp.zeros(max_nodes, dtype=bool)
        self.regret_sums = jnp.zeros((max_nodes, 14 * 2))  # Максимальный размер действия
        self.strategy_sums = jnp.zeros((max_nodes, 14 * 2)) # Максимальный размер действия
        self.num_actions_arr = jnp.zeros(max_nodes, dtype=jnp.int32)
        self.node_counter = 0
        self.nodes_map = {} # {hash(info_set): node_index}
        self.ai_settings = ai_settings if ai_settings is not None else {}


    def get_move(self, game_state: GameState, timeout_event: Event, result: Dict) -> None:
        """
        Выбирает ход (случайный из возможных).
        """
        logger.debug("Inside CFRAgent get_move")

        actions = generate_actions_jax(game_state)
        if actions.shape[0] == 0:
            result["move"] = {"error": "Нет доступных ходов"}
            logger.debug("No actions available, returning error.")
            return

        self.key, subkey = random.split(self.key)
        action_index = int(random.choice(subkey, jnp.arange(actions.shape[0])))
        result["move"] = actions[action_index]  #  Возвращаем JAX-массив
        logger.debug(f"Selected action index: {action_index}")
        logger.debug(f"Final selected move: {result['move']}")

    def train(self, timeout_event: Event, result: Dict) -> None:
        """
        Функция обучения MCCFR (с пакетным обновлением стратегии и jax.vmap).
        """

        def play_one_batch(key):
            """
            Разыгрывает одну партию и возвращает траекторию.
            """
            all_cards = Card.get_all_cards()
            key, subkey = random.split(key)

            #  Преобразуем список объектов Card в список JAX массивов *перед* jnp.array
            all_cards_jax_list = [card_to_array(card) for card in all_cards]
            jax_all_cards = jnp.array(all_cards_jax_list)  #  Создаем JAX массив *из списка JAX массивов*
            all_cards_permuted_jax = random.permutation(subkey, jax_all_cards)  #  Используем jax_all_cards для permutation

            # all_cards = [array_to_card(card_array) for card_array in all_cards_permuted_jax.tolist()] #  Удаляем, т.к. работаем с JAX-массивами
            game_state_p0 = GameState(deck=all_cards_permuted_jax, ai_settings=self.ai_settings)  #  Передаем JAX массив
            game_state_p1 = GameState(deck=all_cards_permuted_jax, ai_settings=self.ai_settings)  #  Передаем JAX массив

            #  Флаги "Фантазии" для каждого игрока
            fantasy_p0 = False
            fantasy_p1 = False

            #  Начальные вероятности (произведения) - 1.0
            pi_0 = 1.0
            pi_1 = 1.0

            trajectory = {
                'info_sets': [],
                'action_indices': [],
                'pi_0': [],
                'pi_1': [],
                'player': [],
                'actions': []
            }

            #  Определяем, кто дилер (в первой партии - случайно)
            nonlocal dealer
            if 'dealer' not in locals():
                key, subkey = random.split(key)
                dealer = int(random.choice(subkey, jnp.array([0, 1])))
            else:
                dealer = 1 - dealer

            #  Определяем, кто ходит первым (тот, кто слева от дилера)
            current_player = 1 - dealer
            current_game_state = game_state_p0 if current_player == 0 else game_state_p1
            opponent_game_state = game_state_p1 if current_player == 0 else game_state_p0
            first_player = current_player

            #  Раздаем начальные 5 карт (с учетом видимости)
            game_state_p0.selected_cards = Hand([array_to_card(c) for c in all_cards_permuted_jax[:5]])
            game_state_p1.selected_cards = Hand([array_to_card(c) for c in all_cards_permuted_jax[5:10]])
            visible_cards_p0 = all_cards_permuted_jax[5:10]
            visible_cards_p1 = all_cards_permuted_jax[:5]

            # Инициализируем remaining_cards *после* раздачи начальных карт
            game_state_p0.initialize_remaining_cards()
            game_state_p1.initialize_remaining_cards()

            cards_dealt = 10

            while not game_state_p0.board.is_full() or not game_state_p1.board.is_full():

                #  Определяем, видит ли текущий игрок карты соперника
                if current_player == 0:
                    visible_opponent_cards = visible_cards_p0
                    if fantasy_p1:
                        visible_opponent_cards = jnp.array([])
                else:
                    visible_opponent_cards = visible_cards_p1
                    if fantasy_p0:
                        visible_opponent_cards = jnp.array([])

                info_set = current_game_state.get_information_set(visible_opponent_cards)

                #  Раздаем карты (если нужно)
                if len(current_game_state.selected_cards.cards) == 0:
                    if current_game_state.board.is_full():
                        num_cards_to_deal = 0
                    elif fantasy_p0 and fantasy_p1:
                        if current_game_state.ai_settings['fantasyType'] == 'progressive':
                            if current_player == 0:
                                num_cards_to_deal = current_game_state.get_progressive_fantasy_cards(jnp.array([card_to_array(card) for card in game_state_p0.board.top]))
                            else:
                                num_cards_to_deal = current_game_state.get_progressive_fantasy_cards(jnp.array([card_to_array(card) for card in game_state_p1.board.top]))
                        else:
                            num_cards_to_deal = 14
                    elif (len(current_game_state.board.top) + len(current_game_state.board.middle) + len(current_game_state.board.bottom) == 5) or \
                         (len(current_game_state.board.top) + len(current_game_state.board.middle) + len(current_game_state.board.bottom) < 13):
                        num_cards_to_deal = 3
                    else:
                        num_cards_to_deal = 0

                    if num_cards_to_deal > 0:
                        new_cards_jax = all_cards_permuted_jax[cards_dealt:cards_dealt + num_cards_to_deal]
                        new_cards_jax = new_cards_jax[jnp.any(new_cards_jax != -1, axis=1)] # ФИЛЬТРУЕМ
                        current_game_state.selected_cards = Hand([array_to_card(c) for c in new_cards_jax])
                        cards_dealt += num_cards_to_deal
                        #  Обновляем видимые карты для соперника (если не в "Фантазии")
                        if current_player == 0 and not fantasy_p1:
                            top_jax = jnp.array([card_to_array(card) for card in opponent_game_state.board.top])
                            middle_jax = jnp.array([card_to_array(card) for card in opponent_game_state.board.middle])
                            bottom_jax = jnp.array([card_to_array(card) for card in opponent_game_state.board.bottom])
                            top_jax = top_jax[jnp.any(top_jax != -1, axis=1)]
                            middle_jax = middle_jax[jnp.any(middle_jax != -1, axis=1)]
                            bottom_jax = bottom_jax[jnp.any(bottom_jax != -1, axis=1)]
                            visible_cards_p0 = jnp.concatenate([top_jax, middle_jax, bottom_jax, new_cards_jax])

                        elif current_player == 1 and not fantasy_p0:
                            top_jax = jnp.array([card_to_array(card) for card in opponent_game_state.board.top])
                            middle_jax = jnp.array([card_to_array(card) for card in opponent_game_state.board.middle])
                            bottom_jax = jnp.array([card_to_array(card) for card in opponent_game_state.board.bottom])
                            top_jax = top_jax[jnp.any(top_jax != -1, axis=1)]
                            middle_jax = middle_jax[jnp.any(middle_jax != -1, axis=1)]
                            bottom_jax = bottom_jax[jnp.any(bottom_jax != -1, axis=1)]
                            visible_cards_p1 = jnp.concatenate([top_jax, middle_jax, bottom_jax, new_cards_jax])

                #  Получаем доступные действия
                actions = generate_actions_jax(current_game_state)
                if not actions.shape[0] == 0:

                    self.key, subkey = random.split(self.key)
                    action_index = int(random.choice(subkey, jnp.arange(actions.shape[0])))
                    # action = actions[action_index]

                    trajectory['info_sets'].append(info_set)
                    trajectory['action_indices'].append(action_index)
                    trajectory['pi_0'].append(pi_0 if current_player == 0 else 0.0)
                    trajectory['pi_1'].append(pi_1 if current_player == 1 else 0.0)
                    trajectory['player'].append(current_player)
                    trajectory['actions'].append(actions)

                    if current_player == 0:
                        pi_0 *= 1.0 / actions.shape[0]
                    else:
                        pi_1 *= 1.0 / actions.shape[0]

                    current_game_state = current_game_state.apply_action(actions[action_index])
                    #  Удаляем карты из selected_cards
                    current_game_state.selected_cards = Hand([])


                #  Меняем текущего игрока
                current_player = 1 - current_player
                current_game_state, opponent_game_state = opponent_game_state, current_game_state

                #  После смены игрока обновляем видимые карты
                if current_player == 0:
                    top_jax = jnp.array([card_to_array(card) for card in opponent_game_state.board.top])
                    middle_jax = jnp.array([card_to_array(card) for card in opponent_game_state.board.middle])
                    bottom_jax = jnp.array([card_to_array(card) for card in opponent_game_state.board.bottom])
                    top_jax = top_jax[jnp.any(top_jax != -1, axis=1)]
                    middle_jax = middle_jax[jnp.any(middle_jax != -1, axis=1)]
                    bottom_jax = bottom_jax[jnp.any(bottom_jax != -1, axis=1)]
                    visible_cards_p0 = jnp.concatenate([top_jax, middle_jax, bottom_jax])
                    if fantasy_p1:
                        visible_cards_p0 = jnp.array([])
                else:
                    top_jax = jnp.array([card_to_array(card) for card in opponent_game_state.board.top])
                    middle_jax = jnp.array([card_to_array(card) for card in opponent_game_state.board.middle])
                    bottom_jax = jnp.array([card_to_array(card) for card in opponent_game_state.board.bottom])
                    top_jax = top_jax[jnp.any(top_jax != -1, axis=1)]
                    middle_jax = middle_jax[jnp.any(middle_jax != -1, axis=1)]
                    bottom_jax = bottom_jax[jnp.any(bottom_jax != -1, axis=1)]
                    visible_cards_p1 = jnp.concatenate([top_jax, middle_jax, bottom_jax])
                    if fantasy_p0:
                        visible_cards_p1 = jnp.array([])


            #  После того, как оба игрока заполнили доски:
            #  1.  Проверяем, попал ли кто-то в "Фантазию"
            if not fantasy_p0 and game_state_p0.is_valid_fantasy_entry_jax(jnp.concatenate([
                jnp.array([card_to_array(card) for card in game_state_p0.board.top]),
                jnp.array([card_to_array(card) for card in game_state_p0.board.middle]),
                jnp.array([card_to_array(card) for card in game_state_p0.board.bottom]),
                jnp.full((1, 2), -1, dtype=jnp.int32)  #  Добавляем пустой слот для discarded
            ])):
                fantasy_p0 = True
            if not fantasy_p1 and game_state_p1.is_valid_fantasy_entry_jax(jnp.concatenate([
                jnp.array([card_to_array(card) for card in game_state_p1.board.top]),
                jnp.array([card_to_array(card) for card in game_state_p1.board.middle]),
                jnp.array([card_to_array(card) for card in game_state_p1.board.bottom]),
                jnp.full((1, 2), -1, dtype=jnp.int32)  #  Добавляем пустой слот для discarded
            ])):
                fantasy_p1 = True

            #  2.  Проверяем, может ли кто-то остаться в "Фантазии"
            if fantasy_p0 and not game_state_p0.is_valid_fantasy_repeat_jax(jnp.concatenate([
                jnp.array([card_to_array(card) for card in game_state_p0.board.top]),
                jnp.array([card_to_array(card) for card in game_state_p0.board.middle]),
                jnp.array([card_to_array(card) for card in game_state_p0.board.bottom]),
                jnp.full((1, 2), -1, dtype=jnp.int32)  #  Добавляем пустой слот для discarded
            ])):
                fantasy_p0 = False  #  Сбрасываем флаг, если не выполнены условия
            if fantasy_p1 and not game_state_p1.is_valid_fantasy_repeat_jax(jnp.concatenate([
                jnp.array([card_to_array(card) for card in game_state_p1.board.top]),
                jnp.array([card_to_array(card) for card in game_state_p1.board.middle]),
                jnp.array([card_to_array(card) for card in game_state_p1.board.bottom]),
                jnp.full((1, 2), -1, dtype=jnp.int32)  #  Добавляем пустой слот для discarded
            ])):
                fantasy_p1 = False

            #  3.  Рассчитываем payoff (с учетом того, кто ходил первым)
            if first_player == 0:
                payoff = float(game_state_p0.get_payoff(opponent_board=game_state_p1.board))
            else:
                payoff = float(game_state_p1.get_payoff(opponent_board=game_state_p0.board))

            trajectory['payoff'] = [payoff] * len(trajectory['info_sets'])

            return trajectory

        play_batch = jax.vmap(play_one_batch)
        dealer = -1 #  Начинаем с -1, чтобы в первой партии дилер был случайным

        for i in range(self.iterations // self.batch_size):
            if timeout_event.is_set():
                logger.info(f"Training interrupted after {i * self.batch_size} iterations due to timeout.")
                break

            self.key, *subkeys = random.split(self.key, num=self.batch_size + 1)
            subkeys = jnp.array(subkeys)

            trajectories = play_batch(subkeys)

            self.update_strategy(trajectories)

            if (i + 1) * self.batch_size % self.save_interval == 0:
                logger.info(f"Iteration {(i + 1) * self.batch_size} of {self.iterations} complete.")
                if (i + 1) * self.batch_size % 10000 == 0:
                    self.save_progress()
                    logger.info(f"Progress saved at iteration {(i + 1) * self.batch_size}")

                if (i + 1) * self.batch_size % 50000 == 0 and self.check_convergence():
                    logger.info(f"CFR agent converged after {(i + 1) * self.batch_size} iterations.")
                    break
            dealer = 1 - dealer #  Меняем дилера для следующего батча

    def update_strategy(self, trajectories):
        """
        Пакетное обновление стратегии на основе накопленных траекторий (JAX-версия).
        """

        #  Объединяем траектории из разных партий в один словарь
        combined_trajectory = {
            'info_sets': [item for trajectory in trajectories for item in trajectory['info_sets']],
            'action_indices': jnp.array([item for trajectory in trajectories for item in trajectory['action_indices']]),
            'pi_0': jnp.array([item for trajectory in trajectories for item in trajectory['pi_0']]),
            'pi_1': jnp.array([item for trajectory in trajectories for item in trajectory['pi_1']]),
            'payoff': jnp.array([item for trajectory in trajectories for item in trajectory['payoff']]),
            'player': jnp.array([item for trajectory in trajectories for item in trajectory['player']]),
            'actions': jnp.array([item for trajectory in trajectories for item in trajectory['actions']])
        }

        #  1. Сначала создаем/обновляем узлы (CFRNode) для всех info_set
        unique_info_sets = set(combined_trajectory['info_sets'])
        # Создаем словарь {info_set: index в массиве actions}
        info_set_to_actions_index = {info_set: i for i, info_set in enumerate(combined_trajectory['info_sets'])}

        for info_set in unique_info_sets:
            info_hash = hash(info_set)
            if info_hash not in self.nodes_map:
                #  Находим индекс первого вхождения info_set в combined_trajectory['info_sets']
                index = combined_trajectory['info_sets'].index(info_set)
                #  Получаем actions, соответствующие этому индексу
                actions = combined_trajectory['actions'][index]

                self.nodes_map[info_hash] = self.node_counter
                #  Инициализируем regret_sum и strategy_sum нулями
                self.regret_sums = self.regret_sums.at[self.node_counter].set(jnp.zeros(actions.shape[0]))
                self.strategy_sums = self.strategy_sums.at[self.node_counter].set(jnp.zeros(actions.shape[0]))
                self.num_actions_arr = self.num_actions_arr.at[self.node_counter].set(actions.shape[0])
                self.nodes_mask = self.nodes_mask.at[self.node_counter].set(True)  #  Узел действителен

                self.node_counter += 1

        #  2. Обновляем regret_sum и strategy_sum (векторизованно)
        def update_node(info_set, action_index, pi_0, pi_1, payoff, player, actions):
            #  Получаем индекс узла по хешу info_set
            node_index = self.nodes_map[hash(info_set)]
            num_actions = self.num_actions_arr[node_index] # Получаем количество действий
            regret_sum = self.regret_sums[node_index]
            strategy_sum = self.strategy_sums[node_index]

            strategy = jnp.maximum(regret_sum, 0)
            normalizing_sum = jnp.sum(strategy)
            strategy = jnp.where(normalizing_sum > 0, strategy / normalizing_sum, jnp.ones(num_actions) / self.num_actions)
            strategy_sum = strategy_sum.at[:num_actions].set(strategy_sum[:num_actions] + (pi_0 if player == 0 else pi_1) * strategy)

            util = jnp.zeros(num_actions)
            util = util.at[action_index].set(payoff if player == 0 else -payoff)
            node_util = jnp.dot(strategy, util)

            if player == 0:
                regret_sum = regret_sum.at[:num_actions].set(regret_sum[:num_actions] + pi_1 * (util - node_util))
            else:
                regret_sum = regret_sum.at[:num_actions].set(regret_sum[:num_actions] + pi_0 * (util - node_util))

            return regret_sum, strategy_sum

        #  Векторизуем update_node
        update_node_vmap = jax.vmap(update_node)

        #  Применяем update_node ко всем элементам траектории
        new_regret_sums, new_strategy_sums = update_node_vmap(
            combined_trajectory['info_sets'],
            combined_trajectory['action_indices'],
            combined_trajectory['pi_0'],
            combined_trajectory['pi_1'],
            combined_trajectory['payoff'],
            combined_trajectory['player'],
            combined_trajectory['actions']
        )

        #  Обновляем массивы regret_sums и strategy_sums
        #  Для этого нужно сначала получить индексы узлов для каждой записи в траектории
        node_indices = jnp.array([self.nodes_map[hash(info_set)] for info_set in combined_trajectory['info_sets']])
        self.regret_sums = self.regret_sums.at[node_indices].set(new_regret_sums)
        self.strategy_sums = self.strategy_sums.at[node_indices].set(new_strategy_sums)

    @jit
    def check_convergence(self) -> bool:
        """
        Проверяет, сошлось ли обучение (средняя стратегия близка к равномерной).
        (JAX-версия)
        """
        #  Вместо цикла по self.nodes.values() используем jnp.where и self.nodes_mask
        valid_indices = jnp.where(self.nodes_mask)[0]  #  Получаем индексы действительных узлов

        #  Функция для проверки сходимости одного узла
        def check_one_node(index):
            num_actions = self.num_actions_arr[index]
            avg_strategy = self.get_average_strategy_by_index(index)  #  Нужна вспомогательная функция
            uniform_strategy = jnp.ones(num_actions) / num_actions
            diff = jnp.mean(jnp.abs(avg_strategy - uniform_strategy))
            return diff > self.stop_threshold  #  True, если НЕ сошлось

        #  Векторизуем проверку
        not_converged = jax.vmap(check_one_node)(valid_indices)

        #  Если хотя бы один узел не сошелся, возвращаем False
        return not jnp.any(not_converged)

    def get_average_strategy_by_index(self, index: int) -> jnp.ndarray:
        """
        Вспомогательная функция для получения средней стратегии по индексу узла.
        """
        strategy_sum = self.strategy_sums[index]
        num_actions = self.num_actions_arr[index]  #  Используем сохраненное количество действий
        normalizing_sum = jnp.sum(strategy_sum)
        return jnp.where(normalizing_sum > 0, strategy_sum / normalizing_sum, jnp.ones(num_actions) / num_actions)

    #  Вспомогательные функции (JAX-версии)
    @jit
    def _get_rank_counts(self, cards_jax: jnp.ndarray) -> jnp.ndarray:
        """Подсчитывает количество карт каждого ранга."""
        ranks = cards_jax[:, 0]
        return jnp.bincount(ranks, minlength=13)

    @jit
    def _get_suit_counts(self, cards_jax: jnp.ndarray) -> jnp.ndarray:
        """Подсчитывает количество карт каждой масти."""
        suits = cards_jax[:, 1]
        return jnp.bincount(suits, minlength=4)

    @jit
    def _is_flush(self, cards_jax: jnp.ndarray) -> bool:
        """Проверяет, является ли набор карт флешем."""
        suits = cards_jax[:, 1]
        return jnp.all(suits == suits[0])  #  Все масти одинаковые

    @jit
    def _is_straight(self, cards_jax: jnp.ndarray) -> bool:
        """Проверяет, является ли набор карт стритом."""
        ranks = jnp.sort(cards_jax[:, 0])
        #  Особый случай: A-5 стрит
        if jnp.array_equal(ranks, jnp.array([0, 1, 2, 3, 12])):
            return True
        return jnp.all(jnp.diff(ranks) == 1)

    @jit
    def _is_straight_flush(self, cards_jax: jnp.ndarray) -> bool:
        return self._is_straight(cards_jax) and self._is_flush(cards_jax)

    @jit
    def _is_royal_flush(self, cards_jax: jnp.ndarray) -> bool:
        if not self._is_flush(cards_jax):
            return False
        ranks = jnp.sort(cards_jax[:, 0])
        return jnp.array_equal(ranks, jnp.array([8, 9, 10, 11, 12]))

    @jit
    def _is_four_of_a_kind(self, cards_jax: jnp.ndarray) -> bool:
        return jnp.any(self._get_rank_counts(cards_jax) == 4)

    @jit
    def _is_full_house(self, cards_jax: jnp.ndarray) -> bool:
        counts = self._get_rank_counts(cards_jax)
        return jnp.any(counts == 3) and jnp.any(counts == 2)

    @jit
    def _is_three_of_a_kind(self, cards_jax: jnp.ndarray) -> bool:
        return jnp.any(self._get_rank_counts(cards_jax) == 3)

    @jit
    def _is_two_pair(self, cards_jax: jnp.ndarray) -> bool:
        return jnp.sum(self._get_rank_counts(cards_jax) == 2) == 2

    @jit
    def _is_one_pair(self, cards_jax: jnp.ndarray) -> bool:
        return jnp.any(self._get_rank_counts(cards_jax) == 2)

    @jit
    def _identify_combination(self, cards_jax: jnp.ndarray) -> int:
        """Определяет тип комбинации (возвращает индекс)."""
        if cards_jax.shape[0] == 0:  #  Пустой набор карт
            return 10

        if cards_jax.shape[0] == 3:
            if self._is_three_of_a_kind(cards_jax):
                return 6  # "three_of_a_kind"
            if self._is_one_pair(cards_jax):
                return 8  # "pair"
            return 9  # high card

        if cards_jax.shape[0] == 5:
            if self._is_royal_flush(cards_jax):
                return 0  # "royal_flush"
            elif self._is_straight_flush(cards_jax):
                return 1  # "straight_flush"
            elif self._is_four_of_a_kind(cards_jax):
                return 2  # "four_of_a_kind"
            elif self._is_full_house(cards_jax):
                return 3  # "full_house"
            elif self._is_flush(cards_jax):
                return 4  # "flush"
            elif self._is_straight(cards_jax):
                return 5  # "straight"
            elif self._is_three_of_a_kind(cards_jax):
                return 6  # "three_of_a_kind"
            elif self._is_two_pair(cards_jax):
                return 7  # "two_pair"
            elif self._is_one_pair(cards_jax):
                return 8  # "pair"
            else:
                return 9  # "high_card"
        return 10

    @jit
    def evaluate_hand(self, cards_jax: jnp.ndarray) -> Tuple[int, float]:
        """
        Оптимизированная оценка покерной комбинации (JAX-версия).
        Возвращает (ранг, score), где меньший ранг = лучшая комбинация.
        """
        if cards_jax.shape[0] == 0:
            return 11, 0.0

        n = cards_jax.shape[0]

        rank_counts = jnp.bincount(cards_jax[:, 0], minlength=13)
        suit_counts = jnp.bincount(cards_jax[:, 1], minlength=4)
        has_flush = jnp.max(suit_counts) == n
        rank_indices = jnp.sort(cards_jax[:, 0])

        is_straight = False
        if jnp.unique(rank_indices).shape[0] == n:
            if jnp.max(rank_indices) - jnp.min(rank_indices) == n - 1:
                is_straight = True
            elif jnp.array_equal(rank_indices, jnp.array([0, 1, 2, 3, 12])):
                is_straight = True

        if n == 3:
            if jnp.max(rank_counts) == 3:
                rank = cards_jax[0, 0]
                return 7, 10.0 + rank
            elif jnp.max(rank_counts) == 2:
                pair_rank_index = jnp.where(rank_counts == 2)[0][0]
                return 8, pair_rank_index / 100.0
            else:
                high_card_rank_index = jnp.max(rank_indices)
                return 9, high_card_rank_index / 100.0

        elif n == 5:
            if has_flush and is_straight:
                if jnp.array_equal(rank_indices, jnp.array([8, 9, 10, 11, 12])):
                    return 1, 25.0  # Роял-флеш
                return 2, 15.0 + jnp.max(rank_indices) / 100.0

            if jnp.max(rank_counts) == 4:
                four_rank_index = jnp.where(rank_counts == 4)[0][0]
                return 3, 10.0 + four_rank_index / 100.0

            if jnp.array_equal(jnp.sort(rank_counts), jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 0])):
                three_rank_index = jnp.where(rank_counts == 3)[0][0]
                return 4, 6.0 + three_rank_index / 100.0

            if has_flush:
                return 5, 4.0 + jnp.max(rank_indices) / 100.0

            if is_straight:
                return 6, 2.0 + jnp.max(rank_indices) / 100.0

            if jnp.max(rank_counts) == 3:
                three_rank_index = jnp.where(rank_counts == 3)[0][0]
                return 7, 2.0 + three_rank_index / 100.0

            pairs = jnp.where(rank_counts == 2)[0]
            if len(pairs) == 2:
                high_pair_index = jnp.max(pairs)
                low_pair_index = jnp.min(pairs)
                return 8, 1.0 + high_pair_index / 100.0 + low_pair_index / 10000.0

            if len(pairs) == 1:
                pair_rank_index = pairs[0]
                return 9, pair_rank_index / 100.0

            return 10, jnp.max(rank_indices) / 100.0

        return 11, 0.0

    # ... (Методы, связанные с узлами CFRNode: get_strategy, get_average_strategy) ...

    def save_progress(self) -> None:
        """Сохраняет прогресс через GitHub."""
        data = {
            "nodes_mask": self.nodes_mask,
            "regret_sums": self.regret_sums,
            "strategy_sums": self.strategy_sums,
            "num_actions_arr": self.num_actions_arr,
            "node_counter": self.node_counter,
            "nodes_map": self.nodes_map,  #  Сохраняем nodes_map
            "iterations": self.iterations,
            "stop_threshold": self.stop_threshold,
            "batch_size": self.batch_size,
            "max_nodes": self.max_nodes,
            "ai_settings": self.ai_settings
        }
        #  Используем функцию из github_utils
        if not save_ai_progress_to_github(data):
            logger.error("Ошибка при сохранении прогресса на GitHub!")


    def load_progress(self) -> None:
        """Загружает прогресс через GitHub."""
        #  Используем функцию из github_utils
        data = load_ai_progress_from_github()
        if data:
            self.nodes_mask = data["nodes_mask"]
            self.regret_sums = data["regret_sums"]
            self.strategy_sums = data["strategy_sums"]
            self.num_actions_arr = data["num_actions_arr"]
            self.node_counter = data["node_counter"]
            self.nodes_map = data["nodes_map"]  #  Загружаем nodes_map
            self.iterations = data["iterations"]
            self.stop_threshold = data.get("stop_threshold", 0.001)  #  Используем значение по умолчанию
            self.batch_size = data.get("batch_size", 1)
            self.max_nodes = data.get("max_nodes", 100000)
            self.ai_settings = data.get("ai_settings", {}) # Загружаем ai_settings
            logger.info("Прогресс AI успешно загружен с GitHub.")
        else:
            logger.warning("Не удалось загрузить прогресс с GitHub.")

class RandomAgent:
    def __init__(self):
        pass

    def get_move(self, game_state: GameState, timeout_event: Event, result: Dict) -> None:
        """
        Выбирает ход (случайный из возможных).
        """
        logger.debug("Inside RandomAgent get_move")

        actions = generate_actions_jax(game_state)
        if actions.shape[0] == 0:
            result["move"] = {"error": "Нет доступных ходов"}
            logger.debug("No actions available, returning error.")
            return

        key = random.PRNGKey(int(time.time()))
        key, subkey = random.split(key)
        action_index = int(random.choice(subkey, jnp.arange(actions.shape[0])))
        result["move"] = actions[action_index]  #  Возвращаем JAX-массив
        logger.debug(f"Selected action index: {action_index}")
        logger.debug(f"Final selected move: {result['move']}")
