import itertools
from collections import defaultdict, Counter  # Counter, возможно, не понадобится
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

        top_rank, _ = self.evaluate_hand(jnp.array([card_to_array(card) for card in self.board.top]))
        middle_rank, _ = self.evaluate_hand(jnp.array([card_to_array(card) for card in self.board.middle]))
        bottom_rank, _ = self.evaluate_hand(jnp.array([card_to_array(card) for card in self.board.bottom]))

        return top_rank > middle_rank or middle_rank > bottom_rank

    def is_valid_fantasy_entry(self, board: Optional[Board] = None) -> bool:
        """Checks if an action leads to a valid fantasy mode entry."""
        if board is None:
            board = self.board

        temp_state = GameState(board=board, ai_settings=self.ai_settings) # Используем self.ai_settings
        if temp_state.is_dead_hand():
            return False

        top_rank, _ = temp_state.evaluate_hand(jnp.array([card_to_array(card) for card in board.top]))
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

        top_rank, _ = temp_state.evaluate_hand(jnp.array([card_to_array(card) for card in board.top]))
        bottom_rank, _ = temp_state.evaluate_hand(jnp.array([card_to_array(card) for card in board.bottom]))
        if self.ai_settings['fantasyType'] == 'progressive':
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

# ... (Остальные классы и функции)

def card_to_array(card: Optional[Card]) -> jnp.ndarray:
    """Преобразует Card в JAX-массив [rank, suit]."""
    if card is None:
        return jnp.array([-1, -1], dtype=jnp.int32)  #  Пустой слот
    return jnp.array([Card.RANKS.index(card.rank), Card.SUITS.index(card.suit)], dtype=jnp.int32)

def array_to_card(card_array: jnp.ndarray) -> Optional[Card]:
    """Преобразует JAX-массив [rank, suit] обратно в Card."""
    if jnp.array_equal(card_array, jnp.array([-1, -1])):
        return None  #  Пустой слот
    return Card(Card.RANKS[card_array[0]], Card.SUITS[card_array[1]])


def generate_placements(cards_jax: jnp.ndarray, board: Board, ai_settings: Dict, max_combinations: int = 10000) -> jnp.ndarray:
    """
    Генерирует все возможные *допустимые* размещения карт на доске (JAX-версия).
    Принимает и возвращает JAX-массивы.  Использует вложенные циклы и itertools.permutations.
    """
    num_cards = cards_jax.shape[0]

    free_slots_top = 3 - len(board.top)
    free_slots_middle = 5 - len(board.middle)
    free_slots_bottom = 5 - len(board.bottom)

    #  Генерируем допустимые комбинации линий (используя вложенные циклы)
    #  Ограничиваем максимальное количество комбинаций, чтобы избежать взрывного роста
    valid_combinations = []
    if num_cards == 1:
        #  Для одной карты - простые варианты
        if free_slots_top > 0:
            valid_combinations.append([0])
        if free_slots_middle > 0:
            valid_combinations.append([1])
        if free_slots_bottom > 0:
            valid_combinations.append([2])

    elif num_cards == 2:
        for c1_line in range(3):
            if (c1_line == 0 and free_slots_top > 0) or (c1_line == 1 and free_slots_middle > 0) or (c1_line == 2 and free_slots_bottom > 0):
                for c2_line in range(3):
                    if c1_line == c2_line:
                        if (c1_line == 0 and free_slots_top > 1) or (c1_line == 1 and free_slots_middle > 1) or (c1_line == 2 and free_slots_bottom > 1):
                            valid_combinations.append([c1_line, c2_line])
                    elif (c2_line == 0 and free_slots_top > 0) or (c2_line == 1 and free_slots_middle > 0) or (c2_line == 2 and free_slots_bottom > 0):
                        valid_combinations.append([c1_line, c2_line])

    elif num_cards == 3:
        for c1_line in range(3):
            if (c1_line == 0 and free_slots_top > 0) or (c1_line == 1 and free_slots_middle > 0) or (c1_line == 2 and free_slots_bottom > 0):
                for c2_line in range(3):
                    if c1_line == c2_line:
                        if (c1_line == 0 and free_slots_top > 1) or (c1_line == 1 and free_slots_middle > 1) or (c1_line == 2 and free_slots_bottom > 1):
                            for c3_line in range(3):
                                if c2_line == c3_line:
                                    if (c1_line == 0 and free_slots_top > 2) or (c1_line == 1 and free_slots_middle > 2) or (c1_line == 2 and free_slots_bottom > 2):
                                        valid_combinations.append([c1_line, c2_line, c3_line])
                                elif (c3_line == 0 and free_slots_top > 0) or (c3_line == 1 and free_slots_middle > 0) or (c3_line == 2 and free_slots_bottom > 0):
                                    valid_combinations.append([c1_line, c2_line, c3_line])
                    elif (c2_line == 0 and free_slots_top > 0) or (c2_line == 1 and free_slots_middle > 0) or (c2_line == 2 and free_slots_bottom > 0):
                        for c3_line in range(3):
                            if c2_line == c3_line:
                                if (c2_line == 0 and free_slots_top > 1) or (c2_line == 1 and free_slots_middle > 1) or (c2_line == 2 and free_slots_bottom > 1):
                                    valid_combinations.append([c1_line, c2_line, c3_line])
                            elif (c3_line == 0 and free_slots_top > 0) or (c3_line == 1 and free_slots_middle > 0) or (c3_line == 2 and free_slots_bottom > 0):
                                valid_combinations.append([c1_line, c2_line, c3_line])

    #  TODO:  Добавить логику для 4, 5, ... карт (вложенные циклы)
    #         Или использовать другой подход (например, рекурсию с мемоизацией)

    else:
        #  Для большего количества карт используем itertools.product (менее эффективно)
        line_combinations = jnp.array(list(itertools.product([0, 1, 2], repeat=num_cards)))
        valid_combinations = []
        for comb in line_combinations:
            counts = jnp.bincount(comb, minlength=3)
            if counts[0] <= free_slots_top and counts[1] <= free_slots_middle and counts[2] <= free_slots_bottom:
                valid_combinations.append(comb)

    valid_combinations = jnp.array(valid_combinations)

    #  Ограничиваем количество комбинаций
    if len(valid_combinations) > max_combinations:
        #  TODO:  Реализовать более умный выбор комбинаций (например, на основе эвристик)
        valid_combinations = valid_combinations[:max_combinations]

    all_placements = []
    for comb in valid_combinations:
        for perm in itertools.permutations(cards_jax):  # Оставляем itertools.permutations
            perm = jnp.array(perm)
            placement = jnp.full((14, 2), -1, dtype=jnp.int32)
            top_indices = jnp.where(comb == 0)[0]
            middle_indices = jnp.where(comb == 1)[0]
            bottom_indices = jnp.where(comb == 2)[0]

            placement = placement.at[top_indices].set(perm[:len(top_indices)])
            placement = placement.at[jnp.array(middle_indices) + 3].set(perm[len(top_indices):len(top_indices) + len(middle_indices)])
            placement = placement.at[jnp.array(bottom_indices) + 8].set(perm[len(top_indices) + len(middle_indices):])
            all_placements.append(placement)

    all_placements = jnp.array(all_placements)

    #  Фильтруем недопустимые размещения (dead hand) - JAX-версия
    is_dead_hand_vmap = jax.vmap(is_dead_hand_jax)  #  Векторизуем функцию проверки
    dead_hands = is_dead_hand_vmap(all_placements, ai_settings)
    valid_placements = all_placements[~dead_hands] #  Используем маску для фильтрации

    return valid_placements


def get_actions(game_state: GameState) -> jnp.ndarray:
    """
    Возвращает JAX-массив возможных действий для данного состояния игры.
    """
    logger.debug("get_actions - START")
    if game_state.is_terminal():
        logger.debug("get_actions - Game is terminal, returning empty actions")
        return jnp.array([])

    num_cards = len(game_state.selected_cards)
    actions = []

    if num_cards > 0:
        selected_cards_jax = jnp.array([[Card.RANKS.index(card.rank), Card.SUITS.index(card.suit)] for card in game_state.selected_cards.cards])

        # Режим фантазии
        if game_state.ai_settings.get("fantasyMode", False):
            #  1.  Сначала проверяем, можем ли мы ОСТАТЬСЯ в "Фантазии"
            can_repeat = False
            if game_state.ai_settings.get("fantasyType") == "progressive":
                #  Для progressive fantasy repeat - сет вверху или каре (и лучше) внизу
                for p in itertools.permutations(range(num_cards)):
                    action = jnp.full((14, 2), -1, dtype=jnp.int32)
                    for i in range(3):
                        if i < len(p):
                            action = action.at[i].set(selected_cards_jax[p[i]])
                    for i in range(5):
                        if 3 + i < len(p):
                            action = action.at[i + 3].set(selected_cards_jax[p[3 + i]])
                    for i in range(5):
                        if 8 + i < len(p):
                            action = action.at[i + 8].set(selected_cards_jax[p[8 + i]])

                    temp_board = Board()
                    temp_board.top = [array_to_card(c) for c in action[:3] if not jnp.array_equal(c, jnp.array([-1, -1]))]
                    temp_board.middle = [array_to_card(c) for c in action[3:8] if not jnp.array_equal(c, jnp.array([-1, -1]))]
                    temp_board.bottom = [array_to_card(c) for c in action[8:13] if not jnp.array_equal(c, jnp.array([-1, -1]))]

                    if game_state.is_valid_fantasy_repeat(temp_board):
                        can_repeat = True
                        actions.append(action)
                        break  #  Если нашли хоть одно, дальше не ищем
            else:
                #  Для обычной "Фантазии" - сет вверху или каре (и лучше) внизу
                for p in itertools.permutations(range(num_cards)):  #  Все перестановки
                    action = jnp.full((14, 2), -1, dtype=jnp.int32)
                    for i in range(3):
                        if i < len(p):
                            action = action.at[i].set(selected_cards_jax[p[i]])  # top
                    for i in range(5):
                        if 3 + i < len(p):
                            action = action.at[i + 3].set(selected_cards_jax[p[3 + i]])  # middle
                    for i in range(5):
                        if 8 + i < len(p):
                            action = action.at[i + 8].set(selected_cards_jax[p[8 + i]])  # bottom

                    #  Создаем временный Board (TODO:  Переделать на JAX)
                    temp_board = Board()
                    temp_board.top = [array_to_card(c) for c in action[:3] if not jnp.array_equal(c, jnp.array([-1, -1]))]
                    temp_board.middle = [array_to_card(c) for c in action[3:8] if not jnp.array_equal(c, jnp.array([-1, -1]))]
                    temp_board.bottom = [array_to_card(c) for c in action[8:13] if not jnp.array_equal(c, jnp.array([-1, -1]))]

                    if game_state.is_valid_fantasy_repeat(temp_board):
                        can_repeat = True
                        actions.append(action)
                        break

            #  2.  Если остаться в "Фантазии" нельзя (или не были в ней),
            #      генерируем все допустимые действия и выбираем лучшее по роялти
            if not can_repeat:
                possible_actions = []
                for p in itertools.permutations(range(num_cards)):
                    action = jnp.full((14, 2), -1, dtype=jnp.int32)
                    for i in range(3):
                        if i < len(p):
                            action = action.at[i].set(selected_cards_jax[p[i]])
                    for i in range(5):
                        if 3 + i < len(p):
                            action = action.at[i + 3].set(selected_cards_jax[p[3 + i]])
                    for i in range(5):
                        if 8 + i < len(p):
                            action = action.at[i + 8].set(selected_cards_jax[p[8 + i]])
                    # discarded (пока не заполняем)

                    #  Добавляем проверку на dead_hand (TODO:  Переделать на JAX)
                    temp_board = Board()
                    temp_board.top = [array_to_card(c) for c in action[:3] if not jnp.array_equal(c, jnp.array([-1, -1]))]
                    temp_board.middle = [array_to_card(c) for c in action[3:8] if not jnp.array_equal(c, jnp.array([-1, -1]))]
                    temp_board.bottom = [array_to_card(c) for c in action[8:13] if not jnp.array_equal(c, jnp.array([-1, -1]))]
                    temp_state = GameState(board=temp_board, ai_settings=game_state.ai_settings)
                    if not temp_state.is_dead_hand():
                        possible_actions.append((action, temp_board))  #  Сохраняем и action, и temp_board

                #  Выбираем действие с максимальным роялти
                if possible_actions:
                    best_action = None
                    best_royalty = -1
                    for action, temp_board in possible_actions:
                        temp_state = GameState(board=temp_board, ai_settings=game_state.ai_settings)
                        royalties = temp_state.calculate_royalties()  #  TODO:  Переделать на JAX
                        total_royalty = sum(royalties.values())
                        if total_royalty > best_royalty:
                            best_royalty = total_royalty
                            best_action = action
                    if best_action is not None:
                        actions.append(best_action)

        # Особый случай: ровно 3 карты
        elif num_cards == 3:
            for discarded_index in range(3):
                indices_to_place = jnp.array([j for j in range(3) if j != discarded_index])
                cards_to_place_jax = selected_cards_jax[indices_to_place]
                discarded_card_jax = selected_cards_jax[discarded_index]

                placements = generate_placements(cards_to_place_jax, game_state.board, game_state.ai_settings)
                for placement in placements:
                    action = placement.at[13].set(discarded_card_jax)
                    actions.append(action)

        # Общий случай
        else:
            placements = generate_placements(selected_cards_jax, game_state.board, game_state.ai_settings)
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

                #  Добавляем все возможные варианты discarded карт
                for i in range(discarded_cards_jax.shape[0] + 1):
                    for discarded_combination in itertools.combinations(discarded_cards_jax, i):
                        action = placement.copy()
                        for j, card_array in enumerate(discarded_combination):
                            action = action.at[13 + j].set(card_array)
                        actions.append(action)


    logger.debug(f"Generated {len(actions)} actions")
    logger.debug("get_actions - END")
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
        self.regret_sums = jnp.zeros((max_nodes, 14 * 2))
        self.strategy_sums = jnp.zeros((max_nodes, 14 * 2))
        self.num_actions_arr = jnp.zeros(max_nodes, dtype=jnp.int32)
        self.node_counter = 0
        self.nodes_map = {} # {hash(info_set): node_index}
        self.ai_settings = ai_settings if ai_settings is not None else {} # <---- ВОТ ЭТА СТРОКА БЫЛА УДАЛЕНА ОШИБОЧНО! ВЕРНИТЕ ЕЕ!


    @jit
    def get_move(self, game_state: GameState, timeout_event: Event, result: Dict) -> None:
        """
        Выбирает ход, используя get_placement.  Для CFRAgent также обновляет стратегию.
        """
        logger.debug("Inside get_move")

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

        #  УДАЛЯЕМ обновление стратегии из get_move
        # if game_state.ai_settings.get("training_mode", False):
        #     ...

    def train(self, timeout_event: Event, result: Dict) -> None:
        """
        Функция обучения MCCFR (с пакетным обновлением стратегии и jax.vmap).
        """

        def play_one_batch(key):
            """
            Разыгрывает одну партию и возвращает траекторию.
            """
            print("CFRAgent.play_one_batch - START")  # Debug print
            all_cards = Card.get_all_cards()
            print(f"CFRAgent.play_one_batch - Initial all_cards type: {type(all_cards)}, first card: {all_cards[0]}") # Debug print
            key, subkey = random.split(key)

            #  Преобразуем список объектов Card в список JAX массивов *перед* jnp.array
            all_cards_jax_list = [card_to_array(card) for card in all_cards] # <---- Преобразование в список JAX массивов
            jax_all_cards = jnp.array(all_cards_jax_list) #  Создаем JAX массив *из списка JAX массивов*
            print(f"CFRAgent.play_one_batch - jax_all_cards type: {type(jax_all_cards)}, dtype: {jax_all_cards.dtype}, shape: {jax_all_cards.shape}, first element: {jax_all_cards[0]}") # Debug print
            all_cards_permuted_jax = random.permutation(subkey, jax_all_cards) #  Используем jax_all_cards для permutation
            print(f"CFRAgent.play_one_batch - all_cards_permuted_jax type: {type(all_cards_permuted_jax)}, dtype: {all_cards_permuted_jax.dtype}, shape: {all_cards_permuted_jax.shape}, first element: {all_cards_permuted_jax[0]}") # Debug print

            # all_cards = random.permutation(subkey, jnp.array(all_cards)) # Original line - replaced
            all_cards = [array_to_card(card_array) for card_array in all_cards_permuted_jax.tolist()] # Modified line - using all_cards_permuted_jax and array_to_card
            print(f"CFRAgent.play_one_batch - all_cards after permutation and tolist type: {type(all_cards)}, first card: {all_cards[0]}") # Debug print


            game_state_p0 = GameState(deck=all_cards, ai_settings=self.ai_settings)
            game_state_p1 = GameState(deck=all_cards, ai_settings=self.ai_settings)

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
            # key, subkey = random.split(key)  #  Уже делили key
            # dealer = int(random.choice(subkey, jnp.array([0, 1])))  #  0 - игрок 0, 1 - игрок 1
            #  В первой партии дилер выбирается случайно, в последующих - меняется
            nonlocal dealer #  Используем nonlocal, т.к. dealer объявлена во внешней функции
            if 'dealer' not in locals():  #  Если дилер еще не определен (первая партия)
                key, subkey = random.split(key)
                dealer = int(random.choice(subkey, jnp.array([0, 1])))
            else:
                dealer = 1 - dealer  #  Меняем дилера

            #  Определяем, кто ходит первым (тот, кто слева от дилера)
            current_player = 1 - dealer
            current_game_state = game_state_p0 if current_player == 0 else game_state_p1
            opponent_game_state = game_state_p1 if current_player == 0 else game_state_p0
            #  Для корректного расчета payoff в конце игры
            first_player = current_player

            #  Раздаем начальные 5 карт (с учетом видимости)
            game_state_p0.selected_cards = Hand(all_cards[:5])
            game_state_p1.selected_cards = Hand(all_cards[5:10])
            #  Игроки видят первые 5 карт друг друга
            visible_cards_p0 = all_cards[5:10]
            visible_cards_p1 = all_cards[:5]

            cards_dealt = 10  #  Счетчик розданных карт

            while not game_state_p0.board.is_full() or not game_state_p1.board.is_full():

                #  Определяем, видит ли текущий игрок карты соперника
                if current_player == 0:
                    visible_opponent_cards = visible_cards_p0
                    if fantasy_p1:
                        visible_opponent_cards = []
                else:  # current_player == 1
                    visible_opponent_cards = visible_cards_p1
                    if fantasy_p0:
                        visible_opponent_cards = []

                #  Получаем info_set с учетом видимых карт
                info_set = current_game_state.get_information_set(visible_opponent_cards)

                #  Раздаем карты (если нужно)
                if len(current_game_state.selected_cards.cards) == 0:
                    if current_game_state.board.is_full():
                        num_cards_to_deal = 0
                    #  Если оба в "Фантазии", раздаем сразу 14 (или сколько нужно)
                    elif fantasy_p0 and fantasy_p1:
                        if current_game_state.ai_settings['fantasyType'] == 'progressive':
                            #  Логика для прогрессивной фантазии (14, 15, 16, 17 карт)
                            if current_player == 0:
                                num_cards_to_deal = self.get_progressive_fantasy_cards(game_state_p0.board)
                            else:
                                num_cards_to_deal = self.get_progressive_fantasy_cards(game_state_p1.board)
                        else:
                            num_cards_to_deal = 14
                    elif (len(current_game_state.board.top) + len(current_game_state.board.middle) + len(current_game_state.board.bottom) == 5) or \
                         (len(current_game_state.board.top) + len(current_game_state.board.middle) + len(current_game_state.board.bottom) < 13):
                        num_cards_to_deal = 3  #  После первых 5 карт и до 13 раздаем по 3
                    else:
                        num_cards_to_deal = 0

                    if num_cards_to_deal > 0:
                        new_cards = all_cards[cards_dealt:cards_dealt + num_cards_to_deal]
                        current_game_state.selected_cards = Hand(new_cards)
                        cards_dealt += num_cards_to_deal
                        #  Обновляем видимые карты для соперника (если не в "Фантазии")
                        if current_player == 0 and not fantasy_p1:
                            visible_cards_p0 = opponent_game_state.board.top + opponent_game_state.board.middle + opponent_game_state.board.bottom + new_cards
                        elif current_player == 1 and not fantasy_p0:
                            visible_cards_p1 = opponent_game_state.board.top + opponent_game_state.board.middle + opponent_game_state.board.bottom + new_cards

                #  Получаем доступные действия
                actions = get_actions(current_game_state)
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

                    current_game_state = current_game_state.apply_action(action_from_array(actions[action_index]))
                    #  Удаляем карты из selected_cards
                    current_game_state.selected_cards = Hand([])


                #  Меняем текущего игрока
                current_player = 1 - current_player
                current_game_state, opponent_game_state = opponent_game_state, current_game_state

                #  После смены игрока обновляем видимые карты
                if current_player == 0:
                    visible_cards_p0 = opponent_game_state.board.top + opponent_game_state.board.middle + opponent_game_state.board.bottom
                    if fantasy_p1:
                        visible_cards_p0 = []
                else:
                    visible_cards_p1 = opponent_game_state.board.top + opponent_game_state.board.middle + opponent_game_state.board.bottom
                    if fantasy_p0:
                        visible_cards_p1 = []


            #  После того, как оба игрока заполнили доски:
            #  1.  Проверяем, попал ли кто-то в "Фантазию"
            if not fantasy_p0 and game_state_p0.is_valid_fantasy_entry():
                fantasy_p0 = True
            if not fantasy_p1 and game_state_p1.is_valid_fantasy_entry():
                fantasy_p1 = True

            #  2.  Проверяем, может ли кто-то остаться в "Фантазии"
            if fantasy_p0 and not game_state_p0.is_valid_fantasy_repeat(game_state_p0.board):
                fantasy_p0 = False  #  Сбрасываем флаг, если не выполнены условия
            if fantasy_p1 and not game_state_p1.is_valid_fantasy_repeat(game_state_p1.board):
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
            strategy = jnp.where(normalizing_sum > 0, strategy / normalizing_sum, jnp.ones(num_actions) / num_actions)
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
    def is_dead_hand_jax(self, placement: jnp.ndarray, ai_settings: Dict) -> bool:
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

        top_rank = self._identify_combination(top_cards)
        middle_rank = self._identify_combination(middle_cards)
        bottom_rank = self._identify_combination(bottom_cards)

        return (top_rank > middle_rank) or (middle_rank > bottom_rank)

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
            "max_nodes": self.max_nodes
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
            logger.info("Прогресс AI успешно загружен с GitHub.")
        else:
            logger.warning("Не удалось загрузить прогресс с GitHub.")

    # ... (RandomAgent - без изменений) ...
    def get_average_strategy_by_index(self, index: int) -> jnp.ndarray:
        """
        Вспомогательная функция для получения средней стратегии по индексу узла.
        """
        strategy_sum = self.strategy_sums[index]
        num_actions = self.num_actions_arr[index]  #  Используем сохраненное количество действий
        normalizing_sum = jnp.sum(strategy_sum)
        return jnp.where(normalizing_sum > 0, strategy_sum / normalizing_sum, jnp.ones(num_actions) / normalizing_sum)

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
        pass

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
