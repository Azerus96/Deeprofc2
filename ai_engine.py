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
from jax import random, jit

# Настройка логирования
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# =============================================================================
# Классы для работы с картами
# =============================================================================

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


# =============================================================================
# Класс GameState и связанные методы
# =============================================================================

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
        self.remaining_cards: List[Card] = self.calculate_remaining_cards()

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
        used_cards.update(self.selected_cards.cards)
        return [card for card in self.deck if card not in used_cards]

    def get_available_cards(self) -> List[Card]:
        """Returns a list of cards that are still available in the deck."""
        available_cards = [card for card in self.deck if card in self.remaining_cards]
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
        new_game_state.remaining_cards = new_game_state.calculate_remaining_cards()
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

        if visible_opponent_cards is not None:
            visible_opponent_str = ",".join(map(card_to_string, sort_cards(visible_opponent_cards)))
        else:
            visible_opponent_str = ""

        return f"T:{top_str}|M:{middle_str}|B:{bottom_str}|D:{discarded_str}|V:{visible_opponent_str}"

    def get_payoff(self, opponent_board: Optional[Board] = None) -> Union[int, Dict[str, int]]:
        """
        Calculates the payoff for the current state.
        If the game is terminal, returns the score difference.
        If the game is not terminal and an opponent_board is provided, returns
        a dictionary with potential payoffs for each possible action.
        """
        if not self.is_terminal():
            if opponent_board is None:
                raise ValueError("Opponent board must be provided for non-terminal states")
            else:
                # TODO: реализовать расчет потенциального выигрыша для каждого действия
                return 0

        if self.is_dead_hand():
            return -1000

        my_royalties = self.calculate_royalties()
        my_total_royalty = sum(my_royalties.values())
        my_line_wins = 0

        if opponent_board is None:
            raise ValueError("Opponent board must be provided for terminal states")

        opponent_royalties = self.calculate_royalties_for_board(opponent_board)
        opponent_total_royalty = sum(opponent_royalties.values())

        if self.evaluate_hand(jnp.array([card_to_array(card) for card in self.board.bottom]))[0] < \
           self.evaluate_hand(jnp.array([card_to_array(card) for card in opponent_board.bottom]))[0]:
            my_line_wins += 1
        elif self.evaluate_hand(jnp.array([card_to_array(card) for card in self.board.bottom]))[0] > \
             self.evaluate_hand(jnp.array([card_to_array(card) for card in opponent_board.bottom]))[0]:
            my_line_wins -= 1

        if self.evaluate_hand(jnp.array([card_to_array(card) for card in self.board.middle]))[0] < \
           self.evaluate_hand(jnp.array([card_to_array(card) for card in opponent_board.middle]))[0]:
            my_line_wins += 1
        elif self.evaluate_hand(jnp.array([card_to_array(card) for card in self.board.middle]))[0] > \
             self.evaluate_hand(jnp.array([card_to_array(card) for card in opponent_board.middle]))[0]:
            my_line_wins -= 1

        if self.evaluate_hand(jnp.array([card_to_array(card) for card in self.board.top]))[0] < \
           self.evaluate_hand(jnp.array([card_to_array(card) for card in opponent_board.top]))[0]:
            my_line_wins += 1
        elif self.evaluate_hand(jnp.array([card_to_array(card) for card in self.board.top]))[0] > \
             self.evaluate_hand(jnp.array([card_to_array(card) for card in opponent_board.top]))[0]:
            my_line_wins -= 1

        if my_line_wins == 3:
            my_line_wins += 3
        elif my_line_wins == -3:
            my_line_wins -= 3

        return (my_total_royalty + my_line_wins) - (opponent_total_royalty - my_line_wins)

    def calculate_royalties_for_board(self, board: Board) -> Dict[str, int]:
        """
        Auxiliary function to calculate royalties for an opponent's board.
        """
        temp_state = GameState(board=board, ai_settings=self.ai_settings)
        return temp_state.calculate_royalties()

    def is_dead_hand(self) -> bool:
        """Checks if the hand is dead (invalid combination order)."""
        if not self.board.is_full():
            return False
        top_rank, _ = self.evaluate_hand(jnp.array([card_to_array(card) for card in self.board.top]))
        middle_rank, _ = self.evaluate_hand(jnp.array([card_to_array(card) for card in self.board.middle]))
        bottom_rank, _ = self.evaluate_hand(jnp.array([card_to_array(card) for card in self.board.bottom]))
        return top_rank > middle_rank or middle_rank > bottom_rank

    def is_valid_fantasy_entry(self, board: Optional[Board] = None) -> bool:
        """Checks if an action leads to a valid fantasy entry."""
        if board is None:
            board = self.board
        temp_state = GameState(board=board, ai_settings=self.ai_settings)
        if temp_state.is_dead_hand():
            return False
        top_rank, _ = temp_state.evaluate_hand(jnp.array([card_to_array(card) for card in board.top]))
        if top_rank == 8:  # pair
            if board.top[0].rank == board.top[1].rank:
                return board.top[0].rank in ["Q", "K", "A"]
        elif top_rank == 7:  # three of a kind
            return True
        return False

    def is_valid_fantasy_repeat(self, board: Optional[Board] = None) -> bool:
        """Checks if an action allows remaining in fantasy mode."""
        if board is None:
            board = self.board
        temp_state = GameState(board=board, ai_settings=self.ai_settings)
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
            if top_rank == 7:
                return True
            if bottom_rank <= 3:
                return True
            return False

    def mark_card_as_used(self, card: Card) -> None:
        """Marks a card as used (placed or discarded)."""
        if card not in self.discarded_cards:
            self.discarded_cards.append(card)

# =============================================================================
# Функции для преобразования карт
# =============================================================================

def card_to_array(card: Optional[Union[Card, str]]) -> jnp.ndarray:
    """
    Преобразует Card в JAX-массив [rank, suit].
    Если передан объект str (например, "2♥"), пытаемся разобрать его.
    """
    if card is None:
        return jnp.array([-1, -1], dtype=jnp.int32)
    if isinstance(card, str):
        for r in Card.RANKS:
            if card.startswith(r):
                suit = card[len(r):]
                if suit in Card.SUITS:
                    card = Card(r, suit)
                    break
        else:
            raise ValueError(f"Cannot parse card string: {card}")
    return jnp.array([Card.RANKS.index(card.rank), Card.SUITS.index(card.suit)], dtype=jnp.int32)

def array_to_card(card_array: jnp.ndarray) -> Optional[Card]:
    """Преобразует JAX-массив [rank, suit] обратно в Card."""
    if jnp.array_equal(card_array, jnp.array([-1, -1])):
        return None
    return Card(Card.RANKS[int(card_array[0])], Card.SUITS[int(card_array[1])])

# =============================================================================
# Функция генерации допустимых размещений карт
# =============================================================================

def generate_placements(cards_jax: jnp.ndarray, board: Board, ai_settings: Dict, max_combinations: int = 10000) -> jnp.ndarray:
    """
    Генерирует все возможные допустимые размещения карт на доске (JAX-версия).
    Возвращает JAX-массивы.
    """
    num_cards = cards_jax.shape[0]
    free_slots_top = 3 - len(board.top)
    free_slots_middle = 5 - len(board.middle)
    free_slots_bottom = 5 - len(board.bottom)
    valid_combinations = []

    if num_cards == 1:
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
    else:
        line_combinations = list(itertools.product([0, 1, 2], repeat=num_cards))
        for comb in line_combinations:
            counts = jnp.bincount(jnp.array(comb), minlength=3)
            if counts[0] <= free_slots_top and counts[1] <= free_slots_middle and counts[2] <= free_slots_bottom:
                valid_combinations.append(list(comb))

    valid_combinations = jnp.array(valid_combinations, dtype=jnp.int32)
    if len(valid_combinations) > max_combinations:
        valid_combinations = valid_combinations[:max_combinations]

    all_placements = []
    for comb in valid_combinations:
        for perm in itertools.permutations(cards_jax.tolist()):
            perm = jnp.array(perm, dtype=jnp.int32)
            placement = jnp.full((14, 2), -1, dtype=jnp.int32)
            top_indices = jnp.where(comb == 0)[0]
            middle_indices = jnp.where(comb == 1)[0]
            bottom_indices = jnp.where(comb == 2)[0]
            placement = placement.at[top_indices].set(perm[:len(top_indices)])
            placement = placement.at[(jnp.array(middle_indices) + 3)].set(perm[len(top_indices):len(top_indices) + len(middle_indices)])
            placement = placement.at[(jnp.array(bottom_indices) + 8)].set(perm[len(top_indices) + len(middle_indices):])
            all_placements.append(placement)
    all_placements = jnp.array(all_placements, dtype=jnp.int32)

    is_dead_hand_vmap = jax.vmap(is_dead_hand_jax)
    dead_hands = is_dead_hand_vmap(all_placements, ai_settings)
    valid_placements = all_placements[~dead_hands]
    return valid_placements

# =============================================================================
# Функция получения возможных действий (actions)
# =============================================================================

def get_actions(game_state: GameState) -> jnp.ndarray:
    logger.debug("get_actions - START")
    if game_state.is_terminal():
        logger.debug("get_actions - Game is terminal, returning empty actions")
        return jnp.array([])

    num_cards = len(game_state.selected_cards)
    actions = []
    if num_cards > 0:
        selected_cards_jax = jnp.array([[Card.RANKS.index(card.rank), Card.SUITS.index(card.suit)] for card in game_state.selected_cards.cards], dtype=jnp.int32)

        if game_state.ai_settings.get("fantasyMode", False):
            can_repeat = False
            if game_state.ai_settings.get("fantasyType") == "progressive":
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
                        break
            else:
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
                        break
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
                    temp_board = Board()
                    temp_board.top = [array_to_card(c) for c in action[:3] if not jnp.array_equal(c, jnp.array([-1, -1]))]
                    temp_board.middle = [array_to_card(c) for c in action[3:8] if not jnp.array_equal(c, jnp.array([-1, -1]))]
                    temp_board.bottom = [array_to_card(c) for c in action[8:13] if not jnp.array_equal(c, jnp.array([-1, -1]))]
                    temp_state = GameState(board=temp_board, ai_settings=game_state.ai_settings)
                    if not temp_state.is_dead_hand():
                        possible_actions.append((action, temp_board))
                if possible_actions:
                    best_action = None
                    best_royalty = -1
                    for action, temp_board in possible_actions:
                        temp_state = GameState(board=temp_board, ai_settings=game_state.ai_settings)
                        royalties = temp_state.calculate_royalties()
                        total_royalty = sum(royalties.values())
                        if total_royalty > best_royalty:
                            best_royalty = total_royalty
                            best_action = action
                    if best_action is not None:
                        actions.append(best_action)
        elif num_cards == 3:
            for discarded_index in range(3):
                indices_to_place = jnp.array([j for j in range(3) if j != discarded_index])
                cards_to_place_jax = selected_cards_jax[indices_to_place]
                discarded_card_jax = selected_cards_jax[discarded_index]
                placements = generate_placements(cards_to_place_jax, game_state.board, game_state.ai_settings)
                for placement in placements:
                    action = placement.at[13].set(discarded_card_jax)
                    actions.append(action)
        else:
            placements = generate_placements(selected_cards_jax, game_state.board, game_state.ai_settings)
            for placement in placements:
                placed_indices = []
                for i in range(13):
                    if not jnp.array_equal(placement[i], jnp.array([-1, -1])):
                        for j in range(selected_cards_jax.shape[0]):
                            if jnp.array_equal(placement[i], selected_cards_jax[j]):
                                placed_indices.append(j)
                                break
                discarded_indices = [i for i in range(selected_cards_jax.shape[0]) if i not in placed_indices]
                discarded_cards_jax = selected_cards_jax[jnp.array(discarded_indices)]
                for i in range(discarded_cards_jax.shape[0] + 1):
                    for discarded_combination in itertools.combinations(discarded_cards_jax.tolist(), i):
                        action = placement.copy()
                        for j, card_array in enumerate(discarded_combination):
                            action = action.at[13 + j].set(jnp.array(card_array, dtype=jnp.int32))
                        actions.append(action)

    logger.debug(f"Generated {len(actions)} actions")
    logger.debug("get_actions - END")
    return jnp.array(actions, dtype=jnp.int32)

# =============================================================================
# Класс CFRNode (если требуется, можно добавить методы get_strategy и get_average_strategy)
# =============================================================================

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
        return jnp.where(normalizing_sum > 0, self.strategy_sum / normalizing_sum, jnp.ones(self.num_actions) / self.num_actions)

# =============================================================================
# Класс CFRAgent
# =============================================================================

class CFRAgent:
    def __init__(self, iterations: int = 500000, stop_threshold: float = 0.001, batch_size: int = 1, max_nodes: int = 100000):
        """
        Инициализация MCCFR агента (с использованием JAX).
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
        self.nodes_map = {}  # {hash(info_set): node_index}

    @jit
    def get_move(self, game_state: GameState, timeout_event: Event, result: Dict) -> None:
        """
        Выбирает ход, используя функцию get_placement.
        """
        logger.debug("Inside CFRAgent get_move")
        move = get_placement(
            game_state.selected_cards.cards,
            game_state.board,
            game_state.discarded_cards,
            game_state.ai_settings,
            self.baseline_evaluation  # Функция оценки
        )
        if move is None:
            result["move"] = {"error": "Нет доступных ходов"}
            logger.debug("No actions available, returning error.")
            return
        result["move"] = move
        logger.debug(f"Final selected move: {move}")

    def train(self, timeout_event: Event, result: Dict) -> None:
        """
        Обучающий цикл MCCFR с пакетным обновлением (использует jax.vmap).
        """
        def play_one_batch(key):
            all_cards = Card.get_all_cards()
            key, subkey = random.split(key)
            all_cards = random.permutation(subkey, jnp.array([ [Card.RANKS.index(card.rank), Card.SUITS.index(card.suit)] for card in all_cards ], dtype=jnp.int32))
            # Преобразуем обратно в объекты Card
            all_cards = [Card(Card.RANKS[int(arr[0])], Card.SUITS[int(arr[1])]) for arr in all_cards.tolist()]
            game_state_p0 = GameState(deck=all_cards, ai_settings=self.ai_settings)
            game_state_p1 = GameState(deck=all_cards, ai_settings=self.ai_settings)
            fantasy_p0 = False
            fantasy_p1 = False
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
            nonlocal dealer
            if dealer == -1:
                key, subkey = random.split(key)
                dealer = int(random.choice(subkey, jnp.array([0, 1])))
            else:
                dealer = 1 - dealer
            current_player = 1 - dealer
            current_game_state = game_state_p0 if current_player == 0 else game_state_p1
            opponent_game_state = game_state_p1 if current_player == 0 else game_state_p0
            first_player = current_player
            game_state_p0.selected_cards = Hand(all_cards[:5])
            game_state_p1.selected_cards = Hand(all_cards[5:10])
            visible_cards_p0 = all_cards[5:10]
            visible_cards_p1 = all_cards[:5]
            cards_dealt = 10
            while not game_state_p0.board.is_full() or not game_state_p1.board.is_full():
                if current_player == 0:
                    visible_opponent_cards = visible_cards_p0 if not fantasy_p1 else []
                else:
                    visible_opponent_cards = visible_cards_p1 if not fantasy_p0 else []
                info_set = current_game_state.get_information_set(visible_opponent_cards)
                if len(current_game_state.selected_cards.cards) == 0:
                    if current_game_state.board.is_full():
                        num_cards_to_deal = 0
                    elif fantasy_p0 and fantasy_p1:
                        if current_game_state.ai_settings['fantasyType'] == 'progressive':
                            num_cards_to_deal = self.get_progressive_fantasy_cards(current_game_state.board)
                        else:
                            num_cards_to_deal = 14
                    elif (len(current_game_state.board.top) + len(current_game_state.board.middle) + len(current_game_state.board.bottom) == 5) or \
                         (len(current_game_state.board.top) + len(current_game_state.board.middle) + len(current_game_state.board.bottom) < 13):
                        num_cards_to_deal = 3
                    else:
                        num_cards_to_deal = 0
                    if num_cards_to_deal > 0:
                        new_cards = all_cards[cards_dealt:cards_dealt + num_cards_to_deal]
                        current_game_state.selected_cards = Hand(new_cards)
                        cards_dealt += num_cards_to_deal
                        if current_player == 0 and not fantasy_p1:
                            visible_cards_p0 = opponent_game_state.board.top + opponent_game_state.board.middle + opponent_game_state.board.bottom + new_cards
                        elif current_player == 1 and not fantasy_p0:
                            visible_cards_p1 = opponent_game_state.board.top + opponent_game_state.board.middle + opponent_game_state.board.bottom + new_cards
                actions = get_actions(current_game_state)
                if actions.shape[0] != 0:
                    self.key, subkey = random.split(self.key)
                    action_index = int(random.choice(subkey, jnp.arange(actions.shape[0])))
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
                    current_game_state.selected_cards = Hand([])
                current_player = 1 - current_player
                current_game_state, opponent_game_state = opponent_game_state, current_game_state
                if current_player == 0:
                    visible_cards_p0 = opponent_game_state.board.top + opponent_game_state.board.middle + opponent_game_state.board.bottom
                    if fantasy_p1:
                        visible_cards_p0 = []
                else:
                    visible_cards_p1 = opponent_game_state.board.top + opponent_game_state.board.middle + opponent_game_state.board.bottom
                    if fantasy_p0:
                        visible_cards_p1 = []
            if not fantasy_p0 and game_state_p0.is_valid_fantasy_entry():
                fantasy_p0 = True
            if not fantasy_p1 and game_state_p1.is_valid_fantasy_entry():
                fantasy_p1 = True
            if fantasy_p0 and not game_state_p0.is_valid_fantasy_repeat(game_state_p0.board):
                fantasy_p0 = False
            if fantasy_p1 and not game_state_p1.is_valid_fantasy_repeat(game_state_p1.board):
                fantasy_p1 = False
            if first_player == 0:
                payoff = float(game_state_p0.get_payoff(opponent_board=game_state_p1.board))
            else:
                payoff = float(game_state_p1.get_payoff(opponent_board=game_state_p0.board))
            trajectory['payoff'] = [payoff] * len(trajectory['info_sets'])
            return trajectory

        play_batch = jax.vmap(play_one_batch)
        dealer = -1
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
            dealer = 1 - dealer

    def update_strategy(self, trajectories):
        """
        Пакетное обновление стратегии на основе накопленных траекторий (JAX-версия).
        """
        combined_trajectory = {
            'info_sets': [item for trajectory in trajectories for item in trajectory['info_sets']],
            'action_indices': jnp.array([item for trajectory in trajectories for item in trajectory['action_indices']]),
            'pi_0': jnp.array([item for trajectory in trajectories for item in trajectory['pi_0']]),
            'pi_1': jnp.array([item for trajectory in trajectories for item in trajectory['pi_1']]),
            'payoff': jnp.array([item for trajectory in trajectories for item in trajectory['payoff']]),
            'player': jnp.array([item for trajectory in trajectories for item in trajectory['player']]),
            'actions': jnp.array([item for trajectory in trajectories for item in trajectory['actions']])
        }
        unique_info_sets = set(combined_trajectory['info_sets'])
        info_set_to_actions_index = {info_set: i for i, info_set in enumerate(combined_trajectory['info_sets'])}
        for info_set in unique_info_sets:
            info_hash = hash(info_set)
            if info_hash not in self.nodes_map:
                index = combined_trajectory['info_sets'].index(info_set)
                actions = combined_trajectory['actions'][index]
                self.nodes_map[info_hash] = self.node_counter
                self.regret_sums = self.regret_sums.at[self.node_counter].set(jnp.zeros(actions.shape[0]))
                self.strategy_sums = self.strategy_sums.at[self.node_counter].set(jnp.zeros(actions.shape[0]))
                self.num_actions_arr = self.num_actions_arr.at[self.node_counter].set(actions.shape[0])
                self.nodes_mask = self.nodes_mask.at[self.node_counter].set(True)
                self.node_counter += 1

        def update_node(info_set, action_index, pi_0, pi_1, payoff, player, actions):
            node_index = self.nodes_map[hash(info_set)]
            num_actions = self.num_actions_arr[node_index]
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

        update_node_vmap = jax.vmap(update_node)
        new_regret_sums, new_strategy_sums = update_node_vmap(
            combined_trajectory['info_sets'],
            combined_trajectory['action_indices'],
            combined_trajectory['pi_0'],
            combined_trajectory['pi_1'],
            combined_trajectory['payoff'],
            combined_trajectory['player'],
            combined_trajectory['actions']
        )
        node_indices = jnp.array([self.nodes_map[hash(info_set)] for info_set in combined_trajectory['info_sets']])
        self.regret_sums = self.regret_sums.at[node_indices].set(new_regret_sums)
        self.strategy_sums = self.strategy_sums.at[node_indices].set(new_strategy_sums)

    @jit
    def check_convergence(self) -> bool:
        """
        Проверяет сходимость обучения (средняя стратегия близка к равномерной).
        """
        valid_indices = jnp.where(self.nodes_mask)[0]
        def check_one_node(index):
            num_actions = self.num_actions_arr[index]
            avg_strategy = self.get_average_strategy_by_index(index)
            uniform_strategy = jnp.ones(num_actions) / num_actions
            diff = jnp.mean(jnp.abs(avg_strategy - uniform_strategy))
            return diff > self.stop_threshold
        not_converged = jax.vmap(check_one_node)(valid_indices)
        return not jnp.any(not_converged)

    def get_average_strategy_by_index(self, index: int) -> jnp.ndarray:
        strategy_sum = self.strategy_sums[index]
        num_actions = self.num_actions_arr[index]
        normalizing_sum = jnp.sum(strategy_sum)
        return jnp.where(normalizing_sum > 0, strategy_sum / normalizing_sum, jnp.ones(num_actions) / num_actions)

    @jit
    def get_progressive_fantasy_cards(self, board: Board) -> int:
        top_cards_jax = jnp.array([card_to_array(card) for card in board.top])
        top_rank, _ = self.evaluate_hand(top_cards_jax)
        if top_rank == 8:  # Pair
            rank = top_cards_jax[0, 0]
            return jnp.where(rank == 12, 16,
                   jnp.where(rank == 11, 15,
                   jnp.where(rank == 10, 14,
                   14)))
        elif top_rank == 7:  # Three of a kind
            return 17
        return 14

    # Вспомогательные функции (JAX-версии) для комбинаций

    @jit
    def _get_rank_counts(self, cards_jax: jnp.ndarray) -> jnp.ndarray:
        ranks = cards_jax[:, 0]
        return jnp.bincount(ranks, minlength=13)

    @jit
    def _get_suit_counts(self, cards_jax: jnp.ndarray) -> jnp.ndarray:
        suits = cards_jax[:, 1]
        return jnp.bincount(suits, minlength=4)

    @jit
    def _is_flush(self, cards_jax: jnp.ndarray) -> bool:
        suits = cards_jax[:, 1]
        return jnp.all(suits == suits[0])

    @jit
    def _is_straight(self, cards_jax: jnp.ndarray) -> bool:
        ranks = jnp.sort(cards_jax[:, 0])
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
        if cards_jax.shape[0] == 0:
            return 10
        if cards_jax.shape[0] == 3:
            if self._is_three_of_a_kind(cards_jax):
                return 6
            if self._is_one_pair(cards_jax):
                return 8
            return 9
        if cards_jax.shape[0] == 5:
            if self._is_royal_flush(cards_jax):
                return 0
            elif self._is_straight_flush(cards_jax):
                return 1
            elif self._is_four_of_a_kind(cards_jax):
                return 2
            elif self._is_full_house(cards_jax):
                return 3
            elif self._is_flush(cards_jax):
                return 4
            elif self._is_straight(cards_jax):
                return 5
            elif self._is_three_of_a_kind(cards_jax):
                return 6
            elif self._is_two_pair(cards_jax):
                return 7
            elif self._is_one_pair(cards_jax):
                return 8
            else:
                return 9
        return 10

    @jit
    def is_dead_hand_jax(self, placement: jnp.ndarray, ai_settings: Dict) -> bool:
        top_cards = placement[:3]
        middle_cards = placement[3:8]
        bottom_cards = placement[8:13]
        top_cards = top_cards[jnp.any(top_cards != -1, axis=1)]
        middle_cards = middle_cards[jnp.any(middle_cards != -1, axis=1)]
        bottom_cards = bottom_cards[jnp.any(bottom_cards != -1, axis=1)]
        if len(top_cards) == 0 or len(middle_cards) == 0 or len(bottom_cards) == 0:
            return False
        top_rank = self._identify_combination(top_cards)
        middle_rank = self._identify_combination(middle_cards)
        bottom_rank = self._identify_combination(bottom_cards)
        return (top_rank > middle_rank) or (middle_rank > bottom_rank)

    @jit
    def evaluate_hand(self, cards_jax: jnp.ndarray) -> Tuple[int, float]:
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
                    return 1, 25.0
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
            if pairs.shape[0] == 2:
                high_pair_index = jnp.max(pairs)
                low_pair_index = jnp.min(pairs)
                return 8, 1.0 + high_pair_index / 100.0 + low_pair_index / 10000.0
            if pairs.shape[0] == 1:
                pair_rank_index = pairs[0]
                return 9, pair_rank_index / 100.0
            return 10, jnp.max(rank_indices) / 100.0
        return 11, 0.0

    @jit
    def calculate_royalties(self) -> Dict[str, int]:
        """
        Расчет роялти по американским правилам (JAX-версия).
        Возвращает словарь для линий: {"top": ..., "middle": ..., "bottom": ...}
        """
        top_cards_jax = jnp.array([card_to_array(card) for card in self.board.top])
        middle_cards_jax = jnp.array([card_to_array(card) for card in self.board.middle])
        bottom_cards_jax = jnp.array([card_to_array(card) for card in self.board.bottom])
        if (top_cards_jax.shape[0] < 3 or middle_cards_jax.shape[0] < 5 or bottom_cards_jax.shape[0] < 5 or self.is_dead_hand()):
            return {"top": 0, "middle": 0, "bottom": 0}
        top_rank, _ = self.evaluate_hand(top_cards_jax)
        middle_rank, _ = self.evaluate_hand(middle_cards_jax)
        bottom_rank, _ = self.evaluate_hand(bottom_cards_jax)
        top_rank_index = None
        if top_rank == 7:
            top_rank_index = top_cards_jax[0, 0]
        elif top_rank == 8:
            top_rank_index = jnp.where(jnp.bincount(top_cards_jax[:, 0], minlength=13) == 2)[0][0]
        def get_royalty(line: int, rank: int, rank_index: Optional[int] = None) -> int:
            top_royalties = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            top_royalties = top_royalties.at[7].set(jnp.where(rank_index is not None, 10 + rank_index, 0))
            top_royalties = top_royalties.at[8].set(jnp.where((rank_index is not None) & (rank_index >= 4), rank_index - 3, 0))
            middle_royalties = jnp.array([0, 50, 30, 20, 12, 8, 4, 2, 0, 0, 0])
            bottom_royalties = jnp.array([0, 25, 15, 10, 6, 4, 2, 0, 0, 0, 0])
            return int(jnp.where(line == 0, top_royalties[rank],
                           jnp.where(line == 1, middle_royalties[rank], bottom_royalties[rank])))
        royalties = {
            "top": get_royalty(0, top_rank, top_rank_index),
            "middle": get_royalty(1, middle_rank),
            "bottom": get_royalty(2, bottom_rank)
        }
        return royalties

    @jit
    def get_line_royalties(self, cards_jax: jnp.ndarray, line: str) -> int:
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
        return int(jnp.where(hand_rank == 1, 25,
               jnp.where(hand_rank == 2, 15,
               jnp.where(hand_rank == 3, 10,
               jnp.where(hand_rank == 4, 6,
               jnp.where(hand_rank == 5, 4,
               jnp.where(hand_rank == 6, 2, 0)))))))

    @jit
    def get_pair_bonus(self, cards_jax: jnp.ndarray) -> int:
        if cards_jax.shape[0] != 3:
            return 0
        ranks = cards_jax[:, 0]
        pair_rank_index = jnp.where(jnp.bincount(ranks, minlength=13) == 2)[0]
        return int(jnp.where(pair_rank_index.size > 0, jnp.maximum(0, pair_rank_index[0] - 4), 0))

    @jit
    def get_high_card_bonus(self, cards_jax: jnp.ndarray) -> int:
        if cards_jax.shape[0] != 3:
            return 0
        ranks = cards_jax[:, 0]
        if jnp.unique(ranks).shape[0] == 3:
            high_card_index = jnp.max(ranks)
            return int(jnp.where(high_card_index == 12, 1, 0))
        return 0

    @jit
    def is_dead_hand_jax(self, placement: jnp.ndarray, ai_settings: Dict) -> bool:
        top_cards = placement[:3]
        middle_cards = placement[3:8]
        bottom_cards = placement[8:13]
        top_cards = top_cards[jnp.any(top_cards != -1, axis=1)]
        middle_cards = middle_cards[jnp.any(middle_cards != -1, axis=1)]
        bottom_cards = bottom_cards[jnp.any(bottom_cards != -1, axis=1)]
        if len(top_cards) == 0 or len(middle_cards) == 0 or len(bottom_cards) == 0:
            return False
        top_rank = self._identify_combination(top_cards)
        middle_rank = self._identify_combination(middle_cards)
        bottom_rank = self._identify_combination(bottom_cards)
        return (top_rank > middle_rank) or (middle_rank > bottom_rank)

    @jit
    def evaluate_move(self, game_state: GameState, action: Dict[str, List[Card]]) -> float:
        # TODO: Реализовать оценку хода
        return 0.0

# =============================================================================
# Класс RandomAgent
# =============================================================================

class RandomAgent:
    def __init__(self):
        pass

    def get_move(self, game_state: GameState, timeout_event: Event, result: Dict) -> None:
        logger.debug("Inside RandomAgent get_move")
        move = get_placement(
            game_state.selected_cards.cards,
            game_state.board,
            game_state.discarded_cards,
            game_state.ai_settings,
            self.baseline_evaluation  # Передаем функцию оценки
        )
        if move is None:
            result["move"] = {"error": "Нет доступных ходов"}
            logger.debug("No actions available in RandomAgent.")
            return
        result["move"] = move
        logger.debug(f"RandomAgent selected move: {move}")

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

# =============================================================================
# Конец файла ai_engine.py
# =============================================================================

def action_from_array(action_array: jnp.ndarray) -> Dict[str, List[Card]]:
    """
    Преобразует JAX-массив действия обратно в словарь.
    """
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
    for i in range(13, min(17, action_array.shape[0])):
        card = array_to_card(action_array[i])
        if card:
            action_dict["discarded"].append(card)
    return action_dict

def get_placement(selected_cards: List[Card], board: Board, discarded_cards: List[Card],
                  ai_settings: Dict, evaluation_fn) -> Optional[Dict[str, List[Card]]]:
    """
    Функция выбора оптимального размещения (stub-реализация).
    Здесь можно интегрировать логику MCCFR.
    Если ход не найден, возвращает None.
    """
    actions = get_actions(GameState(selected_cards=selected_cards, board=board,
                                    discarded_cards=discarded_cards, ai_settings=ai_settings))
    if actions.shape[0] == 0:
        return None
    # Выбираем случайное действие как пример
    idx = int(random.choice(random.PRNGKey(0), jnp.arange(actions.shape[0])))
    chosen_action = actions[idx]
    return action_from_array(chosen_action)
