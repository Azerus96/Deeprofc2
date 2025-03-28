# ai_engine.py

# --- Стандартные импорты ---
import itertools
from collections import defaultdict, Counter
from threading import Event, Thread
import time
import math
import logging
from typing import List, Dict, Tuple, Optional, Union
import concurrent.futures  # Для параллелизма
import os  # Для определения количества CPU

# --- Импорты библиотек машинного обучения и вычислений ---
import jax.numpy as jnp
import jax
from jax import random
from jax import jit
import numpy as np  # Используем numpy для некоторых операций, не требующих JIT

# --- Импорт утилит для GitHub (с обработкой отсутствия) ---
try:
    from github_utils import save_ai_progress_to_github, load_ai_progress_from_github
except ImportError:
    logging.warning("github_utils not found. Saving/Loading progress to GitHub will be disabled.")
    # Определяем заглушки, если модуль не найден
    def save_ai_progress_to_github(*args, **kwargs):
        logging.error("Saving to GitHub is disabled (github_utils not found).")
        return False
    def load_ai_progress_from_github(*args, **kwargs):
        logging.error("Loading from GitHub is disabled (github_utils not found).")
        return None

# --- Настройка логирования ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Класс Card ---
class Card:
    """Представляет игральную карту с рангом и мастью."""
    RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
    SUITS = ["♥", "♦", "♣", "♠"]
    RANK_MAP = {rank: i for i, rank in enumerate(RANKS)}
    SUIT_MAP = {suit: i for i, suit in enumerate(SUITS)}

    def __init__(self, rank: str, suit: str):
        if rank not in self.RANKS:
            raise ValueError(f"Invalid rank: {rank}")
        if suit not in self.SUITS:
            raise ValueError(f"Invalid suit: {suit}")
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
        return [Card(r, s) for r in Card.RANKS for s in Card.SUITS]


# --- Класс Hand ---
class Hand:
    """Представляет руку игрока."""
    def __init__(self, cards: Optional[List[Card]] = None):
        if isinstance(cards, jnp.ndarray):
            cards_list = [array_to_card(c) for c in cards if not jnp.array_equal(c, jnp.array([-1, -1]))]
            self.cards = [c for c in cards_list if c is not None]
        else:
            self.cards = cards if cards is not None else []

    def add_card(self, card: Card) -> None:
        if not isinstance(card, Card):
            raise TypeError("card must be an instance of Card")
        self.cards.append(card)

    def add_cards(self, cards_to_add: List[Card]) -> None:
        for card in cards_to_add:
            self.add_card(card)

    def remove_card(self, card: Card) -> None:
        if not isinstance(card, Card):
            raise TypeError("card must be an instance of Card")
        try:
            self.cards.remove(card)
        except ValueError:
            logger.warning(f"Card {card} not found in hand to remove: {self.cards}")

    def remove_cards(self, cards_to_remove: List[Card]) -> None:
        temp_hand = self.cards[:]
        removed_count = 0
        for card_to_remove in cards_to_remove:
            try:
                temp_hand.remove(card_to_remove)
                removed_count += 1
            except ValueError:
                logger.warning(f"Card {card_to_remove} not found in hand during multi-remove.")
        if removed_count != len(cards_to_remove):
            logger.warning(f"Expected to remove {len(cards_to_remove)}, removed {removed_count}.")
        self.cards = temp_hand

    def __repr__(self) -> str:
        st = lambda cards: ", ".join(map(str, sorted(cards, key=lambda c: (Card.RANK_MAP.get(c.rank, -1), Card.SUIT_MAP.get(c.suit, -1)))))
        return f"Hand: [{st(self.cards)}]"

    def __len__(self) -> int:
        return len(self.cards)

    def __iter__(self):
        return iter(self.cards)

    def __getitem__(self, index: int) -> Card:
        return self.cards[index]

    def to_jax(self) -> jnp.ndarray:
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
        return len(self.top) + len(self.middle) + len(self.bottom)

    def place_card(self, line: str, card: Card) -> None:
        target_line = getattr(self, line, None)
        if target_line is None:
            raise ValueError(f"Invalid line: {line}")
        max_len = 3 if line == "top" else 5
        if len(target_line) >= max_len:
            raise ValueError(f"{line.capitalize()} line is full ({len(target_line)}/{max_len})")
        target_line.append(card)

    def is_full(self) -> bool:
        return len(self.top) == 3 and len(self.middle) == 5 and len(self.bottom) == 5

    def clear(self) -> None:
        self.top = []
        self.middle = []
        self.bottom = []

    def __repr__(self) -> str:
        st = lambda cards: ", ".join(map(str, sorted(cards, key=lambda c: (Card.RANK_MAP.get(c.rank, -1), Card.SUIT_MAP.get(c.suit, -1)))))
        return f"Top: [{st(self.top)}]\nMiddle: [{st(self.middle)}]\nBottom: [{st(self.bottom)}]"

    def get_cards(self, line: str) -> List[Card]:
        if line == "top":
            return self.top
        elif line == "middle":
            return self.middle
        elif line == "bottom":
            return self.bottom
        else:
            # ИСПРАВЛЕНО: Перенос на новую строку
            raise ValueError("Invalid line specified")

    def get_all_cards(self) -> List[Card]:
        return self.top + self.middle + self.bottom

    def get_line_jax(self, line: str) -> jnp.ndarray:
        cards = self.get_cards(line)
        if not cards:
            return jnp.empty((0, 2), dtype=jnp.int32)
        return jnp.array([card_to_array(card) for card in cards], dtype=jnp.int32)

    def to_jax_placement(self) -> jnp.ndarray:
        placement = jnp.full((13, 2), -1, dtype=jnp.int32)
        idx = 0
        for card in self.top:
            if idx < 3:
                placement = placement.at[idx].set(card_to_array(card))
                idx += 1
        idx = 3
        for card in self.middle:
            if idx < 8:
                placement = placement.at[idx].set(card_to_array(card))
                idx += 1
        idx = 8
        for card in self.bottom:
            if idx < 13:
                placement = placement.at[idx].set(card_to_array(card))
                idx += 1
        return placement


# --- Вспомогательные функции для преобразования Card <-> JAX array ---
def card_to_array(card: Optional[Card]) -> jnp.ndarray:
    if card is None:
        return jnp.array([-1, -1], dtype=jnp.int32)
    return jnp.array([Card.RANK_MAP.get(card.rank, -1), Card.SUIT_MAP.get(card.suit, -1)], dtype=jnp.int32)

def array_to_card(card_array: jnp.ndarray) -> Optional[Card]:
    if card_array is None or card_array.shape != (2,) or jnp.array_equal(card_array, jnp.array([-1, -1])):
        return None
    try:
        rank_idx = int(card_array[0])
        suit_idx = int(card_array[1])
        if 0 <= rank_idx < len(Card.RANKS) and 0 <= suit_idx < len(Card.SUITS):
            # ИСПРАВЛЕНО: Перенос на новую строку
            return Card(Card.RANKS[rank_idx], Card.SUITS[suit_idx])
        else:
            # ИСПРАВЛЕНО: Перенос на новую строку
            return None
    except (IndexError, ValueError):
        return None

def action_to_jax(action_dict: Dict[str, List[Card]]) -> jnp.ndarray:
    action_array = jnp.full((17, 2), -1, dtype=jnp.int32)
    idx = 0
    for card in action_dict.get("top", []):
        if idx < 3:
            action_array = action_array.at[idx].set(card_to_array(card))
            idx += 1
    idx = 3
    for card in action_dict.get("middle", []):
        if idx < 8:
            action_array = action_array.at[idx].set(card_to_array(card))
            idx += 1
    idx = 8
    for card in action_dict.get("bottom", []):
        if idx < 13:
            action_array = action_array.at[idx].set(card_to_array(card))
            idx += 1
    idx = 13
    for card in action_dict.get("discarded", []):
        if idx < 17:
            action_array = action_array.at[idx].set(card_to_array(card))
            idx += 1
    return action_array

def action_from_array(action_array: jnp.ndarray) -> Dict[str, List[Card]]:
    if action_array is None or action_array.shape != (17, 2):
        logger.error(f"Invalid shape for action_array: {action_array.shape if action_array is not None else 'None'}")
        return {}
    action_dict = {"top": [], "middle": [], "bottom": [], "discarded": []}
    for i in range(3):
        card = array_to_card(action_array[i])
        if card: action_dict["top"].append(card)
    for i in range(3, 8):
        card = array_to_card(action_array[i])
        if card: action_dict["middle"].append(card)
    for i in range(8, 13):
        card = array_to_card(action_array[i])
        if card: action_dict["bottom"].append(card)
    for i in range(13, 17):
        card = array_to_card(action_array[i])
        if card: action_dict["discarded"].append(card)
    return {k: v for k, v in action_dict.items() if v}


# --- Класс GameState ---
class GameState:
    """Представляет полное состояние игры для одного игрока в определенный момент."""
    def __init__(
        self, selected_cards: Optional[Union[List[Card], Hand]] = None, board: Optional[Board] = None,
        discarded_cards: Optional[List[Card]] = None, ai_settings: Optional[Dict] = None,
        deck: Optional[List[Card]] = None, current_player: int = 0,
        opponent_board: Optional[Board] = None, opponent_discarded: Optional[List[Card]] = None,
    ):
        if isinstance(selected_cards, Hand): self.selected_cards: Hand = selected_cards
        else: self.selected_cards: Hand = Hand(selected_cards)
        self.board: Board = board if board is not None else Board()
        self.discarded_cards: List[Card] = discarded_cards if discarded_cards is not None else []
        self.ai_settings: Dict = ai_settings if ai_settings is not None else {}
        self.current_player: int = current_player; self.deck: List[Card] = deck if deck is not None else []
        self.opponent_board: Board = opponent_board if opponent_board is not None else Board()
        self.opponent_discarded: List[Card] = opponent_discarded if opponent_discarded is not None else []

    def get_current_player(self) -> int: return self.current_player
    def is_terminal(self) -> bool: return self.board.is_full()

    def get_street(self) -> int:
        placed = self.board.get_placed_count()
        if placed == 0: return 1
        if placed == 5: return 2
        if placed == 7: return 3
        if placed == 9: return 4
        if placed == 11: return 5
        if placed == 13: return 6
        if placed < 5: return 1
        logger.warning(f"Unexpected placed cards ({placed}) for street calc.")
        return 0

    def apply_action(self, action: Dict[str, List[Card]]) -> "GameState":
        new_board = Board(); new_board.top=self.board.top[:]; new_board.middle=self.board.middle[:]; new_board.bottom=self.board.bottom[:]
        new_discarded = self.discarded_cards[:]; placed_in_action = []; discarded_in_action = action.get("discarded", [])
        for line in ["top", "middle", "bottom"]:
            cards_to_place = action.get(line, []); placed_in_action.extend(cards_to_place)
            for card in cards_to_place: new_board.place_card(line, card)
        new_discarded.extend(discarded_in_action); played_cards = placed_in_action + discarded_in_action
        new_hand = Hand(self.selected_cards.cards[:]); new_hand.remove_cards(played_cards)
        new_state = GameState(selected_cards=new_hand, board=new_board, discarded_cards=new_discarded,
                              ai_settings=self.ai_settings.copy(), deck=self.deck, current_player=self.current_player,
                              opponent_board=self.opponent_board, opponent_discarded=self.opponent_discarded)
        return new_state

    def get_information_set(self) -> str:
        st = lambda cards: ",".join(map(str, sorted(cards, key=lambda c: (Card.RANK_MAP.get(c.rank, -1), Card.SUIT_MAP.get(c.suit, -1)))))
        street = f"St:{self.get_street()}"; my_board = f"T:{st(self.board.top)}|M:{st(self.board.middle)}|B:{st(self.board.bottom)}"
        my_disc = f"D:{st(self.discarded_cards)}"; opp_board = f"OT:{st(self.opponent_board.top)}|OM:{st(self.opponent_board.middle)}|OB:{st(self.opponent_board.bottom)}"
        opp_disc = f"OD:{st(self.opponent_discarded)}"; return f"{street}|{my_board}|{my_disc}|{opp_board}|{opp_disc}"

    def _calculate_pairwise_score(self, opponent_board: Board) -> int:
        line_score = 0; cmp_b = _compare_hands_py(self.board.get_line_jax("bottom"), opponent_board.get_line_jax("bottom"))
        cmp_m = _compare_hands_py(self.board.get_line_jax("middle"), opponent_board.get_line_jax("middle"))
        cmp_t = _compare_hands_py(self.board.get_line_jax("top"), opponent_board.get_line_jax("top"))
        line_score += cmp_b + cmp_m + cmp_t
        scoop = 3 if cmp_b == 1 and cmp_m == 1 and cmp_t == 1 else (-3 if cmp_b == -1 and cmp_m == -1 and cmp_t == -1 else 0)
        return line_score + scoop

    def get_payoff(self) -> int:
        if not self.is_terminal() or not self.opponent_board.is_full(): logger.warning("get_payoff on non-terminal state(s)."); return 0
        my_place = self.board.to_jax_placement(); opp_place = self.opponent_board.to_jax_placement()
        i_am_dead = is_dead_hand_jax(my_place, self.ai_settings); opp_is_dead = is_dead_hand_jax(opp_place, self.ai_settings)
        my_royalty = 0; opp_royalty = 0
        if not i_am_dead: my_royalty = int(jnp.sum(calculate_royalties_jax(self.board, self.ai_settings)))
        if not opp_is_dead: opp_royalty = int(jnp.sum(calculate_royalties_jax(self.opponent_board, self.ai_settings)))
        if i_am_dead and opp_is_dead: return 0
        if i_am_dead: return -6 - opp_royalty
        if opp_is_dead: return 6 + my_royalty
        pairwise_score = self._calculate_pairwise_score(self.opponent_board); return pairwise_score + my_royalty - opp_royalty

    def is_valid_fantasy_entry(self) -> bool:
        if not self.board.is_full(): return False; place = self.board.to_jax_placement()
        if is_dead_hand_jax(place, self.ai_settings): return False; return is_valid_fantasy_entry_jax(place, self.ai_settings)

    def is_valid_fantasy_repeat(self) -> bool:
        if not self.board.is_full(): return False; place = self.board.to_jax_placement()
        if is_dead_hand_jax(place, self.ai_settings): return False; return is_valid_fantasy_repeat_jax(place, self.ai_settings)

    def get_fantasy_cards_count(self) -> int:
        place = self.board.to_jax_placement(); top_cards = place[0:3][jnp.any(place[0:3] != -1, axis=1)]
        if top_cards.shape[0] != 3: return 0; top_rank, _ = evaluate_hand_jax(top_cards)
        if self.ai_settings.get('fantasyType') == 'progressive':
            if top_rank == 6: return 17
            if top_rank == 8:
                pair_idx = jnp.where(jnp.bincount(top_cards[:, 0], length=13) == 2)[0][0]
                if pair_idx == Card.RANK_MAP['A']: return 16
                if pair_idx == Card.RANK_MAP['K']: return 15
                if pair_idx == Card.RANK_MAP['Q']: return 14
        elif top_rank <= 8:
             pair_idx = jnp.where(jnp.bincount(top_cards[:, 0], length=13) == 2)[0][0] if top_rank == 8 else -1
             if top_rank == 6 or pair_idx >= Card.RANK_MAP['Q']: return 14
        return 0


# --- Вспомогательные JAX функции (оценка рук, роялти, фантазия) ---
@jit
def _get_rank_counts_jax(cards_jax: jnp.ndarray) -> jnp.ndarray:
    if cards_jax.shape[0] == 0: return jnp.zeros(13, dtype=jnp.int32)
    ranks = cards_jax[:, 0]; ranks = jnp.clip(ranks, 0, 12); return jnp.bincount(ranks, length=13)
@jit
def _get_suit_counts_jax(cards_jax: jnp.ndarray) -> jnp.ndarray:
    if cards_jax.shape[0] == 0: return jnp.zeros(4, dtype=jnp.int32)
    suits = cards_jax[:, 1]; suits = jnp.clip(suits, 0, 3); return jnp.bincount(suits, length=4)
@jit
def _is_flush_jax(cards_jax: jnp.ndarray) -> bool:
    if cards_jax.shape[0] != 5: return False; suits = cards_jax[:, 1]; return jnp.all(suits == suits[0])
@jit
def _is_straight_jax(cards_jax: jnp.ndarray) -> bool:
    if cards_jax.shape[0] != 5: return False; ranks = jnp.sort(cards_jax[:, 0])
    if jnp.unique(ranks).shape[0] != 5: return False; is_a5 = jnp.array_equal(ranks, jnp.array([0, 1, 2, 3, 12]))
    is_normal = (ranks[4] - ranks[0]) == 4; return jnp.logical_or(is_a5, is_normal)
@jit
def _is_straight_flush_jax(cards_jax: jnp.ndarray) -> bool:
    if cards_jax.shape[0] != 5: return False; return _is_flush_jax(cards_jax) and _is_straight_jax(cards_jax)
@jit
def _is_royal_flush_jax(cards_jax: jnp.ndarray) -> bool:
    if cards_jax.shape[0] != 5: return False
    if not _is_straight_flush_jax(cards_jax): return False
    ranks = jnp.sort(cards_jax[:, 0]); return jnp.array_equal(ranks, jnp.array([8, 9, 10, 11, 12]))
@jit
def _is_four_of_a_kind_jax(cards_jax: jnp.ndarray) -> bool:
    if cards_jax.shape[0] != 5: return False; return jnp.any(_get_rank_counts_jax(cards_jax) == 4)
@jit
def _is_full_house_jax(cards_jax: jnp.ndarray) -> bool:
    if cards_jax.shape[0] != 5: return False; counts = _get_rank_counts_jax(cards_jax); return jnp.any(counts == 3) and jnp.any(counts == 2)
@jit
def _is_three_of_a_kind_jax(cards_jax: jnp.ndarray) -> bool:
    n = cards_jax.shape[0];
    if n < 3: return False
    counts = _get_rank_counts_jax(cards_jax); has_three = jnp.sum(counts == 3) == 1
    if n == 5:
        has_pair = jnp.sum(counts == 2) == 1; return has_three and not has_pair
    elif n == 3:
        # ИСПРАВЛЕНО: else на новой строке
        return has_three
    else:
        # Для n=4 тройка тоже возможна (3 + 1 кикер)
        if n == 4:
             has_no_pair = jnp.sum(counts == 2) == 0
             return has_three and has_no_pair
        # ИСПРАВЛЕНО: else на новой строке
        return False
@jit
def _is_two_pair_jax(cards_jax: jnp.ndarray) -> bool:
    if cards_jax.shape[0] != 5: return False; return jnp.sum(_get_rank_counts_jax(cards_jax) == 2) == 2
@jit
def _is_one_pair_jax(cards_jax: jnp.ndarray) -> bool:
    n = cards_jax.shape[0];
    if n < 2: return False
    counts = _get_rank_counts_jax(cards_jax); has_one_pair = jnp.sum(counts == 2) == 1; has_no_better = jnp.sum(counts >= 3) == 0
    if n == 5:
        return has_one_pair and has_no_better
    elif n == 3:
        # ИСПРАВЛЕНО: Перенос на новую строку
        return has_one_pair
    elif n == 2:
        # ИСПРАВЛЕНО: Перенос на новую строку
        return has_one_pair
    elif n == 4:
        # ИСПРАВЛЕНО: Перенос на новую строку
        return has_one_pair and has_no_better
    else:
        # ИСПРАВЛЕНО: Перенос на новую строку
        return False
@jit
def _identify_combination_jax(cards_jax: jnp.ndarray) -> int:
    n = cards_jax.shape[0]
    if n == 5:
        if _is_royal_flush_jax(cards_jax): return 0
        if _is_straight_flush_jax(cards_jax): return 1
        if _is_four_of_a_kind_jax(cards_jax): return 2
        if _is_full_house_jax(cards_jax): return 3
        if _is_flush_jax(cards_jax): return 4
        if _is_straight_jax(cards_jax): return 5
        if _is_three_of_a_kind_jax(cards_jax): return 6
        if _is_two_pair_jax(cards_jax): return 7
        if _is_one_pair_jax(cards_jax): return 8
        return 9
    elif n == 3:
        if _is_three_of_a_kind_jax(cards_jax): return 6
        if _is_one_pair_jax(cards_jax): return 8
        return 9
    elif n == 0: return 10
    else: # n=1,2,4
        if n >= 2 and _is_one_pair_jax(cards_jax): return 8
        if n >= 1: return 9
        return 10
@jit
def evaluate_hand_jax(cards_jax: jnp.ndarray) -> Tuple[int, jnp.ndarray]:
    n = cards_jax.shape[0]; default_kickers = jnp.array([-1]*5, dtype=jnp.int32)
    if n == 0: return 10, default_kickers; combination_rank = _identify_combination_jax(cards_jax)
    if combination_rank == 10: return 10, default_kickers
    ranks = cards_jax[:, 0]; rank_counts = _get_rank_counts_jax(cards_jax); sorted_ranks_desc = jnp.sort(ranks)[::-1]
    kickers = jnp.full(5, -1, dtype=jnp.int32); num_to_fill = min(n, 5); kickers = kickers.at[:num_to_fill].set(sorted_ranks_desc[:num_to_fill])
    if combination_rank == 2: four_rank = jnp.where(rank_counts == 4)[0][0]; kicker = jnp.where(rank_counts == 1)[0][0]; kickers = jnp.array([four_rank, kicker, -1, -1, -1], dtype=jnp.int32)
    elif combination_rank == 3: three_rank = jnp.where(rank_counts == 3)[0][0]; pair_rank = jnp.where(rank_counts == 2)[0][0]; kickers = jnp.array([three_rank, pair_rank, -1, -1, -1], dtype=jnp.int32)
    elif combination_rank == 6: three_rank = jnp.where(rank_counts == 3)[0][0]; other_kickers = jnp.sort(ranks[ranks != three_rank])[::-1]; kickers = kickers.at[0].set(three_rank);
    if n == 5: kickers = kickers.at[1:3].set(other_kickers[:2]); kickers = kickers.at[3:].set(-1)
    elif n == 3: kickers = kickers.at[1:].set(-1)
    elif combination_rank == 7: pair_ranks = jnp.sort(jnp.where(rank_counts == 2)[0])[::-1]; kicker = jnp.where(rank_counts == 1)[0][0]; kickers = jnp.array([pair_ranks[0], pair_ranks[1], kicker, -1, -1], dtype=jnp.int32)
    elif combination_rank == 8: pair_rank = jnp.where(rank_counts == 2)[0][0]; other_kickers = jnp.sort(ranks[ranks != pair_rank])[::-1]; kickers = kickers.at[0].set(pair_rank);
    if n == 5: kickers = kickers.at[1:4].set(other_kickers[:3]); kickers = kickers.at[4].set(-1)
    elif n == 3: kickers = kickers.at[1].set(other_kickers[0]); kickers = kickers.at[2:].set(-1)
    elif combination_rank == 5 or combination_rank == 1: is_a5 = jnp.array_equal(jnp.sort(ranks), jnp.array([0, 1, 2, 3, 12])); main_kicker = jnp.where(is_a5, 3, sorted_ranks_desc[0]); kickers = jnp.array([main_kicker, -1, -1, -1, -1], dtype=jnp.int32)
    return combination_rank, kickers
@jit
def is_dead_hand_jax(placement: jnp.ndarray, ai_settings: Dict) -> bool:
    top_cards = placement[0:3]; middle_cards = placement[3:8]; bottom_cards = placement[8:13]
    if jnp.any(placement == -1): return False # Не полная доска не мертвая
    cmp_top_mid = _compare_hands_py(top_cards, middle_cards); cmp_mid_bot = _compare_hands_py(middle_cards, bottom_cards)
    return (cmp_top_mid > 0) or (cmp_mid_bot > 0)
@jit
def calculate_royalties_jax(board: Board, ai_settings: Dict) -> jnp.ndarray:
    if not board.is_full(): return jnp.array([0, 0, 0], dtype=jnp.int32)
    placement_jax = board.to_jax_placement()
    if is_dead_hand_jax(placement_jax, ai_settings): return jnp.array([0, 0, 0], dtype=jnp.int32)
    top_cards = placement_jax[0:3]; middle_cards = placement_jax[3:8]; bottom_cards = placement_jax[8:13]
    top_royalty = 0; top_rank, _ = evaluate_hand_jax(top_cards)
    if top_rank == 6: set_rank_idx = top_cards[0, 0]; top_royalty = 10 + set_rank_idx
    elif top_rank == 8: pair_rank_idx = jnp.where(jnp.bincount(top_cards[:, 0], length=13) == 2)[0][0]; top_royalty = jnp.maximum(0, pair_rank_idx - Card.RANK_MAP['5'])
    middle_rank, _ = evaluate_hand_jax(middle_cards); middle_royalties_map = jnp.array([50, 30, 20, 12, 8, 4, 2, 0, 0, 0, 0], dtype=jnp.int32); middle_royalty = middle_royalties_map[middle_rank]
    bottom_rank, _ = evaluate_hand_jax(bottom_cards); bottom_royalties_map = jnp.array([25, 15, 10, 6, 4, 2, 0, 0, 0, 0, 0], dtype=jnp.int32); bottom_royalty = bottom_royalties_map[bottom_rank]
    return jnp.array([top_royalty, middle_royalty, bottom_royalty], dtype=jnp.int32)
@jit
def is_valid_fantasy_entry_jax(placement: jnp.ndarray, ai_settings: Dict) -> bool:
    top_cards = placement[0:3]; top_rank, _ = evaluate_hand_jax(top_cards)
    if top_rank == 8: pair_rank_idx = jnp.where(jnp.bincount(top_cards[:, 0], length=13) == 2)[0][0]; return pair_rank_idx >= Card.RANK_MAP['Q']
    elif top_rank == 6: return True
    # ИСПРАВЛЕНО: Перенос на новую строку
    return False
@jit
def is_valid_fantasy_repeat_jax(placement: jnp.ndarray, ai_settings: Dict) -> bool:
    top_cards = placement[0:3]; bottom_cards = placement[8:13]
    top_rank, _ = evaluate_hand_jax(top_cards); bottom_rank, _ = evaluate_hand_jax(bottom_cards)
    return (top_rank == 6) or (bottom_rank <= 2)

# --- Функции генерации действий ---
def _generate_placements_recursive(cards_to_place: List[Card], current_board_jax: jnp.ndarray, ai_settings: Dict, card_idx: int, valid_placements: List[jnp.ndarray], max_placements: Optional[int] = 1000):
    if max_placements is not None and len(valid_placements) >= max_placements: return True
    if card_idx == len(cards_to_place):
        placed_count = jnp.sum(jnp.any(current_board_jax != -1, axis=1)); is_potentially_full = (placed_count == 13)
        if is_potentially_full:
            if not is_dead_hand_jax(current_board_jax, ai_settings): valid_placements.append(current_board_jax.copy())
        else: valid_placements.append(current_board_jax.copy())
        return False
    card = cards_to_place[card_idx]; card_arr = card_to_array(card); limit_reached = False
    top_indices = jnp.arange(3); top_occupied = jnp.any(current_board_jax[top_indices] != -1, axis=1); first_free_top = jnp.where(~top_occupied, top_indices, 3)[0]
    if first_free_top < 3:
        next_placement = current_board_jax.at[first_free_top].set(card_arr)
        limit_reached = _generate_placements_recursive(cards_to_place, next_placement, ai_settings, card_idx + 1, valid_placements, max_placements);
        if limit_reached: return True
    mid_indices = jnp.arange(3, 8); mid_occupied = jnp.any(current_board_jax[mid_indices] != -1, axis=1); first_free_mid = jnp.where(~mid_occupied, mid_indices, 8)[0]
    if first_free_mid < 8:
        next_placement = current_board_jax.at[first_free_mid].set(card_arr)
        limit_reached = _generate_placements_recursive(cards_to_place, next_placement, ai_settings, card_idx + 1, valid_placements, max_placements);
        if limit_reached: return True
    bot_indices = jnp.arange(8, 13); bot_occupied = jnp.any(current_board_jax[bot_indices] != -1, axis=1); first_free_bot = jnp.where(~bot_occupied, bot_indices, 13)[0]
    if first_free_bot < 13:
        next_placement = current_board_jax.at[first_free_bot].set(card_arr)
        limit_reached = _generate_placements_recursive(cards_to_place, next_placement, ai_settings, card_idx + 1, valid_placements, max_placements);
        if limit_reached: return True
    return False

def get_actions(game_state: GameState) -> jnp.ndarray:
    logger.debug(f"get_actions - START | Player: {game_state.current_player} | Street: {game_state.get_street()}")
    if game_state.is_terminal(): logger.debug("get_actions - Board is full"); return jnp.empty((0, 17, 2), dtype=jnp.int32)
    hand_cards = game_state.selected_cards.cards; num_cards_in_hand = len(hand_cards)
    if num_cards_in_hand == 0: logger.debug("get_actions - No cards in hand"); return jnp.empty((0, 17, 2), dtype=jnp.int32)
    possible_actions_list = []; street = game_state.get_street(); is_fantasy_turn = game_state.ai_settings.get("in_fantasy_turn", False)
    num_to_place, num_to_discard = 0, 0; placement_limit = 500
    if is_fantasy_turn:
        num_to_place = 13; num_to_discard = num_cards_in_hand - num_to_place
        if num_to_discard < 0: logger.error(f"Fantasy error: Hand={num_cards_in_hand} < 13"); num_to_place=num_cards_in_hand; num_to_discard=0
        num_to_discard = min(num_to_discard, 4); num_to_place = num_cards_in_hand - num_to_discard
        logger.debug(f"Fantasy Action: Hand={num_cards_in_hand}, Place={num_to_place}, Discard={num_to_discard}")
        placement_limit = game_state.ai_settings.get("fantasy_placement_limit", 2000)
    else:
        if street == 1:
            if num_cards_in_hand == 5: num_to_place, num_to_discard = 5, 0
            else: logger.error(f"Street 1 error: Hand={num_cards_in_hand} != 5"); return jnp.empty((0, 17, 2), dtype=jnp.int32)
            placement_limit = game_state.ai_settings.get("street1_placement_limit", 10000) # Увеличенный лимит
            logger.debug(f"Using increased placement limit for Street 1: {placement_limit}")
        elif 2 <= street <= 5:
            if num_cards_in_hand == 3: num_to_place, num_to_discard = 2, 1
            else: logger.error(f"Street {street} error: Hand={num_cards_in_hand} != 3"); return jnp.empty((0, 17, 2), dtype=jnp.int32)
            placement_limit = game_state.ai_settings.get("normal_placement_limit", 500)
        else: logger.error(f"get_actions on invalid street {street}"); return jnp.empty((0, 17, 2), dtype=jnp.int32)
        logger.debug(f"Street {street} Action: Hand={num_cards_in_hand}, Place={num_to_place}, Discard={num_to_discard}")

    action_count_this_hand = 0; max_actions_per_hand = placement_limit * 10
    for cards_to_place_tuple in itertools.combinations(hand_cards, num_to_place):
        if action_count_this_hand >= max_actions_per_hand: logger.warning(f"Max actions per hand ({max_actions_per_hand}) reached."); break
        cards_to_place = list(cards_to_place_tuple); cards_to_discard = [card for card in hand_cards if card not in cards_to_place]
        if len(cards_to_discard) != num_to_discard: continue
        initial_placement_jax = game_state.board.to_jax_placement(); valid_placements_for_combo = []
        limit_was_reached = _generate_placements_recursive(cards_to_place, initial_placement_jax, game_state.ai_settings, 0, valid_placements_for_combo, max_placements=placement_limit)
        if limit_was_reached: logger.warning(f"Placement limit ({placement_limit}) reached for combo: {cards_to_place}")
        discard_jax = jnp.full((4, 2), -1, dtype=jnp.int32)
        for i, card in enumerate(cards_to_discard):
            if i < 4: discard_jax = discard_jax.at[i].set(card_to_array(card))
        for placement_13 in valid_placements_for_combo:
            action_17 = jnp.concatenate((placement_13, discard_jax), axis=0); possible_actions_list.append(action_17); action_count_this_hand += 1
            if action_count_this_hand >= max_actions_per_hand: break
    logger.debug(f"Generated {len(possible_actions_list)} raw actions")
    if not possible_actions_list: logger.warning(f"No valid actions generated for Player {game_state.current_player}!"); return jnp.empty((0, 17, 2), dtype=jnp.int32)
    else:
        # ИСПРАВЛЕНО: Перенос на новую строку
        if not all(a.shape == (17, 2) for a in possible_actions_list):
             logger.error("Inconsistent action shapes generated!"); correct_shape_actions = [a for a in possible_actions_list if a.shape == (17, 2)]
             if not correct_shape_actions: return jnp.empty((0, 17, 2), dtype=jnp.int32)
             return jnp.stack(correct_shape_actions)
        else:
             # ИСПРАВЛЕНО: Перенос на новую строку
             return jnp.stack(possible_actions_list)

# --- Вспомогательные функции для эвристической оценки (Python) ---
def _evaluate_partial_combination_py(cards: List[Card], row_type: str) -> float:
    if not cards: return 0.0; score = 0.0; n = len(cards); ranks = [card.rank for card in cards]; suits = [card.suit for card in cards]
    rank_indices = sorted([r for r in [Card.RANK_MAP.get(rank, -1) for rank in ranks] if r != -1])
    if not rank_indices: return 0.0
    if row_type in ["middle", "bottom"] and n >= 2 and n < 5:
        suit_counts = Counter(suits); max_suit_count = max(suit_counts.values()) if suit_counts else 0
        if max_suit_count >= 3: score += 5.0 * (max_suit_count - 2)
        unique_ranks = sorted(list(set(rank_indices))); un = len(unique_ranks)
        if un >= 2:
            is_connector = all(unique_ranks[i+1] - unique_ranks[i] == 1 for i in range(un - 1)); gaps = sum(unique_ranks[i+1] - unique_ranks[i] - 1 for i in range(un - 1)); span = unique_ranks[-1] - unique_ranks[0] if un > 0 else 0
            if is_connector and span == un - 1: score += 4.0 * un
            elif gaps == 1 and span <= 4: score += 2.0 * un
            if set(unique_ranks).issuperset({0, 1, 2}): score += 3.0;
            if set(unique_ranks).issuperset({0, 1, 2, 3}): score += 5.0;
            if set(unique_ranks).issuperset({9, 10, 11, 12}): score += 4.0;
    rank_counts = Counter(ranks)
    for rank, count in rank_counts.items():
        rank_value = Card.RANK_MAP.get(rank, -1);
        if rank_value == -1: continue
        if count == 2: score += 5.0 + rank_value * 0.5
        elif count == 3: score += 15.0 + rank_value * 1.0
    score += sum(r for r in rank_indices) * 0.1; return score
def _compare_hands_py(hand1_jax: jnp.ndarray, hand2_jax: jnp.ndarray) -> int:
    n1 = hand1_jax.shape[0]; n2 = hand2_jax.shape[0]
    if n1 == 0 and n2 == 0: return 0;
    if n1 == 0: return -1;
    if n2 == 0: return 1;
    rank1, kickers1 = evaluate_hand_jax(hand1_jax); rank2, kickers2 = evaluate_hand_jax(hand2_jax)
    if rank1 < rank2: return 1;
    if rank1 > rank2: return -1;
    kickers1_list = kickers1.tolist(); kickers2_list = kickers2.tolist()
    for k1, k2 in zip(kickers1_list, kickers2_list):
        if k1 > k2: return 1;
        if k1 < k2: return -1;
    return 0
def _is_bottom_stronger_or_equal_py(board: Board) -> bool:
    bottom_jax = board.get_line_jax("bottom"); middle_jax = board.get_line_jax("middle")
    return _compare_hands_py(bottom_jax, middle_jax) >= 0
def _is_middle_stronger_or_equal_py(board: Board) -> bool:
    middle_jax = board.get_line_jax("middle"); top_jax = board.get_line_jax("top")
    return _compare_hands_py(middle_jax, top_jax) >= 0
def heuristic_baseline_evaluation(state: GameState, ai_settings: Dict) -> float:
    is_full = state.board.is_full()
    if is_full:
        placement_jax = state.board.to_jax_placement()
        if is_dead_hand_jax(placement_jax, ai_settings): return -1000.0
    COMBINATION_WEIGHTS = jnp.array([100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 5.0, 0.0], dtype=jnp.float32)
    ROW_MULTIPLIERS = {"top": 1.0, "middle": 1.2, "bottom": 1.5}; total_score = 0.0
    rows_data = {"top": state.board.top, "middle": state.board.middle, "bottom": state.board.bottom}
    max_cards_in_row = {"top": 3, "middle": 5, "bottom": 5}
    for row_name, cards in rows_data.items():
        row_score = 0.0; cards_jax = state.board.get_line_jax(row_name); num_cards_in_row = cards_jax.shape[0]
        if num_cards_in_row > 0:
            rank, kickers = evaluate_hand_jax(cards_jax)
            if rank < len(COMBINATION_WEIGHTS): row_score += COMBINATION_WEIGHTS[rank] + float(jnp.sum(kickers[kickers != -1])) * 0.01
            if num_cards_in_row < max_cards_in_row[row_name]: row_score += _evaluate_partial_combination_py(cards, row_name)
        row_score *= ROW_MULTIPLIERS[row_name]; total_score += row_score
    if state.board.bottom and state.board.middle and _is_bottom_stronger_or_equal_py(state.board): total_score += 15.0
    if state.board.middle and state.board.top and _is_middle_stronger_or_equal_py(state.board): total_score += 10.0
    discard_penalty = 0.0
    for card in state.discarded_cards: rank_value = Card.RANK_MAP.get(card.rank, -1); discard_penalty += (rank_value + 1) * 0.1 if rank_value != -1 else 0
    total_score -= discard_penalty
    current_royalties = calculate_royalties_jax(state.board, ai_settings); total_score += float(jnp.sum(current_royalties)) * 0.5
    return total_score

# --- Класс CFRNode ---
class CFRNode:
    """Узел в дереве CFR, хранящий сожаления и сумму стратегий."""
    def __init__(self, num_actions: int):
        self.regret_sum = jnp.zeros(num_actions, dtype=jnp.float32)
        self.strategy_sum = jnp.zeros(num_actions, dtype=jnp.float32)
        self.num_actions = num_actions
    def get_strategy(self, realization_weight: float) -> jnp.ndarray:
        if self.num_actions == 0: return jnp.array([], dtype=jnp.float32)
        current_regret_sum = self.regret_sum[:self.num_actions]; positive_regret_sum = jnp.maximum(current_regret_sum, 0)
        normalizing_sum = jnp.sum(positive_regret_sum); uniform_strategy = jnp.ones(self.num_actions, dtype=jnp.float32) / self.num_actions
        strategy = jnp.where(normalizing_sum > 0, positive_regret_sum / normalizing_sum, uniform_strategy)
        if realization_weight > 1e-9: self.strategy_sum = self.strategy_sum.at[:self.num_actions].add(realization_weight * strategy)
        return strategy
    def get_average_strategy(self) -> jnp.ndarray:
        if self.num_actions == 0: return jnp.array([], dtype=jnp.float32)
        current_strategy_sum = self.strategy_sum[:self.num_actions]; normalizing_sum = jnp.sum(current_strategy_sum)
        uniform_strategy = jnp.ones(self.num_actions, dtype=jnp.float32) / self.num_actions
        avg_strategy = jnp.where(normalizing_sum > 0, current_strategy_sum / normalizing_sum, uniform_strategy)
        return avg_strategy

# --- Класс CFRAgent ---
class CFRAgent:
    """Агент, использующий Counterfactual Regret Minimization (MCCFR вариант) с параллелизацией на потоках."""
    def __init__(self, iterations: int = 1000000, stop_threshold: float = 0.001, batch_size: int = 64, max_nodes: int = 1000000, ai_settings: Optional[Dict] = None, num_workers: Optional[int] = None):
        self.iterations = iterations
        self.stop_threshold = stop_threshold
        self.save_interval = 2000
        self.convergence_check_interval = 50000
        self.key = random.PRNGKey(int(time.time()))
        self.batch_size = batch_size
        self.max_nodes = max_nodes
        self.nodes_map: Dict[int, CFRNode] = {}
        self.ai_settings = ai_settings if ai_settings is not None else {}
        available_cpus = os.cpu_count() or 1
        self.num_workers = min(num_workers, available_cpus) if num_workers is not None else available_cpus
        logger.info(f"CFRAgent initialized. Iterations={iterations}, Max Nodes={max_nodes}, Stop Threshold={stop_threshold}")
        logger.info(f"Batch Size: {self.batch_size}, Num Workers: {self.num_workers} (Available CPUs: {available_cpus})")
        logger.info(f"AI Settings: {self.ai_settings}")
        logger.info(f"Save Interval: {self.save_interval}, Convergence Check Interval: {self.convergence_check_interval}")

    def get_node(self, info_set: str, num_actions: int) -> Optional[CFRNode]:
        """Получает или создает узел CFR. Возвращает None при достижении лимита или ошибке."""
        info_hash = hash(info_set); node = self.nodes_map.get(info_hash)
        if node is None:
            if len(self.nodes_map) >= self.max_nodes: logger.warning(f"Max nodes ({self.max_nodes}) reached."); return None
            node = CFRNode(num_actions); self.nodes_map[info_hash] = node
        elif node.num_actions != num_actions:
             logger.error(f"CRITICAL: Action count mismatch for hash {info_hash}. Node={node.num_actions}, Req={num_actions}. InfoSet: {info_set[:100]}..."); return None
        return node

    def get_move(self, game_state: GameState, timeout_event: Event, result: Dict) -> None:
        """ Выбирает ход на основе средней стратегии CFR или baseline оценки. """
        logger.debug("Inside CFRAgent get_move"); actions_jax = get_actions(game_state); num_available_actions = actions_jax.shape[0]
        if num_available_actions == 0: result["move"] = {"error": "Нет доступных ходов"}; logger.warning("No actions available in get_move."); return
        info_set = game_state.get_information_set(); node = self.get_node(info_set, num_available_actions); best_action_index = -1
        if node is not None:
            avg_strategy = node.get_average_strategy()
            if avg_strategy.shape[0] == num_available_actions: best_action_index = int(jnp.argmax(avg_strategy)); logger.debug(f"Move using CFR strategy. Node hash {hash(info_set)}. Best idx: {best_action_index}")
            else: logger.error(f"Strategy length mismatch after get_node! Using baseline."); best_action_index = self._get_best_action_baseline(game_state, actions_jax)
        else: logger.debug(f"Node not found/error for hash {hash(info_set)}. Using baseline."); best_action_index = self._get_best_action_baseline(game_state, actions_jax)
        if best_action_index == -1:
             if num_available_actions > 0: logger.warning("Baseline failed or node error, choosing random."); self.key, subkey = random.split(self.key); best_action_index = int(random.choice(subkey, jnp.arange(num_available_actions)))
             else: result["move"] = {"error": "Нет ходов после всех проверок"}; logger.error("Critical error: No actions available."); return
        best_action_array = actions_jax[best_action_index]; move = action_from_array(best_action_array); result["move"] = move; logger.debug(f"Final selected move (idx {best_action_index}): {move}")

    def _get_best_action_baseline(self, game_state: GameState, actions_jax: jnp.ndarray) -> int:
        """ Выбирает лучший ход, используя heuristic_baseline_evaluation. """
        best_score = -float('inf'); best_action_index = -1; key = self.key
        for i, action_array in enumerate(actions_jax):
            action_dict = action_from_array(action_array)
            try:
                next_state = game_state.apply_action(action_dict)
                score = heuristic_baseline_evaluation(next_state, self.ai_settings)
                key, subkey = random.split(key); noise = random.uniform(subkey, minval=-0.01, maxval=0.01); score += noise
                if score > best_score: best_score = score; best_action_index = i
            except Exception as e: logger.exception(f"Error evaluating baseline for action {i}. Action: {action_dict}"); continue
        self.key = key
        if best_action_index == -1 and actions_jax.shape[0] > 0:
             logger.warning("All baseline evaluations failed, choosing random."); key, subkey = random.split(self.key); best_action_index = int(random.choice(subkey, jnp.arange(actions_jax.shape[0]))); self.key = key
        return best_action_index

    def baseline_evaluation(self, state: GameState) -> float:
        """ Обертка для вызова эвристической оценки состояния. """
        return heuristic_baseline_evaluation(state, self.ai_settings)

    # --- Основной метод обучения CFR с параллелизацией на ПОТОКАХ ---
    def train(self, timeout_event: Event, result: Dict) -> None:
        """ Запускает процесс обучения MCCFR на заданное количество итераций или до сходимости, используя параллельные потоки. """
        logger.info(f"Starting CFR training for up to {self.iterations} iterations or until convergence (threshold: {self.stop_threshold})...")
        logger.info(f"Using {self.num_workers} worker threads with batch size {self.batch_size}.") # Уточнено про потоки
        start_time = time.time()
        total_games_processed = 0

        num_batches = self.iterations // self.batch_size
        if self.iterations % self.batch_size != 0: num_batches += 1

        for i in range(num_batches):
            batch_start_time = time.time()
            current_batch_size = min(self.batch_size, self.iterations - total_games_processed)
            if current_batch_size <= 0: break
            if timeout_event.is_set(): logger.info(f"Training interrupted by timeout before batch {i+1}."); break

            keys_batch = random.split(self.key, current_batch_size + 1); self.key = keys_batch[0]; keys_for_workers = keys_batch[1:]
            deck_list = [Card.get_all_cards() for _ in range(current_batch_size)]
            trajectories_batch = []
            logger.debug(f"Starting batch {i+1}/{num_batches} with {current_batch_size} games...")

            # Используем ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                future_to_idx = {executor.submit(self._play_one_game_for_cfr, keys_for_workers[k], deck_list[k]): k for k in range(current_batch_size)}
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx_in_batch = future_to_idx[future]
                    try:
                        trajectory = future.result()
                        if trajectory: trajectories_batch.append(trajectory)
                        else: logger.warning(f"Game simulation {idx_in_batch} in batch {i+1} failed or returned None.")
                    except Exception as exc: logger.error(f'Game simulation {idx_in_batch} in batch {i+1} generated an exception: {exc}', exc_info=True)

            logger.debug(f"Batch {i+1} simulations finished. Processing {len(trajectories_batch)} successful trajectories...")
            batch_updates = 0
            for trajectory in trajectories_batch:
                try:
                    # Обновление стратегии происходит последовательно в основном потоке
                    self._update_strategy_from_trajectory(trajectory)
                    total_games_processed += 1; batch_updates += 1
                except Exception as e: logger.exception(f"Error updating strategy from trajectory in batch {i+1}: {e}")

            batch_time = time.time() - batch_start_time
            logger.info(f"Batch {i+1}/{num_batches} finished in {batch_time:.2f}s ({batch_updates}/{current_batch_size} successful). Total games: {total_games_processed}/{self.iterations}")

            # --- Логирование, Сохранение, Проверка сходимости ---
            processed_before_batch = total_games_processed - batch_updates
            last_save_milestone = processed_before_batch // self.save_interval
            current_save_milestone = total_games_processed // self.save_interval
            if current_save_milestone > last_save_milestone:
                 self.save_progress(total_games_processed)
                 logger.info(f"Progress saved at ~{total_games_processed} games processed.")
            last_conv_check_milestone = processed_before_batch // self.convergence_check_interval
            current_conv_check_milestone = total_games_processed // self.convergence_check_interval
            if current_conv_check_milestone > last_conv_check_milestone:
                if self.check_convergence():
                    logger.info(f"Convergence threshold ({self.stop_threshold}) reached after ~{total_games_processed} games processed.")
                    break

        total_time = time.time() - start_time
        logger.info(f"Training finished after {total_games_processed} successful games processed in {total_time:.2f} seconds.")
        logger.info(f"Total nodes created: {len(self.nodes_map)}")
        self.save_progress(total_games_processed); logger.info("Final progress saved.")

    # --- Симуляция игры ---
    def _play_one_game_for_cfr(self, key: jax.random.PRNGKey, deck: List[Card]) -> Optional[Dict]:
        """ Разыгрывает одну партию OFC Pineapple для сбора траектории для MCCFR. """
        try:
            key, subkey = random.split(key); shuffled_indices = random.permutation(subkey, jnp.arange(52)); shuffled_deck = [deck[i] for i in shuffled_indices]; deck_iter = iter(shuffled_deck)
            game_states = {0: GameState(ai_settings=self.ai_settings, current_player=0, deck=deck), 1: GameState(ai_settings=self.ai_settings, current_player=1, deck=deck)}
            game_states[0].opponent_board = game_states[1].board; game_states[0].opponent_discarded = game_states[1].discarded_cards
            game_states[1].opponent_board = game_states[0].board; game_states[1].opponent_discarded = game_states[0].discarded_cards
            player_fantasies = {0: False, 1: False}; fantasy_cards_count = {0: 0, 1: 0}; player_finished = {0: False, 1: False}
            trajectory = {'states': [], 'actions': [], 'reach_probs': [], 'sampling_probs': [], 'final_payoff': 0}
            reach_p0 = 1.0; reach_p1 = 1.0; current_player = 0; turn_count = 0; max_turns = 60
            while not (player_finished[0] and player_finished[1]):
                turn_count += 1;
                if turn_count > max_turns: logger.error(f"Max turns ({max_turns}) reached."); return None
                if player_finished[current_player]: current_player = 1 - current_player; continue
                state = game_states[current_player]; opponent = 1 - current_player; is_fantasy_turn = player_fantasies[current_player] and fantasy_cards_count[current_player] > 0
                if len(state.selected_cards) == 0:
                    num_to_draw = 0
                    if is_fantasy_turn: num_to_draw = fantasy_cards_count[current_player]; fantasy_cards_count[current_player] = 0; logger.info(f"P{current_player} Fantasy turn: {num_to_draw} cards.")
                    else:
                        street = state.get_street()
                        if street == 1: num_to_draw = 5
                        elif 2 <= street <= 5: num_to_draw = 3
                        else:
                             if not state.is_terminal(): logger.error(f"Cannot draw on street {street} (board not full)."); return None
                             else: logger.info(f"P{current_player} board full."); player_finished[current_player] = True; continue
                    try: drawn_cards = [next(deck_iter) for _ in range(num_to_draw)]; state.selected_cards = Hand(drawn_cards); logger.debug(f"P{current_player} draws {num_to_draw}. Hand: {state.selected_cards}")
                    except StopIteration: logger.error("Deck empty."); player_finished[current_player] = True; continue
                info_set = state.get_information_set(); state.ai_settings["in_fantasy_turn"] = is_fantasy_turn; actions_jax = get_actions(state); num_actions = actions_jax.shape[0]; state.ai_settings["in_fantasy_turn"] = False
                if num_actions == 0:
                    if state.is_terminal(): logger.info(f"P{current_player} no actions, board full."); player_finished[current_player] = True
                    else: logger.error(f"No actions for P{current_player} non-terminal! Info: {info_set[:100]}..."); return None
                if player_finished[current_player]:
                     is_now_in_fantasy = False
                     if not is_fantasy_turn:
                          if state.is_valid_fantasy_entry():
                               f_count = state.get_fantasy_cards_count()
                               if f_count > 0: player_fantasies[current_player] = True; fantasy_cards_count[current_player] = f_count; player_finished[current_player] = False; is_now_in_fantasy = True; logger.info(f"P{current_player} triggers Fantasy ({f_count} cards)!")
                     elif state.is_valid_fantasy_repeat(): fantasy_cards_count[current_player] = 14; player_finished[current_player] = False; is_now_in_fantasy = True; logger.info(f"P{current_player} repeats Fantasy (14 cards)!")
                     if not is_now_in_fantasy: player_fantasies[current_player] = False; fantasy_cards_count[current_player] = 0
                     current_player = opponent; continue
                node = self.get_node(info_set, num_actions); action_index = -1; sampling_prob = 1.0 / num_actions
                if node is not None:
                    current_reach = reach_p0 if current_player == 0 else reach_p1; strategy = node.get_strategy(current_reach)
                    if strategy.shape[0] == num_actions: key, subkey = random.split(key); action_index = int(np.random.choice(np.arange(num_actions), p=np.array(strategy))); sampling_prob = strategy[action_index]
                    else: logger.error(f"Strategy shape mismatch! Using baseline."); action_index = self._get_best_action_baseline(state, actions_jax)
                else: logger.warning("Node error/limit. Using baseline."); action_index = self._get_best_action_baseline(state, actions_jax)
                if action_index == -1: logger.error("Failed to select action. Aborting."); return None
                chosen_action_jax = actions_jax[action_index]; action_dict = action_from_array(chosen_action_jax)
                trajectory['states'].append((hash(info_set), current_player, num_actions)); trajectory['actions'].append(action_index); trajectory['reach_probs'].append((reach_p0, reach_p1)); trajectory['sampling_probs'].append(max(sampling_prob, 1e-9))
                if current_player == 0: reach_p0 *= sampling_prob
                else: reach_p1 *= sampling_prob
                game_states[current_player] = state.apply_action(action_dict)
                game_states[current_player].opponent_board = game_states[opponent].board; game_states[current_player].opponent_discarded = game_states[opponent].discarded_cards
                game_states[opponent].opponent_board = game_states[current_player].board; game_states[opponent].opponent_discarded = game_states[current_player].discarded_cards
                logger.debug(f"P{current_player} applied action {action_index}. Board:\n{game_states[current_player].board}")
                if is_fantasy_turn:
                     logger.info(f"P{current_player} finished Fantasy turn."); player_finished[current_player] = True
                     if game_states[current_player].is_valid_fantasy_repeat(): fantasy_cards_count[current_player] = 14; player_finished[current_player] = False; logger.info(f"P{current_player} repeats Fantasy!")
                     else: player_fantasies[current_player] = False; fantasy_cards_count[current_player] = 0
                elif game_states[current_player].is_terminal():
                     logger.info(f"P{current_player} finished board."); player_finished[current_player] = True
                     if game_states[current_player].is_valid_fantasy_entry():
                          f_count = game_states[current_player].get_fantasy_cards_count()
                          if f_count > 0: player_fantasies[current_player] = True; fantasy_cards_count[current_player] = f_count; player_finished[current_player] = False; logger.info(f"P{current_player} triggers Fantasy ({f_count} cards)!")
                current_player = opponent
            logger.info("Game sim finished. Calculating payoff."); game_states[0].opponent_board = game_states[1].board; game_states[0].opponent_discarded = game_states[1].discarded_cards; game_states[1].opponent_board = game_states[0].board; game_states[1].opponent_discarded = game_states[0].discarded_cards
            final_payoff_p0 = game_states[0].get_payoff(); trajectory['final_payoff'] = final_payoff_p0
            logger.debug(f"Final Payoff P0: {final_payoff_p0}"); logger.debug(f"P0 Board:\n{game_states[0].board}"); logger.debug(f"P1 Board:\n{game_states[1].board}")
            return trajectory
        except Exception as e: logger.exception(f"Error during game simulation: {e}"); return None

    # --- Обновление стратегии ---
    def _update_strategy_from_trajectory(self, trajectory: Dict):
        """ Обновляет сожаления узлов CFR по методу Outcome Sampling MCCFR. """
        final_payoff_p0 = trajectory['final_payoff']; num_steps = len(trajectory['states'])
        for t in range(num_steps):
            info_hash, player, num_actions = trajectory['states'][t]; action_taken_index = trajectory['actions'][t]; reach_p0, reach_p1 = trajectory['reach_probs'][t]; sampling_prob = trajectory['sampling_probs'][t]
            node = self.nodes_map.get(info_hash)
            if node is None: # Пытаемся создать узел, если его нет
                 info_set_placeholder = f"hash_{info_hash}"; node = self.get_node(info_set_placeholder, num_actions)
                 if node is None: logger.warning(f"Node {info_hash} not found/creatable during update. Skipping."); continue
            if node.num_actions != num_actions or sampling_prob < 1e-9:
                 if node.num_actions != num_actions: logger.warning(f"Skipping update for node {info_hash} due to action mismatch.")
                 continue
            payoff_for_player = final_payoff_p0 if player == 0 else -final_payoff_p0; reach_opponent = reach_p1 if player == 0 else reach_p0
            current_regret_sum = node.regret_sum[:num_actions]; positive_regret_sum = jnp.maximum(current_regret_sum, 0); normalizing_sum = jnp.sum(positive_regret_sum); uniform_strategy = jnp.ones(num_actions, dtype=jnp.float32) / num_actions; current_strategy = jnp.where(normalizing_sum > 0, positive_regret_sum / normalizing_sum, uniform_strategy)
            update_multiplier = reach_opponent * (payoff_for_player / sampling_prob); indicator = jnp.zeros(num_actions, dtype=jnp.float32).at[action_taken_index].set(1.0); regret_update = update_multiplier * (indicator - current_strategy)
            node.regret_sum = node.regret_sum.at[:num_actions].add(regret_update)

    # --- Сохранение и загрузка ---
    def save_progress(self, iterations_completed: int) -> None:
        """Сохраняет прогресс CFR (карту узлов и счетчик итераций) через GitHub."""
        try:
            serializable_nodes = {h: {"regret_sum": n.regret_sum.tolist(), "strategy_sum": n.strategy_sum.tolist(), "num_actions": n.num_actions} for h, n in self.nodes_map.items()}
            data_to_save = {"nodes_map_serialized": serializable_nodes, "iterations_completed": iterations_completed, "ai_settings": self.ai_settings, "timestamp": time.time()}
            logger.info(f"Preparing to save {len(serializable_nodes)} nodes. Iterations completed: {iterations_completed}")
            if not save_ai_progress_to_github(data_to_save): logger.error("Error saving progress to GitHub!")
            else: logger.info("Data for saving passed to github_utils.")
        except Exception as e: logger.exception(f"Unexpected error during save_progress: {e}")
    def load_progress(self) -> None:
        """Загружает прогресс CFR (карту узлов) через GitHub."""
        logger.info("Attempting to load AI progress from GitHub...")
        try:
            loaded_data = load_ai_progress_from_github()
            if loaded_data and "nodes_map_serialized" in loaded_data:
                self.nodes_map.clear(); loaded_nodes_map_serialized = loaded_data["nodes_map_serialized"]; loaded_ai_settings = loaded_data.get("ai_settings", {}); iterations_completed = loaded_data.get("iterations_completed", 0); timestamp = loaded_data.get("timestamp", 0)
                logger.info(f"Loaded data: {len(loaded_nodes_map_serialized)} nodes, {iterations_completed} iterations completed.");
                if timestamp > 0: logger.info(f"Data timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))}")
                if loaded_ai_settings != self.ai_settings: logger.warning(f"Loaded AI settings differ! Loaded: {loaded_ai_settings}, Current: {self.ai_settings}")
                num_loaded = 0; num_errors = 0
                for info_hash_str, node_data in loaded_nodes_map_serialized.items():
                    try:
                        info_hash = int(info_hash_str); num_actions = node_data["num_actions"]
                        if len(self.nodes_map) >= self.max_nodes: logger.warning(f"Max nodes ({self.max_nodes}) reached during loading."); break
                        node = CFRNode(num_actions); regret_sum_list = node_data["regret_sum"]; strategy_sum_list = node_data["strategy_sum"]
                        if isinstance(regret_sum_list, list) and isinstance(strategy_sum_list, list) and len(regret_sum_list) >= num_actions and len(strategy_sum_list) >= num_actions:
                             node.regret_sum = node.regret_sum.at[:num_actions].set(jnp.array(regret_sum_list[:num_actions], dtype=jnp.float32))
                             node.strategy_sum = node.strategy_sum.at[:num_actions].set(jnp.array(strategy_sum_list[:num_actions], dtype=jnp.float32))
                             self.nodes_map[info_hash] = node; num_loaded += 1
                        else: logger.warning(f"Data type/length mismatch for node hash {info_hash}. Skipping."); num_errors += 1
                    except Exception as e: logger.exception(f"Error processing loaded node data for hash {info_hash_str}: {e}. Skipping."); num_errors += 1
                logger.info(f"Successfully loaded {num_loaded} nodes.");
                if num_errors > 0: logger.warning(f"Skipped {num_errors} nodes due to errors.")
            else: logger.warning("Failed to load progress from GitHub or data is invalid/empty.")
        except Exception as e: logger.exception(f"Unexpected error during load_progress: {e}")

    # --- Проверка сходимости ---
    def check_convergence(self) -> bool:
        """ Проверяет, сошлось ли обучение (упрощенная проверка). """
        if not self.nodes_map: logger.info("Convergence check: No nodes found."); return False
        total_abs_regret = 0.0; total_actions = 0; num_nodes_checked = 0
        for node in self.nodes_map.values():
            if node.num_actions > 0: current_regrets = node.regret_sum[:node.num_actions]; total_abs_regret += float(jnp.sum(jnp.abs(current_regrets))); total_actions += node.num_actions; num_nodes_checked += 1
        if total_actions == 0: logger.info("Convergence check: No actions found in existing nodes."); return False
        avg_abs_regret = total_abs_regret / total_actions; logger.info(f"Convergence check: Avg absolute regret = {avg_abs_regret:.6f} (threshold: {self.stop_threshold})")
        return avg_abs_regret < self.stop_threshold

# --- Класс RandomAgent ---
class RandomAgent:
    """Агент, выбирающий случайное допустимое действие."""
    def __init__(self): self.key = random.PRNGKey(int(time.time()) + 1); self.ai_settings = {}
    def get_move(self, game_state: GameState, timeout_event: Event, result: Dict) -> None:
        logger.debug("Inside RandomAgent get_move"); actions_jax = get_actions(game_state); num_actions = actions_jax.shape[0]
        if num_actions == 0: result["move"] = {"error": "Нет доступных ходов"}; logger.warning("RandomAgent: No actions available."); return
        self.key, subkey = random.split(self.key); random_index = int(random.choice(subkey, jnp.arange(num_actions))); move = action_from_array(actions_jax[random_index]); result["move"] = move; logger.debug(f"RandomAgent selected action index {random_index}: {move}")
    def baseline_evaluation(self, state: GameState) -> float: current_ai_settings = getattr(self, 'ai_settings', {}); return heuristic_baseline_evaluation(state, current_ai_settings)
    # --- Остальные методы RandomAgent остаются заглушками ---
    def evaluate_move(self, game_state: GameState, action: Dict[str, List[Card]], timeout_event: Event) -> float: logger.warning("Not implemented"); return 0.0
    def shallow_search(self, state: GameState, depth: int, timeout_event: Event) -> float: logger.warning("Not implemented"); return 0.0
    def get_action_value(self, state: GameState, action: Dict[str, List[Card]], timeout_event: Event) -> float: logger.warning("Not implemented"); return 0.0
    def calculate_potential(self, cards: List[Card], line: str, board: Board, available_cards: List[Card]) -> float: logger.warning("Not implemented"); return 0.0
    def is_flush_potential(self, cards: List[Card], available_cards: List[Card]) -> bool: logger.warning("Not implemented"); return False
    def is_straight_potential(self, cards: List[Card], available_cards: List[Card]) -> bool: logger.warning("Not implemented"); return False
    def is_pair_potential(self, cards: List[Card], available_cards: List[Card]) -> bool: logger.warning("Not implemented"); return False
    def evaluate_line_strength(self, cards: List[Card], line: str) -> float: logger.warning("Not implemented"); return 0.0
    def identify_combination(self, cards: List[Card]) -> None: logger.warning("Not implemented"); pass
    def is_bottom_stronger_than_middle(self, state: GameState) -> None: logger.warning("Not implemented"); pass
    def is_middle_stronger_than_top(self, state: GameState) -> None: logger.warning("Not implemented"); pass
    def check_row_strength_rule(self, state: GameState) -> None: logger.warning("Not implemented"); pass
    def save_progress(self) -> None: pass
    def load_progress(self) -> None: pass

# --- Конец файла ai_engine.py ---
