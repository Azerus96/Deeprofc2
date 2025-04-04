# ai_engine_v3.1_padding_fix.py
# Исправлена ошибка NonConcreteBooleanIndexError путем добавления паддинга
# перед вызовом JIT-функций evaluate_hand_jax/compare_hands_jax.
# Основано на ai_engine_v3_jit_fixed.py (1319 строк).

# --- Стандартные импорты ---
import itertools
from collections import defaultdict, Counter
from threading import Event, Thread
import time
import math
import logging
from typing import List, Dict, Tuple, Optional, Union, Any
import concurrent.futures
import os

# --- Импорты библиотек машинного обучения и вычислений ---
import jax.numpy as jnp
import jax
from jax import random
from jax import jit, lax # Импортируем lax для cond/switch
import numpy as np

# --- Импорт утилит для GitHub (с обработкой отсутствия) ---
try:
    from github_utils import save_ai_progress_to_github, load_ai_progress_from_github
except ImportError:
    logging.warning("github_utils not found. Saving/Loading progress to GitHub will be disabled.")
    def save_ai_progress_to_github(*args: Any, **kwargs: Any) -> bool: logging.error("Saving to GitHub is disabled (github_utils not found)."); return False
    def load_ai_progress_from_github(*args: Any, **kwargs: Any) -> Optional[Any]: logging.error("Loading from GitHub is disabled (github_utils not found)."); return None

# --- Настройка логирования ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Класс Card ---
# (Без изменений)
class Card:
    RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]; SUITS = ["♥", "♦", "♣", "♠"]
    RANK_MAP = {rank: i for i, rank in enumerate(RANKS)}; SUIT_MAP = {suit: i for i, suit in enumerate(SUITS)}
    def __init__(self, rank: str, suit: str):
        if rank not in self.RANKS: raise ValueError(f"Invalid rank: {rank}")
        if suit not in self.SUITS: raise ValueError(f"Invalid suit: {suit}")
        self.rank = rank; self.suit = suit
    def __repr__(self) -> str: return f"{self.rank}{self.suit}"
    def __eq__(self, other: object) -> bool:
        if isinstance(other, dict): return self.rank == other.get("rank") and self.suit == other.get("suit")
        return isinstance(other, Card) and self.rank == other.rank and self.suit == other.suit
    def __hash__(self) -> int: return hash((self.rank, self.suit))
    def to_dict(self) -> Dict[str, str]: return {"rank": self.rank, "suit": self.suit}
    @staticmethod
    def from_dict(card_dict: Dict[str, str]) -> "Card": return Card(card_dict["rank"], card_dict["suit"])
    @staticmethod
    def get_all_cards() -> List["Card"]: return [Card(r, s) for r in Card.RANKS for s in Card.SUITS]

# --- Класс Hand ---
# (Без изменений)
class Hand:
    def __init__(self, cards: Optional[List[Card]] = None): self.cards = cards if cards is not None else []
    def add_card(self, card: Card) -> None:
        if not isinstance(card, Card): raise TypeError("card must be an instance of Card")
        self.cards.append(card)
    def add_cards(self, cards_to_add: List[Card]) -> None:
        for card in cards_to_add: self.add_card(card)
    def remove_card(self, card: Card) -> None:
        if not isinstance(card, Card): raise TypeError("card must be an instance of Card")
        try: self.cards.remove(card)
        except ValueError: logger.warning(f"Card {card} not found in hand to remove: {self.cards}")
    def remove_cards(self, cards_to_remove: List[Card]) -> None:
        temp_hand = self.cards[:]; removed_count = 0
        for card_to_remove in cards_to_remove:
            try: temp_hand.remove(card_to_remove); removed_count += 1
            except ValueError: logger.warning(f"Card {card_to_remove} not found in hand during multi-remove.")
        if removed_count != len(cards_to_remove): logger.warning(f"Expected to remove {len(cards_to_remove)}, removed {removed_count}.")
        self.cards = temp_hand
    def __repr__(self) -> str:
        st = lambda cards: ", ".join(map(str, sorted(cards, key=lambda c: (Card.RANK_MAP.get(c.rank, -1), Card.SUIT_MAP.get(c.suit, -1)))))
        return f"Hand: [{st(self.cards)}]"
    def __len__(self) -> int: return len(self.cards)
    def __iter__(self): return iter(self.cards)
    def __getitem__(self, index: int) -> Card: return self.cards[index]
    def to_jax(self) -> jnp.ndarray:
         if not self.cards: return jnp.empty((0, 2), dtype=jnp.int32)
         return jnp.array([card_to_array(card) for card in self.cards], dtype=jnp.int32)

# --- Класс Board ---
# (Без изменений)
class Board:
    def __init__(self): self.top: List[Card] = []; self.middle: List[Card] = []; self.bottom: List[Card] = []
    def get_placed_count(self) -> int: return len(self.top) + len(self.middle) + len(self.bottom)
    def place_card(self, line: str, card: Card) -> None:
        target_line = getattr(self, line, None)
        if target_line is None: raise ValueError(f"Invalid line: {line}")
        max_len = 3 if line == "top" else 5
        if len(target_line) >= max_len: raise ValueError(f"{line.capitalize()} line is full ({len(target_line)}/{max_len})")
        target_line.append(card)
    def is_full(self) -> bool: return len(self.top) == 3 and len(self.middle) == 5 and len(self.bottom) == 5
    def clear(self) -> None: self.top = []; self.middle = []; self.bottom = []
    def __repr__(self) -> str:
        st = lambda cards: ", ".join(map(str, sorted(cards, key=lambda c: (Card.RANK_MAP.get(c.rank, -1), Card.SUIT_MAP.get(c.suit, -1)))))
        return f"Top: [{st(self.top)}]\nMiddle: [{st(self.middle)}]\nBottom: [{st(self.bottom)}]"
    def get_cards(self, line: str) -> List[Card]:
        if line == "top": return self.top
        elif line == "middle": return self.middle
        elif line == "bottom": return self.bottom
        else: raise ValueError("Invalid line specified")
    def get_all_cards(self) -> List[Card]: return self.top + self.middle + self.bottom
    def get_line_jax(self, line: str) -> jnp.ndarray:
        cards = self.get_cards(line)
        if not cards: return jnp.empty((0, 2), dtype=jnp.int32)
        return jnp.array([card_to_array(card) for card in cards], dtype=jnp.int32)
    def to_jax_placement(self) -> jnp.ndarray:
        placement = jnp.full((13, 2), -1, dtype=jnp.int32); idx = 0
        for card in self.top:
             if idx < 3: placement = placement.at[idx].set(card_to_array(card)); idx += 1
        idx = 3
        for card in self.middle:
             if idx < 8: placement = placement.at[idx].set(card_to_array(card)); idx += 1
        idx = 8
        for card in self.bottom:
             if idx < 13: placement = placement.at[idx].set(card_to_array(card)); idx += 1
        return placement
    @staticmethod
    def from_py_placement(placement_list: List[Optional[Card]]) -> "Board":
        board = Board()
        if len(placement_list) != 13: raise ValueError("Placement list must have length 13")
        board.top = [c for c in placement_list[0:3] if c is not None]
        board.middle = [c for c in placement_list[3:8] if c is not None]
        board.bottom = [c for c in placement_list[8:13] if c is not None]
        return board

# --- Вспомогательные функции для преобразования Card <-> JAX array ---
# (Без изменений)
def card_to_array(card: Optional[Card]) -> jnp.ndarray:
    if card is None: return jnp.array([-1, -1], dtype=jnp.int32)
    return jnp.array([Card.RANK_MAP.get(card.rank, -1), Card.SUIT_MAP.get(card.suit, -1)], dtype=jnp.int32)
def array_to_card(card_array: jnp.ndarray) -> Optional[Card]:
    if card_array is None or card_array.shape != (2,) or jnp.array_equal(card_array, jnp.array([-1, -1])): return None
    try:
        rank_idx = int(card_array[0]); suit_idx = int(card_array[1])
        if 0 <= rank_idx < len(Card.RANKS) and 0 <= suit_idx < len(Card.SUITS): return Card(Card.RANKS[rank_idx], Card.SUITS[suit_idx])
        else: return None
    except (IndexError, ValueError): return None
def action_to_jax(action_dict: Dict[str, List[Card]]) -> jnp.ndarray:
    action_array = jnp.full((17, 2), -1, dtype=jnp.int32); idx = 0
    for card in action_dict.get("top", []):
        if idx < 3: action_array = action_array.at[idx].set(card_to_array(card)); idx += 1
    idx = 3; # ... (остальные циклы без изменений) ...
    for card in action_dict.get("middle", []):
        if idx < 8: action_array = action_array.at[idx].set(card_to_array(card)); idx += 1
    idx = 8
    for card in action_dict.get("bottom", []):
        if idx < 13: action_array = action_array.at[idx].set(card_to_array(card)); idx += 1
    idx = 13
    for card in action_dict.get("discarded", []):
        if idx < 17: action_array = action_array.at[idx].set(card_to_array(card)); idx += 1
    return action_array
def action_from_array(action_array: jnp.ndarray) -> Dict[str, List[Card]]:
    if action_array is None or action_array.shape != (17, 2): logger.error(f"Invalid shape for action_array: {action_array.shape if action_array is not None else 'None'}"); return {}
    action_dict = {"top": [], "middle": [], "bottom": [], "discarded": []}
    action_dict["top"] = [card for i in range(3) if (card := array_to_card(action_array[i])) is not None]
    action_dict["middle"] = [card for i in range(3, 8) if (card := array_to_card(action_array[i])) is not None]
    action_dict["bottom"] = [card for i in range(8, 13) if (card := array_to_card(action_array[i])) is not None]
    action_dict["discarded"] = [card for i in range(13, 17) if (card := array_to_card(action_array[i])) is not None]
    return {k: v for k, v in action_dict.items() if v}
def placement_py_to_jax(placement_list: List[Optional[Card]]) -> jnp.ndarray:
    if len(placement_list) != 13: logger.warning(f"Incorrect length for placement_list: {len(placement_list)}. Adjusting to 13."); placement_list = (placement_list + [None]*13)[:13]
    return jnp.array([card_to_array(c) for c in placement_list], dtype=jnp.int32)

# --- Класс GameState ---
# (Без изменений по сравнению с v3)
class GameState:
    def __init__(self, selected_cards: Optional[Union[List[Card], Hand]] = None, board: Optional[Board] = None, discarded_cards: Optional[List[Card]] = None, ai_settings: Optional[Dict] = None, deck: Optional[List[Card]] = None, current_player: int = 0, opponent_board: Optional[Board] = None, opponent_discarded: Optional[List[Card]] = None):
        if isinstance(selected_cards, Hand): self.selected_cards: Hand = selected_cards
        else: self.selected_cards: Hand = Hand(selected_cards)
        self.board: Board = board if board is not None else Board(); self.discarded_cards: List[Card] = discarded_cards if discarded_cards is not None else []
        self.ai_settings: Dict = ai_settings if ai_settings is not None else {}; self.current_player: int = current_player; self.deck: List[Card] = deck if deck is not None else []
        self.opponent_board: Board = opponent_board if opponent_board is not None else Board(); self.opponent_discarded: List[Card] = opponent_discarded if opponent_discarded is not None else []
    def get_current_player(self) -> int: return self.current_player
    def is_terminal(self) -> bool: return self.board.is_full()
    def get_street(self) -> int:
        placed = self.board.get_placed_count()
        if placed == 0: return 1; # ... (остальная логика без изменений) ...
        if placed == 5: return 2; # ...
        if placed == 7: return 3; # ...
        if placed == 9: return 4; # ...
        if placed == 11: return 5; # ...
        if placed == 13: return 6; # ...
        if placed < 5: return 1; # ...
        if placed < 7: return 2; # ...
        if placed < 9: return 3; # ...
        if placed < 11: return 4; # ...
        if placed < 13: return 5; # ...
        logger.warning(f"Unexpected placed cards ({placed}) for street calc."); return 0
    def apply_action(self, action: Dict[str, List[Card]]) -> "GameState":
        new_board = Board(); new_board.top = self.board.top[:]; new_board.middle = self.board.middle[:]; new_board.bottom = self.board.bottom[:]
        new_discarded = self.discarded_cards[:]; placed_in_action = []; discarded_in_action = action.get("discarded", [])
        for line in ["top", "middle", "bottom"]:
            cards_to_place = action.get(line, []); placed_in_action.extend(cards_to_place)
            for card in cards_to_place:
                try: new_board.place_card(line, card)
                except ValueError as e: logger.error(f"Error applying action: {e}. Action: {action}, Board state:\n{self.board}"); raise ValueError(f"Invalid action application: {e}") from e
        new_discarded.extend(discarded_in_action); played_cards = placed_in_action + discarded_in_action
        new_hand = Hand(self.selected_cards.cards[:]); new_hand.remove_cards(played_cards)
        new_state = GameState(selected_cards=new_hand, board=new_board, discarded_cards=new_discarded, ai_settings=self.ai_settings.copy(), deck=self.deck, current_player=self.current_player, opponent_board=self.opponent_board, opponent_discarded=self.opponent_discarded)
        return new_state
    def get_information_set(self) -> str:
        st = lambda cards: ",".join(map(str, sorted(cards, key=lambda c: (Card.RANK_MAP.get(c.rank, -1), Card.SUIT_MAP.get(c.suit, -1)))))
        street = f"St:{self.get_street()}"; my_board = f"T:{st(self.board.top)}|M:{st(self.board.middle)}|B:{st(self.board.bottom)}"; my_disc = f"D:{st(self.discarded_cards)}"
        opp_board = f"OT:{st(self.opponent_board.top)}|OM:{st(self.opponent_board.middle)}|OB:{st(self.opponent_board.bottom)}"; opp_disc = f"OD:{st(self.opponent_discarded)}"
        return f"{street}|{my_board}|{my_disc}|{opp_board}|{opp_disc}"
    def _calculate_pairwise_score(self, opponent_board: Board) -> int:
        line_score = 0
        if not self.board.is_full() or not opponent_board.is_full(): logger.warning("Pairwise score calculated on non-full boards."); return 0
        cmp_b = _compare_hands_py(self.board.bottom, opponent_board.bottom); cmp_m = _compare_hands_py(self.board.middle, opponent_board.middle); cmp_t = _compare_hands_py(self.board.top, opponent_board.top)
        line_score += (1 if cmp_b > 0 else -1 if cmp_b < 0 else 0); line_score += (1 if cmp_m > 0 else -1 if cmp_m < 0 else 0); line_score += (1 if cmp_t > 0 else -1 if cmp_t < 0 else 0)
        scoop = 0
        if cmp_b > 0 and cmp_m > 0 and cmp_t > 0: scoop = 3
        elif cmp_b < 0 and cmp_m < 0 and cmp_t < 0: scoop = -3
        return line_score + scoop
    def get_payoff(self) -> int:
        if not self.is_terminal() or not self.opponent_board.is_full(): logger.warning("get_payoff called on non-terminal state(s)."); return 0
        my_place_jax = self.board.to_jax_placement(); opp_place_jax = self.opponent_board.to_jax_placement()
        i_am_dead = is_dead_hand_jax(my_place_jax); opp_is_dead = is_dead_hand_jax(opp_place_jax)
        my_royalty = jnp.where(i_am_dead, 0, jnp.sum(calculate_royalties_jax(my_place_jax, self.ai_settings)))
        opp_royalty = jnp.where(opp_is_dead, 0, jnp.sum(calculate_royalties_jax(opp_place_jax, self.ai_settings)))
        my_royalty = int(my_royalty); opp_royalty = int(opp_royalty)
        if i_am_dead and opp_is_dead: return 0
        if i_am_dead: return -6 - opp_royalty
        if opp_is_dead: return 6 + my_royalty
        pairwise_score = self._calculate_pairwise_score(self.opponent_board)
        return pairwise_score + my_royalty - opp_royalty
    def is_valid_fantasy_entry(self) -> bool:
        # Используем стандартный Python if, т.к. функция не JIT
        if not self.board.is_full(): return False
        place_jax = self.board.to_jax_placement()
        if bool(is_dead_hand_jax(place_jax)): return False # Конвертируем JAX bool
        return bool(is_valid_fantasy_entry_jax(place_jax, self.ai_settings)) # Конвертируем JAX bool
    def is_valid_fantasy_repeat(self) -> bool:
        # Используем стандартный Python if, т.к. функция не JIT
        if not self.board.is_full(): return False
        place_jax = self.board.to_jax_placement()
        if bool(is_dead_hand_jax(place_jax)): return False # Конвертируем JAX bool
        return bool(is_valid_fantasy_repeat_jax(place_jax, self.ai_settings)) # Конвертируем JAX bool
    def get_fantasy_cards_count(self) -> int:
        # Используем стандартный Python if, т.к. функция не JIT
        if not self.board.is_full(): return 0
        place_jax = self.board.to_jax_placement()
        if bool(is_dead_hand_jax(place_jax)): return 0
        return self._get_fantasy_cards_count_jax_logic(place_jax)
    def _get_fantasy_cards_count_jax_logic(self, place_jax: jnp.ndarray) -> int:
        # Используем стандартный Python if, т.к. функция не JIT
        top_cards_jax = place_jax[0:3]
        if not bool(jnp.sum(jnp.any(top_cards_jax != -1, axis=1)) == 3): return 0
        return self._get_fantasy_cards_count_standard(top_cards_jax)
    def _get_fantasy_cards_count_standard(self, top_cards_jax: jnp.ndarray) -> int:
        # Эта функция вызывается из Python, поэтому Python if здесь допустим
        # --- [ИСПРАВЛЕНИЕ v3.1] Добавляем паддинг перед вызовом evaluate_hand_jax ---
        pad_val = jnp.array([-1, -1], dtype=jnp.int32)
        top_padded = jnp.concatenate([top_cards_jax, jnp.full((2, 2), pad_val)], axis=0)
        top_rank, _ = evaluate_hand_jax(top_padded) # Вызываем с паддингом
        # --- Конец исправления ---
        fantasy_type = self.ai_settings.get('fantasyType', 'standard')
        if fantasy_type == 'standard':
            is_qualifying_pair = False
            if int(top_rank) == 8: # Конвертируем JAX скаляр в int
                rank_counts = _get_rank_counts_jax(top_cards_jax); pair_rank_idx = jnp.argmax(rank_counts == 2)
                is_qualifying_pair = pair_rank_idx >= Card.RANK_MAP['Q']
            is_trips = (int(top_rank) == 6)
            if is_qualifying_pair or is_trips: return 14
            else: return 0
        else: logger.warning(f"Unsupported fantasyType '{fantasy_type}'."); return 0

# --- Вспомогательные JAX функции ---
@jit
def _safe_get_counts(values: jnp.ndarray, length: int) -> jnp.ndarray:
    valid_mask = (values >= 0) & (values < length); valid_values = values[valid_mask]
    return lax.cond(valid_values.shape[0] == 0, lambda: jnp.zeros(length, dtype=jnp.int32), lambda: jnp.bincount(valid_values, length=length))
@jit
def _get_rank_counts_jax(cards_jax: jnp.ndarray) -> jnp.ndarray: # Ожидает (5, 2) из-за паддинга
    ranks = cards_jax[:, 0]; valid_mask = ranks != -1
    return _safe_get_counts(ranks[valid_mask], 13) # Считаем только валидные
@jit
def _get_suit_counts_jax(cards_jax: jnp.ndarray) -> jnp.ndarray: # Ожидает (5, 2)
    suits = cards_jax[:, 1]; valid_mask = suits != -1
    return _safe_get_counts(suits[valid_mask], 4) # Считаем только валидные
@jit
def _is_flush_jax(cards_jax: jnp.ndarray) -> bool: # Ожидает (5, 2)
    is_five_cards = cards_jax.shape[0] == 5; suits = cards_jax[:, 1]; valid_mask = suits != -1
    all_valid = jnp.sum(valid_mask) == 5;
    valid_suits = suits[valid_mask]
    all_same_suit = lax.cond(all_valid, lambda: jnp.all(valid_suits == valid_suits[0]), lambda: False)
    return is_five_cards & all_valid & all_same_suit
@jit
def _is_straight_jax(cards_jax: jnp.ndarray) -> bool: # Ожидает (5, 2)
    is_five_cards = cards_jax.shape[0] == 5; ranks = cards_jax[:, 0]; valid_mask = ranks != -1
    all_valid = jnp.sum(valid_mask) == 5
    def check_straight():
        valid_ranks = ranks[valid_mask] # Индексация OK для статической формы (5,2)
        unique_ranks = jnp.unique(valid_ranks); is_five_unique_ranks = unique_ranks.shape[0] == 5
        sorted_ranks = jnp.sort(unique_ranks)
        is_a5 = jnp.array_equal(sorted_ranks, jnp.array([0, 1, 2, 3, 12], dtype=jnp.int32))
        is_normal = (sorted_ranks[4] - sorted_ranks[0]) == 4
        return is_five_unique_ranks & (is_a5 | is_normal)
    return is_five_cards & all_valid & lax.cond(all_valid, check_straight, lambda: False)
@jit
def _is_straight_flush_jax(cards_jax: jnp.ndarray) -> bool: # Ожидает (5, 2)
    is_five_cards = cards_jax.shape[0] == 5
    return is_five_cards & _is_flush_jax(cards_jax) & _is_straight_jax(cards_jax)
@jit
def _is_royal_flush_jax(cards_jax: jnp.ndarray) -> bool: # Ожидает (5, 2)
    is_five_cards = cards_jax.shape[0] == 5; is_sf = _is_straight_flush_jax(cards_jax)
    ranks = cards_jax[:, 0]; valid_mask = ranks != -1; valid_ranks = jnp.where(valid_mask, ranks, -1)
    has_ace = jnp.any(valid_ranks == 12)
    return is_five_cards & is_sf & has_ace
@jit
def _is_four_of_a_kind_jax(cards_jax: jnp.ndarray) -> bool: # Ожидает (5, 2)
    is_five_cards = cards_jax.shape[0] == 5
    rank_counts = _get_rank_counts_jax(cards_jax); has_four = jnp.any(rank_counts == 4)
    return is_five_cards & has_four
@jit
def _is_full_house_jax(cards_jax: jnp.ndarray) -> bool: # Ожидает (5, 2)
    is_five_cards = cards_jax.shape[0] == 5
    rank_counts = _get_rank_counts_jax(cards_jax); has_three = jnp.any(rank_counts == 3); has_pair = jnp.any(rank_counts == 2)
    return is_five_cards & has_three & has_pair
@jit
def _is_three_of_a_kind_jax(cards_jax: jnp.ndarray) -> bool: # Ожидает (5, 2)
    n = 5; is_valid_size = (n == 5)
    def check_3k():
        rank_counts = _get_rank_counts_jax(cards_jax); has_three = jnp.sum(rank_counts == 3) == 1; has_no_pair = jnp.sum(rank_counts == 2) == 0
        return has_three & has_no_pair
    return lax.cond(is_valid_size, check_3k, lambda: False)
@jit
def _is_two_pair_jax(cards_jax: jnp.ndarray) -> bool: # Ожидает (5, 2)
    is_five_cards = cards_jax.shape[0] == 5
    rank_counts = _get_rank_counts_jax(cards_jax); has_two_pairs = jnp.sum(rank_counts == 2) == 2
    return is_five_cards & has_two_pairs
@jit
def _is_one_pair_jax(cards_jax: jnp.ndarray) -> bool: # Ожидает (5, 2)
    n = 5; is_valid_size = (n == 5)
    def check_1p():
        rank_counts = _get_rank_counts_jax(cards_jax); has_one_pair = jnp.sum(rank_counts == 2) == 1; has_no_better = jnp.sum(rank_counts >= 3) == 0
        return has_one_pair & has_no_better
    return lax.cond(is_valid_size, check_1p, lambda: False)

@jit
def _identify_combination_jax(cards_jax: jnp.ndarray) -> int: # Ожидает (5, 2)
    ranks = cards_jax[:, 0]; valid_mask = ranks != -1; num_valid_cards = jnp.sum(valid_mask)
    def get_rank_for_non_empty(nv):
        def rank_for_5(): return lax.cond(_is_royal_flush_jax(cards_jax), lambda: 0, lambda: lax.cond(_is_straight_flush_jax(cards_jax), lambda: 1, lambda: lax.cond(_is_four_of_a_kind_jax(cards_jax), lambda: 2, lambda: lax.cond(_is_full_house_jax(cards_jax), lambda: 3, lambda: lax.cond(_is_flush_jax(cards_jax), lambda: 4, lambda: lax.cond(_is_straight_jax(cards_jax), lambda: 5, lambda: lax.cond(_is_three_of_a_kind_jax(cards_jax), lambda: 6, lambda: lax.cond(_is_two_pair_jax(cards_jax), lambda: 7, lambda: lax.cond(_is_one_pair_jax(cards_jax), lambda: 8, lambda: 9)))))))))
        def rank_for_3():
            rank_counts = _get_rank_counts_jax(cards_jax); has_three = jnp.sum(rank_counts == 3) == 1; has_one_pair = jnp.sum(rank_counts == 2) == 1
            return lax.cond(has_three, lambda: 6, lambda: lax.cond(has_one_pair, lambda: 8, lambda: 9))
        def rank_for_2():
            rank_counts = _get_rank_counts_jax(cards_jax); has_one_pair = jnp.sum(rank_counts == 2) == 1
            return lax.cond(has_one_pair, lambda: 8, lambda: 9)
        branches = [lambda: 9, rank_for_2, rank_for_3, lambda: 9, rank_for_5]
        safe_index = jnp.clip(nv - 1, 0, len(branches) - 1)
        return lax.switch(safe_index, branches)
    return lax.cond(num_valid_cards == 0, lambda: 10, lambda: get_rank_for_non_empty(num_valid_cards))

@jit
def evaluate_hand_jax(cards_jax: jnp.ndarray) -> Tuple[int, jnp.ndarray]: # Ожидает (5, 2)
    ranks = cards_jax[:, 0]; valid_mask = ranks != -1; num_valid_cards = jnp.sum(valid_mask)
    default_kickers = jnp.full(5, -1, dtype=jnp.int32)

    def compute_rank_and_kickers_padded(cards_5x2, mask_5, nv):
        combination_rank = _identify_combination_jax(cards_5x2)
        rank_counts = _get_rank_counts_jax(cards_5x2)
        valid_ranks = jnp.where(mask_5, ranks, -1)
        valid_ranks_only = valid_ranks[valid_ranks != -1] # Фильтруем -1 для сортировки кикеров
        sorted_valid_ranks_desc = jnp.sort(valid_ranks_only)[::-1]

        # --- Логика кикеров (адаптирована для nv и использует sorted_valid_ranks_desc) ---
        def case_0_1(): is_a5 = jnp.array_equal(jnp.sort(valid_ranks_only), jnp.array([0, 1, 2, 3, 12], dtype=jnp.int32)); main_kicker = jnp.where(is_a5, 3, sorted_valid_ranks_desc[0]); return default_kickers.at[0].set(main_kicker)
        def case_2(): four_rank = jnp.argmax(rank_counts == 4); kicker_rank = jnp.argmax(rank_counts == 1); return default_kickers.at[0].set(four_rank).at[1].set(kicker_rank)
        def case_3(): three_rank = jnp.argmax(rank_counts == 3); pair_rank = jnp.argmax(rank_counts == 2); return default_kickers.at[0].set(three_rank).at[1].set(pair_rank)
        def case_4(): return default_kickers.at[:nv].set(sorted_valid_ranks_desc[:nv])
        def case_5(): is_a5 = jnp.array_equal(jnp.sort(valid_ranks_only), jnp.array([0, 1, 2, 3, 12], dtype=jnp.int32)); main_kicker = jnp.where(is_a5, 3, sorted_valid_ranks_desc[0]); return default_kickers.at[0].set(main_kicker)
        def case_6(): # Трипс
            three_rank = jnp.argmax(rank_counts == 3); k = default_kickers.at[0].set(three_rank)
            k = lax.cond(nv == 5, lambda: k.at[1:3].set(sorted_valid_ranks_desc[1:3]), lambda: k)
            return k
        def case_7(): # Две пары (только nv=5)
            pair_ranks = jnp.sort(jnp.where(rank_counts == 2)[0])[::-1]; kicker_rank = jnp.argmax(rank_counts == 1)
            return default_kickers.at[0].set(pair_ranks[0]).at[1].set(pair_ranks[1]).at[2].set(kicker_rank)
        def case_8(): # Пара
            pair_rank = jnp.argmax(rank_counts == 2); k = default_kickers.at[0].set(pair_rank)
            num_others = nv - 2
            num_kickers_to_set = jnp.minimum(num_others, 3 if nv == 5 else 1 if nv == 3 else 0)
            kickers_available = sorted_valid_ranks_desc[sorted_valid_ranks_desc != pair_rank]
            # Используем статический размер среза и where
            indices = jnp.arange(1, 5); kickers_to_set = jnp.pad(kickers_available, (0, 4 - kickers_available.shape[0]), constant_values=-1)[:4]
            mask_kickers = indices < (1 + num_kickers_to_set)
            k = k.at[1:5].set(jnp.where(mask_kickers, kickers_to_set, k[1:5]))
            return k
        def case_default(): # Хай кард
             return default_kickers.at[:nv].set(sorted_valid_ranks_desc[:nv])

        branches = [case_0_1, case_0_1, case_2, case_3, case_4, case_5, case_6, case_7, case_8, case_default]
        safe_index = jnp.clip(combination_rank, 0, len(branches) - 1)
        computed_kickers = lax.switch(safe_index, branches)

        final_rank = jnp.where(nv == 0, 10, combination_rank)
        final_kickers = jnp.where(nv == 0, default_kickers, computed_kickers)
        return final_rank, final_kickers

    return compute_rank_and_kickers_padded(cards_jax, valid_mask, num_valid_cards)

@jit
def compare_hands_jax(hand1_jax: jnp.ndarray, hand2_jax: jnp.ndarray) -> int: # Ожидает (5, 2)
    rank1, kickers1 = evaluate_hand_jax(hand1_jax); rank2, kickers2 = evaluate_hand_jax(hand2_jax)
    def compare_kickers():
        comparison = jnp.sign(kickers1 - kickers2); mask = comparison != 0
        first_diff_index = jnp.argmax(mask, axis=0)
        result = jnp.where(jnp.any(mask), comparison[first_diff_index], 0)
        return result
    result_final = lax.cond(rank1 < rank2, lambda: 1, lambda: lax.cond(rank1 > rank2, lambda: -1, compare_kickers))
    return result_final

@jit
def is_dead_hand_jax(placement: jnp.ndarray) -> bool: # Ожидает (13, 2)
    top_cards = placement[0:3]; middle_cards = placement[3:8]; bottom_cards = placement[8:13]
    is_top_full = jnp.sum(jnp.any(top_cards != -1, axis=1)) == 3
    is_middle_full = jnp.sum(jnp.any(middle_cards != -1, axis=1)) == 5
    is_bottom_full = jnp.sum(jnp.any(bottom_cards != -1, axis=1)) == 5
    pad_val = jnp.array([-1, -1], dtype=jnp.int32)
    top_padded = jnp.concatenate([top_cards, jnp.full((2, 2), pad_val)], axis=0)
    top_beats_middle = jnp.where(is_top_full & is_middle_full, compare_hands_jax(top_padded, middle_cards) > 0, False)
    middle_beats_bottom = jnp.where(is_middle_full & is_bottom_full, compare_hands_jax(middle_cards, bottom_cards) > 0, False)
    is_dead = top_beats_middle | middle_beats_bottom
    return is_dead

# --- Python versions ---
# --- [ИСПРАВЛЕНИЕ v3.1] Добавлен паддинг в _compare_hands_py ---
def _compare_hands_py(hand1_cards: List[Card], hand2_cards: List[Card]) -> int:
    if not hand1_cards and not hand2_cards: return 0;
    if not hand1_cards: return -1;
    if not hand2_cards: return 1;
    hand1_cards_valid = [c for c in hand1_cards if isinstance(c, Card)]; hand2_cards_valid = [c for c in hand2_cards if isinstance(c, Card)]
    if not hand1_cards_valid and not hand2_cards_valid: return 0;
    if not hand1_cards_valid: return -1;
    if not hand2_cards_valid: return 1;
    hand1_list = [card_to_array(c) for c in hand1_cards_valid]; hand2_list = [card_to_array(c) for c in hand2_cards_valid]
    pad_val = jnp.array([-1, -1], dtype=jnp.int32)
    while len(hand1_list) < 5: hand1_list.append(pad_val)
    while len(hand2_list) < 5: hand2_list.append(pad_val)
    hand1_jax = jnp.stack(hand1_list[:5]); hand2_jax = jnp.stack(hand2_list[:5])
    return int(compare_hands_jax(hand1_jax, hand2_jax))
# --- Конец исправления ---
def _is_dead_hand_py(board: Board) -> bool:
    top_full = len(board.top) == 3; middle_full = len(board.middle) == 5; bottom_full = len(board.bottom) == 5; is_dead = False
    if top_full and middle_full:
        if _compare_hands_py(board.top, board.middle) > 0: is_dead = True # Использует _compare_hands_py с паддингом
    if middle_full and bottom_full:
        if _compare_hands_py(board.middle, board.bottom) > 0: is_dead = True # Использует _compare_hands_py с паддингом
    return is_dead

# --- JIT функции для роялти и фантазии ---
# --- [ИСПРАВЛЕНИЕ v3.1] Добавлен паддинг перед вызовами evaluate_hand_jax ---
@jit
def calculate_royalties_jax(placement_jax: jnp.ndarray, ai_settings: Dict) -> jnp.ndarray:
    top_cards = placement_jax[0:3]; middle_cards = placement_jax[3:8]; bottom_cards = placement_jax[8:13]
    pad_val = jnp.array([-1, -1], dtype=jnp.int32); top_padded = jnp.concatenate([top_cards, jnp.full((2, 2), pad_val)], axis=0)
    is_top_full = jnp.sum(jnp.any(top_cards != -1, axis=1)) == 3
    def top_royalty_calc():
        top_rank, _ = evaluate_hand_jax(top_padded); royalty = jnp.array(0, dtype=jnp.int32)
        royalty = jnp.where(top_rank == 6, 10 + top_cards[0, 0], royalty)
        pair_rank_idx = jnp.argmax(jnp.bincount(jnp.maximum(0, top_cards[:, 0]), length=13) == 2)
        royalty_for_pair = jnp.maximum(0, pair_rank_idx - Card.RANK_MAP['5'])
        royalty = jnp.where(top_rank == 8, royalty_for_pair, royalty); return royalty
    top_royalty = lax.cond(is_top_full, top_royalty_calc, lambda: jnp.array(0, dtype=jnp.int32))
    is_middle_full = jnp.sum(jnp.any(middle_cards != -1, axis=1)) == 5
    def middle_royalty_calc():
        middle_rank, _ = evaluate_hand_jax(middle_cards); middle_royalties_map = jnp.array([50, 30, 20, 12, 8, 4, 2, 0, 0, 0, 0], dtype=jnp.int32)
        safe_rank = jnp.clip(middle_rank, 0, len(middle_royalties_map) - 1); return middle_royalties_map[safe_rank]
    middle_royalty = lax.cond(is_middle_full, middle_royalty_calc, lambda: jnp.array(0, dtype=jnp.int32))
    is_bottom_full = jnp.sum(jnp.any(bottom_cards != -1, axis=1)) == 5
    def bottom_royalty_calc():
        bottom_rank, _ = evaluate_hand_jax(bottom_cards); bottom_royalties_map = jnp.array([25, 15, 10, 6, 4, 2, 0, 0, 0, 0, 0], dtype=jnp.int32)
        safe_rank = jnp.clip(bottom_rank, 0, len(bottom_royalties_map) - 1); return bottom_royalties_map[safe_rank]
    bottom_royalty = lax.cond(is_bottom_full, bottom_royalty_calc, lambda: jnp.array(0, dtype=jnp.int32))
    return jnp.array([top_royalty, middle_royalty, bottom_royalty], dtype=jnp.int32)
@jit
def is_valid_fantasy_entry_jax(placement: jnp.ndarray, ai_settings: Dict) -> bool:
    top_cards = placement[0:3]; pad_val = jnp.array([-1, -1], dtype=jnp.int32); top_padded = jnp.concatenate([top_cards, jnp.full((2, 2), pad_val)], axis=0)
    is_top_full = jnp.sum(jnp.any(top_cards != -1, axis=1)) == 3
    def check_fantasy():
        top_rank, _ = evaluate_hand_jax(top_padded)
        is_qualifying_pair = lax.cond(top_rank == 8, lambda: jnp.argmax(_get_rank_counts_jax(top_cards) == 2) >= Card.RANK_MAP['Q'], lambda: False)
        is_trips = top_rank == 6; return is_trips | is_qualifying_pair
    return lax.cond(is_top_full, check_fantasy, lambda: False)
@jit
def is_valid_fantasy_repeat_jax(placement: jnp.ndarray, ai_settings: Dict) -> bool:
    top_cards = placement[0:3]; bottom_cards = placement[8:13]; pad_val = jnp.array([-1, -1], dtype=jnp.int32); top_padded = jnp.concatenate([top_cards, jnp.full((2, 2), pad_val)], axis=0)
    is_top_full = jnp.sum(jnp.any(top_cards != -1, axis=1)) == 3; is_bottom_full = jnp.sum(jnp.any(bottom_cards != -1, axis=1)) == 5
    def check_repeat():
        top_rank, _ = evaluate_hand_jax(top_padded); bottom_rank, _ = evaluate_hand_jax(bottom_cards)
        trips_on_top = top_rank == 6; quads_or_better_on_bottom = bottom_rank <= 2
        return trips_on_top | quads_or_better_on_bottom
    return lax.cond(is_top_full & is_bottom_full, check_repeat, lambda: False)
# --- Конец исправления ---

# --- Функции генерации действий ---
# (Без изменений по сравнению с v3)
def _is_board_valid_py(board: Board) -> bool:
    middle_ok = True
    if board.top and board.middle:
        if _compare_hands_py(board.top, board.middle) > 0: middle_ok = False
    bottom_ok = True
    if board.middle and board.bottom:
        if _compare_hands_py(board.middle, board.bottom) > 0: bottom_ok = False
    return middle_ok and bottom_ok
def _generate_valid_placements_recursive(current_board: Board, cards_to_place: List[Card], ai_settings: Dict, max_placements_limit: Optional[int] = None) -> List[Board]:
    valid_boards_found = []
    if not cards_to_place: return [current_board]
    card_to_try = cards_to_place[0]; remaining_cards = cards_to_place[1:]
    lines = {"top": 3, "middle": 5, "bottom": 5}
    for line_name, max_len in lines.items():
        if max_placements_limit is not None and len(valid_boards_found) >= max_placements_limit: break
        current_line = getattr(current_board, line_name)
        if len(current_line) < max_len:
            next_board = Board(); next_board.top = current_board.top[:]; next_board.middle = current_board.middle[:]; next_board.bottom = current_board.bottom[:]
            try: next_board.place_card(line_name, card_to_try)
            except ValueError: continue
            if _is_board_valid_py(next_board):
                remaining_limit = None
                if max_placements_limit is not None: remaining_limit = max_placements_limit - len(valid_boards_found);
                if remaining_limit is None or remaining_limit > 0:
                     results_from_branch = _generate_valid_placements_recursive(next_board, remaining_cards, ai_settings, remaining_limit)
                     valid_boards_found.extend(results_from_branch)
    if max_placements_limit is not None: return valid_boards_found[:max_placements_limit]
    else: return valid_boards_found
def get_actions(game_state: GameState) -> jnp.ndarray:
    if game_state.is_terminal(): return jnp.empty((0, 17, 2), dtype=jnp.int32)
    hand_cards = game_state.selected_cards.cards; num_cards_in_hand = len(hand_cards)
    if num_cards_in_hand == 0: logger.warning("get_actions called with empty hand."); return jnp.empty((0, 17, 2), dtype=jnp.int32)
    possible_actions_list_jax = []; street = game_state.get_street(); is_fantasy_turn = game_state.ai_settings.get("in_fantasy_turn", False)
    num_to_place, num_to_discard = 0, 0; placement_limit = 500
    if is_fantasy_turn: num_to_place = min(num_cards_in_hand, 13); num_to_discard = num_cards_in_hand - num_to_place; placement_limit = game_state.ai_settings.get("fantasy_placement_limit", 2000); logger.debug(f"Fantasy turn: Placing {num_to_place}, Discarding {num_to_discard}. Limit: {placement_limit}")
    elif street == 1:
        if num_cards_in_hand == 5: num_to_place, num_to_discard = 5, 0
        else: logger.error(f"Street 1 error: Hand={num_cards_in_hand} != 5."); return jnp.empty((0, 17, 2), dtype=jnp.int32)
        placement_limit = game_state.ai_settings.get("street1_placement_limit", 10000); logger.debug(f"Street 1: Placing {num_to_place}, Discarding {num_to_discard}. Limit: {placement_limit}")
    elif 2 <= street <= 5:
        if num_cards_in_hand == 3: num_to_place, num_to_discard = 2, 1
        else: logger.error(f"Street {street} error: Hand={num_cards_in_hand} != 3."); return jnp.empty((0, 17, 2), dtype=jnp.int32)
        placement_limit = game_state.ai_settings.get("normal_placement_limit", 500); logger.debug(f"Street {street}: Placing {num_to_place}, Discarding {num_to_discard}. Limit: {placement_limit}")
    else: logger.error(f"get_actions called on invalid street {street}."); return jnp.empty((0, 17, 2), dtype=jnp.int32)
    action_count_total = 0; max_total_actions = placement_limit * 20
    if street == 1 and num_to_place == 5:
        logger.debug("Using Itertools for Street 1 action generation.")
        cards_to_place = hand_cards; cards_to_discard = []; discard_jax = jnp.full((4, 2), -1, dtype=jnp.int32)
        if game_state.board.get_placed_count() != 0: logger.error("Cannot handle non-empty board with Street 1 itertools logic."); return jnp.empty((0, 17, 2), dtype=jnp.int32)
        available_slots = list(range(13))
        for slot_indices_tuple in itertools.combinations(available_slots, 5):
            if action_count_total >= max_total_actions: logger.warning(f"Max total actions ({max_total_actions}) reached during St1 combo iteration."); break
            slot_indices = list(slot_indices_tuple)
            for card_permutation_tuple in itertools.permutations(cards_to_place):
                if action_count_total >= max_total_actions: logger.warning(f"Max total actions ({max_total_actions}) reached during St1 perm iteration."); break
                current_placement_py = [None] * 13; valid_placement_attempt = True
                try:
                    for i, slot_idx in enumerate(slot_indices): current_placement_py[slot_idx] = card_permutation_tuple[i]
                except IndexError: valid_placement_attempt = False; continue
                if valid_placement_attempt:
                    placement_13_jax = placement_py_to_jax(current_placement_py)
                    action_17 = jnp.concatenate((placement_13_jax, discard_jax), axis=0)
                    possible_actions_list_jax.append(action_17); action_count_total += 1
            if action_count_total >= max_total_actions: break
    elif num_to_place > 0 :
        logger.debug(f"Using NEW Recursive generator for Street {street}/Fantasy. Placing {num_to_place}, Discarding {num_to_discard}")
        initial_board_obj = Board(); initial_board_obj.top = game_state.board.top[:]; initial_board_obj.middle = game_state.board.middle[:]; initial_board_obj.bottom = game_state.board.bottom[:]
        for cards_to_place_tuple in itertools.combinations(hand_cards, num_to_place):
            if action_count_total >= max_total_actions: logger.warning(f"Max total actions ({max_total_actions}) reached during combo iteration."); break
            cards_to_place = list(cards_to_place_tuple); cards_to_discard = [card for card in hand_cards if card not in cards_to_place]
            if len(cards_to_discard) != num_to_discard: continue
            discard_jax = jnp.full((4, 2), -1, dtype=jnp.int32)
            for i, card in enumerate(cards_to_discard):
                if i < 4: discard_jax = discard_jax.at[i].set(card_to_array(card))
            board_copy_for_recursion = Board(); board_copy_for_recursion.top = initial_board_obj.top[:]; board_copy_for_recursion.middle = initial_board_obj.middle[:]; board_copy_for_recursion.bottom = initial_board_obj.bottom[:]
            current_limit = placement_limit
            valid_final_boards_for_combo = _generate_valid_placements_recursive(board_copy_for_recursion, cards_to_place, game_state.ai_settings, max_placements_limit=current_limit)
            if len(valid_final_boards_for_combo) >= current_limit: logger.warning(f"Placement limit ({current_limit}) reached for combo: {cards_to_place}")
            for final_board in valid_final_boards_for_combo:
                placement_13_jax = final_board.to_jax_placement()
                action_17 = jnp.concatenate((placement_13_jax, discard_jax), axis=0)
                possible_actions_list_jax.append(action_17); action_count_total += 1
                if action_count_total >= max_total_actions: break
            if action_count_total >= max_total_actions: break
    num_generated = len(possible_actions_list_jax)
    if num_generated == 0: logger.warning(f"No valid actions generated for P{game_state.current_player} on St {street}. Hand: {hand_cards}, Board:\n{game_state.board}"); return jnp.empty((0, 17, 2), dtype=jnp.int32)
    else:
        logger.info(f"Generated {num_generated} actions for P{game_state.current_player} on St {street}.")
        correct_shape_actions = []
        for i, a in enumerate(possible_actions_list_jax):
            if hasattr(a, 'shape') and a.shape == (17, 2): correct_shape_actions.append(a)
            else: logger.error(f"Action at index {i} has incorrect shape or type: {type(a)}, shape={getattr(a, 'shape', 'N/A')}")
        if len(correct_shape_actions) != num_generated:
             logger.error(f"Correctly shaped actions: {len(correct_shape_actions)}/{num_generated}")
             if not correct_shape_actions: return jnp.empty((0, 17, 2), dtype=jnp.int32)
             return jnp.stack(correct_shape_actions)
        else: return jnp.stack(possible_actions_list_jax)

# --- Вспомогательные функции для эвристической оценки (Python) ---
def _evaluate_partial_combination_py(cards: List[Card], row_type: str) -> float:
    if not cards: return 0.0; score = 0.0; n = len(cards); ranks = [card.rank for card in cards]; suits = [card.suit for card in cards]; rank_indices = sorted([r for r in [Card.RANK_MAP.get(rank, -1) for rank in ranks] if r != -1]);
    if not rank_indices: return 0.0
    if row_type in ["middle", "bottom"] and n >= 2 and n < 5:
        suit_counts = Counter(suits); max_suit_count = max(suit_counts.values()) if suit_counts else 0
        if max_suit_count >= 3: score += 5.0 * (max_suit_count - 2)
        unique_ranks = sorted(list(set(rank_indices))); un = len(unique_ranks)
        if un >= 2:
            gaps = 0; span = unique_ranks[-1] - unique_ranks[0]
            for i in range(un - 1): gaps += (unique_ranks[i+1] - unique_ranks[i] - 1)
            if gaps == 0 and span == un - 1: score += 4.0 * un
            elif gaps == 1 and span <= 4: score += 2.0 * un
            if set(unique_ranks).issuperset({0, 1, 2}): score += 3.0;
            if set(unique_ranks).issuperset({0, 1, 2, 3}): score += 5.0
            if set(unique_ranks).issuperset({9, 10, 11, 12}): score += 4.0
    rank_counts = Counter(ranks)
    for rank, count in rank_counts.items():
        rank_value = Card.RANK_MAP.get(rank, -1);
        if rank_value == -1: continue
        if count == 2: score += 5.0 + rank_value * 0.5
        elif count == 3: score += 15.0 + rank_value * 1.0
    score += sum(r for r in rank_indices) * 0.1; return score
def _is_bottom_stronger_or_equal_py(board: Board) -> bool:
    if not board.bottom or not board.middle: return True
    return _compare_hands_py(board.bottom, board.middle) >= 0
def _is_middle_stronger_or_equal_py(board: Board) -> bool:
    if not board.middle or not board.top: return True
    return _compare_hands_py(board.middle, board.top) >= 0
def heuristic_baseline_evaluation(state: GameState, ai_settings: Dict) -> float:
    board = state.board; is_full = board.is_full(); total_score = 0.0; placement_jax = board.to_jax_placement()
    if is_full and bool(is_dead_hand_jax(placement_jax)): return -1000.0
    COMBINATION_WEIGHTS = jnp.array([100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 5.0, 0.0], dtype=jnp.float32)
    ROW_MULTIPLIERS = {"top": 1.0, "middle": 1.2, "bottom": 1.5}; max_cards_in_row = {"top": 3, "middle": 5, "bottom": 5}
    rows_data = {"top": board.top, "middle": board.middle, "bottom": board.bottom}
    for row_name, cards_list in rows_data.items():
        row_score = 0.0; num_cards_in_row = len(cards_list); max_cards = max_cards_in_row[row_name]
        if num_cards_in_row > 0:
            if num_cards_in_row == max_cards:
                # --- [ИСПРАВЛЕНИЕ v3.1] Паддинг перед evaluate_hand_jax ---
                cards_list_padded = cards_list[:]
                pad_val_arr = jnp.array([-1, -1], dtype=jnp.int32)
                target_len = 5
                cards_to_pad = target_len - len(cards_list_padded)
                if cards_to_pad > 0:
                    padding = jnp.tile(pad_val_arr, (cards_to_pad, 1))
                    cards_jax_unpadded = jnp.array([card_to_array(c) for c in cards_list_padded], dtype=jnp.int32)
                    cards_jax = jnp.concatenate([cards_jax_unpadded, padding], axis=0)
                else:
                    cards_jax = jnp.array([card_to_array(c) for c in cards_list_padded[:target_len]], dtype=jnp.int32)
                # --- Конец исправления ---

                rank, kickers = evaluate_hand_jax(cards_jax) # Вызываем с паддингом
                safe_rank = jnp.clip(rank, 0, len(COMBINATION_WEIGHTS) - 1); row_score += float(COMBINATION_WEIGHTS[safe_rank])
                row_score += float(jnp.sum(kickers[kickers != -1])) * 0.01
            else: row_score += _evaluate_partial_combination_py(cards_list, row_name)
        total_score += row_score * ROW_MULTIPLIERS[row_name]
    if board.bottom and board.middle:
         if _is_bottom_stronger_or_equal_py(board): total_score += 15.0
         else: total_score -= 50.0
    if board.middle and board.top:
         if _is_middle_stronger_or_equal_py(board): total_score += 10.0
         else: total_score -= 50.0
    discard_penalty = 0.0
    for card in state.discarded_cards: rank_value = Card.RANK_MAP.get(card.rank, -1); discard_penalty += (rank_value + 1) * 0.1 if rank_value != -1 else 0
    total_score -= discard_penalty
    if is_full: current_royalties = calculate_royalties_jax(placement_jax, ai_settings); total_score += float(jnp.sum(current_royalties)) * 0.5
    return total_score

# --- Класс CFRNode ---
# (Добавлена нормализация стратегий для большей надежности)
class CFRNode:
    def __init__(self, num_actions: int):
        if num_actions < 0: raise ValueError("Number of actions cannot be negative")
        self.num_actions = num_actions
        if self.num_actions > 0: self.regret_sum = jnp.zeros(self.num_actions, dtype=jnp.float32); self.strategy_sum = jnp.zeros(self.num_actions, dtype=jnp.float32)
        else: self.regret_sum = jnp.array([], dtype=jnp.float32); self.strategy_sum = jnp.array([], dtype=jnp.float32)
    def get_strategy(self, realization_weight: float) -> jnp.ndarray:
        if self.num_actions == 0: return jnp.array([], dtype=jnp.float32)
        positive_regret_sum = jnp.maximum(self.regret_sum, 0); normalizing_sum = jnp.sum(positive_regret_sum)
        # --- [ИСПРАВЛЕНИЕ v3.1] Защита от деления на ноль и нормализация ---
        uniform_strategy = jnp.ones(self.num_actions, dtype=jnp.float32) / self.num_actions
        strategy = jnp.where(normalizing_sum > 1e-9, positive_regret_sum / normalizing_sum, uniform_strategy)
        strategy_sum_val = jnp.sum(strategy)
        strategy = jnp.where(strategy_sum_val > 1e-9, strategy / strategy_sum_val, uniform_strategy)
        # --- Конец исправления ---
        if realization_weight > 1e-9: self.strategy_sum = self.strategy_sum + (realization_weight * strategy)
        return strategy
    def get_average_strategy(self) -> jnp.ndarray:
        if self.num_actions == 0: return jnp.array([], dtype=jnp.float32)
        normalizing_sum = jnp.sum(self.strategy_sum)
        # --- [ИСПРАВЛЕНИЕ v3.1] Защита от деления на ноль и нормализация ---
        uniform_strategy = jnp.ones(self.num_actions, dtype=jnp.float32) / self.num_actions
        avg_strategy = jnp.where(normalizing_sum > 1e-9, self.strategy_sum / normalizing_sum, uniform_strategy)
        avg_strategy_sum_val = jnp.sum(avg_strategy)
        avg_strategy = jnp.where(avg_strategy_sum_val > 1e-9, avg_strategy / avg_strategy_sum_val, uniform_strategy)
        # --- Конец исправления ---
        return avg_strategy

# --- Класс CFRAgent ---
# (Изменения в _play_one_game_for_cfr для конвертации JAX bool в Python bool)
class CFRAgent:
    def __init__(self, iterations: int = 1000000, stop_threshold: float = 0.001, batch_size: int = 64, max_nodes: int = 1000000, ai_settings: Optional[Dict] = None, num_workers: Optional[int] = None):
        self.iterations = iterations; self.stop_threshold = stop_threshold; self.save_interval = 2000; self.convergence_check_interval = 50000
        self.key = random.PRNGKey(int(time.time())); self.batch_size = batch_size; self.max_nodes = max_nodes; self.nodes_map: Dict[int, CFRNode] = {}
        self.ai_settings = ai_settings if ai_settings is not None else {}; available_cpus = os.cpu_count() or 1
        if num_workers is None: self.num_workers = available_cpus
        else: self.num_workers = min(num_workers, available_cpus)
        logger.info(f"CFRAgent initialized. Iterations={iterations}, Max Nodes={max_nodes}, Stop Threshold={stop_threshold}")
        logger.info(f"Batch Size: {self.batch_size}, Num Workers: {self.num_workers} (Available CPUs: {available_cpus})"); logger.info(f"AI Settings: {self.ai_settings}")
        logger.info(f"Save Interval: {self.save_interval} games, Convergence Check Interval: {self.convergence_check_interval} games")
    def get_node(self, info_set: str, num_actions: int) -> Optional[CFRNode]:
        if num_actions < 0: logger.error(f"Negative actions: {num_actions}"); return None
        info_hash = hash(info_set); node = self.nodes_map.get(info_hash)
        if node is None:
            if len(self.nodes_map) >= self.max_nodes: logger.warning(f"Max nodes ({self.max_nodes}) reached."); return None
            try: node = CFRNode(num_actions); self.nodes_map[info_hash] = node
            except ValueError as e: logger.error(f"Error creating CFRNode: {e}"); return None
        elif node.num_actions != num_actions:
            logger.error(f"CRITICAL: Action count mismatch for hash {info_hash}. Node had {node.num_actions}, Requested={num_actions}. Discarding old node. InfoSet: {info_set[:100]}...")
            if len(self.nodes_map) >= self.max_nodes and info_hash not in self.nodes_map: logger.warning(f"Max nodes reached while replacing node {info_hash}."); return None
            try: node = CFRNode(num_actions); self.nodes_map[info_hash] = node; logger.warning(f"Replaced node {info_hash} with {num_actions} actions.")
            except ValueError as e: logger.error(f"Error replacing CFRNode: {e}"); return None
        return node
    def get_move(self, game_state: GameState, timeout_event: Event, result: Dict) -> None:
        start_time = time.time(); actions_jax = get_actions(game_state); num_available_actions = actions_jax.shape[0]
        logger.info(f"get_move: Found {num_available_actions} actions.")
        if num_available_actions == 0: result["move"] = {"error": "Нет доступных ходов"}; logger.warning("No actions available in get_move."); return
        info_set = game_state.get_information_set(); node = self.get_node(info_set, num_available_actions); best_action_index = -1
        if node is not None:
            logger.debug(f"Node found for info_set hash {hash(info_set)}. Using average strategy.")
            avg_strategy = node.get_average_strategy()
            if avg_strategy.shape[0] == num_available_actions and node.num_actions > 0: best_action_index = int(jnp.argmax(avg_strategy)); logger.debug(f"Chose action {best_action_index} based on max avg strategy.")
            else: logger.error(f"Strategy length mismatch or empty strategy! Node actions: {node.num_actions}, Available: {num_available_actions}, Strategy shape: {avg_strategy.shape}. Using baseline."); best_action_index = self._get_best_action_baseline(game_state, actions_jax)
        else: logger.warning(f"Node not found or error for info_set hash {hash(info_set)}. Using baseline evaluation."); best_action_index = self._get_best_action_baseline(game_state, actions_jax)
        if best_action_index < 0 or best_action_index >= num_available_actions:
             if num_available_actions > 0: logger.warning(f"Best action index {best_action_index} invalid. Choosing random action."); self.key, subkey = random.split(self.key); best_action_index = int(random.choice(subkey, jnp.arange(num_available_actions)))
             else: result["move"] = {"error": "Нет ходов после всех проверок"}; logger.error("Critical error: No actions available."); return
        best_action_array = actions_jax[best_action_index]; move = action_from_array(best_action_array)
        result["move"] = move; end_time = time.time(); logger.info(f"get_move finished in {end_time - start_time:.4f}s. Chosen action index: {best_action_index}. Move: {move}")
    def _get_best_action_baseline(self, game_state: GameState, actions_jax: jnp.ndarray) -> int:
        best_score = -float('inf'); best_action_index = -1; current_key = self.key
        logger.debug(f"Evaluating {actions_jax.shape[0]} actions using baseline heuristic...")
        for i, action_array in enumerate(actions_jax):
            action_dict = action_from_array(action_array)
            if not action_dict: logger.warning(f"Skipping baseline eval for invalid action array index {i}"); continue
            try:
                next_state = game_state.apply_action(action_dict); score = heuristic_baseline_evaluation(next_state, self.ai_settings)
                current_key, subkey = random.split(current_key); noise = random.uniform(subkey, minval=-0.01, maxval=0.01); score += float(noise)
                if score > best_score: best_score = score; best_action_index = i
            except Exception as e: logger.exception(f"Error evaluating baseline for action {i}. Action: {action_dict}\nError: {e}"); continue
        self.key = current_key
        if best_action_index == -1 and actions_jax.shape[0] > 0: logger.warning("All baseline evaluations failed. Choosing random action."); self.key, subkey = random.split(self.key); best_action_index = int(random.choice(subkey, jnp.arange(actions_jax.shape[0])))
        logger.debug(f"Baseline evaluation complete. Best action index: {best_action_index}, Score: {best_score:.4f}"); return best_action_index
    def baseline_evaluation(self, state: GameState) -> float:
        if not isinstance(state, GameState): logger.error("Invalid state passed to baseline_evaluation."); return -float('inf')
        return heuristic_baseline_evaluation(state, self.ai_settings)
    def train(self, timeout_event: Event, result: Dict) -> None:
        logger.info(f"Starting CFR training for up to {self.iterations} iterations..."); logger.info(f"Using {self.num_workers} worker threads with batch size {self.batch_size}.")
        start_time = time.time(); total_games_processed = 0; self.load_progress()
        num_batches = (self.iterations + self.batch_size - 1) // self.batch_size; logger.info(f"Total batches to run: {num_batches}")
        for i in range(num_batches):
            batch_start_time = time.time(); current_batch_size = min(self.batch_size, self.iterations - (i * self.batch_size))
            if current_batch_size <= 0: break
            if timeout_event.is_set(): logger.info(f"Training interrupted by timeout before batch {i+1}."); break
            keys_batch = random.split(self.key, current_batch_size + 1); self.key = keys_batch[0]; keys_for_workers = keys_batch[1:]
            deck_template = Card.get_all_cards(); deck_list = [deck_template[:] for _ in range(current_batch_size)]; trajectories_batch = []
            games_in_batch_started = i * self.batch_size; logger.info(f"Starting Batch {i+1}/{num_batches} ({current_batch_size} games, Total started: {games_in_batch_started})...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                future_to_idx = {executor.submit(self._play_one_game_for_cfr, keys_for_workers[k], deck_list[k]): k for k in range(current_batch_size)}
                num_sim_success = 0; num_sim_failed = 0
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx_in_batch = future_to_idx[future]
                    try:
                        trajectory = future.result()
                        if trajectory: trajectories_batch.append(trajectory); num_sim_success += 1
                        else: logger.warning(f"Game simulation {idx_in_batch} in batch {i+1} returned None."); num_sim_failed += 1
                    except Exception as exc: logger.error(f'Game simulation {idx_in_batch} in batch {i+1} generated an exception: {exc}', exc_info=True); num_sim_failed += 1
            logger.info(f"Batch {i+1} simulations finished. Success: {num_sim_success}, Failed: {num_sim_failed}.")
            batch_updates = 0
            if trajectories_batch:
                logger.info(f"Processing {len(trajectories_batch)} trajectories for strategy update...")
                for trajectory in trajectories_batch:
                    try: self._update_strategy_from_trajectory(trajectory); total_games_processed += 1; batch_updates += 1
                    except Exception as e: logger.exception(f"Error updating strategy from trajectory in batch {i+1}: {e}")
            else: logger.warning(f"No successful trajectories to process in batch {i+1}.")
            batch_time = time.time() - batch_start_time; logger.info(f"Batch {i+1}/{num_batches} finished in {batch_time:.2f}s. Strategy updates: {batch_updates}. Total games processed: {total_games_processed}")
            processed_before_batch = total_games_processed - batch_updates
            last_save_milestone = (processed_before_batch // self.save_interval); current_save_milestone = (total_games_processed // self.save_interval)
            if current_save_milestone > last_save_milestone: logger.info(f"Reached save interval. Saving progress at {total_games_processed} games."); self.save_progress(total_games_processed)
            last_conv_check_milestone = (processed_before_batch // self.convergence_check_interval); current_conv_check_milestone = (total_games_processed // self.convergence_check_interval)
            if current_conv_check_milestone > last_conv_check_milestone:
                logger.info(f"Reached convergence check interval. Checking convergence at {total_games_processed} games.")
                if self.check_convergence(): logger.info(f"Convergence threshold reached after {total_games_processed} games. Stopping training."); break
            if timeout_event.is_set(): logger.info(f"Training interrupted by timeout after batch {i+1}."); break
        total_time = time.time() - start_time; logger.info(f"Training finished. Total successful games processed: {total_games_processed} in {total_time:.2f} seconds.")
        logger.info(f"Total CFR nodes created: {len(self.nodes_map)}"); logger.info("Saving final progress..."); self.save_progress(total_games_processed); logger.info("Final progress saved.")
        result["status"] = "Training completed"; result["nodes_count"] = len(self.nodes_map); result["games_processed"] = total_games_processed
    def _play_one_game_for_cfr(self, key: jax.random.PRNGKey, deck: List[Card]) -> Optional[Dict]:
        game_start_time = time.time()
        try:
            key, subkey = random.split(key); shuffled_indices = random.permutation(subkey, jnp.arange(52)); shuffled_deck = [deck[int(i)] for i in shuffled_indices]; deck_iter = iter(shuffled_deck)
            game_states = {0: GameState(ai_settings=self.ai_settings.copy(), current_player=0, deck=shuffled_deck), 1: GameState(ai_settings=self.ai_settings.copy(), current_player=1, deck=shuffled_deck)}
            game_states[0].opponent_board = game_states[1].board; game_states[0].opponent_discarded = game_states[1].discarded_cards; game_states[1].opponent_board = game_states[0].board; game_states[1].opponent_discarded = game_states[0].discarded_cards
            player_fantasies = {0: False, 1: False}; fantasy_cards_count = {0: 0, 1: 0}; player_finished = {0: False, 1: False}
            trajectory = {'states': [], 'actions': [], 'reach_probs': [], 'sampling_probs': [], 'final_payoff': 0}
            reach_p0 = 1.0; reach_p1 = 1.0; current_player = 0; turn_count = 0; max_turns = 60
            while not (player_finished[0] and player_finished[1]):
                turn_count += 1
                if turn_count > max_turns: logger.error(f"Max turns ({max_turns}) reached."); return None
                if player_finished[current_player]: current_player = 1 - current_player; continue
                state = game_states[current_player]; opponent = 1 - current_player; is_fantasy_turn_now = player_fantasies[current_player]
                if len(state.selected_cards) == 0:
                    num_to_draw = 0
                    if is_fantasy_turn_now:
                        num_to_draw = fantasy_cards_count[current_player]
                        if num_to_draw <= 0: logger.error(f"P{current_player} in fantasy but card count is {num_to_draw}. Resetting."); player_fantasies[current_player] = False; player_finished[current_player] = True; current_player = opponent; continue
                        fantasy_cards_count[current_player] = 0; logger.debug(f"P{current_player} Fantasy turn: drawing {num_to_draw} cards.")
                    else:
                        street = state.get_street()
                        if street == 1: num_to_draw = 5
                        elif 2 <= street <= 5: num_to_draw = 3
                        elif street == 6:
                             if state.is_terminal():
                                 logger.debug(f"P{current_player} board full (Street {street}). Finishing player."); player_finished[current_player] = True; current_place_jax = state.board.to_jax_placement()
                                 # --- [ИСПРАВЛЕНИЕ v3.1] Конвертация JAX bool в Python bool ---
                                 if not bool(is_dead_hand_jax(current_place_jax)):
                                     if bool(state.is_valid_fantasy_entry()):
                                 # --- Конец исправления ---
                                         f_count = state.get_fantasy_cards_count()
                                         if f_count > 0: logger.info(f"P{current_player} QUALIFIES for Fantasy ({f_count} cards) on turn end!"); player_fantasies[current_player] = True; fantasy_cards_count[current_player] = f_count; player_finished[current_player] = False
                                         else: logger.warning(f"P{current_player} met fantasy entry but got 0 cards?")
                                 else: logger.debug(f"P{current_player} finished dead. No fantasy check.")
                                 current_player = opponent; continue
                             else: logger.error(f"Street {street} but board not terminal."); return None
                        else: logger.error(f"Invalid street {street} for drawing."); return None
                    try: drawn_cards = [next(deck_iter) for _ in range(num_to_draw)]; state.selected_cards = Hand(drawn_cards); logger.debug(f"P{current_player} drew {num_to_draw}. New Hand: {state.selected_cards}")
                    except StopIteration: logger.error("Deck empty during draw."); player_finished[0]=True; player_finished[1]=True; continue
                info_set = state.get_information_set(); state.ai_settings["in_fantasy_turn"] = is_fantasy_turn_now
                actions_jax = get_actions(state); num_actions = actions_jax.shape[0]; state.ai_settings["in_fantasy_turn"] = False
                if num_actions == 0:
                    if state.is_terminal():
                        logger.debug(f"P{current_player} has no actions and board is full. Finishing player."); player_finished[current_player] = True; current_place_jax = state.board.to_jax_placement()
                        # --- [ИСПРАВЛЕНИЕ v3.1] Конвертация JAX bool в Python bool ---
                        if not bool(is_dead_hand_jax(current_place_jax)):
                            if is_fantasy_turn_now:
                                if bool(state.is_valid_fantasy_repeat()):
                        # --- Конец исправления ---
                                    f_repeat_count = 14; logger.info(f"P{current_player} REPEATS Fantasy ({f_repeat_count} cards)!"); player_fantasies[current_player] = True; fantasy_cards_count[current_player] = f_repeat_count; player_finished[current_player] = False
                                else: logger.info(f"P{current_player} finished Fantasy, did not repeat."); player_fantasies[current_player] = False; fantasy_cards_count[current_player] = 0
                            else:
                        # --- [ИСПРАВЛЕНИЕ v3.1] Конвертация JAX bool в Python bool ---
                                if bool(state.is_valid_fantasy_entry()):
                        # --- Конец исправления ---
                                    f_count = state.get_fantasy_cards_count()
                                    if f_count > 0: logger.info(f"P{current_player} QUALIFIES for Fantasy ({f_count} cards) after no actions!"); player_fantasies[current_player] = True; fantasy_cards_count[current_player] = f_count; player_finished[current_player] = False
                                    else: logger.warning(f"P{current_player} met fantasy entry but got 0 cards (no actions)?")
                        else: logger.debug(f"P{current_player} finished dead (no actions). No fantasy check."); player_fantasies[current_player] = False; fantasy_cards_count[current_player] = 0
                        current_player = opponent; continue
                    else: logger.error(f"No actions for P{current_player} in non-terminal state!"); return None
                node = self.get_node(info_set, num_actions); action_index = -1; sampling_prob = 1.0 / num_actions
                if node is not None:
                    current_reach = reach_p0 if current_player == 0 else reach_p1; strategy = node.get_strategy(current_reach)
                    if strategy.shape[0] == num_actions and node.num_actions > 0 and abs(jnp.sum(strategy) - 1.0) < 1e-5: key, subkey = random.split(key); action_index = int(np.random.choice(np.arange(num_actions), p=np.array(strategy))); sampling_prob = strategy[action_index]
                    else: logger.warning(f"Strategy shape/sum mismatch ({strategy.shape[0]} vs {num_actions}, sum={jnp.sum(strategy):.4f}). Using baseline."); action_index = self._get_best_action_baseline(state, actions_jax)
                else: logger.warning(f"Node error/limit for node {hash(info_set)}. Using baseline."); action_index = self._get_best_action_baseline(state, actions_jax)
                if action_index < 0 or action_index >= num_actions: logger.error(f"Invalid action index {action_index}."); return None
                chosen_action_jax = actions_jax[action_index]; action_dict = action_from_array(chosen_action_jax)
                if not action_dict: logger.error(f"Invalid action_dict from index {action_index}."); return None
                trajectory['states'].append((hash(info_set), current_player, num_actions)); trajectory['actions'].append(action_index); trajectory['reach_probs'].append((reach_p0, reach_p1)); trajectory['sampling_probs'].append(max(float(sampling_prob), 1e-9))
                if current_player == 0: reach_p0 *= sampling_prob
                else: reach_p1 *= sampling_prob
                try:
                    game_states[current_player] = state.apply_action(action_dict); new_state = game_states[current_player]; opp_state = game_states[opponent]
                    new_state.opponent_board = opp_state.board; new_state.opponent_discarded = opp_state.discarded_cards; opp_state.opponent_board = new_state.board; opp_state.opponent_discarded = new_state.discarded_cards
                except Exception as e: logger.exception(f"Error applying action {action_index} ({action_dict}): {e}"); return None
                current_state_after_action = game_states[current_player]
                if current_state_after_action.is_terminal():
                    logger.debug(f"P{current_player} finished board after action {action_index}."); player_finished[current_player] = True; current_place_jax = current_state_after_action.board.to_jax_placement()
                    # --- [ИСПРАВЛЕНИЕ v3.1] Конвертация JAX bool в Python bool ---
                    if not bool(is_dead_hand_jax(current_place_jax)):
                        if is_fantasy_turn_now:
                            if bool(current_state_after_action.is_valid_fantasy_repeat()):
                    # --- Конец исправления ---
                                f_repeat_count = 14; logger.info(f"P{current_player} REPEATS Fantasy ({f_repeat_count} cards) after action!"); player_fantasies[current_player] = True; fantasy_cards_count[current_player] = f_repeat_count; player_finished[current_player] = False
                            else: logger.info(f"P{current_player} finished Fantasy, did not repeat."); player_fantasies[current_player] = False; fantasy_cards_count[current_player] = 0
                        else:
                    # --- [ИСПРАВЛЕНИЕ v3.1] Конвертация JAX bool в Python bool ---
                            if bool(current_state_after_action.is_valid_fantasy_entry()):
                    # --- Конец исправления ---
                                f_count = current_state_after_action.get_fantasy_cards_count()
                                if f_count > 0: logger.info(f"P{current_player} QUALIFIES for Fantasy ({f_count} cards) after action!"); player_fantasies[current_player] = True; fantasy_cards_count[current_player] = f_count; player_finished[current_player] = False
                                else: logger.warning(f"P{current_player} met fantasy entry but got 0 cards (after action)?")
                    else: logger.debug(f"P{current_player} finished dead after action. No fantasy check."); player_fantasies[current_player] = False; fantasy_cards_count[current_player] = 0
                current_player = opponent
            logger.debug("Game simulation loop finished.")
            game_states[0].opponent_board = game_states[1].board; game_states[0].opponent_discarded = game_states[1].discarded_cards; game_states[1].opponent_board = game_states[0].board; game_states[1].opponent_discarded = game_states[0].discarded_cards
            final_payoff_p0 = 0
            if game_states[0].is_terminal() and game_states[1].is_terminal(): final_payoff_p0 = game_states[0].get_payoff()
            else: logger.error(f"Game ended but boards not terminal? P0 Full: {game_states[0].is_terminal()}, P1 Full: {game_states[1].is_terminal()}")
            trajectory['final_payoff'] = final_payoff_p0; game_end_time = time.time(); logger.debug(f"Game simulation took {game_end_time - game_start_time:.4f}s. Final Payoff P0: {final_payoff_p0}")
            return trajectory
        except Exception as e: logger.exception(f"Unhandled error during game simulation: {e}"); return None
    def _update_strategy_from_trajectory(self, trajectory: Dict):
        final_payoff_p0 = trajectory['final_payoff']; num_steps = len(trajectory['states'])
        if num_steps == 0: logger.warning("Empty trajectory update."); return
        for t in range(num_steps):
            info_hash, player, num_actions = trajectory['states'][t]; action_taken_index = trajectory['actions'][t]; reach_p0, reach_p1 = trajectory['reach_probs'][t]; sampling_prob = trajectory['sampling_probs'][t]
            node = self.nodes_map.get(info_hash)
            if node is None: continue
            if node.num_actions == 0: logger.warning(f"Node {info_hash} has 0 actions."); continue
            if node.num_actions != num_actions: logger.error(f"Action count mismatch node {info_hash} ({node.num_actions} vs {num_actions})."); continue
            if action_taken_index < 0 or action_taken_index >= num_actions: logger.error(f"Invalid action index {action_taken_index} for node {info_hash}."); continue
            if sampling_prob < 1e-9: logger.warning(f"Near-zero sampling prob ({sampling_prob:.2E}) for node {info_hash}."); continue
            payoff_for_player = final_payoff_p0 if player == 0 else -final_payoff_p0; reach_opponent = reach_p1 if player == 0 else reach_p0
            update_weight = payoff_for_player * (reach_opponent / sampling_prob)
            # Используем get_strategy для получения текущей стратегии (она нормализована)
            current_strategy = node.get_strategy(0.0) # Не обновляем strategy_sum здесь
            indicator = jnp.zeros(num_actions, dtype=jnp.float32).at[action_taken_index].set(1.0); regret_update = update_weight * (indicator - current_strategy)
            node.regret_sum = node.regret_sum + regret_update
            # Обновление strategy_sum происходит в get_strategy при выборе хода в симуляции
    def save_progress(self, iterations_completed: int) -> None:
        if 'save_ai_progress_to_github' not in globals(): logger.error("github_utils not available."); return
        logger.info(f"Preparing to save progress. Nodes: {len(self.nodes_map)}, Iterations: {iterations_completed}")
        try:
            serializable_nodes = {}
            for h, n in self.nodes_map.items():
                 regret_list = n.regret_sum.tolist() if n.num_actions > 0 else []; strategy_list = n.strategy_sum.tolist() if n.num_actions > 0 else []
                 serializable_nodes[h] = {"regret_sum": regret_list, "strategy_sum": strategy_list, "num_actions": n.num_actions}
            data_to_save = {"nodes_map_serialized": serializable_nodes, "iterations_completed": iterations_completed, "ai_settings": self.ai_settings, "timestamp": time.time()}
            logger.info(f"Data prepared ({len(serializable_nodes)} nodes). Calling github_utils...")
            if not save_ai_progress_to_github(data_to_save): logger.error("Saving progress via github_utils failed!")
            else: logger.info("Data passed to github_utils for saving.")
        except Exception as e: logger.exception(f"Unexpected error during save_progress preparation: {e}")
    def load_progress(self) -> None:
        if 'load_ai_progress_from_github' not in globals(): logger.error("github_utils not available."); return
        logger.info("Attempting to load AI progress from GitHub...")
        try:
            loaded_data = load_ai_progress_from_github()
            if loaded_data and isinstance(loaded_data, dict) and "nodes_map_serialized" in loaded_data:
                loaded_nodes_map_serialized = loaded_data["nodes_map_serialized"]; loaded_ai_settings = loaded_data.get("ai_settings", {}); iterations_completed = loaded_data.get("iterations_completed", 0); timestamp = loaded_data.get("timestamp", 0)
                logger.info(f"Loaded data. Nodes: {len(loaded_nodes_map_serialized)}, Iterations: {iterations_completed}.")
                if timestamp > 0: logger.info(f"Data timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))}")
                if loaded_ai_settings != self.ai_settings: logger.warning("Loaded AI settings differ from current agent settings!"); logger.warning(f"--- Loaded: {loaded_ai_settings}"); logger.warning(f"--- Current: {self.ai_settings}")
                self.nodes_map.clear(); num_loaded = 0; num_errors = 0; num_skipped_max_nodes = 0
                for info_hash_str, node_data in loaded_nodes_map_serialized.items():
                    if len(self.nodes_map) >= self.max_nodes: num_skipped_max_nodes += 1; continue
                    try:
                        info_hash = int(info_hash_str); num_actions = node_data.get("num_actions"); regret_sum_list = node_data.get("regret_sum"); strategy_sum_list = node_data.get("strategy_sum")
                        if not isinstance(num_actions, int) or num_actions < 0: logger.warning(f"Invalid num_actions ({num_actions}) for {info_hash}. Skip."); num_errors += 1; continue
                        if not isinstance(regret_sum_list, list) or not isinstance(strategy_sum_list, list): logger.warning(f"Invalid sum type for {info_hash}. Skip."); num_errors += 1; continue
                        if num_actions > 0 and (len(regret_sum_list) != num_actions or len(strategy_sum_list) != num_actions): logger.warning(f"Length mismatch for {info_hash}. NA={num_actions}, RL={len(regret_sum_list)}, SL={len(strategy_sum_list)}. Skip."); num_errors += 1; continue
                        node = CFRNode(num_actions)
                        if num_actions > 0:
                            try: node.regret_sum = jnp.array(regret_sum_list, dtype=jnp.float32); node.strategy_sum = jnp.array(strategy_sum_list, dtype=jnp.float32)
                            except (ValueError, TypeError) as arr_err: logger.error(f"Error converting arrays for {info_hash}: {arr_err}. Skip."); num_errors += 1; continue
                        self.nodes_map[info_hash] = node; num_loaded += 1
                    except (ValueError, KeyError, TypeError) as e: logger.exception(f"Error processing node '{info_hash_str}': {e}. Skip."); num_errors += 1
                logger.info(f"Successfully loaded {num_loaded} nodes.")
                if num_errors > 0: logger.warning(f"Skipped {num_errors} nodes due to errors.")
                if num_skipped_max_nodes > 0: logger.warning(f"Skipped {num_skipped_max_nodes} nodes due to max_nodes limit ({self.max_nodes}).")
            else: logger.warning("Failed to load progress or data invalid/empty. Starting fresh.")
        except Exception as e: logger.exception(f"Unexpected error during load_progress: {e}")
    def check_convergence(self) -> bool:
        if not self.nodes_map: logger.info("Convergence check: No nodes."); return False
        total_abs_regret = 0.0; total_actions_in_nodes = 0; num_nodes_checked = 0
        for node in self.nodes_map.values():
            if node.num_actions > 0:
                if node.regret_sum.shape == (node.num_actions,): total_abs_regret += float(jnp.sum(jnp.abs(node.regret_sum))); total_actions_in_nodes += node.num_actions; num_nodes_checked += 1
                else: logger.warning(f"Regret sum shape mismatch for node {hash}: expected ({node.num_actions},), got {node.regret_sum.shape}. Skipping.");
        if total_actions_in_nodes == 0: logger.info("Convergence check: No actions in nodes."); return False
        # Добавим защиту от деления на ноль
        avg_abs_regret_per_action = total_abs_regret / total_actions_in_nodes if total_actions_in_nodes > 0 else 0.0
        logger.info(f"Convergence check: Avg absolute regret per action = {avg_abs_regret_per_action:.6f} (Threshold: {self.stop_threshold}) over {num_nodes_checked} nodes.")
        is_converged = avg_abs_regret_per_action < self.stop_threshold
        if is_converged: logger.info("Convergence threshold reached!")
        return is_converged

# --- Конец файла ai_engine_v3.1_padding_fix.py ---
