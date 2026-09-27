"""
橋牌 RAG 專案版本共用工具模組。
提供手牌分析、叫牌歷史格式化、合約描述等功能。
"""

from typing import List, Dict, Tuple, Optional, Any

SUIT_NAMES = {'S': 'Spades', 'H': 'Hearts', 'D': 'Diamonds', 'C': 'Clubs', 'N': 'No Trump'}
SEAT_NAMES = {'N': 'North', 'E': 'East', 'S': 'South', 'W': 'West'}

def count_hcp(hand: List[str]) -> int:
    """計算高牌點 (A=4, K=3, Q=2, J=1)
    
    參數:
        hand: 手牌列表，例如 ['SA', 'HK']
        
    回傳:
        高牌點總和
    """
    pts = {'A': 4, 'K': 3, 'Q': 2, 'J': 1}
    return sum(pts.get(card[1], 0) for card in hand)

def hand_shape(hand: List[str]) -> Tuple[int, int, int, int]:
    """回傳牌型 (S, H, D, C) 各花色張數
    
    參數:
        hand: 手牌列表
        
    回傳:
        四元組 (黑桃張數, 紅心張數, 方塊張數, 梅花張數)
    """
    counts = {'S': 0, 'H': 0, 'D': 0, 'C': 0}
    for card in hand:
        counts[card[0]] += 1
    return (counts['S'], counts['H'], counts['D'], counts['C'])

def is_balanced(shape: Tuple[int, int, int, int]) -> bool:
    """是否平均型 (4-3-3-3, 4-4-3-2, 5-3-3-2)
    
    參數:
        shape: 牌型元組
        
    回傳:
        如果是平均型則回傳 True，否則回傳 False
    """
    sorted_shape = sorted(list(shape), reverse=True)
    return sorted_shape in [[4, 3, 3, 3], [4, 4, 3, 2], [5, 3, 3, 2]]

def describe_hand(hand: List[str]) -> str:
    """將手牌描述成自然語言。
    
    hand: ['SA', 'SK', 'HQ', 'HJ', 'H9', 'DK', 'D8', 'D5', 'CA', 'CJ', 'C7', 'C4', 'C2']
    回傳: '18 HCP, 5-3-3-2 shape. Spades: AK. Hearts: QJ9. Diamonds: K85. Clubs: AJ742.'
    """
    hcp = count_hcp(hand)
    shape = hand_shape(hand)
    sorted_shape = sorted(list(shape), reverse=True)
    shape_str = f"{sorted_shape[0]}-{sorted_shape[1]}-{sorted_shape[2]}-{sorted_shape[3]}"
    
    suits = {'S': [], 'H': [], 'D': [], 'C': []}
    for card in hand:
        if card[0] in suits:
            suits[card[0]].append(card[1])
        
    def format_suit(suit_cards: List[str]) -> str:
        # 為了正確排序：A K Q J T 9 8 7 6 5 4 3 2
        order = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10, '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2}
        sorted_cards = sorted(suit_cards, key=lambda x: order.get(x, 0), reverse=True)
        return "".join(sorted_cards) if sorted_cards else "void"
        
    spades = format_suit(suits['S'])
    hearts = format_suit(suits['H'])
    diamonds = format_suit(suits['D'])
    clubs = format_suit(suits['C'])
    
    return f"{hcp} HCP, {shape_str} shape. Spades: {spades}. Hearts: {hearts}. Diamonds: {diamonds}. Clubs: {clubs}."

def format_auction(auction: List[Dict[str, str]], dealer: str) -> str:
    """將叫牌歷史格式化成人類易讀的形式。
    
    auction: [{'seat':'N','bid':'1C'}, {'seat':'E','bid':'P'}, {'seat':'S','bid':'1H'}]
    dealer: 'N'
    回傳: 'N: 1C - E: Pass - S: 1H - W: ?'
    """
    seats = ['N', 'E', 'S', 'W']
    try:
        current_idx = seats.index(dealer)
    except ValueError:
        current_idx = 0
        
    parts = []
    for call in auction:
        seat = call.get('seat', '')
        bid = call.get('bid', '')
        if bid == 'P':
            bid_str = 'Pass'
        elif bid == 'X':
            bid_str = 'Double'
        elif bid == 'XX':
            bid_str = 'Redouble'
        else:
            bid_str = bid
        parts.append(f"{seat}: {bid_str}")
        
        try:
            current_idx = (seats.index(seat) + 1) % 4
        except ValueError:
            pass
        
    next_seat = seats[current_idx]
    parts.append(f"{next_seat}: ?")
    
    return " - ".join(parts)

def format_contract(contract: Dict[str, Any]) -> str:
    """將合約格式化。
    
    contract: {'level': 3, 'strain': 'N', 'declarer': 'S', 'doubled': 0}
    回傳: '3NT by South, undoubled'
    """
    level = contract.get('level')
    strain = contract.get('strain')
    declarer = contract.get('declarer')
    doubled = contract.get('doubled', 0)
    
    if level == 0 or strain is None:
        return "Passed Out"
        
    strain_str = "NT" if strain == 'N' else strain
    declarer_str = SEAT_NAMES.get(declarer, declarer)
    
    if doubled == 1:
        double_str = "doubled"
    elif doubled == 2:
        double_str = "redoubled"
    else:
        double_str = "undoubled"
        
    return f"{level}{strain_str} by {declarer_str}, {double_str}"

def format_current_trick(current_trick: Dict[str, Any]) -> str:
    """格式化當前墩。
    
    current_trick: {'leader': 'W', 'cards': [{'seat': 'W', 'card': 'H3'}, {'seat': 'N', 'card': 'HK'}]}
    回傳: 'Led by West: H3, N: HK, (waiting for E, S)'
    """
    leader = current_trick.get('leader')
    cards = current_trick.get('cards', [])
    
    if not leader or not cards:
        return "New trick"
        
    leader_str = SEAT_NAMES.get(leader, leader)
    
    parts = []
    for idx, play in enumerate(cards):
        seat = play.get('seat', '')
        card = play.get('card', '')
        if idx == 0:
            parts.append(f"Led by {leader_str}: {card}")
        else:
            parts.append(f"{seat}: {card}")
            
    seats = ['N', 'E', 'S', 'W']
    try:
        leader_idx = seats.index(leader)
    except ValueError:
        leader_idx = 0
        
    played_seats = [play.get('seat') for play in cards]
    waiting_for = []
    
    for i in range(4):
        seat = seats[(leader_idx + i) % 4]
        if seat not in played_seats:
            waiting_for.append(seat)
            
    if waiting_for:
        parts.append(f"(waiting for {', '.join(waiting_for)})")
        
    return ", ".join(parts)
