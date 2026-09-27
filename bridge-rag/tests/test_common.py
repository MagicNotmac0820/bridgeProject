"""versions/common.py:手牌特徵與局面格式化。"""

from versions.common import (count_hcp, describe_hand, format_auction, format_contract,
                             format_current_trick, hand_shape, is_balanced)


def test_count_hcp():
    assert count_hcp(['SA', 'HK', 'DQ', 'CJ', 'C2']) == 10
    assert count_hcp(['S2', 'H3', 'D4', 'C5']) == 0


def test_hand_shape():
    assert hand_shape(['SA', 'S2', 'HK', 'H2', 'DQ', 'D2', 'CJ', 'C2']) == (2, 2, 2, 2)
    assert hand_shape(['SA', 'SK', 'SQ', 'SJ', 'ST']) == (5, 0, 0, 0)


def test_is_balanced():
    assert is_balanced((4, 3, 3, 3))
    assert is_balanced((4, 4, 3, 2))
    assert is_balanced((3, 2, 5, 3))
    assert not is_balanced((5, 4, 2, 2))
    assert not is_balanced((6, 3, 2, 2))


def test_describe_hand():
    hand = ['SA', 'SK', 'HQ', 'HJ', 'H9', 'DK', 'D8', 'D5', 'CA', 'CJ', 'C7', 'C4', 'C2']
    desc = describe_hand(hand)
    assert "18 HCP, 5-3-3-2 shape" in desc
    assert "Spades: AK." in desc and "Hearts: QJ9." in desc
    assert "Diamonds: K85." in desc and "Clubs: AJ742." in desc


def test_describe_hand_void():
    hand = ['SA', 'SK', 'SQ', 'SJ', 'ST', 'S9', 'S8', 'HA', 'HK', 'HQ', 'DA', 'DK', 'DQ']
    assert "Clubs: void." in describe_hand(hand)


def test_format_auction():
    auction = [{'seat': 'N', 'bid': '1C'}, {'seat': 'E', 'bid': 'P'}, {'seat': 'S', 'bid': '1H'}]
    assert format_auction(auction, 'N') == "N: 1C - E: Pass - S: 1H - W: ?"
    assert format_auction([], 'E') == "E: ?"


def test_format_contract():
    assert format_contract({'level': 3, 'strain': 'N', 'declarer': 'S', 'doubled': 0}) == "3NT by South, undoubled"
    assert format_contract({'level': 4, 'strain': 'S', 'declarer': 'E', 'doubled': 1}) == "4S by East, doubled"


def test_format_current_trick():
    trick = {'leader': 'W', 'cards': [{'seat': 'W', 'card': 'H3'}, {'seat': 'N', 'card': 'HK'}]}
    assert format_current_trick(trick) == "Led by West: H3, N: HK, (waiting for E, S)"
    assert format_current_trick({'leader': 'N', 'cards': []}) == "New trick"
