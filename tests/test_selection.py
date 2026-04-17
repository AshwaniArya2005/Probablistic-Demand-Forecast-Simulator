"""The pre-registered decision rules, checked on hand-picked numbers (not on project results)."""
import pytest

from selection import choose_floor, choose_id_arm, keep_year_ago, simplest_within

CFGS = ["simple", "middle", "complex"]      # listed simplest first


def test_simplest_within_tolerance():
    assert simplest_within(CFGS, [0.4100, 0.4090, 0.4080]) == "simple"       # all within 0.002 of the best: simplest wins
    assert simplest_within(CFGS, [0.4100, 0.4090, 0.4079]) == "middle"       # simple is 0.0021 behind: out; middle is 0.0011 behind: in
    assert simplest_within(CFGS, [0.4300, 0.4200, 0.4000]) == "complex"      # clearly best
    assert simplest_within(CFGS, [0.4000, 0.4300, 0.4200]) == "simple"       # simplest is also the best


def test_boundary_of_the_tolerance_is_inclusive():
    assert simplest_within(["a", "b"], [0.405, 0.403]) == "a"                # exactly 0.002 behind: still within


def test_floor_takes_the_lowest_wape_and_exact_ties_go_to_one_over_28():
    f = {1 / 56: 0.4102, 1 / 28: 0.4110, 1 / 14: 0.4150}
    assert choose_floor(f) == 1 / 56
    assert choose_floor({1 / 56: 0.4100, 1 / 28: 0.4100, 1 / 14: 0.4150}) == 1 / 28      # tie at 4 decimals
    assert choose_floor({1 / 56: 0.4100, 1 / 28: 0.4200, 1 / 14: 0.4100}) == 1 / 56      # tie not involving 1/28: the smaller floor


def test_year_ago_kept_only_for_a_gain_of_at_least_0_003():
    assert keep_year_ago(0.400, 0.403) and keep_year_ago(0.400, 0.410)
    assert not keep_year_ago(0.400, 0.4029) and not keep_year_ago(0.405, 0.400)


@pytest.mark.parametrize("w,expected", [
    ({"no_ids": 0.400, "no_item": 0.400, "all": 0.400}, "no_ids"),                   # no evidence for ids
    ({"no_ids": 0.405, "no_item": 0.401, "all": 0.410}, "no_item"),                   # no_item better than no_ids by 0.004; all is worse
    ({"no_ids": 0.410, "no_item": 0.409, "all": 0.400}, "all"),                       # all beats no_ids by 0.010, no_item only 0.001 behind no_ids
    ({"no_ids": 0.4030, "no_item": 0.4029, "all": 0.4000}, "all"),                    # all beats the chosen no_ids by exactly 0.003
    ({"no_ids": 0.4029, "no_item": 0.4029, "all": 0.4000}, "no_ids"),                 # all beats no_ids by only 0.0029
    ({"no_ids": 0.410, "no_item": 0.405, "all": 0.403}, "no_item"),                   # all beats no_item by 0.002 only: stay with fewer ids
])
def test_id_arm_rule(w, expected):
    assert choose_id_arm(w) == expected
