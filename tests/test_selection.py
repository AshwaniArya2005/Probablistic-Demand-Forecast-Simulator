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


from selection import add_group, adopt_normalised, choose_floor_rel, simplest_within_rel


def test_relative_tie_prefers_the_simpler():
    assert simplest_within_rel(CFGS, [1.0100, 1.0040, 1.0000]) == "middle"      # simple is 1.0% behind (out), middle 0.4% (in)
    assert simplest_within_rel(CFGS, [1.0049, 1.0030, 1.0000]) == "simple"      # all within 0.5%
    assert simplest_within_rel(CFGS, [1.0600, 1.0300, 1.0000]) == "complex"


def test_floor_ties_go_to_the_default_only_if_it_is_tied():
    assert choose_floor_rel({1 / 28: 0.400, 1 / 14: 0.399, 1 / 7: 0.4001}) == 1 / 14       # all tied: default
    assert choose_floor_rel({1 / 28: 0.400, 1 / 14: 0.401, 1 / 7: 0.405}) == 1 / 14        # 1/28 and 1/14 tied: default
    assert choose_floor_rel({1 / 28: 0.410, 1 / 14: 0.404, 1 / 7: 0.400}) == 1 / 7         # default 0.9% behind: the best floor


def test_normalisation_needs_both_mean_and_median_to_win_by_one_percent():
    assert adopt_normalised(1.00, 0.50, 0.98, 0.49)                              # 2% better on both
    assert not adopt_normalised(1.00, 0.50, 0.98, 0.498)                         # median only 0.4% better
    assert not adopt_normalised(1.00, 0.50, 0.995, 0.45)                         # mean only 0.5% better
    assert not adopt_normalised(1.00, 0.50, 0.50, 0.505)                         # mean wins hugely (one series), median loses: stay raw


def test_group_added_only_for_a_half_percent_gain():
    assert add_group(1.000, 0.994) and not add_group(1.000, 0.996) and not add_group(1.000, 1.010)
