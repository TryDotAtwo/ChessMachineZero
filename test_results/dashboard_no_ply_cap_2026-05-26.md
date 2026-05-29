# Dashboard No Forced Ply Cap 2026-05-26

## Scope

- change_id=dashboard_no_forced_ply_cap_v11
- user_request=Remove the half-move limit that forces dashboard games into a draw.
- behavior_change=Current Percepta dashboard/self-play runtime no longer sets adjudication_cap_reached from max_plies.
- retained_rules=checkmate, stalemate, fifty-move rule, threefold repetition, insufficient material.
- compatibility=Dashboard API still accepts max_plies for older callers, but the value is ignored by the current uncapped dashboard game loop and snapshot reports max_plies=null.

## TDD Evidence

- initial_test=`python -m pytest -p no:cacheprovider tests\test_dashboard.py::test_dashboard_selfplay_has_no_forced_ply_limit_draw -q` => failed because snapshot max_plies was 1 and the old cap path still existed.
- fixed_test=`python -m pytest -p no:cacheprovider tests\test_dashboard.py::test_dashboard_selfplay_has_no_forced_ply_limit_draw -q` => 1 passed.

## Verification

- dashboard_tests=`python -m pytest -p no:cacheprovider tests\test_dashboard.py -q` => 12 passed.
- full_pytest=`python -m pytest -p no:cacheprovider -q` => 145 passed.
- dashboard_warning_check=`python -m pytest -p no:cacheprovider -W error tests\test_dashboard.py -q` => 12 passed.
- full_warning_check=`python -m pytest -p no:cacheprovider -W error -q` => 145 passed.
- collected_tests=145.

## Boundary Check

- search=`rg -n "adjudication_cap_reached=self\.ply|self\.ply \+ 1 >= self\.max_plies|max_plies must be positive" src/chess_machine_zero/dashboard src/chess_machine_zero/model/percepta_parametric_selfplay.py tests/test_dashboard.py`
- result=no matches.

## Files

- updated=src/chess_machine_zero/model/percepta_parametric_selfplay.py
- updated=src/chess_machine_zero/dashboard/state.py
- updated=src/chess_machine_zero/dashboard/server.py
- updated=tests/test_dashboard.py
