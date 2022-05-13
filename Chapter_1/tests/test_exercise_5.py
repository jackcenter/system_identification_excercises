import pytest

from Chapter_1 import exercise_5

def test_exercise_5ab():
    simulations = 10000
    results_combined = exercise_5.run_exercise_5a(simulations, False)
    assert len(results_combined) == simulations
    assert exercise_5.run_exercise_5b(results_combined) == 0

# TODO: write tests for structures 
