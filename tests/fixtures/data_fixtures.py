"""
Test data fixtures for ShedBoxAI processor testing.

This module provides reusable test data fixtures for various testing scenarios.
"""

from typing import Any, Dict

import pandas as pd
import pytest


@pytest.fixture
def sample_users_data():
    """Sample user DataFrame for testing."""
    return pd.DataFrame(
        {
            "name": ["John", "Jane", "Bob", "Alice", "Charlie"],
            "age": [25, 30, 35, 28, 45],
            "city": ["NYC", "LA", "Chicago", "Boston", "Seattle"],
            "salary": [50000, 75000, 90000, 65000, 120000],
            "department": [
                "Engineering",
                "Sales",
                "Engineering",
                "Marketing",
                "Engineering",
            ],
        }
    )


@pytest.fixture
def sample_products_data():
    """Sample product DataFrame for testing."""
    return pd.DataFrame(
        {
            "product_id": [1, 2, 3, 4, 5],
            "name": ["Laptop", "Mouse", "Keyboard", "Monitor", "Headphones"],
            "price": [999.99, 25.99, 79.99, 299.99, 149.99],
            "category": [
                "Electronics",
                "Accessories",
                "Accessories",
                "Electronics",
                "Audio",
            ],
            "in_stock": [True, True, False, True, True],
            "rating": [4.5, 4.2, 4.8, 4.1, 4.6],
        }
    )


@pytest.fixture
def sample_multi_source_data(sample_users_data, sample_products_data):
    """Multi-source test data dictionary."""
    return {"users": sample_users_data, "products": sample_products_data}


@pytest.fixture
def large_dataset():
    """Larger dataset for performance testing."""
    import numpy as np

    np.random.seed(42)  # For reproducible tests

    n_rows = 1000
    return pd.DataFrame(
        {
            "id": range(n_rows),
            "value": np.random.randint(1, 100, n_rows),
            "category": np.random.choice(["A", "B", "C", "D"], n_rows),
            "score": np.random.normal(50, 15, n_rows),
            "active": np.random.choice([True, False], n_rows),
        }
    )


@pytest.fixture
def empty_dataframe():
    """Empty DataFrame for edge case testing."""
    return pd.DataFrame()


@pytest.fixture
def single_row_data():
    """Single row DataFrame for edge case testing."""
    return pd.DataFrame({"id": [1], "name": ["Test"], "value": [100]})
