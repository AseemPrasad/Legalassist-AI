"""user_feedback needs indexes on the columns the dashboards filter and sort by."""
from sqlalchemy import create_engine, inspect

from db.base import Base
from db.models.feedback import UserFeedback


def _indexed_columns():
    engine = create_engine("sqlite://")
    # Only create the table under test; its FKs are not enforced on SQLite.
    Base.metadata.create_all(engine, tables=[UserFeedback.__table__])
    insp = inspect(engine)
    return [tuple(ix["column_names"]) for ix in insp.get_indexes("user_feedback")]


def test_created_at_is_indexed():
    assert ("created_at",) in _indexed_columns()


def test_user_id_created_at_composite_index():
    assert ("user_id", "created_at") in _indexed_columns()


def test_user_id_still_indexed():
    assert ("user_id",) in _indexed_columns()
