from abc import ABC, abstractmethod
from typing import Iterable, Optional


class BaseDataStructure(ABC):
    """
    Abstract base class for trees that store strings.

    Subclasses must implement the following methods:
        - insert(value: str) -> bool: Insert a value if not already present.
        - contains(value: str) -> bool: Check if a value exists.
        - delete(value: str) -> bool: Remove a value if present.
    """

    def __init__(self, values: Optional[Iterable[str]] = None):
        """
        Optional helper: concrete subclasses may call super().__init__(values)
        to allow bulk initialization via an iterable of strings.
        """
        self.rotations_insert = 0
        self.rotations_delete = 0
        if values:
            for v in values:
                self.insert(v)

    @property
    def total_rotations(self) -> int:
        """Total number of rotations performed (insert + delete)."""
        return self.rotations_insert + self.rotations_delete

    def reset_metrics(self) -> None:
        """Reset all rotation metrics."""
        self.rotations_insert = 0
        self.rotations_delete = 0

    @abstractmethod
    def insert(self, value: str) -> bool:
        """
        Insert a value into the tree.

        Parameters
        ----------
        value: str
            The value to insert.

        Returns
        -------
        bool
            True if insertion succeeded or was acknowledged,
                  False if insertion failed (e.g., structure is full).
        """
        raise NotImplementedError

    @abstractmethod
    def contains(self, value: str) -> bool:
        """
        Check whether a value exists in the tree.

        Parameters
        ----------
        value: str
            The value to check.

        Returns
        -------
        bool
            True if `value` is present.
        """
        raise NotImplementedError

    def __contains__(self, value: str) -> bool:
        """
        Enable `value in dataStructure` syntax by delegating to `contains()`.

        Parameters
        ----------
        value: str
            The value to check for membership.

        Returns
        -------
        bool
            True if `value` is present, False otherwise.
        """
        return self.contains(value)

    @abstractmethod
    def delete(self, value: str) -> bool:
        """
        Remove a value from the tree.

        Parameters
        ----------
        value: str
            The value to delete.

        Returns
        -------
        bool
            True if the value was removed, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def validate(self, value: str) -> bool:
        """
        Validate tree-constraint invariant.

        Raises
        ------
        AssertionError
            If any tree property is violated.
        """
        raise NotImplementedError
