"""Benchmark task ladder v1.0.0.

Each TaskDef carries:
  - an immutable ProblemSpec (acceptance tests)
  - test source materialized by the Referee
  - a CANDIDATE POOL of plausible implementations.

Pool semantics (documented limitation): with no live LLM, the pool simulates a
generation distribution. Index 0 models a greedy sample and is deliberately
often FLAWED, later indices progressively more correct. The SAME seeded
permutation is shared by both treatments of a trial (paired design). Pools are
fixed at benchmark freeze; neither system sees them as privileged data --
agents only draw unseen entries in fixed order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

from forgemind.core.problem import AcceptanceTest, ProblemSpec


@dataclass(frozen=True)
class TaskDef:
    task_id: str
    version: str
    difficulty: int          # 1..4 in this ladder
    description: str
    module_path: str         # e.g. "solution.py"
    required_functions: tuple[str, ...]
    spec: ProblemSpec
    test_source: str
    candidate_pool: tuple[str, ...]

    def test_files(self) -> dict[str, str]:
        return {"__fm_acceptance_tests.py": self.test_source}


# --------------------------------------------------------------------------
# helpers to build specs tersely
# --------------------------------------------------------------------------

def _spec(task_id, description, functions, tests, difficulty_note=""):
    return ProblemSpec(
        problem_id=task_id,
        description=description,
        requirements=tuple(f"implement {f}" for f in functions),
        acceptance_tests=tuple(AcceptanceTest(t.split("::")[-1], t) for t in tests),
        interface_constraints={f: f"function: {f}" for f in functions},
    )


def _wrap(body: str, fname: str = "f") -> str:
    return body


# ==========================================================================
# LEVEL 1 -- simple functions
# ==========================================================================

_L1A_TESTS = '''
def test_basic():
    from solution import reverse_string
    assert reverse_string("abc") == "cba"

def test_empty():
    from solution import reverse_string
    assert reverse_string("") == ""

def test_palindrome():
    from solution import reverse_string
    assert reverse_string("abba") == "abba"

def test_unicode():
    from solution import reverse_string
    assert reverse_string("héllo") == "olléh"
'''

_L1A_POOL = (
    # greedy draw: subtly wrong (drops last char handling via slice off-by-one)
    "def reverse_string(s):\n    return s[-1] + s[:-1]\n",
    # wrong: reverses words not chars
    "def reverse_string(s):\n    return ' '.join(w[::-1] for w in s.split(' '))\n",
    # correct
    "def reverse_string(s):\n    return s[::-1]\n",
)

task_l1a = TaskDef(
    task_id="L1A_reverse_string", version="1.0.0", difficulty=1,
    description="Reverse a string exactly.",
    module_path="solution.py",
    required_functions=("reverse_string",),
    spec=_spec("L1A_reverse_string", "reverse a string",
               ("reverse_string",),
               ("test_basic", "test_empty", "test_palindrome", "test_unicode")),
    test_source=_L1A_TESTS,
    candidate_pool=_L1A_POOL,
)


_L1B_TESTS = '''
def test_positive():
    from solution import is_palindrome_number
    assert is_palindrome_number(121) is True

def test_negative():
    from solution import is_palindrome_number
    assert is_palindrome_number(-121) is False

def test_single_digit():
    from solution import is_palindrome_number
    assert is_palindrome_number(7) is True

def test_not_palindrome():
    from solution import is_palindrome_number
    assert is_palindrome_number(10) is False
'''

_L1B_POOL = (
    # greedy: negative handled wrong (string of -121 reversed == '121-', but
    # author forgot negatives -> actually returns True for -121? no: '-121'[::-1]
    # == '121-' != '-121', so False. This one is CORRECT-by-accident.
    "def is_palindrome_number(n):\n    return str(n) == str(n)[::-1]\n",
    # wrong: ignores sign so -121 passes
    "def is_palindrome_number(n):\n    s = str(abs(n))\n    return s == s[::-1]\n",
    # correct explicit
    "def is_palindrome_number(n):\n    if n < 0:\n        return False\n"
    "    s = str(n)\n    return s == s[::-1]\n",
)

task_l1b = TaskDef(
    task_id="L1B_palindrome_number", version="1.0.0", difficulty=1,
    description="Palindrome integer check; negatives are not palindromes.",
    module_path="solution.py",
    required_functions=("is_palindrome_number",),
    spec=_spec("L1B_palindrome_number", "palindrome number",
               ("is_palindrome_number",),
               ("test_positive", "test_negative", "test_single_digit",
                "test_not_palindrome")),
    test_source=_L1B_TESTS,
    candidate_pool=_L1B_POOL,
)


# ==========================================================================
# LEVEL 2 -- algorithms
# ==========================================================================

_L2A_TESTS = '''
def test_sorted_basic():
    from solution import binary_search
    assert binary_search([1,3,5,7,9], 5) == 2

def test_first():
    from solution import binary_search
    assert binary_search([1,3,5], 1) == 0

def test_missing():
    from solution import binary_search
    assert binary_search([1,3,5], 4) == -1

def test_empty():
    from solution import binary_search
    assert binary_search([], 1) == -1

def test_large():
    from solution import binary_search
    xs = list(range(0, 20000, 2))
    assert binary_search(xs, 13332) == 6666
    assert binary_search(xs, 13331) == -1
'''

_L2A_POOL = (
    # greedy: classic mid calculation overflow-style bug ported: uses
    # (lo+hi)//2 (fine in py) BUT loop condition wrong -> misses first elem
    "def binary_search(xs, target):\n"
    "    lo, hi = 0, len(xs) - 1\n"
    "    while lo < hi:\n"
    "        mid = (lo + hi) // 2\n"
    "        if xs[mid] == target:\n            return mid\n"
    "        elif xs[mid] < target:\n            lo = mid + 1\n"
    "        else:\n            hi = mid - 1\n"
    "    return -1\n",
    # wrong: linear scan fallback disguised (correct but O(n); still passes)
    "def binary_search(xs, target):\n"
    "    for i, x in enumerate(xs):\n"
    "        if x == target:\n            return i\n"
    "    return -1\n",
    # correct proper bisection
    "def binary_search(xs, target):\n"
    "    lo, hi = 0, len(xs)\n"
    "    while lo < hi:\n"
    "        mid = (lo + hi) // 2\n"
    "        if xs[mid] == target:\n            return mid\n"
    "        elif xs[mid] < target:\n            lo = mid + 1\n"
    "        else:\n            hi = mid\n"
    "    return -1\n",
)

task_l2a = TaskDef(
    task_id="L2A_binary_search", version="1.0.0", difficulty=2,
    description="Binary search on sorted list; -1 if absent.",
    module_path="solution.py",
    required_functions=("binary_search",),
    spec=_spec("L2A_binary_search", "binary search",
               ("binary_search",),
               ("test_sorted_basic", "test_first", "test_missing",
                "test_empty", "test_large")),
    test_source=_L2A_TESTS,
    candidate_pool=_L2A_POOL,
)


_L2B_TESTS = '''
def test_fib_small():
    from solution import fib
    assert [fib(i) for i in range(7)] == [0, 1, 1, 2, 3, 5, 8]

def test_fib_ten():
    from solution import fib
    assert fib(10) == 55

def test_fib_twenty():
    from solution import fib
    assert fib(20) == 6765

def test_fib_zero():
    from solution import fib
    assert fib(0) == 0
'''

_L2B_POOL = (
    # greedy: exponential recursion without memo -> times out on n=20? no,
    # fib(20) recursion is fine (~10^4 calls). But make it WRONG base:
    "def fib(n):\n    if n <= 1:\n        return 1\n"
    "    return fib(n-1) + fib(n-2)\n",   # fib(0)=1 wrong
    # wrong: off-by-one iteration
    "def fib(n):\n    a, b = 1, 1\n    for _ in range(n):\n        a, b = b, a+b\n    return a\n",
    # correct iterative
    "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a+b\n    return a\n",
)

task_l2b = TaskDef(
    task_id="L2B_fibonacci", version="1.0.0", difficulty=2,
    description="fib(n) with fib(0)=0.",
    module_path="solution.py",
    required_functions=("fib",),
    spec=_spec("L2B_fibonacci", "fibonacci", ("fib",),
               ("test_fib_small", "test_fib_ten", "test_fib_twenty",
                "test_fib_zero")),
    test_source=_L2B_TESTS,
    candidate_pool=_L2B_POOL,
)


_L2C_TESTS = '''
def test_mixed():
    from solution import roman_to_int
    assert roman_to_int("III") == 3
    assert roman_to_int("IV") == 4
    assert roman_to_int("IX") == 9
    assert roman_to_int("LVIII") == 58
    assert roman_to_int("MCMXCIV") == 1994

def test_single():
    from solution import roman_to_int
    assert roman_to_int("M") == 1000
'''

_L2C_POOL = (
    # greedy: ignores subtractive notation
    "def roman_to_int(s):\n"
    "    vals = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}\n"
    "    return sum(vals[c] for c in s)\n",
    # wrong: subtracts whenever smaller precedes larger, double-subtracts
    "def roman_to_int(s):\n"
    "    vals = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}\n"
    "    total = 0\n"
    "    for i, c in enumerate(s):\n"
    "        v = vals[c]\n"
    "        if i+1 < len(s) and vals[s[i+1]] > v:\n            total -= v\n"
    "        else:\n            total += v\n"
    "    return total\n",
    # correct standard
    "def roman_to_int(s):\n"
    "    vals = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}\n"
    "    total = 0\n"
    "    for i in range(len(s)):\n"
    "        if i+1 < len(s) and vals[s[i]] < vals[s[i+1]]:\n"
    "            total -= vals[s[i]]\n"
    "        else:\n            total += vals[s[i]]\n"
    "    return total\n",
)

task_l2c = TaskDef(
    task_id="L2C_roman_numerals", version="1.0.0", difficulty=2,
    description="Convert Roman numerals to integers (subtractive notation).",
    module_path="solution.py",
    required_functions=("roman_to_int",),
    spec=_spec("L2C_roman_numerals", "roman numerals",
               ("roman_to_int",), ("test_mixed", "test_single")),
    test_source=_L2C_TESTS,
    candidate_pool=_L2C_POOL,
)


# ==========================================================================
# LEVEL 3 -- stateful / multi-function / invariants
# ==========================================================================

_L3A_TESTS = '''
def _m():
    from solution import BankAccount
    return BankAccount(100)

def test_deposit_withdraw():
    acc = _m()
    acc.deposit(50)
    acc.withdraw(30)
    assert acc.balance == 120

def test_overdraft():
    acc = _m()
    try:
        acc.withdraw(200)
        raised = False
    except ValueError:
        raised = True
    assert raised and acc.balance == 100

def test_negative_deposit():
    acc = _m()
    try:
        acc.deposit(-5)
        raised = False
    except ValueError:
        raised = True
    assert raised

def test_history():
    acc = _m()
    acc.deposit(10)
    acc.withdraw(5)
    assert len(acc.history()) == 2
'''

_L3A_POOL = (
    # greedy: no validation, no history
    "class BankAccount:\n"
    "    def __init__(self, balance=0):\n        self._b = balance\n"
    "    @property\n    def balance(self):\n        return self._b\n"
    "    def deposit(self, amt):\n        self._b += amt\n"
    "    def withdraw(self, amt):\n        self._b -= amt\n"
    "    def history(self):\n        return []\n",
    # wrong: validates but records deposits twice, missing withdraw guard order
    "class BankAccount:\n"
    "    def __init__(self, balance=0):\n        self._b = balance\n"
    "        self._h = []\n"
    "    @property\n    def balance(self):\n        return self._b\n"
    "    def deposit(self, amt):\n"
    "        if amt <= 0:\n            raise ValueError('bad')\n"
    "        self._b += amt\n"
    "        self._h.append(('dep', amt))\n"
    "        self._h.append(('dup', amt))\n"
    "    def withdraw(self, amt):\n        self._b -= amt\n"
    "        self._h.append(('wd', amt))\n"
    "    def history(self):\n        return list(self._h)\n",
    # correct
    "class BankAccount:\n"
    "    def __init__(self, balance=0):\n"
    "        if balance < 0:\n            raise ValueError('negative start')\n"
    "        self._b = balance\n        self._h = []\n"
    "    @property\n    def balance(self):\n        return self._b\n"
    "    def deposit(self, amt):\n"
    "        if amt <= 0:\n            raise ValueError('must be positive')\n"
    "        self._b += amt\n        self._h.append(('deposit', amt))\n"
    "    def withdraw(self, amt):\n"
    "        if amt <= 0:\n            raise ValueError('must be positive')\n"
    "        if amt > self._b:\n            raise ValueError('insufficient')\n"
    "        self._b -= amt\n        self._h.append(('withdraw', amt))\n"
    "    def history(self):\n        return list(self._h)\n",
)

task_l3a = TaskDef(
    task_id="L3A_bank_account", version="1.0.0", difficulty=3,
    description="BankAccount class: validation + overdraft + operation history.",
    module_path="solution.py",
    required_functions=("BankAccount",),
    spec=_spec("L3A_bank_account", "bank account class",
               ("BankAccount",),
               ("test_deposit_withdraw", "test_overdraft",
                "test_negative_deposit", "test_history")),
    test_source=_L3A_TESTS,
    candidate_pool=_L3A_POOL,
)


_L3B_TESTS = '''
def test_lru_order():
    from solution import LRUCache
    c = LRUCache(2)
    c.put("a", 1)
    c.put("b", 2)
    assert c.get("a") == 1     # a now most-recent
    c.put("c", 3)              # evicts b
    assert c.get("b") == -1
    assert c.get("a") == 1
    assert c.get("c") == 3

def test_lru_update_refreshes():
    from solution import LRUCache
    c = LRUCache(2)
    c.put("a", 1)
    c.put("b", 2)
    c.put("a", 10)             # refresh a
    c.put("c", 3)              # evicts b
    assert c.get("a") == 10
    assert c.get("b") == -1

def test_capacity_one():
    from solution import LRUCache
    c = LRUCache(1)
    c.put("x", 1)
    c.put("y", 2)
    assert c.get("x") == -1
    assert c.get("y") == 2
'''

_L3B_POOL = (
    # greedy: FIFO not LRU (no refresh on get/put-update)
    "class LRUCache:\n"
    "    def __init__(self, capacity):\n"
    "        self.cap = capacity\n        self.d = {}\n        self.order = []\n"
    "    def get(self, k):\n"
    "        if k in self.d:\n            return self.d[k]\n"
    "        return -1\n"
    "    def put(self, k, v):\n"
    "        if k not in self.d and len(self.d) >= self.cap:\n"
    "            old = self.order.pop(0)\n            del self.d[old]\n"
    "        if k not in self.d:\n            self.order.append(k)\n"
    "        self.d[k] = v\n",
    # wrong: refresh on get but evicts newest instead of oldest
    "class LRUCache:\n"
    "    def __init__(self, capacity):\n"
    "        self.cap = capacity\n        self.d = {}\n        self.order = []\n"
    "    def get(self, k):\n"
    "        if k in self.d:\n"
    "            self.order.remove(k)\n            self.order.append(k)\n"
    "            return self.d[k]\n"
    "        return -1\n"
    "    def put(self, k, v):\n"
    "        if k in self.d:\n            self.order.remove(k)\n"
    "        elif len(self.d) >= self.cap:\n"
    "            newest = self.order.pop()\n            del self.d[newest]\n"
    "        self.order.append(k)\n        self.d[k] = v\n",
    # correct LRU
    "class LRUCache:\n"
    "    def __init__(self, capacity):\n"
    "        self.cap = capacity\n        self.d = {}\n        self.order = []\n"
    "    def _touch(self, k):\n"
    "        if k in self.order:\n            self.order.remove(k)\n"
    "        self.order.append(k)\n"
    "    def get(self, k):\n"
    "        if k in self.d:\n            self._touch(k)\n            return self.d[k]\n"
    "        return -1\n"
    "    def put(self, k, v):\n"
    "        self._touch(k)\n        self.d[k] = v\n"
    "        if len(self.d) > self.cap:\n"
    "            oldest = self.order.pop(0)\n            del self.d[oldest]\n",
)

task_l3b = TaskDef(
    task_id="L3B_lru_cache", version="1.0.0", difficulty=3,
    description="LRU cache with get(-1 default)/put and recency refresh.",
    module_path="solution.py",
    required_functions=("LRUCache",),
    spec=_spec("L3B_lru_cache", "lru cache", ("LRUCache",),
               ("test_lru_order", "test_lru_update_refreshes",
                "test_capacity_one")),
    test_source=_L3B_TESTS,
    candidate_pool=_L3B_POOL,
)


_L3C_TESTS = '''
def test_push_pop():
    from solution import MinStack
    s = MinStack()
    s.push(5)
    s.push(2)
    s.push(7)
    assert s.min() == 2
    assert s.pop() == 7
    assert s.min() == 2

def test_min_after_removal():
    from solution import MinStack
    s = MinStack()
    s.push(3)
    s.push(1)
    s.push(4)
    s.pop()
    assert s.min() == 1
    s.pop()
    assert s.min() == 3

def test_empty_min_raises():
    from solution import MinStack
    import pytest
    s = MinStack()
    try:
        s.min()
        raised = False
    except IndexError:
        raised = True
    assert raised

def test_duplicates():
    from solution import MinStack
    s = MinStack()
    s.push(2)
    s.push(2)
    s.pop()
    assert s.min() == 2
'''

_L3C_POOL = (
    # greedy: min recomputed by scanning copy; pop loses min correctness when
    # implemented via sort... make it wrong: min() pops the stack!
    "class MinStack:\n"
    "    def __init__(self):\n        self.items = []\n"
    "    def push(self, x):\n        self.items.append(x)\n"
    "    def pop(self):\n        return self.items.pop()\n"
    "    def min(self):\n        return min(self.items.pop())\n",
    # wrong: cached global min never restored after pop
    "class MinStack:\n"
    "    def __init__(self):\n        self.items = []\n        self.m = None\n"
    "    def push(self, x):\n"
    "        self.items.append(x)\n"
    "        if self.m is None or x < self.m:\n            self.m = x\n"
    "    def pop(self):\n        return self.items.pop()\n"
    "    def min(self):\n"
    "        if not self.items:\n            raise IndexError('empty')\n"
    "        return self.m\n",
    # correct auxiliary-min stack
    "class MinStack:\n"
    "    def __init__(self):\n        self.items = []\n        self.mins = []\n"
    "    def push(self, x):\n"
    "        self.items.append(x)\n"
    "        self.mins.append(x if not self.mins or x < self.mins[-1]\n"
    "                        else self.mins[-1])\n"
    "    def pop(self):\n"
    "        self.mins.pop()\n"
    "        return self.items.pop()\n"
    "    def min(self):\n"
    "        if not self.mins:\n            raise IndexError('empty')\n"
    "        return self.mins[-1]\n",
)

task_l3c = TaskDef(
    task_id="L3C_min_stack", version="1.0.0", difficulty=3,
    description="MinStack: push/pop/min all O(1), min tracks removals.",
    module_path="solution.py",
    required_functions=("MinStack",),
    spec=_spec("L3C_min_stack", "min stack", ("MinStack",),
               ("test_push_pop", "test_min_after_removal",
                "test_empty_min_raises", "test_duplicates")),
    test_source=_L3C_TESTS,
    candidate_pool=_L3C_POOL,
)


# ==========================================================================
# LEVEL 4 -- multi-file / interfaces / regression preservation
# ==========================================================================

_L4A_TESTS = '''
def test_pipeline():
    from solution import parse_csv_line, format_record
    fields = parse_csv_line('"Doe, John",42,AT')
    assert fields == ["Doe, John", "42", "AT"]
    rec = {"name": "Doe, John", "age": "42", "country": "AT"}
    out = format_record(fields, ["name", "age", "country"])
    assert out["name"] == "Doe, John"

def test_roundtrip():
    from solution import parse_csv_line, format_record
    original = ["x,y", "1", "z"]
    line = '"' + original[0] + '",1,z'
    assert parse_csv_line(line)[0] == "x,y"

def test_format_missing_col():
    from solution import format_record
    rec = format_record(["a"], ["name"])
    assert rec["name"] == "a"
'''

_L4A_POOL = (
    # greedy: naive split breaks quoted commas
    "def parse_csv_line(line):\n    return line.split(',')\n\n"
    "def format_record(fields, columns):\n"
    "    return {c: (fields[i] if i < len(fields) else None)\n"
    "            for i, c in enumerate(columns)}\n",
    # wrong parser: strips quotes globally, mangles embedded quotes
    "def parse_csv_line(line):\n"
    "    parts = []\n"
    "    cur = ''\n"
    "    in_q = False\n"
    "    for ch in line:\n"
    "        if ch == '\"':\n            in_q = not in_q\n"
    "        elif ch == ',' and not in_q:\n"
    "            parts.append(cur)\n            cur = ''\n"
    "        else:\n            cur += ch\n"
    "    parts.append(cur)\n"
    "    return parts\n\n"
    "def format_record(fields, columns):\n"
    "    return {c: (fields[i] if i < len(fields) else None)\n"
    "            for i, c in enumerate(columns)}\n",
    # correct state-machine CSV field parser
    "def parse_csv_line(line):\n"
    "    parts, cur, in_q = [], [], False\n"
    "    i = 0\n"
    "    while i < len(line):\n"
    "        ch = line[i]\n"
    "        if ch == '\"':\n"
    "            if in_q and i + 1 < len(line) and line[i+1] == '\"':\n"
    "                cur.append('\"')\n"
    "                i += 1\n"
    "            else:\n                in_q = not in_q\n"
    "        elif ch == ',' and not in_q:\n"
    "            parts.append(''.join(cur)); cur = []\n"
    "        else:\n            cur.append(ch)\n"
    "        i += 1\n"
    "    parts.append(''.join(cur))\n"
    "    return parts\n\n"
    "def format_record(fields, columns):\n"
    "    return {c: (fields[i] if i < len(fields) else None)\n"
    "            for i, c in enumerate(columns)}\n",
)

task_l4a = TaskDef(
    task_id="L4A_csv_module", version="1.0.0", difficulty=4,
    description="CSV quoted-field parser + record formatter (two interfaces).",
    module_path="solution.py",
    required_functions=("parse_csv_line", "format_record"),
    spec=_spec("L4A_csv_module", "csv parsing module",
               ("parse_csv_line", "format_record"),
               ("test_pipeline", "test_roundtrip", "test_format_missing_col")),
    test_source=_L4A_TESTS,
    candidate_pool=_L4A_POOL,
)


ALL_TASKS: tuple[TaskDef, ...] = (
    task_l1a, task_l1b, task_l2a, task_l2b, task_l2c,
    task_l3a, task_l3b, task_l3c, task_l4a,
)
