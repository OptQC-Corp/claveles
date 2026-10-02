# Copyright 2026 OptQC Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for what the feedforward verifier accepts, and for the constructs it has to keep out.

The verifier is the only thing standing between a feedforward function and the interpreter running it:
the source is executed on the machine that runs the circuit, and the built-ins are reachable there
whatever globals it is given. So the point of these tests is not only that the useful constructs are
accepted, but that everything else is refused *by default* rather than by being enumerated.
"""

from __future__ import annotations

import ast

import pytest

from claveles.feedforward.verification import verify_feedforward


def _wrap(body: str) -> str:
    """Put a function body into a one-argument function the verifier can be handed.

    Args:
        body: The body's lines, already indented relative to the function.

    Returns:
        str: The function source.
    """
    return f"def f(m):\n{body}\n"


ACCEPTED = {
    "arithmetic": "    return m * 2.0 + 1.0",
    "power": "    return m**2",
    "unary minus": "    return -m",
    "unary invert": "    return ~int(m)",
    "not": "    return float(not m)",
    "comparison": "    return float(m > 0)",
    "chained comparison": "    return float(0 < m < 1)",
    "equality": "    return float(m == 0.0)",
    "and": "    return float(m > 0 and m < 1)",
    "or": "    return float(m > 1 or m < 0)",
    "conditional expression": "    return 1.0 if m > 0 else 0.0",
    "builtin call": "    return abs(m)",
    "math import and call": "    from math import atan\n    return atan(m)",
    "math import with alias": "    from math import atan as arctan\n    return arctan(m)",
    "assignment": "    a = m * 2.0\n    return a",
    "augmented assignment": "    a = m\n    a += 1.0\n    return a",
    "walrus": "    return (a := m * 2.0) + a",
    "tuple unpacking from divmod": "    a, _ = divmod(m, 5)\n    return a",
    "real part": "    a = complex(m, 1.0)\n    return a.real",
    "imag part": "    a = complex(m, 1.0)\n    return a.imag",
}

REFUSED = {
    "attribute access": ("    return (0.0).__class__", "attribute '__class__'"),
    "attribute on the argument": ("    return m.__class__", "attribute '__class__'"),
    "a method looked up": ("    a = complex(m, 1.0)\n    return a.conjugate", "attribute 'conjugate'"),
    "subscripting": ("    return [1.0, 2.0][0]", "subscripting"),
    "a dict literal": ("    return {'a': 1.0}['a']", "subscripting"),
    "a list literal returned": ("    a = [1.0]\n    return m", "a list literal"),
    "a set literal": ("    a = {1.0}\n    return m", "a set literal"),
    "an f-string": ("    return f'{m}'", "an f-string"),
    "a lambda": ("    a = lambda: 1.0\n    return m", "a lambda"),
    "a comprehension": ("    a = [x for x in [m]]\n    return m", "a comprehension"),
    "a generator expression": ("    a = (x for x in [m])\n    return m", "a generator expression"),
    "is": ("    return float(m is None)", "Only the following comparisons"),
    "is not": ("    return float(m is not None)", "Only the following comparisons"),
    "in": ("    return float(m in [1.0, 2.0])", "Only the following comparisons"),
    "not in": ("    return float(m not in [1.0, 2.0])", "Only the following comparisons"),
    "argument unpacking": ("    return pow(*[m, 2])", "argument unpacking"),
    # The dict form reaches the verifier as an `ast.keyword` rather than an `ast.Starred`, so it is
    # refused by its own check in visit_Call rather than by generic_visit. Both forms are listed so
    # that the two spellings of unpacking are known to be refused alike.
    "keyword argument unpacking": ("    a = m\n    return pow(m, **a)", "argument unpacking"),
    "a statement that is only an expression": ("    m > 0\n    return m", "Allowed expressions are limited to"),
    "an attribute as an assignment target": ("    m.x = 1.0\n    return m", "either local variables or unpacked"),
    "an attribute inside an unpacked tuple": (
        "    a, m.x = divmod(m, 5)\n    return a",
        "Assignment targets must be local variables",
    ),
    "an attribute as an augmented assignment target": ("    m.x += 1.0\n    return m", "must be local variables"),
    "a subscript as an assignment target": ("    m[0] = 1.0\n    return m", "either local variables or unpacked"),
    "a wildcard import": ("    from math import *\n    return m", r"Importing all \(\* wildcard\) is not allowed"),
    "an import of another module": ("    from os import path\n    return m", "Only the built-in math module"),
}

#: Whole-function sources whose *signature* or *decorator* is what has to be refused, so unlike
#: REFUSED they cannot be expressed as a body handed to `_wrap`.
REFUSED_DEFINITIONS = {
    "no argument": ("def f():\n    return 1.0\n", "must have a single argument"),
    "two arguments": ("def f(m, n):\n    return m\n", "must have a single argument"),
    "a positional-only argument": ("def f(a, /, m):\n    return m\n", "must not have positional-only arguments"),
    "a variable length argument": ("def f(m, *args):\n    return m\n", "must not have variable length arguments"),
    "a keyword-only argument": ("def f(m, *, k):\n    return m\n", "must not have keyword-only arguments"),
    "a default value": ("def f(m=1.0):\n    return m\n", "must not have defaults"),
    "two decorators": ("@feedforward\n@other\ndef f(m):\n    return m\n", "at most one decorator"),
    "a decorator that is not a name": ("@mod.feedforward\ndef f(m):\n    return m\n", "Decorator must be a name"),
    "a decorator that is called": ("@feedforward()\ndef f(m):\n    return m\n", "Decorator must be a name"),
    "another decorator": ("@other\ndef f(m):\n    return m\n", "Decorator must be 'feedforward'"),
    "two functions": ("def f(m):\n    return m\ndef g(m):\n    return m\n", "single function definition"),
    "something that is not a function": ("m = 1.0\n", "single function definition"),
    "an async function": ("async def f(m):\n    return m\n", "single function definition"),
    "two return statements": ("def f(m):\n    return m\n    return m\n", "exactly one return statement"),
    "a tuple returned": ("def f(m):\n    return m, m\n", "cannot be a tuple or None"),
    "a bare return": ("def f(m):\n    return\n", "cannot be a tuple or None"),
    "no return at the end": ("def f(m):\n    a = m\n", "must end with a return statement"),
}


@pytest.mark.parametrize(("label", "body"), ACCEPTED.items(), ids=ACCEPTED.keys())
def test_an_accepted_construct_verifies(label: str, body: str) -> None:
    assert verify_feedforward(_wrap(body)) == set()


@pytest.mark.parametrize(("label", "case"), REFUSED.items(), ids=REFUSED.keys())
def test_a_refused_construct_is_rejected(label: str, case: tuple[str, str]) -> None:
    body, expected = case
    with pytest.raises(ValueError, match=expected):
        verify_feedforward(_wrap(body))


@pytest.mark.parametrize(("label", "case"), REFUSED_DEFINITIONS.items(), ids=REFUSED_DEFINITIONS.keys())
def test_a_refused_definition_is_rejected(label: str, case: tuple[str, str]) -> None:
    source, expected = case
    with pytest.raises(ValueError, match=expected):
        verify_feedforward(source)


def test_a_keyword_argument_still_has_its_value_checked() -> None:
    """Refusing `**` must not be mistaken for refusing keyword arguments, whose values still matter."""
    assert verify_feedforward(_wrap("    return pow(m, exp=2)")) == set()
    with pytest.raises(ValueError, match="Only the following functions can be called"):
        verify_feedforward(_wrap("    return pow(m, exp=open('/etc/passwd'))"))


# The operator whitelists below name every operator their AST category has, so the accompanying
# "only these are allowed" rejection cannot be reached from any source Python will parse. These tests
# pin that down: if a future Python adds an operator, the whitelist stops being exhaustive and the
# rejection becomes live, which is the moment to decide whether the new operator belongs in it.
@pytest.mark.parametrize(
    ("category", "allowed"),
    [
        (ast.unaryop, {ast.USub, ast.UAdd, ast.Invert, ast.Not}),
        (ast.boolop, {ast.And, ast.Or}),
        (
            ast.operator,
            {
                ast.Add,
                ast.Sub,
                ast.Mult,
                ast.Div,
                ast.FloorDiv,
                ast.Mod,
                ast.LShift,
                ast.RShift,
                ast.BitAnd,
                ast.BitXor,
                ast.BitOr,
                ast.Pow,
                ast.MatMult,
            },
        ),
    ],
    ids=["unary", "boolean", "binary"],
)
def test_an_operator_whitelist_covers_its_whole_category(category: type[ast.AST], allowed: set[type[ast.AST]]) -> None:
    assert set(category.__subclasses__()) == allowed


def test_the_comparison_whitelist_deliberately_does_not() -> None:
    """Unlike the three above, comparisons are a real subset: identity and membership stay out."""
    refused = set(ast.cmpop.__subclasses__()) - {ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq}
    assert refused == {ast.Is, ast.IsNot, ast.In, ast.NotIn}


def test_a_name_the_function_assigned_cannot_be_called() -> None:
    """Otherwise an object picked up elsewhere could be bound to a name and invoked through it."""
    with pytest.raises(ValueError, match="Only the following functions can be called"):
        verify_feedforward(_wrap("    a = m\n    return a(m)"))


def test_a_built_in_cannot_be_bound_to_a_name_either() -> None:
    """Reading `abs` as a value is refused before the question of calling it arises."""
    with pytest.raises(ValueError, match="neither defined within the function"):
        verify_feedforward(_wrap("    g = abs\n    return g(m)"))


def test_reaching_an_arbitrary_callable_is_refused() -> None:
    """The shape a sandbox escape takes: bind something reached by attribute, then call the name."""
    with pytest.raises(ValueError, match="attribute '__class__'"):
        verify_feedforward(_wrap("    g = (0.0).__class__\n    return g(1)"))
    with pytest.raises(ValueError, match="attribute '__subclasses__'"):
        verify_feedforward(
            _wrap("    s = (0.0).__class__.__bases__[0].__subclasses__\n    t = s()\n    return 0.0"),
        )


def test_an_unrecognized_construct_is_refused_rather_than_ignored() -> None:
    """A construct nobody thought about has to be a rejection, not a gap."""
    with pytest.raises(ValueError, match="cannot use a tuple literal"):
        verify_feedforward(_wrap("    a = (1.0, 2.0)\n    return m"))


def test_the_statement_form_of_if_stays_out() -> None:
    """And says what to write instead, since reaching for `if` is how a decision gets written first."""
    with pytest.raises(ValueError, match=r"conditional expression.*1\.0 if m > 0 else 0\.0"):
        verify_feedforward("def f(m):\n    if m > 0:\n        return 1.0\n    return 0.0\n")


def test_another_disallowed_statement_still_gets_the_general_message() -> None:
    with pytest.raises(ValueError, match="can only include return statements"):
        verify_feedforward("def f(m):\n    while m > 0:\n        m = m - 1.0\n    return m\n")


def test_a_declared_parameter_is_still_readable() -> None:
    assert verify_feedforward(_wrap("    return float(m > threshold)"), {"threshold"}) == {"threshold"}


def test_an_undeclared_name_is_still_refused() -> None:
    with pytest.raises(ValueError, match="neither defined within the function"):
        verify_feedforward(_wrap("    return float(m > threshold)"))
