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
#
# Portions of this file are derived from MQC3,
# originally licensed under the MIT License.

"""Verification of feedforward functions."""

import ast

#: How a rejected construct is described, for the constructs a caller is most likely to reach for.
#: Anything absent falls back to the name of its syntax node.
_CONSTRUCT_NAMES = {
    "Subscript": "subscripting",
    "List": "a list literal",
    "Tuple": "a tuple literal",
    "Dict": "a dict literal",
    "Set": "a set literal",
    "JoinedStr": "an f-string",
    "Lambda": "a lambda",
    "ListComp": "a comprehension",
    "SetComp": "a comprehension",
    "DictComp": "a comprehension",
    "GeneratorExp": "a generator expression",
    "Starred": "argument unpacking",
    "Await": "await",
    "Yield": "yield",
    "YieldFrom": "yield from",
    "Slice": "a slice",
}


def _refusal(kind: str) -> str:
    """Describe why a construct is refused, in the one wording every refusal by construct uses.

    Args:
        kind: The name of the syntax node, as a key of :data:`_CONSTRUCT_NAMES`.

    Returns:
        str: The message to raise.
    """
    described = _CONSTRUCT_NAMES.get(kind, f"`{kind}`")
    return (
        f"Feedforward functions cannot use {described}. They are limited to arithmetic on the "
        "measured value, comparisons, and calls to the permitted built-in and math functions."
    )


class _FeedForwardVerifier(ast.NodeVisitor):
    """Verifier class that checks if the body of a function definition is safe for feedforward.

    Anything the visitor does not explicitly accept is refused. The source is executed on the machine
    that runs the circuit, in an environment where the built-ins are reachable whatever globals it is
    given, so this check is the only thing keeping a feedforward function to arithmetic. Letting an
    unrecognized construct through would therefore not be a gap in the description of the check but a
    gap in the check itself: attribute access alone is enough to reach any object the interpreter can
    see.
    """

    def __init__(self, allowed_free_names: set[str] | None = None) -> None:
        self.local_names = set()
        # Of those, the ones that may be *called*. Only what was imported from `math` qualifies: a
        # feedforward function has no way to define a function of its own, so a name it assigned holds
        # a value, and treating it as callable is what lets an object picked up elsewhere be invoked.
        self.callable_names: set[str] = set()
        self.return_stmt_count = 0
        # Names the function may read from the surrounding scope. Each has to have been declared as a
        # feedforward parameter, so that a mistyped name is refused rather than resolved to whatever
        # else happens to be in scope.
        self.allowed_free_names = allowed_free_names or set()
        self.used_free_names: set[str] = set()

    def generic_visit(self, node: ast.AST) -> None:
        """Refuse any construct without a visitor of its own.

        Args:
            node (ast.AST): The syntax node reached.

        Raises:
            ValueError: Always.
        """
        raise ValueError(_refusal(type(node).__name__))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        allowed_stmt = (ast.Return, ast.Assign, ast.AugAssign, ast.ImportFrom, ast.Expr)

        self._verify_args_of_func_def(node)
        for arg in node.args.args:
            self.local_names.add(arg.arg)
        self._verify_decorator_of_func_def(node)

        if not isinstance(node.body[-1], ast.Return):
            msg = "The function definition must end with a return statement."
            raise ValueError(msg)  # noqa: TRY004

        for stmt in node.body:
            if isinstance(stmt, ast.If):
                # Named on its own, because reaching for `if` is how a decision gets written first and
                # the general message does not say what to write instead.
                msg = (
                    "Feedforward functions cannot use an `if` statement. Write the decision as a "
                    "conditional expression, as in `return 1.0 if m > 0 else 0.0`, or as arithmetic on "
                    "a comparison, as in `return float(m > 0)`."
                )
                raise ValueError(msg)  # noqa: TRY004
            if not isinstance(stmt, allowed_stmt):
                msg = (
                    "Feedforward functions can only include return statements, assignments, "
                    "augmented assignments, import-from statements, and expression statements."
                )
                raise ValueError(msg)  # noqa: TRY004
            self.visit(stmt)

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is None or isinstance(node.value, ast.Tuple):
            msg = "The return value must be a single value and cannot be a tuple or None."
            raise ValueError(msg)
        self.return_stmt_count += 1
        if self.return_stmt_count > 1:
            msg = "Function body must contain exactly one return statement."
            raise ValueError(msg)

        self.visit(node.value)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.local_names.add(target.id)
            elif isinstance(target, ast.Tuple):  # Unpacking a tuple
                for element in target.elts:
                    if isinstance(element, ast.Name):
                        self.local_names.add(element.id)
                    else:
                        msg = "Assignment targets must be local variables."
                        raise ValueError(msg)  # noqa: TRY004
            else:
                msg = "Assignment targets must be either local variables or unpacked tuples."
                raise ValueError(msg)  # noqa: TRY004

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.value)
        if isinstance(node.target, ast.Name):
            self.local_names.add(node.target.id)
        else:
            msg = "Augmented assignment targets must be local variables."
            raise ValueError(msg)  # noqa: TRY004

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module != "math":
            msg = "Only the built-in math module is allowed for import."
            raise ValueError(msg)

        for name in node.names:
            if name.name == "*":
                msg = "Importing all (* wildcard) is not allowed."
                raise ValueError(msg)
            imported = name.asname or name.name
            self.local_names.add(imported)
            self.callable_names.add(imported)

    def visit_Expr(self, node: ast.Expr) -> None:
        allowed_expr = (ast.UnaryOp, ast.BinOp, ast.Call, ast.Constant, ast.Name, ast.NamedExpr)
        if not isinstance(node.value, allowed_expr):
            msg = (
                "Allowed expressions are limited to: "
                "unary operations, binary operations, function calls, constants, names, and named expressions."
            )
            raise ValueError(msg)  # noqa: TRY004
        self.visit(node.value)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        allowed_unary_op = (ast.USub, ast.UAdd, ast.Invert, ast.Not)
        if not isinstance(node.op, allowed_unary_op):
            msg = "Only the following unary operations are allowed: +, -, ~, not."
            raise ValueError(msg)  # noqa: TRY004
        self.visit(node.operand)

    def visit_Compare(self, node: ast.Compare) -> None:
        """Accept a comparison between values, which is how a decision is expressed.

        Identity and membership are refused. ``is`` says nothing useful about numbers, and ``in``
        reaches for a container's own ``__contains__``, neither of which is arithmetic on a
        measurement.

        Args:
            node (ast.Compare): The comparison.

        Raises:
            ValueError: If an operator other than an ordering or equality test is used.
        """
        allowed_compare_op = (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)
        for op in node.ops:
            if not isinstance(op, allowed_compare_op):
                msg = "Only the following comparisons are allowed: <, <=, >, >=, ==, !=."
                raise ValueError(msg)  # noqa: TRY004

        self.visit(node.left)
        for comparator in node.comparators:
            self.visit(comparator)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        """Accept ``and`` and ``or``, which combine comparisons.

        Args:
            node (ast.BoolOp): The boolean operation.

        Raises:
            ValueError: If the operator is neither ``and`` nor ``or``.
        """
        if not isinstance(node.op, (ast.And, ast.Or)):
            msg = "Only the following boolean operations are allowed: and, or."
            raise ValueError(msg)  # noqa: TRY004

        for value in node.values:
            self.visit(value)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        """Accept a conditional expression, which is how a decision picks between two values.

        The statement form of ``if`` stays out: a conditional expression covers what a feedforward
        function needs, and keeping the body to a single expression is what makes it checkable.

        Args:
            node (ast.IfExp): The conditional expression.
        """
        self.visit(node.test)
        self.visit(node.body)
        self.visit(node.orelse)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        allowed_binary_op = (
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
        )
        if not isinstance(node.op, allowed_binary_op):
            msg = "Only the following binary operations are allowed: +, -, *, /, //, %, <<, >>, &, ^, |, **, @."
            raise ValueError(msg)  # noqa: TRY004

        self.visit(node.left)
        self.visit(node.right)

    def visit_Call(self, node: ast.Call) -> None:
        allowed_builtin_func = {"abs", "bool", "complex", "divmod", "float", "int", "pow", "round"}
        if not isinstance(node.func, ast.Name) or (node.func.id not in (allowed_builtin_func | self.callable_names)):
            msg = (
                "Only the following functions can be called: abs, bool, complex, divmod, float, int, pow, round, "
                "and functions imported from the math module. A name the function assigned cannot be called, "
                "since it holds a value rather than a function."
            )
            raise ValueError(msg)

        for arg in node.args:
            self.visit(arg)

        for kwarg in node.keywords:
            # `f(**mapping)` reaches here as a keyword whose `arg` is None, and has to be named
            # explicitly: unlike `f(*seq)`, which arrives as an `ast.Starred` argument and so is
            # refused by `generic_visit`, the dict form is an `ast.keyword` like any other and only its
            # value would be looked at. Both are the same unpacking, and are refused the same way.
            if kwarg.arg is None:
                raise ValueError(_refusal("Starred"))
            self.visit(kwarg.value)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Accept the real and imaginary parts of a number, and nothing else.

        These are wanted: a gate parameter may be complex, so a function that produces one needs to be
        able to take it apart. Both yield a number, so nothing new becomes reachable through them,
        which is what separates them from attribute access in general. An attribute is never callable
        here either, since a call has to name what it calls.

        Args:
            node (ast.Attribute): The attribute access.

        Raises:
            ValueError: If any other attribute is named.
        """
        if node.attr not in {"real", "imag"}:
            msg = (
                f"Feedforward functions cannot read the attribute {node.attr!r}. Only `real` and `imag`, "
                "the parts of a number, are available."
            )
            raise ValueError(msg)
        self.visit(node.value)

    def visit_Constant(self, node: ast.Constant) -> None:
        pass

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in self.local_names:
            return
        if node.id in self.allowed_free_names:
            self.used_free_names.add(node.id)
            return
        msg = (
            f"The name {node.id!r} is neither defined within the function nor declared as a feedforward "
            "parameter. Declare it with claveles.feedforward.ff_parameters to read it from the surrounding scope."
        )
        raise ValueError(msg)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        if not isinstance(node.target, ast.Name):
            msg = "Assignment targets must be local variables."
            raise ValueError(msg)  # noqa: TRY004
        self.local_names.add(node.target.id)

    def _verify_args_of_func_def(self, func_def: ast.FunctionDef) -> None:
        """Verify that arguments of a function definition is safe for feedforward.

        Args:
            func_def (ast.FunctionDef): A function definition to verify.

        Raises:
            ValueError: If the function is not safe.
        """
        if len(func_def.args.args) != 1:
            msg = "Function must have a single argument."
            raise ValueError(msg)
        if func_def.args.posonlyargs:
            msg = "Function must not have positional-only arguments."
            raise ValueError(msg)
        if func_def.args.vararg:
            msg = "Function must not have variable length arguments."
            raise ValueError(msg)
        if func_def.args.kwonlyargs:
            msg = "Function must not have keyword-only arguments."
            raise ValueError(msg)
        if func_def.args.kw_defaults:
            msg = "Function must not have keyword defaults."
            raise ValueError(msg)
        if func_def.args.defaults:
            msg = "Function must not have defaults."
            raise ValueError(msg)

    def _verify_decorator_of_func_def(self, func_def: ast.FunctionDef) -> None:
        """Verify that decorators of a function definition is safe for feedforward.

        Args:
            func_def (ast.FunctionDef): A function definition to verify.

        Raises:
            ValueError: If the function is not safe.
        """
        if len(func_def.decorator_list) > 1:
            msg = "Function must have at most one decorator."
            raise ValueError(msg)
        if func_def.decorator_list:
            decorator = func_def.decorator_list[0]
            if not isinstance(decorator, ast.Name):
                msg = "Decorator must be a name."
                raise ValueError(msg)
            if decorator.id != "feedforward":
                msg = "Decorator must be 'feedforward'."
                raise ValueError(msg)


def verify_feedforward(f: str, allowed_free_names: set[str] | None = None) -> set[str]:
    """Verify whether the given function string is safe for use as a feedforward function.

    **Anything not listed below is refused.** The function is sent as source and run on the machine that
    executes the circuit, where the built-ins are reachable whatever globals it is given, so this check
    is the only thing keeping it to arithmetic. A construct that went unrecognized would therefore be a
    hole rather than an omission from this list: reading one attribute is enough to reach any object the
    interpreter can see.

    - It must define exactly one function.
    - The function must have a single argument (no default values, no `*args`, `**kwargs`, or keyword-only arguments).
    - Only the following statements are allowed: `return`, `assign`, `augassign`, `import from math`, and expression statements.
    - The function body must end with a single `return` statement, which may not return a tuple.
    - Only the following expressions are allowed:
      - Constants, and names defined in the function or declared as a feedforward parameter.
      - Unary operators `+`, `-`, `~`, `not`.
      - Binary operators `+`, `-`, `*`, `/`, `//`, `%`, `<<`, `>>`, `&`, `^`, `|`, `**`, `@`.
      - Comparisons `<`, `<=`, `>`, `>=`, `==`, `!=`. `is` and `in` are refused: neither is arithmetic on a measurement, and `in` reaches for a container's own `__contains__`.
      - `and`, `or`, and the conditional expression `a if c else b`. The statement form of `if` is not allowed; the conditional expression covers what a feedforward function needs.
      - Calls to `abs`, `bool`, `complex`, `divmod`, `float`, `int`, `pow`, `round`, and to names imported from `math`. A name the function assigned cannot be called: it holds a value, and treating it as callable is what would let an object picked up elsewhere be invoked.
      - The `real` and `imag` parts of a number, which a function producing a complex gate parameter needs. No other attribute is readable.
      - Assignment expressions (`:=`).
    - Every name must be defined within the function body, or declared as a feedforward parameter.

    Notably refused: attribute access other than `real` and `imag`, subscripting, list, tuple, dict and
    set literals, f-strings, lambdas, comprehensions and generator expressions, and argument unpacking
    in either form (`f(*args)` and `f(**kwargs)` alike).

    Args:
        f (str): The source code string of the function to verify.
        allowed_free_names (set[str] | None): Names the function may read from the surrounding scope,
            each declared as a feedforward parameter. Any other free name is refused.

    Returns:
        set[str]: The declared parameters the function actually reads.

    Raises:
        ValueError: If the function violates any of the feedforward constraints.
    """  # noqa: E501
    parsed = ast.parse(f, feature_version=(3, 10))

    if len(parsed.body) != 1 or not isinstance(parsed.body[0], ast.FunctionDef):
        msg = "Function must be a single function definition."
        raise ValueError(msg)

    verifier = _FeedForwardVerifier(allowed_free_names)
    verifier.visit(parsed.body[0])
    return verifier.used_free_names
