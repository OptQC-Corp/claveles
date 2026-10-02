# noqa: INP001
"""Configuration of documentation."""

from docutils.nodes import Node
from sphinx.application import Sphinx
from sphinx.writers.html5 import HTML5Translator

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Claveles"
copyright = "2025 RIKEN"  # noqa: A001
author = "RIKEN"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.bibtex",
    "myst_nb",
]
source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}
autodoc_default_options = {
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
autodoc_typehints = "description"
# If the execution time of notebook is long, increase this value
nb_execution_timeout = 1200

numfig = True

templates_path = []
exclude_patterns = []

# Two different classes really do share a short name: `Operation` in both `circuit.ops` and
# `graph.ops`, `MeasuredVariable` in both `circuit.ops` and `machinery.macronode_angle`. A short
# `Operation` in a docstring therefore matches more than one target, and Sphinx picks one.
#
# This hides the ambiguity only. A reference that resolves to *nothing* is reported under
# `subtype=<role>` -- `ref.class`, `ref.meth` and so on -- so `sphinx-build -n` still reports every
# dangling reference with this entry in place. Run it that way after renaming anything public.
suppress_warnings = ["ref.python"]

# Names that appear in a rendered signature but have no page of their own, and are not meant to: the
# type variables that give the generics their shape, and the mixin that carries the arithmetic on a
# feedforward parameter. Listed so that `sphinx-build -n` reports only references worth fixing.
nitpick_ignore = [
    ("py:class", "claveles.feedforward._LinearOperatorMixin"),
    ("py:class", "claveles.feedforward._Var"),
    ("py:class", "claveles.feedforward._VarElem"),
    ("py:class", "claveles.pb.io.T"),
    ("py:class", "claveles.utility.K"),
    ("py:class", "claveles.utility.V"),
    ("py:obj", "claveles.feedforward._Var"),
    ("py:obj", "claveles.feedforward._VarElem"),
    ("py:obj", "claveles.pb.io.T"),
    ("py:obj", "claveles.utility.K"),
    ("py:obj", "claveles.utility.V"),
]

myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

# Backend-support badges reused across gates.md. Defined here so the badge
# markup (including the hover tooltip in the ``title`` attribute) stays in one
# place. The classes match those emitted by sphinx-design's ``{bdg-*}`` roles,
# so the custom.css text-color override applies to these badges too.
myst_substitutions = {
    "bdg_hw": (
        '<span class="sd-sphinx-override sd-badge sd-bg-warning sd-bg-text-warning" '
        'title="Executable on the real machine (convertible to the graph and machinery representations)">'
        "Hardware</span>"
    ),
    "bdg_bosonic": (
        '<span class="sd-sphinx-override sd-badge sd-bg-primary sd-bg-text-primary" '
        'title="Applicable to states in the bosonic (Gaussian) representation">'
        "Bosonic</span>"
    ),
    "bdg_fock": (
        '<span class="sd-sphinx-override sd-badge sd-bg-success sd-bg-text-success" '
        'title="Applicable to states in the Fock representation">'
        "Fock</span>"
    ),
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- Bibliography -------------------------------------------------
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "unsrt"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["custom.css"]


# -- Machine translation -----------------------------------------------------

# Classes that mark a piece of the page as something a machine translator must leave alone.
#
# ``notranslate`` is put there by Sphinx itself, on inline literals, cross-reference names,
# highlighted code blocks and MathJax spans. ``sig`` is the class on an API signature, which Sphinx
# does not mark; without it, a translated reference page renames the parameters of every function.
UNTRANSLATABLE_CLASSES = frozenset({"notranslate", "sig"})


class UntranslatableAwareHTMLTranslator(HTML5Translator):
    """Emit ``translate="no"`` alongside the classes that mean "do not translate this"."""

    def starttag(self, node: Node, tagname: str, suffix: str = "\n", empty: bool = False, **attributes: object) -> str:  # noqa: FBT001, FBT002
        """Add ``translate="no"`` to a tag that carries one of :data:`UNTRANSLATABLE_CLASSES`.

        Returns:
            The start tag, with the attribute added when the classes ask for it.
        """
        if not any(name.lower() == "translate" for name in attributes):
            classes = set(node.get("classes", ()))
            for name, value in attributes.items():
                if name.lower() == "class":
                    classes.update(str(value).split())
            if classes & UNTRANSLATABLE_CLASSES:
                attributes["translate"] = "no"
        return super().starttag(node, tagname, suffix, empty, **attributes)


def setup(app: Sphinx) -> None:
    """Register the translator that marks code and math as untranslatable.

    The ``notranslate`` class is a Google Translate convention, and the browser translation built
    into Chrome goes by the standard ``translate`` attribute instead -- so a page read through it
    has its identifiers, code blocks and formulas translated as if they were prose, which is where
    ``FockState`` stops being a name and turns into a phrase. The attribute is inherited, so marking
    the element Sphinx already classed is enough to cover what is nested inside it.
    """
    app.set_translator("html", UntranslatableAwareHTMLTranslator, override=True)
