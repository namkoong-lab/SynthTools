"""Template sanity tests.

Guards against two classes of regressions:
  1. Brace-mismatch bugs where `.format()` crashes on a template (caused by
     unescaped `{` `}` in JSON examples).
  2. Fancy unicode quotes in any template.

Every YAML template under prompt_templates/ is loaded and `.format()`-ed with a
full kwargs dict derived from its declared schema.
"""

import re
from pathlib import Path

import pytest
import yaml

PROMPT_ROOT = Path(__file__).resolve().parent.parent / "prompt_templates"
FANCY = ["\u2018", "\u2019", "\u201c", "\u201d"]


def _all_templates():
    for p in sorted(PROMPT_ROOT.rglob("*.yml")):
        yield p
    for p in sorted(PROMPT_ROOT.rglob("*.yaml")):
        yield p


TEMPLATE_PATHS = list(_all_templates())
assert TEMPLATE_PATHS, f"No templates found under {PROMPT_ROOT}"


def _load_templates(path: Path):
    """Return every templated string in the YAML as a list of (label, text).

    Most files have a single `template:` field. Some (e.g. the chat-format
    task_evolver t1 template) define multiple named fields like
    `system_template` and `final_user_template`. We validate each.
    """
    data = yaml.safe_load(path.read_text())
    out = []
    if isinstance(data, dict):
        for key in ("template", "system_template", "final_user_template"):
            if key in data and isinstance(data[key], str):
                out.append((key, data[key]))
    return out


def _placeholders(template: str):
    """Return every {name} placeholder, after collapsing {{/}} escapes."""
    names = set()
    i = 0
    while i < len(template):
        if template[i] == "{":
            if i + 1 < len(template) and template[i + 1] == "{":
                i += 2
                continue
            end = template.find("}", i)
            if end == -1:
                break
            name = template[i + 1:end]
            if name and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
                names.add(name)
            i = end + 1
        else:
            i += 1
    return names


@pytest.mark.parametrize("path", TEMPLATE_PATHS, ids=[str(p.relative_to(PROMPT_ROOT)) for p in TEMPLATE_PATHS])
def test_template_format_renders(path: Path):
    templates = _load_templates(path)
    if not templates:
        pytest.skip(f"no string templates in {path.name}")
    for label, template in templates:
        names = _placeholders(template)
        kwargs = {n: f"<{n}>" for n in names}
        try:
            template.format(**kwargs)
        except (KeyError, IndexError, ValueError) as e:
            pytest.fail(
                f"Template {path.relative_to(PROMPT_ROOT)} (key={label}) failed .format(): {e}"
            )


@pytest.mark.parametrize("path", TEMPLATE_PATHS, ids=[str(p.relative_to(PROMPT_ROOT)) for p in TEMPLATE_PATHS])
def test_template_no_fancy_quotes(path: Path):
    text = path.read_text()
    for q in FANCY:
        assert q not in text, f"Fancy quote {q!r} in {path.relative_to(PROMPT_ROOT)}"
