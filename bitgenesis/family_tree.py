"""Store, order, and render explicitly stated family relationships."""

from __future__ import annotations

import json
import re
from pathlib import Path
from xml.sax.saxutils import escape

_NAME = r"(?P<{group}>(?:[A-ZÀ-ÖØ-Þ][\w'’-]*)(?:\s+[A-ZÀ-ÖØ-Þ][\w'’-]*){{0,2}})"
_CHILD_FIRST = re.compile(
    r"^\s*"
    + _NAME.format(group="child")
    + r"\s+(?:è|e'|is)\s+(?:il\s+)?"
    r"(?:figli[oa]|child|son|daughter)\s+(?:di|of)\s+"
    + _NAME.format(group="parent")
    + r"\s*[.!?]?\s*$",
    re.IGNORECASE,
)
_PARENT_FIRST = re.compile(
    r"^\s*"
    + _NAME.format(group="parent")
    + r"\s+(?:è|e'|is)\s+(?:il\s+|la\s+)?"
    r"(?:padre|madre|genitore|father|mother|parent)\s+(?:di|of)\s+"
    + _NAME.format(group="child")
    + r"\s*[.!?]?\s*$",
    re.IGNORECASE,
)
_FIRST_PERSON_PARENT = re.compile(
    r"^\s*(?:mio\s+padre|mia\s+madre|mio\s+genitore|"
    r"my\s+father|my\s+mother|my\s+parent)\s+"
    r"(?:si\s+chiama|è|e'|is)\s+"
    + _NAME.format(group="parent")
    + r"\s*[.!?]?\s*$",
    re.IGNORECASE,
)
_FIRST_PERSON_CHILD = re.compile(
    r"^\s*(?:mio\s+figlio|mia\s+figlia|my\s+son|my\s+daughter|"
    r"my\s+child)\s+(?:si\s+chiama|è|e'|is)\s+"
    + _NAME.format(group="child")
    + r"\s*[.!?]?\s*$",
    re.IGNORECASE,
)
_PERSONAL_PARENT = re.compile(
    r"^\s*"
    + _NAME.format(group="parent")
    + r"\s+(?:è|e'|is)\s+(?:mio\s+padre|mia\s+madre|"
    r"mio\s+genitore|my\s+father|my\s+mother|my\s+parent)"
    r"\s*[.!?]?\s*$",
    re.IGNORECASE,
)
_PARENT_OF = re.compile(
    r"^\s*(?:il\s+padre|la\s+madre|il\s+genitore|"
    r"the\s+father|the\s+mother|the\s+parent)\s+(?:di|of)\s+"
    + _NAME.format(group="child")
    + r"\s+(?:è|e'|is)\s+"
    + _NAME.format(group="parent")
    + r"\s*[.!?]?\s*$",
    re.IGNORECASE,
)
_RELATION_PATTERNS = (
    (_CHILD_FIRST, "child", "parent"),
    (_PARENT_FIRST, "child", "parent"),
    (_FIRST_PERSON_PARENT, "Io", "parent"),
    (_FIRST_PERSON_CHILD, "child", "Io"),
    (_PERSONAL_PARENT, "Io", "parent"),
    (_PARENT_OF, "child", "parent"),
)
_MAX_PEOPLE = 500
_MAX_NAME_LENGTH = 80


def empty_family_tree() -> dict[str, object]:
    """Return a new, empty tree document."""
    return {"version": 1, "people": [], "parent_child": []}


def read_family_tree(path: Path) -> dict[str, object]:
    """Read and validate the saved family tree."""
    path = Path(path)
    if not path.exists():
        return empty_family_tree()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ValueError(f"Il file dell'albero genealogico non è valido: {error}") from error
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError("Il formato del file dell'albero genealogico non è supportato.")
    people = data.get("people")
    relations = data.get("parent_child")
    if not isinstance(people, list) or not isinstance(relations, list):
        raise ValueError("Il file dell'albero genealogico ha una struttura non valida.")
    if any(not isinstance(person, str) for person in people):
        raise ValueError("Il file dell'albero genealogico contiene nomi non validi.")
    if any(
        not isinstance(relation, dict)
        or not isinstance(relation.get("parent"), str)
        or not isinstance(relation.get("child"), str)
        for relation in relations
    ):
        raise ValueError("Il file dell'albero genealogico contiene relazioni non valide.")
    tree = {
        "version": 1,
        "people": people,
        "parent_child": [
            {"parent": relation["parent"], "child": relation["child"]}
            for relation in relations
        ],
    }
    _validate_tree(tree)
    return tree


def parse_family_relations(message: str) -> list[tuple[str, str]]:
    """Extract parent-child relations only from complete, explicit statements."""
    relations: list[tuple[str, str]] = []
    for sentence in re.split(r"[\n;]+|(?<=[.!?])\s+", message.strip()):
        sentence = sentence.strip()
        for pattern, child_group, parent_group in _RELATION_PATTERNS:
            match = pattern.fullmatch(sentence)
            if match is None:
                continue
            child = (
                "Io"
                if child_group == "Io"
                else match.group(child_group).strip()
            )
            parent = (
                "Io"
                if parent_group == "Io"
                else match.group(parent_group).strip()
            )
            relations.append((parent, child))
            break
    return relations


def is_family_tree_statement(message: str) -> bool:
    """Return whether every nonempty sentence is a recognized family relation."""
    sentences = [
        sentence.strip()
        for sentence in re.split(r"[\n;]+|(?<=[.!?])\s+", message.strip())
        if sentence.strip()
    ]
    return bool(sentences) and all(
        any(pattern.fullmatch(sentence) for pattern, _, _ in _RELATION_PATTERNS)
        for sentence in sentences
    )


def add_parent_child(
    tree: dict[str, object],
    parent: str,
    child: str,
) -> bool:
    """Add an explicitly provided parent-child relation, rejecting cycles."""
    parent_name = _clean_name(parent)
    child_name = _clean_name(child)
    people = tree["people"]
    relations = tree["parent_child"]
    assert isinstance(people, list)
    assert isinstance(relations, list)
    if parent_name.casefold() == child_name.casefold():
        raise ValueError("Una persona non può essere genitore di sé stessa.")

    parent_name = _canonical_name(people, parent_name)
    child_name = _canonical_name(people, child_name)
    if any(
        relation["parent"].casefold() == parent_name.casefold()
        and relation["child"].casefold() == child_name.casefold()
        for relation in relations
    ):
        return False

    candidate = {
        "version": 1,
        "people": list(people),
        "parent_child": [
            *relations,
            {"parent": parent_name, "child": child_name},
        ],
    }
    if parent_name not in people and child_name not in people:
        if len(people) + 2 > _MAX_PEOPLE:
            raise ValueError(f"L'albero non può superare {_MAX_PEOPLE} persone.")
        candidate["people"] = [*people, parent_name, child_name]
    elif parent_name not in people or child_name not in people:
        if len(people) + 1 > _MAX_PEOPLE:
            raise ValueError(f"L'albero non può superare {_MAX_PEOPLE} persone.")
        candidate["people"] = [
            *people,
            *(
                [parent_name]
                if parent_name not in people
                else [child_name]
            ),
        ]
    _validate_tree(candidate)
    tree["people"] = candidate["people"]
    tree["parent_child"] = candidate["parent_child"]
    return True


def remove_parent_child(
    tree: dict[str, object],
    parent: str,
    child: str,
) -> bool:
    """Remove a relation and prune people no longer connected to any relation."""
    people = tree["people"]
    relations = tree["parent_child"]
    assert isinstance(people, list)
    assert isinstance(relations, list)
    parent_key = _clean_name(parent).casefold()
    child_key = _clean_name(child).casefold()
    remaining = [
        relation
        for relation in relations
        if not (
            relation["parent"].casefold() == parent_key
            and relation["child"].casefold() == child_key
        )
    ]
    if len(remaining) == len(relations):
        return False
    connected = {
        name.casefold()
        for relation in remaining
        for name in (relation["parent"], relation["child"])
    }
    tree["parent_child"] = remaining
    tree["people"] = [
        person for person in people if person.casefold() in connected
    ]
    return True


def save_family_tree(path: Path, tree: dict[str, object]) -> None:
    """Atomically save the tree JSON and its ordered SVG visualization."""
    _validate_tree(tree)
    path = Path(path)
    svg_path = path.with_suffix(".svg")
    json_temp = path.with_name(f".{path.name}.tmp")
    svg_temp = svg_path.with_name(f".{svg_path.name}.tmp")
    try:
        json_temp.write_text(
            json.dumps(tree, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        svg_temp.write_text(render_family_tree_svg(tree), encoding="utf-8")
        json_temp.replace(path)
        svg_temp.replace(svg_path)
    finally:
        if json_temp.exists():
            json_temp.unlink()
        if svg_temp.exists():
            svg_temp.unlink()


def render_family_tree_svg(tree: dict[str, object]) -> str:
    """Render a generation-ordered, escaped SVG family tree."""
    generations = _ordered_generations(tree)
    if not generations:
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="180" '
            'viewBox="0 0 640 180" role="img" '
            'aria-label="Albero genealogico vuoto">'
            '<rect width="100%" height="100%" rx="16" fill="#111a2b"/>'
            '<text x="320" y="95" text-anchor="middle" fill="#9aa9bd" '
            'font-family="sans-serif" font-size="18">'
            "Aggiungi relazioni per creare l'albero genealogico.</text></svg>"
        )

    card_width = 180
    card_height = 66
    horizontal_gap = 48
    vertical_gap = 98
    margin_x = 36
    margin_y = 40
    width = max(
        640,
        max(len(generation) for generation in generations) * card_width
        + (max(len(generation) for generation in generations) - 1) * horizontal_gap
        + margin_x * 2,
    )
    height = margin_y * 2 + len(generations) * card_height + (len(generations) - 1) * vertical_gap
    positions: dict[str, tuple[float, float]] = {}
    for generation_index, generation in enumerate(generations):
        row_width = (
            len(generation) * card_width
            + (len(generation) - 1) * horizontal_gap
        )
        start_x = (width - row_width) / 2
        y = margin_y + generation_index * (card_height + vertical_gap)
        for index, person in enumerate(generation):
            positions[person.casefold()] = (
                start_x + index * (card_width + horizontal_gap),
                y,
            )

    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Albero genealogico ordinato per generazioni">',
        '<rect width="100%" height="100%" rx="16" fill="#111a2b"/>',
    ]
    relations = tree["parent_child"]
    assert isinstance(relations, list)
    for relation in relations:
        parent_x, parent_y = positions[relation["parent"].casefold()]
        child_x, child_y = positions[relation["child"].casefold()]
        lines.append(
            f'<path d="M {parent_x + card_width / 2} {parent_y + card_height} '
            f'V {parent_y + card_height + vertical_gap / 2} '
            f'H {child_x + card_width / 2} V {child_y}" '
            'fill="none" stroke="#73e0b0" stroke-width="2"/>'
        )
    for generation_index, generation in enumerate(generations):
        for person in generation:
            x, y = positions[person.casefold()]
            lines.extend(
                (
                    f'<rect x="{x}" y="{y}" width="{card_width}" '
                    f'height="{card_height}" rx="12" fill="#1b2940" '
                    'stroke="#73e0b0"/>',
                    f'<text x="{x + card_width / 2}" y="{y + 28}" '
                    'text-anchor="middle" fill="#e5e7eb" '
                    'font-family="sans-serif" font-size="16" '
                    'textLength="160" lengthAdjust="spacingAndGlyphs">'
                    f"{escape(person)}</text>",
                    f'<text x="{x + card_width / 2}" y="{y + 49}" '
                    'text-anchor="middle" fill="#9aa9bd" '
                    'font-family="sans-serif" font-size="12">'
                    f"Generazione {generation_index + 1}</text>",
                )
            )
    lines.append("</svg>")
    return "\n".join(lines)


def _clean_name(name: str) -> str:
    value = " ".join(name.split())
    if not value or len(value) > _MAX_NAME_LENGTH:
        raise ValueError(
            f"Il nome deve contenere da 1 a {_MAX_NAME_LENGTH} caratteri."
        )
    if any(ord(character) < 32 for character in value):
        raise ValueError("Il nome contiene caratteri non validi.")
    return value


def _canonical_name(people: list[object], name: str) -> str:
    for person in people:
        assert isinstance(person, str)
        if person.casefold() == name.casefold():
            return person
    return name


def _validate_tree(tree: dict[str, object]) -> None:
    people = tree.get("people")
    relations = tree.get("parent_child")
    if not isinstance(people, list) or not isinstance(relations, list):
        raise ValueError("L'albero genealogico ha una struttura non valida.")
    canonical = {
        person.casefold(): person
        for person in people
        if isinstance(person, str)
    }
    if len(canonical) != len(people) or len(people) > _MAX_PEOPLE:
        raise ValueError("L'albero genealogico contiene persone duplicate o troppe persone.")
    if any(_clean_name(person) != person for person in people):
        raise ValueError("L'albero genealogico contiene nomi non validi.")
    parents: dict[str, set[str]] = {name: set() for name in canonical}
    seen_relations: set[tuple[str, str]] = set()
    for relation in relations:
        if not isinstance(relation, dict):
            raise ValueError("L'albero genealogico contiene una relazione non valida.")
        parent = relation.get("parent")
        child = relation.get("child")
        if (
            not isinstance(parent, str)
            or not isinstance(child, str)
            or parent.casefold() not in canonical
            or child.casefold() not in canonical
            or parent.casefold() == child.casefold()
        ):
            raise ValueError("L'albero genealogico contiene una relazione non valida.")
        relation_key = (parent.casefold(), child.casefold())
        if relation_key in seen_relations:
            raise ValueError("L'albero genealogico contiene relazioni duplicate.")
        seen_relations.add(relation_key)
        parents[relation_key[1]].add(relation_key[0])

    resolved: dict[str, int] = {}
    active: set[str] = set()

    def generation(person: str) -> int:
        if person in resolved:
            return resolved[person]
        if person in active:
            raise ValueError("La relazione creerebbe un ciclo nell'albero genealogico.")
        active.add(person)
        level = max(
            (generation(parent) + 1 for parent in parents[person]),
            default=0,
        )
        active.remove(person)
        resolved[person] = level
        return level

    for person in canonical:
        generation(person)


def _ordered_generations(tree: dict[str, object]) -> list[list[str]]:
    people = tree["people"]
    relations = tree["parent_child"]
    assert isinstance(people, list)
    assert isinstance(relations, list)
    if not people:
        return []
    parents: dict[str, set[str]] = {person.casefold(): set() for person in people}
    names = {person.casefold(): person for person in people}
    for relation in relations:
        parents[relation["child"].casefold()].add(relation["parent"].casefold())
    resolved: dict[str, int] = {}

    def generation(person: str) -> int:
        if person not in resolved:
            resolved[person] = max(
                (generation(parent) + 1 for parent in parents[person]),
                default=0,
            )
        return resolved[person]

    for person in parents:
        generation(person)
    ordered: list[list[str]] = []
    for level in range(max(resolved.values()) + 1):
        ordered.append(
            sorted(
                (
                    names[person]
                    for person, person_level in resolved.items()
                    if person_level == level
                ),
                key=str.casefold,
            )
        )
    return ordered
