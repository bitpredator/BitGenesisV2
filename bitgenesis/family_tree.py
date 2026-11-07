"""Store, order, and render explicitly stated family relationships."""

from __future__ import annotations

import json
import re
from pathlib import Path
from xml.sax.saxutils import escape

_NAME_BODY = r"(?:[A-ZÀ-ÖØ-Þ][\w'’-]*)(?:\s+[A-ZÀ-ÖØ-Þ][\w'’-]*){0,2}"
_NAME = r"(?P<{group}>" + _NAME_BODY.replace("{", "{{").replace("}", "}}") + r")"
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
_PARENT_FIRST_MULTIPLE = re.compile(
    r"^\s*"
    + _NAME.format(group="parent")
    + r"\s+(?:è|e'|is)\s+(?:il\s+|la\s+)?"
    r"(?:padre|madre|genitore|father|mother|parent)\s+(?:di|of)\s+"
    r"(?P<children>"
    + _NAME_BODY
    + r"(?:\s+(?:e|and)\s+(?:(?:di|of)\s+)?"
    + _NAME_BODY
    + r")+)"
    r"\s*[.!?]?\s*$",
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
_SIBLINGS_TOGETHER = re.compile(
    r"^\s*"
    + _NAME.format(group="first")
    + r"\s+(?:e|and)\s+"
    + _NAME.format(group="second")
    + r"\s+(?:sono|are)\s+"
    r"(?:fratell[io]|sorell[ae]|fratelli e sorelle|siblings)"
    r"\s*[.!?]?\s*$",
    re.IGNORECASE,
)
_SIBLING_OF = re.compile(
    r"^\s*"
    + _NAME.format(group="first")
    + r"\s+(?:è|e'|is)\s+(?:il\s+|la\s+)?"
    r"(?:fratell[io]|sorell[ae]|brother|sister|sibling)\s+"
    r"(?:di|of)\s+"
    + _NAME.format(group="second")
    + r"\s*[.!?]?\s*$",
    re.IGNORECASE,
)
_SIBLING_POSSESSIVE = re.compile(
    r"^\s*"
    + _NAME.format(group="first")
    + r"\s+(?:è|e'|is)\s+"
    + _NAME.format(group="second")
    + r"(?:'s|’s)\s+"
    r"(?:fratell[io]|sorell[ae]|brother|sister|sibling)"
    r"\s*[.!?]?\s*$",
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
_SIBLING_PATTERNS = (
    _SIBLINGS_TOGETHER,
    _SIBLING_OF,
    _SIBLING_POSSESSIVE,
)
_SPOUSES_TOGETHER = re.compile(
    r"^\s*"
    + _NAME.format(group="first")
    + r"\s+(?:e|and)\s+"
    + _NAME.format(group="second")
    + r"\s+(?:sono|are)\s+"
    r"(?:marito e moglie|moglie e marito|coniugi|husband and wife|"
    r"wife and husband|spouses|married)"
    r"\s*[.!?]?\s*$",
    re.IGNORECASE,
)
_SPOUSE_OF = re.compile(
    r"^\s*"
    + _NAME.format(group="first")
    + r"\s+(?:è|e'|is)\s+(?:il\s+|la\s+)?"
    r"(?:marito|moglie|coniuge|husband|wife|spouse)\s+"
    r"(?:di|of)\s+"
    + _NAME.format(group="second")
    + r"\s*[.!?]?\s*$",
    re.IGNORECASE,
)
_MARRIED_TO = re.compile(
    r"^\s*"
    + _NAME.format(group="first")
    + r"\s+(?:è|e'|is)\s+(?:sposato|sposata|sposat[oa]|married)\s+"
    r"(?:con|a|to)\s+"
    + _NAME.format(group="second")
    + r"\s*[.!?]?\s*$",
    re.IGNORECASE,
)
_SPOUSE_PATTERNS = (_SPOUSES_TOGETHER, _SPOUSE_OF, _MARRIED_TO)
_MAX_PEOPLE = 500
_MAX_NAME_LENGTH = 80


def empty_family_tree() -> dict[str, object]:
    """Return a new, empty tree document."""
    return {
        "version": 1,
        "people": [],
        "parent_child": [],
        "sibling_pairs": [],
        "spouse_pairs": [],
    }


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
    siblings = data.get("sibling_pairs", [])
    spouses = data.get("spouse_pairs", [])
    if (
        not isinstance(people, list)
        or not isinstance(relations, list)
        or not isinstance(siblings, list)
        or not isinstance(spouses, list)
    ):
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
    if any(
        not isinstance(pair, dict)
        or not isinstance(pair.get("first"), str)
        or not isinstance(pair.get("second"), str)
        for pair in siblings
    ):
        raise ValueError("Il file dell'albero genealogico contiene fratellanze non valide.")
    if any(
        not isinstance(pair, dict)
        or not isinstance(pair.get("first"), str)
        or not isinstance(pair.get("second"), str)
        for pair in spouses
    ):
        raise ValueError("Il file dell'albero genealogico contiene coniugi non validi.")
    tree = {
        "version": 1,
        "people": people,
        "parent_child": [
            {"parent": relation["parent"], "child": relation["child"]}
            for relation in relations
        ],
        "sibling_pairs": [
            {"first": pair["first"], "second": pair["second"]}
            for pair in siblings
        ],
        "spouse_pairs": [
            {"first": pair["first"], "second": pair["second"]}
            for pair in spouses
        ],
    }
    _validate_tree(tree)
    return tree


def parse_family_relations(message: str) -> list[tuple[str, str]]:
    """Extract parent-child relations only from complete, explicit statements."""
    relations: list[tuple[str, str]] = []
    for sentence in re.split(r"[\n;]+|(?<=[.!?])\s+", message.strip()):
        sentence = sentence.strip()
        multiple_children = _PARENT_FIRST_MULTIPLE.fullmatch(sentence)
        if multiple_children is not None:
            children = re.split(
                r"\s+(?:e|and)\s+(?:(?:di|of)\s+)?",
                multiple_children.group("children"),
                flags=re.IGNORECASE,
            )
            parent = multiple_children.group("parent").strip()
            relations.extend((parent, child.strip()) for child in children)
            continue
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


def parse_sibling_relations(message: str) -> list[tuple[str, str]]:
    """Extract sibling pairs from explicit Italian or English statements."""
    pairs: list[tuple[str, str]] = []
    for sentence in re.split(r"[\n;]+|(?<=[.!?])\s+", message.strip()):
        sentence = sentence.strip()
        for pattern in _SIBLING_PATTERNS:
            match = pattern.fullmatch(sentence)
            if match is not None:
                pairs.append(
                    (
                        match.group("first").strip(),
                        match.group("second").strip(),
                    )
                )
                break
    return pairs


def parse_spouse_relations(message: str) -> list[tuple[str, str]]:
    """Extract married pairs from explicit Italian or English statements."""
    pairs: list[tuple[str, str]] = []
    for sentence in re.split(r"[\n;]+|(?<=[.!?])\s+", message.strip()):
        sentence = sentence.strip()
        for pattern in _SPOUSE_PATTERNS:
            match = pattern.fullmatch(sentence)
            if match is not None:
                pairs.append(
                    (
                        match.group("first").strip(),
                        match.group("second").strip(),
                    )
                )
                break
    return pairs


def is_family_tree_statement(message: str) -> bool:
    """Return whether every nonempty sentence is a recognized family relation."""
    sentences = [
        sentence.strip()
        for sentence in re.split(r"[\n;]+|(?<=[.!?])\s+", message.strip())
        if sentence.strip()
    ]
    return bool(sentences) and all(
        _PARENT_FIRST_MULTIPLE.fullmatch(sentence) is not None
        or any(pattern.fullmatch(sentence) for pattern, _, _ in _RELATION_PATTERNS)
        or any(pattern.fullmatch(sentence) for pattern in _SIBLING_PATTERNS)
        or any(pattern.fullmatch(sentence) for pattern in _SPOUSE_PATTERNS)
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
    siblings = tree.get("sibling_pairs", [])
    assert isinstance(siblings, list)
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
        "sibling_pairs": list(tree["sibling_pairs"]),
        "spouse_pairs": list(tree["spouse_pairs"]),
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


def add_sibling_pair(
    tree: dict[str, object],
    first: str,
    second: str,
) -> bool:
    """Add an explicitly stated sibling pair at the same family-tree level."""
    first_name = _clean_name(first)
    second_name = _clean_name(second)
    if first_name.casefold() == second_name.casefold():
        raise ValueError("Una persona non può essere fratello o sorella di sé stessa.")

    people = tree["people"]
    pairs = tree["sibling_pairs"]
    assert isinstance(people, list)
    assert isinstance(pairs, list)
    first_name = _canonical_name(people, first_name)
    second_name = _canonical_name(people, second_name)
    pair_key = frozenset((first_name.casefold(), second_name.casefold()))
    if any(
        frozenset((pair["first"].casefold(), pair["second"].casefold()))
        == pair_key
        for pair in pairs
    ):
        return False

    candidate = {
        "version": 1,
        "people": list(people),
        "parent_child": list(tree["parent_child"]),
        "sibling_pairs": [
            *pairs,
            {"first": first_name, "second": second_name},
        ],
        "spouse_pairs": list(tree["spouse_pairs"]),
    }
    for name in (first_name, second_name):
        if not any(person.casefold() == name.casefold() for person in candidate["people"]):
            if len(candidate["people"]) >= _MAX_PEOPLE:
                raise ValueError(f"L'albero non può superare {_MAX_PEOPLE} persone.")
            candidate["people"].append(name)
    _validate_tree(candidate)
    tree["people"] = candidate["people"]
    tree["sibling_pairs"] = candidate["sibling_pairs"]
    return True


def add_spouse_pair(
    tree: dict[str, object],
    first: str,
    second: str,
) -> bool:
    """Add an explicitly stated spouse pair."""
    first_name = _clean_name(first)
    second_name = _clean_name(second)
    if first_name.casefold() == second_name.casefold():
        raise ValueError("Una persona non può essere coniuge di sé stessa.")

    people = tree["people"]
    pairs = tree["spouse_pairs"]
    assert isinstance(people, list)
    assert isinstance(pairs, list)
    first_name = _canonical_name(people, first_name)
    second_name = _canonical_name(people, second_name)
    pair_key = frozenset((first_name.casefold(), second_name.casefold()))
    if any(
        frozenset((pair["first"].casefold(), pair["second"].casefold()))
        == pair_key
        for pair in pairs
    ):
        return False

    candidate = {
        "version": 1,
        "people": list(people),
        "parent_child": list(tree["parent_child"]),
        "sibling_pairs": list(tree["sibling_pairs"]),
        "spouse_pairs": [
            *pairs,
            {"first": first_name, "second": second_name},
        ],
    }
    for name in (first_name, second_name):
        if not any(person.casefold() == name.casefold() for person in candidate["people"]):
            if len(candidate["people"]) >= _MAX_PEOPLE:
                raise ValueError(f"L'albero non può superare {_MAX_PEOPLE} persone.")
            candidate["people"].append(name)
    _validate_tree(candidate)
    tree["people"] = candidate["people"]
    tree["spouse_pairs"] = candidate["spouse_pairs"]
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
    tree["parent_child"] = remaining
    _prune_unrelated_people(tree)
    return True


def remove_sibling_pair(
    tree: dict[str, object],
    first: str,
    second: str,
) -> bool:
    """Remove a sibling link and prune people no longer in any relationship."""
    first_key = _clean_name(first).casefold()
    second_key = _clean_name(second).casefold()
    pairs = tree["sibling_pairs"]
    assert isinstance(pairs, list)
    remaining = [
        pair
        for pair in pairs
        if frozenset((pair["first"].casefold(), pair["second"].casefold()))
        != frozenset((first_key, second_key))
    ]
    if len(remaining) == len(pairs):
        return False
    tree["sibling_pairs"] = remaining
    _prune_unrelated_people(tree)
    return True


def remove_spouse_pair(
    tree: dict[str, object],
    first: str,
    second: str,
) -> bool:
    """Remove a spouse pair and prune people no longer in any relationship."""
    first_key = _clean_name(first).casefold()
    second_key = _clean_name(second).casefold()
    pairs = tree["spouse_pairs"]
    assert isinstance(pairs, list)
    remaining = [
        pair
        for pair in pairs
        if frozenset((pair["first"].casefold(), pair["second"].casefold()))
        != frozenset((first_key, second_key))
    ]
    if len(remaining) == len(pairs):
        return False
    tree["spouse_pairs"] = remaining
    _prune_unrelated_people(tree)
    return True


def _prune_unrelated_people(tree: dict[str, object]) -> None:
    people = tree["people"]
    relations = tree["parent_child"]
    pairs = tree["sibling_pairs"]
    spouses = tree["spouse_pairs"]
    assert isinstance(people, list)
    assert isinstance(relations, list)
    assert isinstance(pairs, list)
    assert isinstance(spouses, list)
    connected = {
        name.casefold()
        for relation in relations
        for name in (relation["parent"], relation["child"])
    }
    connected.update(
        name.casefold()
        for pair in pairs
        for name in (pair["first"], pair["second"])
    )
    connected.update(
        name.casefold()
        for pair in spouses
        for name in (pair["first"], pair["second"])
    )
    tree["people"] = [
        person for person in people if person.casefold() in connected
    ]


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
    siblings = tree["sibling_pairs"]
    assert isinstance(relations, list)
    assert isinstance(siblings, list)
    for relation in relations:
        parent_x, parent_y = positions[relation["parent"].casefold()]
        child_x, child_y = positions[relation["child"].casefold()]
        lines.append(
            f'<path d="M {parent_x + card_width / 2} {parent_y + card_height} '
            f'V {parent_y + card_height + vertical_gap / 2} '
            f'H {child_x + card_width / 2} V {child_y}" '
            'fill="none" stroke="#73e0b0" stroke-width="2"/>'
        )
    for pair in siblings:
        first_x, first_y = positions[pair["first"].casefold()]
        second_x, second_y = positions[pair["second"].casefold()]
        connector_y = first_y + card_height + 28
        first_center = first_x + card_width / 2
        second_center = second_x + card_width / 2
        lines.append(
            f'<path d="M {first_center} {first_y + card_height} '
            f'V {connector_y} H {second_center} '
            f'V {second_y + card_height}" fill="none" stroke="#91a9ff" '
            'stroke-width="2" stroke-dasharray="6 4"/>'
        )
    for pair in tree["spouse_pairs"]:
        first_x, first_y = positions[pair["first"].casefold()]
        second_x, second_y = positions[pair["second"].casefold()]
        if first_x < second_x:
            left_edge, right_edge = first_x + card_width, second_x
        else:
            left_edge, right_edge = second_x + card_width, first_x
        center_y = first_y + card_height / 2
        midpoint = (left_edge + right_edge) / 2
        lines.extend(
            (
                f'<path d="M {left_edge} {center_y - 3} H {right_edge}" '
                'fill="none" stroke="#f0a6ca" stroke-width="2"/>',
                f'<path d="M {left_edge} {center_y + 3} H {right_edge}" '
                'fill="none" stroke="#f0a6ca" stroke-width="2"/>',
                f'<text x="{midpoint}" y="{center_y - 9}" '
                'text-anchor="middle" fill="#f0a6ca" '
                'font-family="sans-serif" font-size="11">coniugi</text>',
            )
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
    siblings = tree.get("sibling_pairs", [])
    spouses = tree.get("spouse_pairs", [])
    if (
        not isinstance(people, list)
        or not isinstance(relations, list)
        or not isinstance(siblings, list)
        or not isinstance(spouses, list)
    ):
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

    sibling_parent = {name: name for name in canonical}

    def find_sibling_root(person: str) -> str:
        while sibling_parent[person] != person:
            sibling_parent[person] = sibling_parent[sibling_parent[person]]
            person = sibling_parent[person]
        return person

    seen_pairs: set[frozenset[str]] = set()
    for relation_name, pairs in (("fratellanza", siblings), ("coniugio", spouses)):
        for pair in pairs:
            if not isinstance(pair, dict):
                raise ValueError(
                    f"L'albero genealogico contiene una relazione di {relation_name} non valida."
                )
            first = pair.get("first")
            second = pair.get("second")
            if (
                not isinstance(first, str)
                or not isinstance(second, str)
                or first.casefold() not in canonical
                or second.casefold() not in canonical
                or first.casefold() == second.casefold()
            ):
                raise ValueError(
                    f"L'albero genealogico contiene una relazione di {relation_name} non valida."
                )
            pair_key = frozenset((first.casefold(), second.casefold()))
            if pair_key in seen_pairs:
                raise ValueError(
                    "L'albero genealogico contiene relazioni di coppia duplicate o incompatibili."
                )
            seen_pairs.add(pair_key)
            first_root = find_sibling_root(first.casefold())
            second_root = find_sibling_root(second.casefold())
            sibling_parent[second_root] = first_root

    parents_by_sibling_group: dict[str, set[str]] = {
        find_sibling_root(person): set() for person in canonical
    }
    for child, person_parents in parents.items():
        child_group = find_sibling_root(child)
        for parent in person_parents:
            parent_group = find_sibling_root(parent)
            if parent_group == child_group:
                raise ValueError(
                    "La relazione è incompatibile con la fratellanza dichiarata."
                )
            parents_by_sibling_group[child_group].add(parent_group)

    resolved: dict[str, int] = {}
    active: set[str] = set()

    def generation(group: str) -> int:
        if group in resolved:
            return resolved[group]
        if group in active:
            raise ValueError("La relazione creerebbe un ciclo nell'albero genealogico.")
        active.add(group)
        level = max(
            (
                generation(parent_group) + 1
                for parent_group in parents_by_sibling_group[group]
            ),
            default=0,
        )
        active.remove(group)
        resolved[group] = level
        return level

    for group in parents_by_sibling_group:
        generation(group)


def _ordered_generations(tree: dict[str, object]) -> list[list[str]]:
    _validate_tree(tree)
    people = tree["people"]
    relations = tree["parent_child"]
    assert isinstance(people, list)
    assert isinstance(relations, list)
    if not people:
        return []
    siblings = tree.get("sibling_pairs", [])
    assert isinstance(siblings, list)
    spouses = tree["spouse_pairs"]
    assert isinstance(spouses, list)
    names = {person.casefold(): person for person in people}
    sibling_parent = {name: name for name in names}

    def find_sibling_root(person: str) -> str:
        while sibling_parent[person] != person:
            sibling_parent[person] = sibling_parent[sibling_parent[person]]
            person = sibling_parent[person]
        return person

    for pair in siblings:
        first_root = find_sibling_root(pair["first"].casefold())
        second_root = find_sibling_root(pair["second"].casefold())
        sibling_parent[second_root] = first_root
    for pair in spouses:
        first_root = find_sibling_root(pair["first"].casefold())
        second_root = find_sibling_root(pair["second"].casefold())
        sibling_parent[second_root] = first_root

    parents: dict[str, set[str]] = {
        find_sibling_root(person): set() for person in names
    }
    for relation in relations:
        parent_group = find_sibling_root(relation["parent"].casefold())
        child_group = find_sibling_root(relation["child"].casefold())
        parents[child_group].add(parent_group)
    resolved: dict[str, int] = {}

    def generation(group: str, active: set[str] | None = None) -> int:
        if group in resolved:
            return resolved[group]
        if active is None:
            active = set()
        if group in active:
            raise ValueError("La relazione creerebbe un ciclo nell'albero genealogico.")
        active.add(group)
        resolved[group] = max(
            (generation(parent, active) + 1 for parent in parents[group]),
            default=0,
        )
        active.remove(group)
        return resolved[group]

    for person in parents:
        generation(person)
    ordered: list[list[str]] = []
    for level in range(max(resolved.values()) + 1):
        people_at_level = [
            person
            for person in names
            if resolved[find_sibling_root(person)] == level
        ]
        ordered.append(
            sorted(
                (names[person] for person in people_at_level),
                key=str.casefold,
            )
        )
    return ordered
