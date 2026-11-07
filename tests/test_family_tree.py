import tempfile
import unittest
from pathlib import Path

from bitgenesis.family_tree import (
    add_parent_child,
    add_sibling_pair,
    add_spouse_pair,
    empty_family_tree,
    is_family_tree_statement,
    parse_family_relations,
    parse_sibling_relations,
    parse_spouse_relations,
    read_family_tree,
    remove_parent_child,
    remove_sibling_pair,
    remove_spouse_pair,
    render_family_tree_svg,
    save_family_tree,
)


class FamilyTreeTests(unittest.TestCase):
    def test_extracts_only_explicit_parent_child_statements(self) -> None:
        self.assertEqual(
            parse_family_relations(
                "TestPersonB è figlio di TestPersonA. TestPersonA è la madre di TestPersonB."
            ),
            [("TestPersonA", "TestPersonB"), ("TestPersonA", "TestPersonB")],
        )
        self.assertEqual(
            parse_family_relations(
                "TestPersonH è il padre di TestPersonI e di TestPersonJ"
            ),
            [("TestPersonH", "TestPersonI"), ("TestPersonH", "TestPersonJ")],
        )
        self.assertEqual(
            parse_family_relations("TestPersonH è padre di TestPersonI e TestPersonJ."),
            [("TestPersonH", "TestPersonI"), ("TestPersonH", "TestPersonJ")],
        )
        self.assertTrue(
            is_family_tree_statement(
                "TestPersonH è il padre di TestPersonI e di TestPersonJ"
            )
        )
        self.assertEqual(
            parse_family_relations("Mio padre è TestPersonF; Mia figlia è TestPersonD."),
            [("TestPersonF", "Io"), ("Io", "TestPersonD")],
        )
        self.assertEqual(
            parse_family_relations("My mother is TestPersonE. TestPersonC is my father."),
            [("TestPersonE", "Io"), ("TestPersonC", "Io")],
        )
        self.assertEqual(
            parse_family_relations("Ieri ho parlato con TestPersonB."),
            [],
        )
        self.assertFalse(
            is_family_tree_statement(
                "TestPersonB è figlio di TestPersonA. Qual è il suo cognome?"
            )
        )
        self.assertTrue(
            is_family_tree_statement("TestPersonB è figlio di TestPersonA; TestPersonA è madre di TestPersonB.")
        )

    def test_extracts_explicit_sibling_statements(self) -> None:
        self.assertEqual(
            parse_sibling_relations(
                "TestPersonA e TestPersonB sono fratelli. TestPersonC is TestPersonE's brother."
            ),
            [("TestPersonA", "TestPersonB"), ("TestPersonC", "TestPersonE")],
        )
        self.assertEqual(
            parse_sibling_relations("TestPersonA è sorella di TestPersonB; TestPersonC and TestPersonE are siblings."),
            [("TestPersonA", "TestPersonB"), ("TestPersonC", "TestPersonE")],
        )
        self.assertTrue(is_family_tree_statement("TestPersonA e TestPersonB sono fratelli."))
        self.assertFalse(is_family_tree_statement("TestPersonA e TestPersonB giocano insieme."))

    def test_extracts_explicit_spouse_statements(self) -> None:
        self.assertEqual(
            parse_spouse_relations("TestPersonH e TestPersonK sono marito e moglie."),
            [("TestPersonH", "TestPersonK")],
        )
        self.assertEqual(
            parse_spouse_relations("TestPersonE è moglie di TestPersonC; TestPersonC is married to TestPersonE."),
            [("TestPersonE", "TestPersonC"), ("TestPersonC", "TestPersonE")],
        )
        self.assertTrue(
            is_family_tree_statement("TestPersonH e TestPersonK sono marito e moglie")
        )

    def test_adds_unique_relations_and_rejects_cycles(self) -> None:
        tree = empty_family_tree()
        self.assertTrue(add_parent_child(tree, "TestPersonA", "TestPersonB"))
        self.assertFalse(add_parent_child(tree, "testpersona", "TESTPERSONB"))
        self.assertTrue(add_parent_child(tree, "TestPersonB", "TestPersonC"))

        with self.assertRaisesRegex(ValueError, "ciclo"):
            add_parent_child(tree, "TestPersonC", "TestPersonA")

        self.assertEqual(
            tree["parent_child"],
            [
                {"parent": "TestPersonA", "child": "TestPersonB"},
                {"parent": "TestPersonB", "child": "TestPersonC"},
            ],
        )

    def test_saves_and_orders_svg_by_generation(self) -> None:
        tree = empty_family_tree()
        add_parent_child(tree, "TestPersonA", "TestPersonB")
        add_parent_child(tree, "TestPersonB", "TestPersonC")
        add_parent_child(tree, "TestPersonA", "TestPersonG")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "AlberoGenealogico.json"

            save_family_tree(path, tree)

            self.assertEqual(read_family_tree(path), tree)
            svg = path.with_suffix(".svg").read_text(encoding="utf-8")
            self.assertLess(svg.index(">TestPersonA</text>"), svg.index(">TestPersonG</text>"))
            self.assertLess(svg.index(">TestPersonA</text>"), svg.index(">TestPersonB</text>"))
            self.assertLess(svg.index(">TestPersonB</text>"), svg.index(">TestPersonC</text>"))
            self.assertIn("Generazione 1", svg)
            self.assertIn("Generazione 3", svg)
            self.assertFalse(path.with_name(".AlberoGenealogico.json.tmp").exists())
            self.assertFalse(path.with_name(".AlberoGenealogico.svg.tmp").exists())

    def test_siblings_are_saved_at_the_same_generation_and_rendered(self) -> None:
        tree = empty_family_tree()
        add_parent_child(tree, "TestPersonA", "TestPersonB")
        add_sibling_pair(tree, "TestPersonB", "TestPersonC")
        self.assertFalse(add_sibling_pair(tree, "TESTPERSONC", "testpersonb"))

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "AlberoGenealogico.json"
            save_family_tree(path, tree)

            restored = read_family_tree(path)
            self.assertEqual(
                restored["sibling_pairs"],
                [{"first": "TestPersonB", "second": "TestPersonC"}],
            )
            svg = path.with_suffix(".svg").read_text(encoding="utf-8")
            person_b = svg.index(">TestPersonB</text>")
            person_c = svg.index(">TestPersonC</text>")
            self.assertLess(svg.index("Generazione 2", person_b), person_c)
            self.assertIn('stroke="#91a9ff"', svg)
            self.assertIn('stroke-dasharray="6 4"', svg)

    def test_sibling_groups_inherit_known_generation(self) -> None:
        tree = empty_family_tree()
        add_parent_child(tree, "TestPersonA", "TestPersonB")
        add_sibling_pair(tree, "TestPersonB", "TestPersonC")
        add_parent_child(tree, "TestPersonC", "TestPersonD")

        svg = render_family_tree_svg(tree)

        self.assertLess(svg.index(">TestPersonB</text>"), svg.index(">TestPersonD</text>"))
        self.assertLess(svg.index(">TestPersonC</text>"), svg.index(">TestPersonD</text>"))
        self.assertIn("Generazione 3", svg)

    def test_spouses_are_saved_at_the_same_generation_and_rendered(self) -> None:
        tree = empty_family_tree()
        add_parent_child(tree, "TestPersonH", "TestPersonI")
        self.assertTrue(add_spouse_pair(tree, "TestPersonH", "TestPersonK"))
        self.assertFalse(add_spouse_pair(tree, "testpersonk", "TESTPERSONH"))

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "AlberoGenealogico.json"
            save_family_tree(path, tree)

            restored = read_family_tree(path)
            self.assertEqual(
                restored["spouse_pairs"],
                [{"first": "TestPersonH", "second": "TestPersonK"}],
            )
            svg = path.with_suffix(".svg").read_text(encoding="utf-8")
            self.assertLess(
                svg.index("Generazione 1", svg.index(">TestPersonH</text>")),
                svg.index(">TestPersonK</text>"),
            )
            self.assertIn('stroke="#f0a6ca"', svg)
            self.assertIn(">coniugi</text>", svg)

    def test_removing_spouse_pair_prunes_disconnected_people(self) -> None:
        tree = empty_family_tree()
        add_spouse_pair(tree, "TestPersonH", "TestPersonK")

        self.assertTrue(remove_spouse_pair(tree, "TestPersonK", "TestPersonH"))
        self.assertEqual(tree["people"], [])
        self.assertFalse(remove_spouse_pair(tree, "TestPersonH", "TestPersonK"))

    def test_removing_sibling_pair_prunes_disconnected_people(self) -> None:
        tree = empty_family_tree()
        add_sibling_pair(tree, "TestPersonA", "TestPersonB")

        self.assertTrue(remove_sibling_pair(tree, "TestPersonB", "TestPersonA"))
        self.assertEqual(tree["people"], [])
        self.assertFalse(remove_sibling_pair(tree, "TestPersonA", "TestPersonB"))

    def test_reads_legacy_tree_without_sibling_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "AlberoGenealogico.json"
            path.write_text(
                '{"version": 1, "people": ["TestPersonA", "TestPersonB"], '
                '"parent_child": [{"parent": "TestPersonA", "child": "TestPersonB"}]}',
                encoding="utf-8",
            )

            self.assertEqual(
                read_family_tree(path)["sibling_pairs"],
                [],
            )
            self.assertEqual(read_family_tree(path)["spouse_pairs"], [])

    def test_escapes_names_in_svg(self) -> None:
        tree = empty_family_tree()
        add_parent_child(tree, "A&B", "Figlio <uno>")

        svg = render_family_tree_svg(tree)

        self.assertIn("A&amp;B", svg)
        self.assertIn("Figlio &lt;uno&gt;", svg)

    def test_removing_relation_prunes_disconnected_people(self) -> None:
        tree = empty_family_tree()
        add_parent_child(tree, "TestPersonA", "TestPersonB")
        add_parent_child(tree, "TestPersonB", "TestPersonC")

        self.assertTrue(remove_parent_child(tree, "TestPersonA", "TestPersonB"))
        self.assertEqual(tree["people"], ["TestPersonB", "TestPersonC"])
        self.assertFalse(remove_parent_child(tree, "TestPersonA", "TestPersonB"))


if __name__ == "__main__":
    unittest.main()
