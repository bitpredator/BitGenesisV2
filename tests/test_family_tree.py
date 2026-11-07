import tempfile
import unittest
from pathlib import Path

from bitgenesis.family_tree import (
    add_parent_child,
    empty_family_tree,
    is_family_tree_statement,
    parse_family_relations,
    read_family_tree,
    remove_parent_child,
    render_family_tree_svg,
    save_family_tree,
)


class FamilyTreeTests(unittest.TestCase):
    def test_extracts_only_explicit_parent_child_statements(self) -> None:
        self.assertEqual(
            parse_family_relations(
                "Carlo è figlio di Anna. Anna è la madre di Carlo."
            ),
            [("Anna", "Carlo"), ("Anna", "Carlo")],
        )
        self.assertEqual(
            parse_family_relations("Mio padre è Marco; Mia figlia è Giulia."),
            [("Marco", "Io"), ("Io", "Giulia")],
        )
        self.assertEqual(
            parse_family_relations("My mother is Maria. Luca is my father."),
            [("Maria", "Io"), ("Luca", "Io")],
        )
        self.assertEqual(
            parse_family_relations("Ieri ho parlato con Carlo."),
            [],
        )
        self.assertFalse(
            is_family_tree_statement(
                "Carlo è figlio di Anna. Qual è il suo cognome?"
            )
        )
        self.assertTrue(
            is_family_tree_statement("Carlo è figlio di Anna; Anna è madre di Carlo.")
        )

    def test_adds_unique_relations_and_rejects_cycles(self) -> None:
        tree = empty_family_tree()
        self.assertTrue(add_parent_child(tree, "Anna", "Carlo"))
        self.assertFalse(add_parent_child(tree, "anna", "CARLO"))
        self.assertTrue(add_parent_child(tree, "Carlo", "Luca"))

        with self.assertRaisesRegex(ValueError, "ciclo"):
            add_parent_child(tree, "Luca", "Anna")

        self.assertEqual(
            tree["parent_child"],
            [
                {"parent": "Anna", "child": "Carlo"},
                {"parent": "Carlo", "child": "Luca"},
            ],
        )

    def test_saves_and_orders_svg_by_generation(self) -> None:
        tree = empty_family_tree()
        add_parent_child(tree, "Anna", "Carlo")
        add_parent_child(tree, "Carlo", "Luca")
        add_parent_child(tree, "Anna", "Beatrice")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "AlberoGenealogico.json"

            save_family_tree(path, tree)

            self.assertEqual(read_family_tree(path), tree)
            svg = path.with_suffix(".svg").read_text(encoding="utf-8")
            self.assertLess(svg.index(">Anna</text>"), svg.index(">Beatrice</text>"))
            self.assertLess(svg.index(">Anna</text>"), svg.index(">Carlo</text>"))
            self.assertLess(svg.index(">Carlo</text>"), svg.index(">Luca</text>"))
            self.assertIn("Generazione 1", svg)
            self.assertIn("Generazione 3", svg)
            self.assertFalse(path.with_name(".AlberoGenealogico.json.tmp").exists())
            self.assertFalse(path.with_name(".AlberoGenealogico.svg.tmp").exists())

    def test_escapes_names_in_svg(self) -> None:
        tree = empty_family_tree()
        add_parent_child(tree, "A&B", "Figlio <uno>")

        svg = render_family_tree_svg(tree)

        self.assertIn("A&amp;B", svg)
        self.assertIn("Figlio &lt;uno&gt;", svg)

    def test_removing_relation_prunes_disconnected_people(self) -> None:
        tree = empty_family_tree()
        add_parent_child(tree, "Anna", "Carlo")
        add_parent_child(tree, "Carlo", "Luca")

        self.assertTrue(remove_parent_child(tree, "Anna", "Carlo"))
        self.assertEqual(tree["people"], ["Carlo", "Luca"])
        self.assertFalse(remove_parent_child(tree, "Anna", "Carlo"))


if __name__ == "__main__":
    unittest.main()
