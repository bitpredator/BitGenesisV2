import http.client
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from bitgenesis.config import load_config
from bitgenesis.web import LocalHTTPServer, WebApplication


class WebInterfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        root = Path(self.temporary_directory.name)
        self.model_path = root / "models" / "bitgenesis.pt"
        self.tokenizer_path = root / "models" / "tokenizer.json"
        self.genesis_path = root / "Genesis.md"
        config = replace(
            load_config(),
            datasets_dir=root / "datasets",
            models_dir=root / "models",
        )
        self.app = WebApplication(
            config,
            self.model_path,
            self.tokenizer_path,
            max_new_tokens=80,
            temperature=1.0,
            effort="auto",
            genesis_path=self.genesis_path,
        )
        self.server = LocalHTTPServer(("127.0.0.1", 0), self.app)
        self.addCleanup(self.server.server_close)
        self.server_thread = threading.Thread(
            target=self.server.serve_forever,
            daemon=True,
        )
        self.server_thread.start()
        self.addCleanup(self.server_thread.join, 2)
        self.addCleanup(self.server.shutdown)

    def request(
        self,
        method: str,
        path: str,
        body: dict[str, object] | None = None,
        *,
        origin: str | None = None,
    ) -> tuple[int, dict[str, object] | str]:
        connection = http.client.HTTPConnection(
            "127.0.0.1",
            self.server.server_port,
            timeout=5,
        )
        self.addCleanup(connection.close)
        headers = {}
        payload = None
        if body is not None:
            payload = json.dumps(body)
            headers["Content-Type"] = "application/json"
        if origin is not None:
            headers["Origin"] = origin
        connection.request(method, path, body=payload, headers=headers)
        response = connection.getresponse()
        content = response.read().decode("utf-8")
        if response.getheader("Content-Type", "").startswith("application/json"):
            return response.status, json.loads(content)
        return response.status, content

    def test_serves_browser_page_and_reports_model_status(self) -> None:
        page_status, page = self.request("GET", "/")
        status_code, status = self.request("GET", "/api/status")

        self.assertEqual(page_status, 200)
        self.assertIsInstance(page, str)
        self.assertIn("BitGenesis", page)
        self.assertIn('<link rel="stylesheet" href="/style.css">', page)
        stylesheet_status, stylesheet = self.request("GET", "/style.css")
        self.assertEqual(stylesheet_status, 200)
        self.assertIsInstance(stylesheet, str)
        self.assertIn(":root", stylesheet)
        self.assertEqual(status_code, 200)
        self.assertEqual(
            status,
            {
                "model_ready": False,
                "training_status": "idle",
                "training_log": "",
            },
        )

    def test_rejects_cross_origin_requests(self) -> None:
        status, response = self.request(
            "POST",
            "/api/reset",
            {},
            origin=f"http://example.test:{self.server.server_port}",
        )

        self.assertEqual(status, 403)
        self.assertIn("error", response)

    def test_chat_reports_missing_model_without_loading_it(self) -> None:
        with patch("bitgenesis.generate.ConversationChat") as chat_class:
            status, response = self.request(
                "POST",
                "/api/chat",
                {"message": "Ciao", "effort": "auto", "agent": "chat"},
                origin=f"http://127.0.0.1:{self.server.server_port}",
            )

        self.assertEqual(status, 409)
        self.assertIn("Modello non disponibile", response["error"])
        chat_class.assert_not_called()

    def test_chat_uses_local_model_and_selected_profile(self) -> None:
        self.model_path.parent.mkdir(parents=True)
        self.model_path.touch()
        self.tokenizer_path.touch()
        with patch("bitgenesis.generate.ConversationChat") as chat_class:
            chat_class.return_value.reply.return_value = "Ciao, Utente!"
            status, response = self.request(
                "POST",
                "/api/chat",
                {
                    "message": "Scrivi una funzione Python",
                    "effort": "alto",
                    "agent": "coding",
                },
                origin=f"http://127.0.0.1:{self.server.server_port}",
            )

        self.assertEqual(status, 200)
        self.assertEqual(response, {"response": "Ciao, Utente!", "effort": "alto"})
        chat_class.assert_called_once_with(self.model_path, self.tokenizer_path)
        call = chat_class.return_value.reply.call_args
        self.assertIn("Istruzioni per l'agente coding", call.args[0])
        self.assertEqual(call.kwargs["max_new_tokens"], 160)

    def test_saves_and_reads_project_instructions(self) -> None:
        status, response = self.request(
            "POST",
            "/api/instructions",
            {"action": "replace", "text": "Usa risposte concise."},
            origin=f"http://127.0.0.1:{self.server.server_port}",
        )

        self.assertEqual(status, 200)
        self.assertEqual(
            response["instructions"],
            "Usa risposte concise.",
        )
        self.assertEqual(
            self.genesis_path.read_text(encoding="utf-8"),
            "# Istruzioni permanenti per BitGenesis\n\nUsa risposte concise.\n",
        )

    def test_rejects_invalid_chat_settings(self) -> None:
        status, response = self.request(
            "POST",
            "/api/chat",
            {"message": "Ciao", "effort": "impossibile"},
            origin=f"http://127.0.0.1:{self.server.server_port}",
        )

        self.assertEqual(status, 400)
        self.assertIn("Livello di impegno non valido", response["error"])


if __name__ == "__main__":
    unittest.main()
