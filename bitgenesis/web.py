"""Serve a local browser interface for BitGenesis."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import threading
from typing import TYPE_CHECKING, cast
from urllib.parse import urlsplit
import webbrowser

from bitgenesis.agents import (
    AGENT_INSTRUCTIONS,
    AgentName,
    simple_greeting,
    self_introduction,
)
from bitgenesis.config import Config
from bitgenesis.effort import (
    EffortLevel,
    choose_effort,
    effort_token_budget,
    is_effort_level,
)
from bitgenesis.genesis import read_instructions, update_instructions

if TYPE_CHECKING:
    from bitgenesis.generate import ConversationChat

_LOCAL_USER_NAME = "Utente"
_MAX_REQUEST_BYTES = 64 * 1024
_MAX_LOG_CHARS = 20_000


class WebRequestError(Exception):
    """An expected request failure with an HTTP status code."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class WebApplication:
    """Hold the in-memory state for one local browser session."""

    def __init__(
        self,
        config: Config,
        model_path: Path,
        tokenizer_path: Path,
        *,
        max_new_tokens: int,
        temperature: float,
        effort: EffortLevel,
        genesis_path: Path | None = None,
    ) -> None:
        self.config = config
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.effort = effort
        self.genesis_path = (
            Path.cwd() / "Genesis.md"
            if genesis_path is None
            else Path(genesis_path)
        )
        self.history: list[tuple[str, str]] = []
        self.chat: ConversationChat | None = None
        self.training_status = "idle"
        self.training_log = ""
        self.training_process: subprocess.Popen[str] | None = None
        self.lock = threading.RLock()

    def status(self) -> dict[str, object]:
        with self.lock:
            return {
                "model_ready": (
                    self.model_path.is_file() and self.tokenizer_path.is_file()
                ),
                "training_status": self.training_status,
                "training_log": self.training_log,
            }

    def instructions(self) -> str:
        try:
            return read_instructions(self.genesis_path)
        except (OSError, UnicodeError) as error:
            raise WebRequestError(
                500, f"Impossibile leggere le istruzioni: {error}"
            ) from error

    def save_instructions(self, action: str, text: str) -> str:
        if action not in {"append", "replace"}:
            raise WebRequestError(400, "Azione per le istruzioni non valida.")
        if not text.strip():
            raise WebRequestError(400, "Le istruzioni non possono essere vuote.")
        try:
            update_instructions(
                self.genesis_path,
                text,
                replace=action == "replace",
            )
        except ValueError as error:
            raise WebRequestError(
                400, f"Impossibile aggiornare le istruzioni: {error}"
            ) from error
        except OSError as error:
            raise WebRequestError(
                500, f"Impossibile aggiornare le istruzioni: {error}"
            ) from error
        return self.instructions()

    def reset_chat(self) -> None:
        with self.lock:
            self.history.clear()

    def reply(self, body: dict[str, object]) -> dict[str, object]:
        message = body.get("message")
        effort = body.get("effort", self.effort)
        agent = body.get("agent", "chat")
        if not isinstance(message, str) or not message.strip():
            raise WebRequestError(400, "Scrivi un messaggio prima di inviarlo.")
        if len(message) > 16_000:
            raise WebRequestError(400, "Il messaggio supera il limite consentito.")
        if not isinstance(effort, str) or not is_effort_level(effort):
            raise WebRequestError(400, "Livello di impegno non valido.")
        if not isinstance(agent, str) or agent not in {"chat", "coding"}:
            raise WebRequestError(400, "Profilo non valido.")

        selected_agent = cast(AgentName, agent)
        introduction = self_introduction(message.strip(), selected_agent)
        if introduction is not None:
            with self.lock:
                self.history.append((message.strip(), introduction))
            return {
                "response": introduction,
                "effort": "non applicato",
                "generated": False,
            }

        with self.lock:
            if self.training_status == "running":
                raise WebRequestError(
                    409, "Attendi la fine dell'addestramento prima di chattare."
                )
            if not (self.model_path.is_file() and self.tokenizer_path.is_file()):
                raise WebRequestError(
                    409,
                    "Modello non disponibile. Usa il pannello Addestramento "
                    "per creare un modello conversazionale.",
                )

            greeting = simple_greeting(message.strip(), selected_agent)
            if greeting is not None:
                self.history.append((message.strip(), greeting))
                return {
                    "response": greeting,
                    "effort": "non applicato",
                    "generated": False,
                }

            try:
                from bitgenesis.generate import (
                    ContextLengthError,
                    ConversationChat,
                )

                if self.chat is None:
                    self.chat = ConversationChat(
                        self.model_path,
                        self.tokenizer_path,
                    )
                project_instructions = self.instructions()
            except WebRequestError:
                raise
            except Exception as error:
                raise WebRequestError(
                    500, f"Impossibile caricare il modello locale: {error}"
                ) from error

            clean_message = message.strip()
            prompt_sections = []
            if agent == "coding":
                prompt_sections.append(
                    "Istruzioni per l'agente coding:\n"
                    f"{AGENT_INSTRUCTIONS['coding']}"
                )
            if project_instructions:
                prompt_sections.append(
                    "Istruzioni permanenti del progetto da Genesis.md:\n"
                    f"{project_instructions}"
                )
            prompt = (
                "\n\n".join(prompt_sections)
                + f"\n\nRichiesta dell'utente: {clean_message}"
                if prompt_sections
                else clean_message
            )
            selected_effort = choose_effort(clean_message, cast(EffortLevel, effort))
            token_budget = effort_token_budget(
                self.max_new_tokens,
                selected_effort,
            )
            chat_history = [
                *self.history,
                (
                    f"Regola per questa chat locale: chiama l'utente "
                    f"{_LOCAL_USER_NAME}.",
                    f"In questa chat locale chiamerò l'utente {_LOCAL_USER_NAME}.",
                ),
            ]
            try:
                response = self.chat.reply(
                    prompt,
                    chat_history,
                    max_new_tokens=token_budget,
                    temperature=self.temperature,
                )
            except ContextLengthError as error:
                raise WebRequestError(400, str(error)) from error
            except Exception as error:
                raise WebRequestError(
                    500, f"Generazione della risposta non riuscita: {error}"
                ) from error
            self.history.append((clean_message, response))

            return {
                "response": response or "[risposta vuota]",
                "effort": selected_effort,
                "generated": True,
            }

    def start_training(self, body: dict[str, object]) -> dict[str, str]:
        steps = body.get("steps", 1000)
        if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 100_000:
            raise WebRequestError(
                400, "I passi di addestramento devono essere tra 1 e 100000."
            )
        if not self.config.datasets_dir.is_dir() or not any(
            path.is_file()
            for path in self.config.datasets_dir.rglob("*.jsonl")
        ):
            raise WebRequestError(
                400,
                "Non trovo dati conversazionali .jsonl in "
                f"{self.config.datasets_dir}.",
            )

        with self.lock:
            if self.training_status == "running":
                raise WebRequestError(409, "L'addestramento è già in corso.")
            self.training_status = "running"
            self.training_log = "Avvio dell'addestramento conversazionale...\n"
            worker = threading.Thread(
                target=self._run_training,
                args=(steps,),
                name="bitgenesis-training",
                daemon=True,
            )
            worker.start()
        return {"training_status": "running"}

    def _run_training(self, steps: int) -> None:
        command = [
            sys.executable,
            "-u",
            "-m",
            "bitgenesis.train",
            "--data-format",
            "conversations",
            "--steps",
            str(steps),
        ]
        try:
            process = subprocess.Popen(
                command,
                cwd=Path.cwd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            with self.lock:
                self.training_process = process
            if process.stdout is None:
                raise RuntimeError("Impossibile leggere l'output dell'addestramento.")
            with process.stdout:
                for line in process.stdout:
                    with self.lock:
                        self.training_log = (
                            self.training_log + line
                        )[-_MAX_LOG_CHARS:]
            return_code = process.wait()
        except (OSError, RuntimeError) as error:
            with self.lock:
                self.training_log = (
                    self.training_log + f"\nAddestramento non avviato: {error}"
                )[-_MAX_LOG_CHARS:]
                self.training_status = "failed"
                self.training_process = None
            return

        with self.lock:
            self.training_status = "completed" if return_code == 0 else "failed"
            if return_code == 0:
                self.training_log += "\nAddestramento completato."
                self.chat = None
                self.history.clear()
            else:
                self.training_log += (
                    f"\nAddestramento terminato con codice {return_code}."
                )
            self.training_log = self.training_log[-_MAX_LOG_CHARS:]
            self.training_process = None


class LocalHTTPServer(ThreadingHTTPServer):
    """HTTP server bound to the local machine only."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], app: WebApplication) -> None:
        self.app = app
        super().__init__(address, LocalRequestHandler)


class LocalRequestHandler(BaseHTTPRequestHandler):
    server: LocalHTTPServer

    def do_GET(self) -> None:
        if not self._is_local_request():
            self._send_json(403, {"error": "Richiesta locale non valida."})
            return
        route = urlsplit(self.path).path
        if route == "/" or route == "/index.html":
            try:
                page = Path(__file__).with_name("web.html").read_bytes()
            except OSError as error:
                self._send_json(500, {"error": f"Interfaccia non disponibile: {error}"})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self._send_security_headers()
            self.end_headers()
            self.wfile.write(page)
        elif route == "/style.css":
            try:
                stylesheet = Path(__file__).with_name("style.css").read_bytes()
            except OSError as error:
                self._send_json(500, {"error": f"Foglio di stile non disponibile: {error}"})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/css; charset=utf-8")
            self.send_header("Content-Length", str(len(stylesheet)))
            self._send_security_headers()
            self.end_headers()
            self.wfile.write(stylesheet)
        elif route == "/api/status":
            self._send_json(200, self.server.app.status())
        elif route == "/api/instructions":
            try:
                self._send_json(
                    200,
                    {"instructions": self.server.app.instructions()},
                )
            except WebRequestError as error:
                self._send_json(error.status, {"error": str(error)})
        else:
            self._send_json(404, {"error": "Pagina non trovata."})

    def do_POST(self) -> None:
        if not self._is_local_request() or not self._is_same_origin_request():
            self._send_json(403, {"error": "Richiesta locale non valida."})
            return
        try:
            body = self._read_json_body()
            route = urlsplit(self.path).path
            if route == "/api/chat":
                result = self.server.app.reply(body)
            elif route == "/api/train":
                result = self.server.app.start_training(body)
            elif route == "/api/reset":
                self.server.app.reset_chat()
                result = {"status": "ok"}
            elif route == "/api/instructions":
                action = body.get("action")
                text = body.get("text")
                if not isinstance(action, str) or not isinstance(text, str):
                    raise WebRequestError(
                        400, "Specifica un'azione e il testo delle istruzioni."
                    )
                result = {
                    "instructions": self.server.app.save_instructions(action, text)
                }
            else:
                self._send_json(404, {"error": "Endpoint non trovato."})
                return
        except WebRequestError as error:
            self._send_json(error.status, {"error": str(error)})
            return
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
            self._send_json(400, {"error": f"Richiesta non valida: {error}"})
            return
        except Exception as error:
            self._send_json(500, {"error": f"Operazione non riuscita: {error}"})
            return
        self._send_json(200, result)

    def _read_json_body(self) -> dict[str, object]:
        content_length = self.headers.get("Content-Length")
        if content_length is None:
            raise WebRequestError(411, "La richiesta deve includere un corpo.")
        try:
            size = int(content_length)
        except ValueError as error:
            raise WebRequestError(400, "Dimensione della richiesta non valida.") from error
        if size < 0 or size > _MAX_REQUEST_BYTES:
            raise WebRequestError(413, "La richiesta supera il limite consentito.")
        if self.headers.get_content_type() != "application/json":
            raise WebRequestError(415, "Il contenuto deve essere JSON.")
        data = json.loads(self.rfile.read(size).decode("utf-8"))
        if not isinstance(data, dict):
            raise WebRequestError(400, "Il corpo della richiesta deve essere un oggetto.")
        return cast(dict[str, object], data)

    def _is_local_request(self) -> bool:
        host_header = self.headers.get("Host", "")
        try:
            host = urlsplit(f"//{host_header}")
            port = host.port
        except ValueError:
            return False
        return (
            host.hostname in {"127.0.0.1", "localhost"}
            and port == self.server.server_port
        )

    def _is_same_origin_request(self) -> bool:
        origin = self.headers.get("Origin")
        if origin is None:
            return True
        try:
            parsed_origin = urlsplit(origin)
            origin_port = parsed_origin.port
        except ValueError:
            return False
        host_header = self.headers.get("Host", "")
        return (
            parsed_origin.scheme == "http"
            and parsed_origin.netloc.casefold() == host_header.casefold()
            and parsed_origin.hostname in {"127.0.0.1", "localhost"}
            and origin_port == self.server.server_port
        )

    def _send_json(self, status: int, data: dict[str, object]) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self._send_security_headers()
        self.end_headers()
        self.wfile.write(payload)

    def _send_security_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self'; "
            "script-src 'self' 'unsafe-inline'; connect-src 'self'; "
            "form-action 'self'; frame-ancestors 'none'",
        )

    def log_message(self, format: str, *args: object) -> None:
        print(f"BitGenesis web: {format % args}")


def run_web_server(
    *,
    config: Config,
    model_path: Path,
    tokenizer_path: Path,
    max_new_tokens: int,
    temperature: float,
    effort: EffortLevel,
    port: int = 8765,
    open_browser: bool = True,
) -> None:
    """Start the local-only browser interface and optionally open it."""
    app = WebApplication(
        config,
        model_path,
        tokenizer_path,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        effort=effort,
    )
    server = LocalHTTPServer(("127.0.0.1", port), app)
    address = f"http://127.0.0.1:{server.server_port}"
    print(f"BitGenesis è disponibile nel browser: {address}")
    if open_browser:
        webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nInterfaccia BitGenesis arrestata.")
    finally:
        with app.lock:
            process = app.training_process
        if process is not None and process.poll() is None:
            process.terminate()
        server.server_close()


__all__ = ("WebApplication", "run_web_server")
