"""Unit and integration tests for the Inkling-2-Mini local LLM toolkit."""

from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from toolkit.chat import (
    StreamPrinter,
    _save_transcript,
    build_direct_command,
    compose_user_prompt,
    parse_lora_scale_arg,
)
from toolkit.cli import build_parser, main
from toolkit.client import ChatResult, ServerClient, ServerClientError
from toolkit.config import ConfigError, format_lora_status, load_config, validate_effort
from toolkit.server import ServerStatus, build_server_command, read_log_tail


class FakeHTTPResponse:
    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self._buf = io.BytesIO(body)

    def read(self) -> bytes:
        return self._buf.read()

    def readline(self) -> bytes:
        return self._buf.readline()


class FakeHTTPConnection:
    def __init__(self, handler) -> None:
        self._handler = handler
        self.last_request: dict | None = None
        self.closed = False

    def request(self, method: str, path: str, body: bytes | None = None, headers: dict | None = None) -> None:
        self.last_request = {
            "method": method,
            "path": path,
            "body": body,
            "headers": headers or {},
        }

    def getresponse(self) -> FakeHTTPResponse:
        assert self.last_request is not None
        return self._handler(self.last_request)

    def close(self) -> None:
        self.closed = True


class ConfigTests(unittest.TestCase):
    def test_load_repo_config(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        cfg = load_config(repo_root, env={})
        self.assertEqual(cfg.paths.model.name, "Inkling-2-Mini-Q8_0.gguf")
        self.assertEqual(cfg.paths.lora.name, "Inkling-2-Mini-Uncensored-LoRA.gguf")
        self.assertEqual(cfg.lora.scale, 1.0)
        self.assertEqual(cfg.server.port, 9931)
        self.assertEqual(cfg.server_base_url, "http://127.0.0.1:9931")
        self.assertEqual(cfg.api_base_url, "http://127.0.0.1:9931/v1")

    def test_env_overrides_and_wildcard_host(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        cfg = load_config(
            repo_root,
            env={
                "INKLING_HOST": "0.0.0.0",
                "INKLING_PORT": "8088",
                "INKLING_LORA_SCALE": "0.75",
                "INKLING_EFFORT": "high",
                "INKLING_AUTO_START": "false",
            },
        )
        self.assertEqual(cfg.server.host, "0.0.0.0")
        self.assertEqual(cfg.client_host, "127.0.0.1")
        self.assertEqual(cfg.server_base_url, "http://127.0.0.1:8088")
        self.assertAlmostEqual(cfg.lora.scale, 0.75)
        self.assertEqual(cfg.generation.reasoning_effort, "high")
        self.assertFalse(cfg.server.auto_start)

    def test_invalid_effort_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            validate_effort("ultra")

    def test_format_lora_status(self) -> None:
        self.assertEqual(format_lora_status(1.0), "1.0 (abliterated)")
        self.assertEqual(format_lora_status(0.0), "0.0 (base)")
        self.assertEqual(format_lora_status(0.65), "0.65 (custom)")
        self.assertEqual(format_lora_status(None), "none")


class CommandBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parent.parent
        self.cfg = load_config(self.repo_root, env={})

    def test_server_command_default_abliterated(self) -> None:
        cmd = build_server_command(self.cfg)
        self.assertEqual(cmd[0], str(self.cfg.paths.llama_server))
        self.assertIn("--lora-scaled", cmd)
        idx = cmd.index("--lora-scaled")
        self.assertEqual(cmd[idx + 1], f"{self.cfg.paths.lora}:1")
        self.assertIn("--webui", cmd)
        self.assertIn("--jinja", cmd)
        self.assertIn("--reasoning-effort", cmd)
        self.assertEqual(cmd[cmd.index("--reasoning-effort") + 1], "medium")

    def test_server_command_base_mode_preloads_unapplied_lora(self) -> None:
        cmd = build_server_command(self.cfg, base_only=True, webui=False)
        self.assertNotIn("--lora-scaled", cmd)
        self.assertIn("--lora", cmd)
        self.assertIn("--lora-init-without-apply", cmd)
        self.assertIn("--no-webui", cmd)

    def test_server_command_no_lora(self) -> None:
        cmd = build_server_command(self.cfg, no_lora=True, extra_args=["--metrics"])
        self.assertNotIn("--lora", cmd)
        self.assertNotIn("--lora-scaled", cmd)
        self.assertEqual(cmd[-1], "--metrics")

    def test_direct_command_abliterated_vs_base_vs_attach(self) -> None:
        cmd_ab = build_direct_command(self.cfg, prompt="hi", single_turn=True, effort="none")
        self.assertEqual(cmd_ab[0], str(self.cfg.paths.llama_cli))
        self.assertIn("--lora-scaled", cmd_ab)
        self.assertIn("--single-turn", cmd_ab)
        self.assertIn("--reasoning-effort", cmd_ab)
        self.assertEqual(cmd_ab[cmd_ab.index("--reasoning-effort") + 1], "none")
        self.assertNotIn("--reasoning-budget", cmd_ab)

        cmd_base = build_direct_command(self.cfg, base_only=True)
        self.assertNotIn("--lora-scaled", cmd_base)

        cmd_attach = build_direct_command(self.cfg, attach=True, system="Be terse")
        self.assertIn("--server-base", cmd_attach)
        self.assertEqual(cmd_attach[cmd_attach.index("--server-base") + 1], "http://127.0.0.1:9931")
        self.assertNotIn("--model", cmd_attach)
        self.assertIn("--system-prompt", cmd_attach)


class ClientTests(unittest.TestCase):
    def test_sse_streaming_and_request_payload(self) -> None:
        recorded: dict = {}
        sse_body = (
            b'data: {"choices":[{"delta":{"reasoning_content":"Step 1. "}}]}\n\n'
            b'data: {"choices":[{"delta":{"reasoning_content":"Step 2."}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":"Final "}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":"answer."},"finish_reason":"stop"}],'
            b'"timings":{"prompt_n":12,"prompt_per_second":120.5,"predicted_n":8,"predicted_per_second":45.2}}\n\n'
            b"data: [DONE]\n\n"
        )

        def handler(req: dict) -> FakeHTTPResponse:
            recorded.update(req)
            return FakeHTTPResponse(200, sse_body)

        client = ServerClient("http://127.0.0.1:9931")
        reasoning_chunks: list[str] = []
        content_chunks: list[str] = []

        with mock.patch.object(client, "_connect", return_value=FakeHTTPConnection(handler)):
            res = client.chat_completion(
                [{"role": "user", "content": "hello"}],
                model="inkling-abliterated",
                stream=True,
                reasoning_effort="low",
                lora_scale=0.5,
                temperature=0.7,
                max_tokens=128,
                on_reasoning=reasoning_chunks.append,
                on_content=content_chunks.append,
            )

        sent = json.loads(recorded["body"].decode("utf-8"))
        self.assertEqual(sent["reasoning_effort"], "low")
        self.assertEqual(sent["lora"], [{"id": 0, "scale": 0.5}])
        self.assertEqual(sent["max_tokens"], 128)
        self.assertEqual(res.reasoning, "Step 1. Step 2.")
        self.assertEqual(res.content, "Final answer.")
        self.assertEqual(res.finish_reason, "stop")
        self.assertAlmostEqual(res.timings["predicted_per_second"], 45.2)
        self.assertEqual(reasoning_chunks, ["Step 1. ", "Step 2."])
        self.assertEqual(content_chunks, ["Final ", "answer."])

    def test_effort_none_does_not_force_zero_budget(self) -> None:
        recorded: dict = {}

        def handler(req: dict) -> FakeHTTPResponse:
            recorded.update(req)
            return FakeHTTPResponse(
                200,
                b'{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}',
            )

        client = ServerClient("http://127.0.0.1:9931")
        with mock.patch.object(client, "_connect", return_value=FakeHTTPConnection(handler)):
            res = client.chat_completion(
                [{"role": "user", "content": "hi"}],
                stream=False,
                reasoning_effort="none",
            )
        sent = json.loads(recorded["body"].decode("utf-8"))
        self.assertEqual(sent["reasoning_effort"], "none")
        self.assertNotIn("reasoning_budget_tokens", sent)
        self.assertEqual(res.content, "ok")

    def test_health_and_lora_adapters(self) -> None:
        state = {"scale": 1.0}

        def handler(req: dict) -> FakeHTTPResponse:
            if req["path"] == "/health":
                return FakeHTTPResponse(200, b'{"status":"ok"}')
            if req["path"] == "/lora-adapters" and req["method"] == "GET":
                payload = json.dumps([{"id": 0, "path": "lora.gguf", "scale": state["scale"]}]).encode()
                return FakeHTTPResponse(200, payload)
            if req["path"] == "/lora-adapters" and req["method"] == "POST":
                body = json.loads(req["body"].decode())
                state["scale"] = body[0]["scale"]
                return FakeHTTPResponse(200, b"{}")
            return FakeHTTPResponse(404, b'{"error":"not found"}')

        client = ServerClient("http://127.0.0.1:9931")
        with mock.patch.object(client, "_connect", side_effect=lambda _t: FakeHTTPConnection(handler)):
            ok, detail = client.health()
            self.assertTrue(ok)
            self.assertEqual(detail, "ok")
            adapters = client.set_lora_scale(0.0)
            self.assertEqual(adapters[0]["scale"], 0.0)

    def test_error_response_raises(self) -> None:
        def handler(_req: dict) -> FakeHTTPResponse:
            return FakeHTTPResponse(503, b'{"error":{"message":"Loading model"}}')

        client = ServerClient("http://127.0.0.1:9931")
        with mock.patch.object(client, "_connect", return_value=FakeHTTPConnection(handler)):
            ok, detail = client.health()
            self.assertFalse(ok)
            self.assertEqual(detail, "Loading model")
            with self.assertRaises(ServerClientError):
                client.props()


class ChatAndCLITests(unittest.TestCase):
    def test_stream_printer_piped_isolates_stdout(self) -> None:
        out = io.StringIO()
        err = io.StringIO()
        printer = StreamPrinter(show_thinking=True, stdout=out, stderr=err, piped=True)
        printer.on_reasoning("internal thought")
        printer.on_content("clean answer")
        printer.finish()
        self.assertEqual(out.getvalue(), "clean answer\n")
        self.assertIn("internal thought", err.getvalue())

    def test_parse_lora_scale_aliases(self) -> None:
        self.assertEqual(parse_lora_scale_arg("on"), 1.0)
        self.assertEqual(parse_lora_scale_arg("abliterated"), 1.0)
        self.assertEqual(parse_lora_scale_arg("off"), 0.0)
        self.assertEqual(parse_lora_scale_arg("base"), 0.0)
        self.assertAlmostEqual(parse_lora_scale_arg("0.42"), 0.42)
        with self.assertRaises(ConfigError):
            parse_lora_scale_arg("nope")

    def test_compose_prompt_and_save_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sample = tmp_path / "note.txt"
            sample.write_text("file body", encoding="utf-8")
            combined = compose_user_prompt("summarize", stdin_text="piped text\n", files=[sample])
            self.assertIn("--- File:", combined)
            self.assertIn("file body", combined)
            self.assertIn("piped text", combined)
            self.assertTrue(combined.endswith("summarize"))

            md_out = tmp_path / "chat.md"
            json_out = tmp_path / "chat.json"
            msgs = [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "reasoning_content": "hmm", "content": "hello"},
            ]
            _save_transcript(md_out, msgs)
            _save_transcript(json_out, msgs)
            self.assertIn("<summary>Thinking</summary>", md_out.read_text(encoding="utf-8"))
            self.assertEqual(len(json.loads(json_out.read_text(encoding="utf-8"))), 2)

    def test_install_and_uninstall_symlink(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as tmp:
            rc_install = main(["install", "--dir", tmp], repo_root=repo_root)
            self.assertEqual(rc_install, 0)
            link = Path(tmp) / "inkling"
            self.assertTrue(link.is_symlink())
            self.assertEqual(link.resolve(), (repo_root / "inkling").resolve())

            rc_uninstall = main(["uninstall", "--dir", tmp], repo_root=repo_root)
            self.assertEqual(rc_uninstall, 0)
            self.assertFalse(link.exists())

    def test_cli_config_and_env_json(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        buf = io.StringIO()
        with mock.patch("sys.stdout", buf):
            rc = main(["config", "--json"], repo_root=repo_root)
        self.assertEqual(rc, 0)
        parsed = json.loads(buf.getvalue())
        self.assertTrue(parsed["paths"]["model"]["exists"])
        self.assertTrue(parsed["paths"]["lora"]["exists"])
        self.assertTrue(parsed["paths"]["llama_cli"]["exists"])
        self.assertTrue(parsed["paths"]["llama_server"]["exists"])

        buf_env = io.StringIO()
        fake_status = ServerStatus(
            running=False,
            healthy=False,
            managed=False,
            pid=None,
            base_url="http://127.0.0.1:9931",
            api_url="http://127.0.0.1:9931/v1",
            health_detail="unreachable",
        )
        with mock.patch("toolkit.cli.get_server_status", return_value=fake_status), mock.patch("sys.stdout", buf_env):
            rc_env = main(["env", "--json"], repo_root=repo_root)
        self.assertEqual(rc_env, 0)
        env_data = json.loads(buf_env.getvalue())
        self.assertEqual(env_data["OPENAI_BASE_URL"], "http://127.0.0.1:9931/v1")
        self.assertEqual(env_data["OPENAI_MODEL"], "inkling-abliterated")

    def test_parser_accepts_subcommands_and_passthrough(self) -> None:
        parser = build_parser()
        ns_direct = parser.parse_args(["direct", "--base", "-e", "low", "--", "--verbose-prompt"])
        self.assertTrue(ns_direct.base)
        self.assertEqual(ns_direct.effort, "low")
        self.assertEqual(ns_direct.extra_args, ["--", "--verbose-prompt"])

        ns_ask = parser.parse_args(["ask", "--lora-scale", "0.8", "explain", "this"])
        self.assertEqual(ns_ask.lora_scale, "0.8")
        self.assertEqual(ns_ask.prompt, ["explain", "this"])

        ns_code = parser.parse_args(["code", "--yolo", "-e", "high", "-w", "/tmp", "fix", "tests"])
        self.assertTrue(ns_code.yolo)
        self.assertEqual(ns_code.effort, "high")
        self.assertEqual(ns_code.workspace, "/tmp")
        self.assertEqual(ns_code.prompt, ["fix", "tests"])

        from toolkit.cli import build_code_command

        repo_root = Path(__file__).resolve().parent.parent
        cfg = load_config(repo_root, env={})
        cmd = build_code_command(cfg, "http://127.0.0.1:9931", ns_code)
        self.assertIn("--mode", cmd)
        self.assertEqual(cmd[cmd.index("--mode") + 1], "yolo")
        self.assertNotIn("--max-tokens", cmd)

    def test_read_log_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "server.log"
            p.write_text("\n".join(f"line {i}" for i in range(100)), encoding="utf-8")
            tail = read_log_tail(p, lines=5).splitlines()
            self.assertEqual(tail, ["line 95", "line 96", "line 97", "line 98", "line 99"])


if __name__ == "__main__":
    unittest.main()
