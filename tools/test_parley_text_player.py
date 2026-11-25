"""Verify the bundled parley-text model player and certification manifest fixture.

Runs the real Parley server with four parley-scripted baseline seats and one
parley-text seat (configured from coworld_manifest_template.json) against a
deterministic local OpenAI-compatible HTTP /v1/chat/completions fixture.
"""

import asyncio
import json
import os
import re
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "players" / "parley-text"))

from coworld.examples.parley.evidence import read_capture_archive  # noqa: E402
from coworld.examples.parley.inference import InferenceSettings  # noqa: E402


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


async def run_fixture_episode(
    game_bin: str,
    player_bin: str,
    output_dir: Path,
    *,
    fenced: bool,
    missing_endpoint: bool = False,
) -> dict:
    manifest = json.loads((ROOT / "coworld_manifest_template.json").read_text())
    players_by_id = {entry["id"]: entry for entry in manifest["player"]}
    assert "parley-scripted" in players_by_id, "parley-scripted baseline must remain selectable"
    assert "parley-text" in players_by_id, "parley-text model player must be declared"
    assert manifest["player"][0]["id"] == "parley-scripted", "baseline stays first for no-LLM smoke"

    text_player = players_by_id["parley-text"]
    text_env = text_player["env"]
    assert text_env.get("COWORLD_LLM_ENABLED") == "true"
    assert text_env.get("COWORLD_LLM_MODEL"), "COWORLD_LLM_MODEL must be set"
    assert text_env.get("PARLEY_OPERATOR_PROMPT", "").strip(), "explicit strategy prompt required"
    settings = InferenceSettings.model_validate_json(text_env.get("PARLEY_INFERENCE", "{}"))
    assert settings.top_p is None
    assert settings.output_format == "text"
    assert settings.output_tokens == 1024

    cert_players = manifest["certification"]["players"]
    cert_ids = [slot["player_id"] for slot in cert_players]
    for entry in manifest["player"]:
        assert entry["id"] in cert_ids, f"bundled player {entry['id']} missing certification slot"
    text_slot = cert_ids.index("parley-text")

    output_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
    calls: list[dict] = []

    async def handle_http(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            head_text = head.decode("utf-8", errors="replace")
            request_line = head_text.splitlines()[0]
            assert request_line.startswith("POST /v1/chat/completions "), request_line
            slot_match = re.search(r"(?im)^x-coworld-player-slot:\s*(\d+)\s*$", head_text)
            assert slot_match and int(slot_match.group(1)) == text_slot, head_text
            length_match = re.search(r"(?im)^content-length:\s*(\d+)\s*$", head_text)
            assert length_match, head_text
            body = json.loads(await reader.readexactly(int(length_match.group(1))))
            assert body["model"] == text_env["COWORLD_LLM_MODEL"]
            assert body["max_tokens"] == 1024
            assert "top_p" not in body
            assert "response_format" not in body
            user_prompt = body["messages"][1]["content"]
            target = re.search(r'<one of "([^"]+)"', user_prompt)
            action: dict[str, str] = {"say": "Playing for the win."}
            if target:
                action["shoot"] = target.group(1)
                action["aim"] = "head"
            content = f"```json\n{json.dumps(action)}\n```" if fenced else json.dumps(action)
            payload = {
                "id": f"chatcmpl-fixture-{len(calls)}",
                "model": body["model"],
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 120, "completion_tokens": 24},
            }
            raw = json.dumps(payload).encode("utf-8")
            calls.append({"request": body, "action": action, "fenced": fenced})
            response = (
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: application/json\r\n"
                b"Connection: close\r\n"
                b"X-Request-Id: req-fixture-" + str(len(calls)).encode("ascii") + b"\r\n"
                b"Content-Length: " + str(len(raw)).encode("ascii") + b"\r\n\r\n" + raw
            )
            writer.write(response)
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    http_server = await asyncio.start_server(handle_http, "127.0.0.1", 0)
    http_port = http_server.sockets[0].getsockname()[1]
    game_port = _free_port()

    config = dict(manifest["certification"]["game_config"])
    config["tokens"] = [f"seat-token-{i}" for i in range(len(cert_players))]
    config["episodeTimeoutSeconds"] = 60
    config["llmTimeoutSeconds"] = 10
    (output_dir / "config.json").write_text(json.dumps(config))

    game_env = {k: v for k, v in os.environ.items() if not k.startswith("COWORLD_LLM_")}
    game_env.update(
        COGAME_HOST="127.0.0.1",
        COGAME_PORT=str(game_port),
        COGAME_CONFIG_URI=(output_dir / "config.json").as_uri(),
        COGAME_RESULTS_URI=(output_dir / "results.json").as_uri(),
        COGAME_SAVE_REPLAY_URI=(output_dir / "replay.json").as_uri(),
        COGAME_SAVE_TRAJECTORY_URI=(output_dir / "trajectory.jsonl").as_uri(),
        COGAME_PLAYER_FAILURE_URI=(output_dir / "player_failure.json").as_uri(),
    )

    player_procs: list[asyncio.subprocess.Process] = []
    try:
        with (output_dir / "game.log").open("w") as game_log:
            game_proc = await asyncio.create_subprocess_exec(
                game_bin, env=game_env, stdout=game_log, stderr=asyncio.subprocess.STDOUT
            )
            async with asyncio.timeout(15):
                while "serving on" not in (output_dir / "game.log").read_text():
                    assert game_proc.returncode is None, (output_dir / "game.log").read_text()
                    await asyncio.sleep(0.05)

            for slot, seat in enumerate(cert_players):
                ws_url = f"ws://127.0.0.1:{game_port}/player?slot={slot}&token={config['tokens'][slot]}"
                spec = players_by_id[seat["player_id"]]
                p_env = {k: v for k, v in os.environ.items() if not k.startswith("COWORLD_LLM_") and not k.startswith("PARLEY_")}
                p_env.update(spec.get("env", {}))
                p_env["COWORLD_PLAYER_WS_URL"] = ws_url
                p_env["COWORLD_PLAYER_ARTIFACT_UPLOAD_URL"] = (output_dir / f"player-{slot}.zip").as_uri()
                if seat["player_id"] == "parley-scripted":
                    cmd = [player_bin]
                else:
                    p_env["PYTHONPATH"] = str(ROOT / "players" / "parley-text")
                    if not missing_endpoint:
                        p_env["COWORLD_LLM_ENDPOINT"] = f"http://127.0.0.1:{http_port}"
                    cmd = [sys.executable, "-m", "coworld.examples.parley.player"]
                log_file = (output_dir / f"player-{slot}.log").open("w")
                proc = await asyncio.create_subprocess_exec(
                    *cmd, env=p_env, stdout=log_file, stderr=asyncio.subprocess.STDOUT
                )
                player_procs.append(proc)

            async with asyncio.timeout(45):
                game_rc = await game_proc.wait()
                player_rcs = await asyncio.gather(*(p.wait() for p in player_procs))
    finally:
        http_server.close()
        await http_server.wait_closed()

    if missing_endpoint:
        assert game_rc != 0, "Missing COWORLD_LLM_ENDPOINT must fail episode without fallback"
        assert player_rcs[text_slot] != 0, "parley-text must fail when COWORLD_LLM_ENDPOINT is absent"
        assert not (output_dir / "results.json").exists(), "Failed episode must not publish results"
        assert len(calls) == 0
        return {"mode": "missing_endpoint", "game_rc": game_rc, "player_rc": player_rcs[text_slot]}

    assert game_rc == 0, f"Game failed ({game_rc}):\n" + (output_dir / "game.log").read_text()
    assert all(rc == 0 for rc in player_rcs), f"Player exit codes: {player_rcs}"
    assert len(calls) > 0, "parley-text must make genuine HTTP model calls"
    results = json.loads((output_dir / "results.json").read_text())
    assert len(results["scores"]) == 5
    assert (output_dir / "replay.json").exists()
    captures = read_capture_archive((output_dir / f"player-{text_slot}.zip").read_bytes())
    assert len(captures) == len(calls), f"captures {len(captures)} != calls {len(calls)}"
    trajectory = [json.loads(line) for line in (output_dir / "trajectory.jsonl").read_text().splitlines() if line.strip()]
    llm_decisions = [row for row in trajectory if row.get("seat") == text_slot and row.get("origin") == "llm"]
    assert len(llm_decisions) == len(calls), f"trajectory llm decisions {len(llm_decisions)} != calls {len(calls)}"
    return {
        "mode": "fenced" if fenced else "bare",
        "calls": len(calls),
        "rounds": results["rounds"],
        "turns": results["turns"],
    }


async def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(f"usage: {sys.argv[0]} <parley-bin> <parley-player-bin> <output-dir>")
    game_bin, player_bin, out_root = sys.argv[1], sys.argv[2], Path(sys.argv[3]).resolve()
    summary = [
        await run_fixture_episode(game_bin, player_bin, out_root / "bare", fenced=False),
        await run_fixture_episode(game_bin, player_bin, out_root / "fenced", fenced=True),
        await run_fixture_episode(game_bin, player_bin, out_root / "missing-endpoint", fenced=False, missing_endpoint=True),
    ]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
