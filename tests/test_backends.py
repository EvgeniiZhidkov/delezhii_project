"""Exercise subprocess handling with a fake CLI; never contact a model."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from chores_e1.backends import CodexBackend
from chores_e1.config import ModelConfig


class BackendTests(unittest.TestCase):
    def make_cli(self, root, body):
        path = root / "fake-codex"
        path.write_text(f"#!{sys.executable}\n" + '''import json,os,sys,time
from pathlib import Path
if '--version' in sys.argv:
 print('fixture-cli 1.0'); sys.exit(0)
if 'login' in sys.argv:
 print('Logged in using ChatGPT'); sys.exit(0)
''' + body, encoding="utf-8")
        path.chmod(0o700)
        return path

    def test_fresh_profile_complete_answer_and_model_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            auth_home = root / "original-home"
            auth_home.mkdir()
            (auth_home / "auth.json").write_text('{"fixture":"not-a-real-credential"}')
            (auth_home / "config.toml").write_text('some_user_config = true')
            executable = self.make_cli(root, '''
home = Path(os.environ['CODEX_HOME'])
assert home != Path(''' + repr(str(auth_home)) + ''')
assert (home/'auth.json').is_file()
assert not (home/'config.toml').exists()
assert 'OPENAI_API_KEY' not in os.environ
assert '--ignore-user-config' in sys.argv and '--ignore-rules' in sys.argv
assert sys.stdin.read() == 'fixture prompt'
reply = Path(sys.argv[sys.argv.index('--output-last-message')+1])
reply.write_text('fixture response')
sessions = home/'sessions'; sessions.mkdir()
(sessions/'rollout.jsonl').write_text(json.dumps({'type':'turn_context','payload':{'model':'fixture-model'}})+'\\n')
print(json.dumps({'type':'turn.completed','usage':{'input_tokens':3,'output_tokens':2}}))
''')
            call = root / "call"
            with patch.dict(os.environ, {"CODEX_HOME": str(auth_home), "OPENAI_API_KEY": "fixture-ignored"}):
                backend = CodexBackend(str(executable))
                reply = backend("fixture prompt", ModelConfig(), call)
            self.assertIsNone(reply.error)
            self.assertEqual(reply.raw, "fixture response")
            self.assertEqual(reply.resolved_models, ["fixture-model"])
            self.assertEqual(reply.usage["input_tokens"], 3)
            self.assertTrue((call / "stdout.jsonl").is_file())
            self.assertTrue((call / "sessions" / "rollout.jsonl").is_file())
            self.assertFalse(list(call.rglob("auth.json")))

    def test_tool_violation_and_timeout_keep_diagnostics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = self.make_cli(root, '''
print(json.dumps({'type':'item.completed','item':{'type':'command_execution'}}),flush=True)
print(json.dumps({'type':'turn.completed'}),flush=True)
''')
            backend = CodexBackend(str(executable))
            reply = backend("prompt", ModelConfig(), root / "violation")
            self.assertIn("политика", reply.error)
            self.assertIsNone(reply.raw)
            self.assertIn("command_execution", (root / "violation" / "stdout.jsonl").read_text())
            executable = self.make_cli(root, '''
print('started fixture',flush=True)
time.sleep(2)
''')
            reply = CodexBackend(str(executable))("prompt", ModelConfig(timeout=0.1), root / "timeout")
            self.assertIn("Таймаут", reply.error)
            self.assertIn("started fixture", (root / "timeout" / "stdout.jsonl").read_text())


if __name__ == "__main__":
    unittest.main()
