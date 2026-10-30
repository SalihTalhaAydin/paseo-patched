# voice-patched

Salih's fork of Paseo with voice fixes. It runs as a second daemon on port 6768 with home `~/.paseo-patched`. The packaged app's daemon on 6767 (`~/.paseo`) is not touched.

- `origin`: https://github.com/SalihTalhaAydin/paseo (this branch)
- `upstream`: https://github.com/getpaseo/paseo
- Base: upstream `8c6de842a` (v0.14.0)

## Fixes

| Commit      | Fix                                                                                                                                                                       |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `21744e1e8` | OpenCode agents get the `speak` tool in voice mode.                                                                                                                       |
| `d80b90ece` | Strip the WAV header the Kokoro TTS server returns for `pcm`, which caused a pop after each sentence.                                                                     |
| `cb571d2bf` | Send no `language` to STT when it is `auto`. The Parakeet server returns 500 when `language` is sent.                                                                     |
| `ed9a7801a` | Dictation sends no default transcription prompt.                                                                                                                          |
| `66c760a96` | STT requests time out at 8 s with one retry and log `STT transcription timing`. Audio spoken during an in-flight request goes to the next segment instead of being wiped. |
| `c138fd52a` | Voice turns queue in order. New speech keeps the turn still waiting, and a transcript that arrives after the 10 s timeout is delivered once.                              |

## Run

`~/.paseo-patched/config.json` is a copy of `~/.paseo/config.json` with `daemon.listen` set to `127.0.0.1:6768` and STT `language` set to `auto`.

```bash
npm run build --workspace=@getpaseo/server
cd packages/server
PASEO_HOME=$HOME/.paseo-patched PASEO_LISTEN=127.0.0.1:6768 \
PASEO_LOCAL_MODELS_DIR=$HOME/.paseo/models/local-speech \
nohup node dist/scripts/supervisor-entrypoint.js >> $HOME/.paseo-patched/stdout.log 2>&1 &
```

After a rebuild, restart only this daemon: `paseo --home ~/.paseo-patched daemon restart`. Check it with `curl http://127.0.0.1:6768/api/health`. It does not start at login.

## Update from upstream

```bash
git fetch upstream
git rebase <new-release-tag>
npm install
```

Then rebuild and restart as above. Drop a fix once upstream ships its own.
