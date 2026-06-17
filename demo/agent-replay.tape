# VHS tape: live agentic capture -> byte-identical offline replay, zero GPU.
# Render: vhs demo/agent-replay.tape   (brew install vhs)
#
# PRE-FLIGHT (everything slow happens before recording):
#   1. Capture rig up + ready:        just agentic-capture-up   (wait for 4/4)
#   2. Port-forward capture rig:      kubectl -n weaton-dev port-forward deploy/trace-capture-h200-agentic 8000:8000
#   3. Run prep: demo/prep.sh         (act-1 dry run NOT needed; this tape records the real act 1)
#   4. Replay rig up (sim waiting):   just replay-up
#   5. Port-forward replay rig on :8001 AFTER act 1's trace is loaded
#      (the tape pauses at the interstitial; see demo/README.md for the beat sheet)

Output demo/agent-replay.gif
Set FontSize 16
Set Width 1400
Set Height 800
Set Theme "Catppuccin Mocha"
Set TypingSpeed 40ms

# ---- Act 1: a real agent against a real GPU --------------------------------
Type "# Act 1: Claude Code -> vLLM frontend -> recording tap -> H200 GPU"
Enter
Sleep 1s
Type "kubectl -n weaton-dev get pods -l llm-d.ai/guide=trace-capture-agentic"
Enter
Sleep 3s
Type "demo/run-agent.sh http://localhost:8000 /tmp/demo-live"
Enter
# Real GPU inference for a few agent turns; trim render-time as needed.
Sleep 90s
Type "cat /tmp/demo-live/greet.py && python3 /tmp/demo-live/greet.py"
Enter
Sleep 3s
Type "# Every token the agent saw was recorded on the wire:"
Enter
Type "just agentic-capture-fetch /tmp/demo-trace.jsonl && head -c 400 /tmp/demo-trace.jsonl"
Enter
Sleep 4s

# ---- Interstitial: kill the GPU --------------------------------------------
Type "# Now: no more GPU. Scale it to zero and load the trace into the simulator."
Enter
Type "just agentic-capture-down && just replay-load-trace /tmp/demo-trace.jsonl"
Enter
Sleep 5s
Type "kubectl -n weaton-dev get pods -l llm-d.ai/guide=trace-capture-agentic"
Enter
Sleep 2s
Type "# ^ no pods, no GPU anywhere. The replay rig is a CPU frontend + a simulator."
Enter
Sleep 2s

# ---- Act 2: the same agent, fully offline ----------------------------------
Type "# Act 2: the SAME agent command, against the simulator (port 8001)"
Enter
Type "demo/run-agent.sh http://localhost:8001 /tmp/demo-offline"
Enter
# Replay is fast (no real inference); pacing still depends on --latency-trace.
Sleep 30s
Type "diff /tmp/demo-live/greet.py /tmp/demo-offline/greet.py && echo 'IDENTICAL: replayed byte-for-byte, zero GPU'"
Enter
Sleep 5s
