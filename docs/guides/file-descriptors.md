# File descriptors: limit, diagnostic, connection cap

## What the service does

- At startup (`start_server`), it lifts the soft `RLIMIT_NOFILE` to
  `min(hard limit, target)`, target = `ORCHESTRATOR_NOFILE_TARGET` (default
  65536). macOS refuses values above `kern.maxfilesperproc` even with an
  unlimited hard limit, so it steps down (61440, 10240, 4096, 1024). The result
  is logged (`File descriptors: nofile soft 256 -> 10240 (hard unlimited)`);
  a limit that could not be lifted is a `warn`, never silent.
- Every 60 s it logs the usage: a `warn` at 80 %, when the table is full
  (EMFILE/ENFILE: "measurement impossible", a saturation alert), or when the
  meter is blind for any other reason (no `/dev/fd` in a container or sandbox:
  "measurement unavailable (errno N)", status DEGRADED, static cap in place, not
  a saturation alert). The same `warn` is logged at startup. The numbers
  (`open_fds`, `fd_soft_limit`, `fd_percent`, `fd_alert`, `fd_measure_error`,
  `connections`, `max_connections`, `connections_rejected`, `accept_errors`)
  are served to authenticated users by `GET /api/diagnostics/resources`. The
  public `GET /health` carries no number: it only turns `"degraded"` when the
  alert is up or the meter is blind. `/dev/fd` is held open from startup so
  measuring needs no free descriptor.
- Inbound connections are capped: cap = soft limit - (descriptors that are not
  inbound connections, measured at each admission) - 64 reserve (floor 4), and
  optionally `ORCHESTRATOR_MAX_CONNECTIONS` as a lower ceiling. When the meter
  is blind the cap falls back to `limit - max(64, 25 %)`. Beyond the cap a
  connection gets a complete `503` with `Retry-After: 1` and is closed;
  established connections (WebSocket, SSE) are never cut. Loopback peers keep a
  quota of `ORCHESTRATOR_LOOPBACK_RESERVE` (default 8, clamped to half the 64
  margin, logged at startup) connections OVER the cap, so the health probe, the
  desktop app and local MCP clients are not locked out. It is a TEMPORARY
  overshoot: a connection admitted on the quota is closed by the guard after
  `ORCHESTRATOR_LOOPBACK_MAX_AGE_SECS` (default 10), so long-lived local
  WebSockets or keep-alives cannot hold the quota and exclude the probe for good
  (a local client that keeps reconnecting can still occupy it). A quota rather
  than a second health listener: no extra port, every local client benefits. A
  LAN health probe is not covered. The cap is sized from a measurement, not a
  guarantee: a sudden burst of non-inbound descriptors larger than the reserve
  can still hit EMFILE; accept then retries every 100 ms.
- `ORCHESTRATOR_CONN_STATS=1` logs, every 60 s, the top peers by remote address
  (connections, time since the first one of the current series, connections
  admitted). It is bounded (256 peers, then an aggregate counter) and forgets a
  peer when its last connection closes. It says WHO and SINCE WHEN; it does not
  say what the connections are.
- WebSockets (`/ws/chat`, `/ws/events`, `/ws/activity`, `/ws/run`) now require a
  valid `Pong` (echoing the Ping's 8-byte sequence) within 60 s, and every write
  fails after 60 s without progress, so a peer that stops answering or reading
  frees its descriptor.

## Deployment fix (NOT applied: restarting the service is a human decision)

A launchd job inherits a soft limit of 256. With the lift in code, the job needs
nothing more as long as the hard limit allows it (`launchctl limit maxfiles`
shows an unlimited hard limit). To also raise it from the plist, add to
`~/Library/LaunchAgents/land.ffs.orchestrator.plist`, inside the top-level dict:

```xml
<key>SoftResourceLimits</key>
<dict><key>NumberOfFiles</key><integer>10240</integer></dict>
<key>HardResourceLimits</key>
<dict><key>NumberOfFiles</key><integer>10240</integer></dict>
```

then `launchctl unload` / `launchctl load` the job (a restart). System-wide:
`sudo launchctl limit maxfiles 10240 unlimited`.

## Known costs and limits (documented, not fixed)

- Every admission (and every refusal) lists `/dev/fd`: O(open descriptors),
  tens of microseconds at ~250 descriptors, proportionally more at 10 000.
  `/health` lists once, the diagnostics endpoint twice (usage, then the cap).
  The refusal log line is rate limited and its text is only built when it is
  actually written.
- Child processes (agent CLIs) inherit the lifted limit (up to 65536). A program
  that uses `select(2)` with `FD_SETSIZE` (1024) can misbehave with descriptors
  above 1023: lower `ORCHESTRATOR_NOFILE_TARGET` if one is found.
- A suspended mobile tab or a dead link is closed 60 to 90 s after its last
  valid Pong (two Ping periods, checked at the tick); the frontend is expected
  to reconnect.
- A write is bounded by 60 s plus the time a client at 8 KiB/s would need for
  the message (WebSocket sinks do not expose per-byte progress), and never more
  than 5 minutes whatever the size; a client slower than that is cut.
- The `select!` loops are `biased` (client input first, then the ping tick): a
  client that floods frames can starve the tick of ITS OWN connection (no Ping,
  hence no liveness check) and delay the server's events to it. It affects only
  that connection.
- `max_connections` in the diagnostics is the cap measured at that instant and
  can differ by one from the cap used for a concurrent admission.
- The peer address of a refused connection is logged only when
  `ORCHESTRATOR_CONN_STATS` is set.
