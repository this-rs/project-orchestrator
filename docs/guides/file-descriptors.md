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
  is blind the cap falls back to `limit - max(64, 25 %)` (floor 16). Beyond the
  cap a connection gets a complete `503` with `Retry-After: 1` and is closed;
  established connections (WebSocket, SSE) are never cut. The cap is sized from
  a measurement, not a guarantee: a sudden burst of non-inbound descriptors
  larger than the reserve can still hit EMFILE; accept then retries every 100 ms.
- Loopback quota. Loopback peers (the health probe, the desktop app, local MCP
  clients) keep a quota of `ORCHESTRATOR_LOOPBACK_RESERVE` (default 8, clamped to
  half the 64 margin, logged at startup) connections OVER the cap. The quota is
  for SHORT REQUESTS ONLY. Every response on a quota connection carries
  `Connection: close` (the connection closes after its response, no keep-alive);
  a WebSocket upgrade, or a response whose `Content-Type` is `text/event-stream`,
  gets a clean `503` (the SSE refusal is applied AFTER the handler ran, so its
  effects are produced; harmless while no production route answers with an event
  stream; the desktop therefore never reaches `onopen` through the quota and its reconnect
  backoff keeps growing). The MCP Streamable HTTP requests are served: a POST
  (`Accept: application/json, text/event-stream`), and the GET
  (`Accept: text/event-stream`) whose 405 tells the client there is no server stream. HTTP/2 in clear text
  (h2c) is refused on the quota: it has no `Connection: close`, and hyper-util's
  `http2` feature is enabled in this build (checked with
  `cargo tree -e features -i hyper-util`). What holds a quota place, honestly (the quota is meant for short requests; these
  are the ways a place stays taken, the last one is NOT bounded):
  - an incomplete request head (nothing received, or partial headers): until
    `ORCHESTRATOR_LOOPBACK_MAX_AGE_SECS` (default 10, minimum 1), the safety net
    against slow-loris, which closes on the read side only and only when no byte is
    available;
  - once hyper has parsed the head (it calls the service only then) nothing in the
    quota cuts the REQUEST: a slow request body holds the place until the route
    timeout (60 s for protected routes, `HTTP_REQUEST_TIMEOUT_SECS`; 20 s for public
    routes, `PUBLIC_TIMEOUT_SECS`; not measured by a test here);
  - a client that stops READING a response (asset, export...): until a write makes
    no progress for `ORCHESTRATOR_LOOPBACK_WRITE_STALL_SECS` (default 30, minimum
    1). A no-progress delay, not a duration cap: a slow client that keeps reading is
    never cut. Only quota connections are subject to it.
  - a LONG non-SSE response that the server itself produces slowly (a slow chunked
    stream) to a client that keeps reading: nothing bounds it on the quota (no write
    stalls, the client advances; the route timeout covers the wait for the
    response, not its streaming). A route that does this should not be reached
    through the quota.
  A LAN health probe is not covered. A quota rather than a second health listener:
  no extra port, every local client benefits.
- `ORCHESTRATOR_CONN_STATS=1` logs, every 60 s, the top peers by remote address
  (connections, time since the first one of the current series, connections
  admitted). It is bounded (256 peers, then an aggregate counter) and forgets a
  peer when its last connection closes. It says WHO and SINCE WHEN; it does not
  say what the connections are.
- WebSockets (`/ws/chat`, `/ws/events`, `/ws/activity`, `/ws/run`) now require a
  valid `Pong` (echoing the Ping's 8-byte sequence) within 60 s, and every write
  fails after 60 s plus the time an 8 KiB/s client would need for the message
  (at most 5 minutes), so a peer that stops answering or reading
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
