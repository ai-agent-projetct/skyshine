"""The post-show report: what actually happened, from the flight log.

Distinct from the validation report on purpose. That one says what the
plan promised; this one says what the night delivered - how far into the
show the fleet got, which drones raised which conditions and when, what
the operator overrode, and the state every aircraft finished in. It is
built only from the flight log, so it cannot quietly describe the design
instead of the flight.
"""

from __future__ import annotations

import time
from html import escape as esc

SEVERE_EVENTS = {"emergency", "override", "collective_rth", "error", "deviation"}


def _airframe(log: dict, index: int) -> str:
    """Which registered airframe flew slot `index`, if the log knows."""
    frames = log.get("airframes") or []
    return str(frames[index]) if 0 <= index < len(frames) and frames[index] else ""


def summarise(log: dict) -> dict:
    """The numbers the report is built on, also useful on their own."""
    events = log.get("events", [])
    alerts = [e for e in events if e.get("kind") == "alert"]
    by_drone: dict[int, list[str]] = {}
    for e in alerts:
        d = e.get("data", {})
        if "drone" in d:
            by_drone.setdefault(int(d["drone"]), []).append(d.get("condition", "?"))

    final = log.get("final_state", [])
    batteries = [d.get("battery_percent", 0.0) for d in final]
    deviations = [d.get("deviation", 0.0) for d in final]
    reached = float(log.get("show_time_reached", 0.0))
    duration = float(log.get("duration", 0.0)) or 1.0

    if log.get("completed"):
        outcome = "completed"
    elif any(e.get("kind") == "collective_rth" for e in events):
        outcome = "returned home"
    elif any(e.get("kind") == "emergency" for e in events):
        outcome = "aborted"
    elif reached > 0:
        outcome = "incomplete"
    else:
        outcome = "not flown"

    return {
        "outcome": outcome,
        "show_time_reached": round(reached, 1),
        "duration": round(duration, 1),
        "percent_flown": round(100.0 * min(reached / duration, 1.0), 1),
        "drones": int(log.get("drone_count", len(final))),
        "drones_with_issues": len(by_drone),
        "alerts": len(alerts),
        "emergencies": sum(1 for e in events if e.get("kind") == "emergency"),
        "overrides": len(log.get("overrides", [])),
        "min_battery": round(min(batteries), 1) if batteries else None,
        "max_deviation": round(max(deviations), 2) if deviations else None,
        "offline_at_end": sum(1 for d in final if not d.get("connected", True)),
        "issues_by_drone": {k: sorted(set(v)) for k, v in sorted(by_drone.items())},
    }


def post_show_html(log: dict, *, registry_summary: dict | None = None) -> str:
    s = summarise(log)
    started = log.get("start_unix")
    when = (
        time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(started))
        if started else "never started"
    )
    verdict_class = {
        "completed": "ok", "returned home": "warn", "aborted": "bad",
        "incomplete": "warn", "not flown": "idle",
    }[s["outcome"]]

    overrides = "".join(
        f"<li>{esc(str(o))}</li>" for o in log.get("overrides", [])
    ) or "<li class=none>None.</li>"

    issues = "".join(
        f"<tr><td>{d}</td><td>{esc(', '.join(kinds))}</td>"
        f"<td>{esc(_airframe(log, d))}</td></tr>"
        for d, kinds in s["issues_by_drone"].items()
    ) or "<tr><td colspan=3 class=none>No drone raised a condition.</td></tr>"

    timeline = "".join(
        f"<tr class='{'sev' if e.get('kind') in SEVERE_EVENTS else ''}'>"
        f"<td class=mono>{esc(e.get('iso', '')[11:19])}</td>"
        f"<td>{esc(e.get('kind', ''))}</td><td>{esc(e.get('message', ''))}</td></tr>"
        for e in log.get("events", [])
        if e.get("kind") != "alert_cleared"
    )

    final_rows = "".join(
        f"<tr class='{'' if d.get('connected', True) else 'sev'}'>"
        f"<td>{d.get('index')}</td>"
        f"<td>{esc(_airframe(log, d.get('index', -1)))}</td>"
        f"<td>{'online' if d.get('connected') else 'OFFLINE'}</td>"
        f"<td>{esc(str(d.get('mode', '')))}</td>"
        f"<td class=num>{d.get('altitude_agl', 0):.1f}</td>"
        f"<td class=num>{d.get('battery_percent', 0):.0f}</td>"
        f"<td class=num>{d.get('deviation', 0):.2f}</td>"
        f"<td>{esc('; '.join(d.get('faults', [])))}</td></tr>"
        for d in log.get("final_state", [])
    )

    fleet_block = ""
    if registry_summary:
        fleet_block = "<h2>Fleet after this show</h2><table>" + "".join(
            f"<tr><td>{esc(k.replace('_', ' '))}</td><td class=num>{esc(str(v))}</td></tr>"
            for k, v in registry_summary.items() if k != "assignments"
        ) + "</table>"

    safety = log.get("safety") or {}
    sm = safety.get("metrics", {})

    return f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1">
<title>Post-show report - {esc(log.get('show', ''))}</title><style>
*{{box-sizing:border-box}}
body{{font:13px/1.5 system-ui,sans-serif;color:#111;background:#fff;max-width:960px;
margin:0 auto;padding:28px 18px}}
h1{{font-size:21px;margin:0 0 3px}}
h2{{font-size:12px;text-transform:uppercase;letter-spacing:.09em;color:#666;
margin:26px 0 8px;border-bottom:1px solid #ddd;padding-bottom:4px}}
.verdict{{display:inline-block;padding:4px 14px;border-radius:4px;font-weight:700;
letter-spacing:.06em;text-transform:uppercase}}
.ok{{background:#e3f6ea;color:#17683c;border:1px solid #9ad4b2}}
.warn{{background:#fff5df;color:#7a5200;border:1px solid #e6c674}}
.bad{{background:#fdeaec;color:#8a1b28;border:1px solid #e3a3ac}}
.idle{{background:#f1f3f6;color:#555;border:1px solid #ccd}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px;
margin-top:14px}}
.kpi{{border:1px solid #e3e6ec;border-radius:6px;padding:9px 11px}}
.kpi b{{display:block;font-size:19px}}
.kpi span{{font-size:11px;color:#666;text-transform:uppercase;letter-spacing:.05em}}
table{{width:100%;border-collapse:collapse}}
th,td{{text-align:left;padding:4px 6px;border-bottom:1px solid #eee;vertical-align:top}}
th{{color:#666;font-weight:600;font-size:11px;text-transform:uppercase}}
td.num{{text-align:right;font-family:ui-monospace,Consolas,monospace}}
.mono{{font-family:ui-monospace,Consolas,monospace;white-space:nowrap}}
tr.sev td{{background:#fdf3f4}}
.none{{color:#888}}
ul{{margin:4px 0;padding-left:20px}}
.scroll{{max-height:420px;overflow:auto;border:1px solid #eee;border-radius:6px}}
footer{{margin-top:30px;padding-top:12px;border-top:1px solid #ddd;color:#666;font-size:11px}}
@media print{{.scroll{{max-height:none;overflow:visible}} tr{{break-inside:avoid}}}}
</style></head><body>
<h1>Post-show report</h1>
<p>{esc(log.get('show', ''))} &middot; {s['drones']} drones &middot; {esc(log.get('driver', ''))}
&middot; {esc(when)}</p>
<p><span class="verdict {verdict_class}">{esc(s['outcome'])}</span>
&nbsp;{s['show_time_reached']} s of {s['duration']} s flown ({s['percent_flown']}%)</p>

<div class=kpis>
 <div class=kpi><b>{s['drones_with_issues']}</b><span>drones with issues</span></div>
 <div class=kpi><b>{s['alerts']}</b><span>conditions raised</span></div>
 <div class=kpi><b>{s['emergencies']}</b><span>emergency actions</span></div>
 <div class=kpi><b>{s['overrides']}</b><span>overrides</span></div>
 <div class=kpi><b>{'-' if s['max_deviation'] is None else s['max_deviation']}</b><span>max deviation m</span></div>
 <div class=kpi><b>{'-' if s['min_battery'] is None else s['min_battery']}</b><span>min battery %</span></div>
 <div class=kpi><b>{s['offline_at_end']}</b><span>offline at end</span></div>
</div>

<h2>Plan that was flown</h2><table>
<tr><td>Mission checksum</td><td class=mono>{esc(str(log.get('mission_checksum') or 'none'))}</td></tr>
<tr><td>Safety verdict at upload</td><td>{'PASS' if safety.get('passed') else ('FAIL' if safety else 'not validated')}</td></tr>
<tr><td>Designed minimum separation</td><td class=num>{esc(str(sm.get('min_separation_m', '-')))} m</td></tr>
</table>

<h2>Overrides</h2><ul>{overrides}</ul>

<h2>Drones that raised a condition</h2><table>
<tr><th>Drone</th><th>Conditions</th><th>Airframe</th></tr>{issues}</table>

<h2>Event log</h2><div class=scroll><table>
<tr><th>UTC</th><th>Kind</th><th>Detail</th></tr>{timeline}</table></div>

<h2>Final state</h2><div class=scroll><table>
<tr><th>#</th><th>Airframe</th><th>Link</th><th>Mode</th><th>Alt m</th>
<th>Bat %</th><th>Dev m</th><th>Faults</th></tr>{final_rows}</table></div>

{fleet_block}

<footer>Generated by Sky Shine from the flight log. Every figure here was
recorded during the flight; none is taken from the design. Overrides are
listed in full because they are the decisions a review will ask about.</footer>
</body></html>"""
