use cmz_engine_sys::{Engine, MovePacket, TraceOp, TracePacket};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};

pub const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeDashboardSnapshot {
    pub contract_json: String,
    pub white_readable_log: Vec<String>,
    pub white_token_trace: Vec<String>,
    pub black_readable_log: Vec<String>,
    pub black_token_trace: Vec<String>,
    pub packet_count: usize,
    pub policy_selected_move_uci: String,
    pub policy_decoder_used: bool,
    pub trace_verified_legal: bool,
}

impl NativeDashboardSnapshot {
    pub fn to_json(&self) -> String {
        format!(
            "{{\"runtime\":\"rust_native_dashboard\",\"rule_compute\":\"native_trace_stream_only\",\"python_hot_path\":false,\"contract\":{},\"white_readable_log\":[{}],\"white_token_trace\":[{}],\"black_readable_log\":[{}],\"black_token_trace\":[{}],\"packet_count\":{},\"policy_selected_move_uci\":\"{}\",\"policy_decoder_used\":{},\"trace_verified_legal\":{}}}",
            self.contract_json,
            json_string_array(&self.white_readable_log),
            json_string_array(&self.white_token_trace),
            json_string_array(&self.black_readable_log),
            json_string_array(&self.black_token_trace),
            self.packet_count,
            json_escape(&self.policy_selected_move_uci),
            self.policy_decoder_used,
            self.trace_verified_legal
        )
    }
}

pub fn build_snapshot(engine: &mut Engine, fen: &str) -> Result<NativeDashboardSnapshot, String> {
    let contract_json = engine.percepta_contract_json()?;
    let selection = engine.policy_select_move(fen)?;
    let packet_count = engine.begin_legal_trace_stream(fen)?;
    let mut trace = Vec::with_capacity(packet_count);
    for _ in 0..packet_count {
        let packet = engine.decode_next_legal_trace_packet()?;
        trace.push(packet);
        if packet.op == TraceOp::ProgramHalt {
            break;
        }
    }
    let readable = trace.iter().map(readable_packet).collect::<Vec<_>>();
    let token_trace = trace.iter().map(token_packet).collect::<Vec<_>>();
    Ok(NativeDashboardSnapshot {
        contract_json,
        white_readable_log: readable,
        white_token_trace: token_trace,
        black_readable_log: Vec::new(),
        black_token_trace: Vec::new(),
        packet_count: trace.len(),
        policy_selected_move_uci: selection.move_packet.to_uci(),
        policy_decoder_used: selection.policy_decoder_used,
        trace_verified_legal: selection.trace_verified_legal,
    })
}

pub fn serve(host: &str, port: u16, fen: &str) -> Result<(), String> {
    let listener = TcpListener::bind((host, port)).map_err(|err| err.to_string())?;
    for stream in listener.incoming() {
        let stream = stream.map_err(|err| err.to_string())?;
        handle_connection(stream, fen)?;
    }
    Ok(())
}

fn handle_connection(mut stream: TcpStream, fen: &str) -> Result<(), String> {
    let mut request = [0u8; 2048];
    let read_len = stream.read(&mut request).map_err(|err| err.to_string())?;
    let request_text = String::from_utf8_lossy(&request[..read_len]);
    let path = request_text
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .unwrap_or("/");

    if path == "/api/snapshot" {
        let mut engine = Engine::new()?;
        let body = build_snapshot(&mut engine, fen)?.to_json();
        write_response(&mut stream, "200 OK", "application/json", &body)?;
    } else {
        write_response(
            &mut stream,
            "200 OK",
            "text/html; charset=utf-8",
            DASHBOARD_HTML,
        )?;
    }
    Ok(())
}

fn write_response(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    body: &str,
) -> Result<(), String> {
    let response = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream
        .write_all(response.as_bytes())
        .map_err(|err| err.to_string())
}

fn readable_packet(packet: &TracePacket) -> String {
    match packet.op {
        TraceOp::Candidate => MovePacket::from_move_id(packet.a0, packet.commit)
            .map(|move_packet| format!("shared_transformer candidate {}", move_packet.to_uci()))
            .unwrap_or_else(|_| format!("shared_transformer candidate_id {}", packet.a0)),
        TraceOp::LegalSet => format!(
            "shared_transformer legal_set id={} legal={}",
            packet.a0, packet.a1
        ),
        TraceOp::ProgramHalt => "shared_transformer program_halt".to_string(),
        _ => format!("shared_transformer op={:?}", packet.op),
    }
}

fn token_packet(packet: &TracePacket) -> String {
    let tokens = packet.to_tokens();
    format!(
        "{:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x}",
        tokens[0], tokens[1], tokens[2], tokens[3], tokens[4], tokens[5], tokens[6]
    )
}

fn json_string_array(values: &[String]) -> String {
    values
        .iter()
        .map(|value| format!("\"{}\"", json_escape(value)))
        .collect::<Vec<_>>()
        .join(",")
}

fn json_escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}

const DASHBOARD_HTML: &str = r#"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ChessMachineZero Native Dashboard</title>
  <style>
    :root { color-scheme: light; --bg: #f4f6f1; --panel: #fbfcf8; --ink: #162016; --muted: #66705f; --line: #ccd5c6; --white: #f2bb38; --black: #29372f; --trace: #0077aa; --readable: #097342; }
    body { margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--ink); }
    main { max-width: 1120px; margin: 24px auto; padding: 0 18px; }
    header { display: flex; align-items: baseline; justify-content: space-between; gap: 16px; margin-bottom: 16px; }
    h1 { font-size: 18px; line-height: 1.2; margin: 0; letter-spacing: 0; }
    .meter { color: var(--muted); font: 13px/1.2 ui-monospace, SFMono-Regular, Consolas, monospace; }
    .grid { display: grid; grid-template-columns: 320px 1fr; gap: 16px; align-items: start; }
    .board { display: grid; grid-template-columns: repeat(8, 1fr); aspect-ratio: 1; border: 1px solid #aeb8a8; background: white; }
    .sq { display: grid; place-items: center; font-size: 22px; font-weight: 700; }
    .sq:nth-child(16n+1), .sq:nth-child(16n+3), .sq:nth-child(16n+5), .sq:nth-child(16n+7), .sq:nth-child(16n+10), .sq:nth-child(16n+12), .sq:nth-child(16n+14), .sq:nth-child(16n+16) { background: #e8ede4; }
    .journals { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    .journal { background: var(--panel); border: 1px solid var(--line); border-left: 5px solid var(--white); border-radius: 8px; overflow: hidden; }
    .journal.black { border-left-color: var(--black); }
    .journal-head { display: flex; justify-content: space-between; gap: 12px; padding: 12px 14px 8px; }
    .journal-title { font-weight: 800; }
    .tabs { display: grid; grid-template-columns: 1fr 1fr; border-top: 1px solid var(--line); }
    button { border: 0; border-right: 1px solid var(--line); background: #eef1eb; padding: 10px 8px; font: 700 13px/1.1 ui-sans-serif, system-ui; cursor: pointer; }
    button.active { background: var(--panel); color: #000; }
    pre { height: 280px; margin: 0; padding: 0 12px 12px; overflow: auto; scrollbar-gutter: stable; scroll-behavior: auto; white-space: pre; font: 13px/1.45 ui-monospace, SFMono-Regular, Consolas, monospace; color: var(--readable); }
    pre.token { color: var(--trace); }
    @media (max-width: 860px) { .grid, .journals { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<main>
  <header><h1>ChessMachineZero Native Trace VM</h1><div class="meter" id="meter">loading</div></header>
  <section class="grid">
    <div class="board" id="board"></div>
    <div class="journals">
      <article class="journal"><div class="journal-head"><div class="journal-title">White</div><div class="meter" id="white-meter"></div></div><pre id="white-log" tabindex="0"></pre><div class="tabs"><button class="active" data-target="white-log">Readable log</button><button data-target="white-token">Token trace</button></div><pre id="white-token" class="token" tabindex="0" hidden></pre></article>
      <article class="journal black"><div class="journal-head"><div class="journal-title">Black</div><div class="meter" id="black-meter"></div></div><pre id="black-log" tabindex="0"></pre><div class="tabs"><button class="active" data-target="black-log">Readable log</button><button data-target="black-token">Token trace</button></div><pre id="black-token" class="token" tabindex="0" hidden></pre></article>
    </div>
  </section>
</main>
<script>
const scrollState = new Map();
function atBottom(el) { return Math.abs(el.scrollHeight - el.clientHeight - el.scrollTop) < 4; }
function setText(id, lines) { const el = document.getElementById(id); const follow = !scrollState.has(id) || atBottom(el); const top = el.scrollTop; el.textContent = lines.join('\n'); if (follow) el.scrollTop = el.scrollHeight; else el.scrollTop = top; }
document.querySelectorAll('pre').forEach(el => el.addEventListener('scroll', () => scrollState.set(el.id, el.scrollTop)));
document.querySelectorAll('button').forEach(btn => btn.addEventListener('click', () => { const parent = btn.closest('.journal'); parent.querySelectorAll('button').forEach(x => x.classList.remove('active')); btn.classList.add('active'); parent.querySelectorAll('pre').forEach(pre => pre.hidden = pre.id !== btn.dataset.target); }));
function drawBoard() { const pieces = ['♜','♞','♝','♛','♚','♝','♞','♜','♟','♟','♟','♟','♟','♟','♟','♟','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','♙','♙','♙','♙','♙','♙','♙','♙','♖','♘','♗','♕','♔','♗','♘','♖']; document.getElementById('board').innerHTML = pieces.map(p => `<div class="sq">${p}</div>`).join(''); }
async function refresh() { const res = await fetch('/api/snapshot'); const data = await res.json(); document.getElementById('meter').textContent = `packets=${data.packet_count} HullKVCache 2D`; document.getElementById('white-meter').textContent = `packets=${data.white_token_trace.length}`; document.getElementById('black-meter').textContent = `packets=${data.black_token_trace.length}`; setText('white-log', data.white_readable_log); setText('white-token', data.white_token_trace); setText('black-log', data.black_readable_log); setText('black-token', data.black_token_trace); }
drawBoard(); refresh(); setInterval(refresh, 1200);
</script>
</body>
</html>"#;

#[cfg(test)]
mod tests {
    use super::{build_snapshot, START_FEN};
    use cmz_engine_sys::Engine;

    #[test]
    fn rust_dashboard_snapshot_uses_native_trace_stream_without_python_hot_path() {
        let mut engine = Engine::new().unwrap();
        let snapshot = build_snapshot(&mut engine, START_FEN).unwrap();

        assert!(snapshot.contract_json.contains("\"python_hot_path\":false"));
        assert!(snapshot
            .contract_json
            .contains("\"long_context_cache\":\"HullKVCache\""));
        assert_eq!(snapshot.packet_count, snapshot.white_token_trace.len());
        assert!(snapshot.policy_decoder_used);
        assert!(snapshot.trace_verified_legal);
        assert!(!snapshot.policy_selected_move_uci.is_empty());
        assert!(snapshot
            .white_readable_log
            .iter()
            .any(|line| line.contains("candidate")));
        assert!(snapshot
            .white_token_trace
            .iter()
            .all(|line| line.split_whitespace().count() == 7));
    }
}
