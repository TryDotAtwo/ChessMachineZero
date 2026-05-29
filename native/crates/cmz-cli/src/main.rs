use cmz_engine_sys::Engine;

const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

fn main() {
    let mut engine = Engine::new().expect("native engine must initialize");
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();
    let contract_mode = args.first().is_some_and(|arg| arg == "--contract");
    if contract_mode {
        println!(
            "{}",
            engine
                .percepta_contract_json()
                .expect("Percepta contract must be available")
        );
        return;
    }
    let trace_mode = args.first().is_some_and(|arg| arg == "--trace");
    if trace_mode {
        args.remove(0);
    }
    let fen = if args.is_empty() {
        START.to_string()
    } else {
        args.join(" ")
    };
    println!("cuda_available={}", engine.cuda_available());
    println!(
        "runtime_mode={}",
        engine.runtime_mode().expect("runtime mode")
    );
    if engine.cuda_available() {
        println!(
            "cuda_device={}",
            engine.cuda_device_name().expect("device name")
        );
    }
    if trace_mode {
        let trace = engine
            .legal_trace_packets(&fen)
            .expect("legal trace generation must succeed");
        println!("legal_trace_packet_count={}", trace.len());
        println!("legal_trace_token_count={}", trace.len() * 7);
        for packet in trace {
            let tokens = packet.to_tokens();
            println!(
                "{:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x}",
                tokens[0], tokens[1], tokens[2], tokens[3], tokens[4], tokens[5], tokens[6]
            );
        }
    } else {
        let moves = engine
            .legal_moves_uci(&fen)
            .expect("legal move generation must succeed");
        println!("legal_count={}", moves.len());
        for mv in moves {
            println!("{mv}");
        }
    }
}
