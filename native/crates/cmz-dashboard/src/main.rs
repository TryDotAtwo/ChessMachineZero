use cmz_dashboard::{serve, START_FEN};

fn main() {
    let mut host = "0.0.0.0".to_string();
    let mut port = 8768u16;
    let mut fen = START_FEN.to_string();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--host" => host = args.next().expect("--host requires a value"),
            "--port" => {
                port = args
                    .next()
                    .expect("--port requires a value")
                    .parse()
                    .expect("--port must be a u16")
            }
            "--fen" => fen = args.next().expect("--fen requires a value"),
            _ => panic!("unknown cmz-dashboard argument: {arg}"),
        }
    }
    serve(&host, port, &fen).expect("native dashboard server failed");
}
