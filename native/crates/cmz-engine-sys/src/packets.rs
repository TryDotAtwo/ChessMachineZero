use std::convert::TryFrom;

pub const SQUARE_COUNT: u32 = 64;
pub const PROMO_BASE: u32 = 5;
const FILES: &[u8; 8] = b"abcdefgh";
const RANKS: &[u8; 8] = b"12345678";

pub mod move_flag {
    pub const QUIET: u32 = 0;
    pub const CAPTURE: u32 = 1;
    pub const EP: u32 = 2;
    pub const CASTLE: u32 = 4;
    pub const PROMOTION: u32 = 8;
    pub const CHECK: u32 = 16;
    pub const MATE_CANDIDATE: u32 = 32;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Promo {
    None = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
    Queen = 4,
}

impl TryFrom<u32> for Promo {
    type Error = String;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::None),
            1 => Ok(Self::Knight),
            2 => Ok(Self::Bishop),
            3 => Ok(Self::Rook),
            4 => Ok(Self::Queen),
            _ => Err(format!("promotion out of range: {value}")),
        }
    }
}

impl Promo {
    fn from_uci(byte: u8) -> Result<Self, String> {
        match byte {
            b'n' => Ok(Self::Knight),
            b'b' => Ok(Self::Bishop),
            b'r' => Ok(Self::Rook),
            b'q' => Ok(Self::Queen),
            _ => Err(format!("invalid UCI promotion: {}", byte as char)),
        }
    }

    fn uci_suffix(self) -> &'static str {
        match self {
            Self::None => "",
            Self::Knight => "n",
            Self::Bishop => "b",
            Self::Rook => "r",
            Self::Queen => "q",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MovePacket {
    pub from_sq: u32,
    pub to_sq: u32,
    pub promo: Promo,
    pub flags: u32,
}

impl MovePacket {
    pub const WIDTH: usize = 4;

    pub fn new(from_sq: u32, to_sq: u32, promo: Promo, flags: u32) -> Result<Self, String> {
        if from_sq >= SQUARE_COUNT {
            return Err(format!("from_sq out of range: {from_sq}"));
        }
        if to_sq >= SQUARE_COUNT {
            return Err(format!("to_sq out of range: {to_sq}"));
        }
        let flags = if promo == Promo::None {
            flags
        } else {
            flags | move_flag::PROMOTION
        };
        Ok(Self {
            from_sq,
            to_sq,
            promo,
            flags,
        })
    }

    pub fn move_id(self) -> u32 {
        self.from_sq * SQUARE_COUNT * PROMO_BASE + self.to_sq * PROMO_BASE + self.promo as u32
    }

    pub fn to_tuple(self) -> [u32; Self::WIDTH] {
        [self.from_sq, self.to_sq, self.promo as u32, self.flags]
    }

    pub fn from_tuple(values: &[u32]) -> Result<Self, String> {
        if values.len() != Self::WIDTH {
            return Err(format!(
                "MovePacket requires {} fields, got {}",
                Self::WIDTH,
                values.len()
            ));
        }
        Self::new(values[0], values[1], Promo::try_from(values[2])?, values[3])
    }

    pub fn from_move_id(move_id: u32, flags: u32) -> Result<Self, String> {
        let max_move_id = SQUARE_COUNT * SQUARE_COUNT * PROMO_BASE;
        if move_id >= max_move_id {
            return Err(format!("move_id out of range: {move_id}"));
        }
        let from_sq = move_id / (SQUARE_COUNT * PROMO_BASE);
        let rem = move_id % (SQUARE_COUNT * PROMO_BASE);
        let to_sq = rem / PROMO_BASE;
        let promo = rem % PROMO_BASE;
        Self::new(from_sq, to_sq, Promo::try_from(promo)?, flags)
    }

    pub fn from_uci(uci: &str, flags: u32) -> Result<Self, String> {
        let bytes = uci.as_bytes();
        if bytes.len() != 4 && bytes.len() != 5 {
            return Err(format!("invalid UCI move length: {uci}"));
        }
        let from_sq = square_index(&uci[0..2])?;
        let to_sq = square_index(&uci[2..4])?;
        let promo = if bytes.len() == 5 {
            Promo::from_uci(bytes[4])?
        } else {
            Promo::None
        };
        Self::new(from_sq, to_sq, promo, flags)
    }

    pub fn to_uci(self) -> String {
        format!(
            "{}{}{}",
            square_name(self.from_sq).expect("validated from_sq"),
            square_name(self.to_sq).expect("validated to_sq"),
            self.promo.uci_suffix()
        )
    }
}

pub fn square_name(square: u32) -> Result<String, String> {
    if square >= SQUARE_COUNT {
        return Err(format!("square out of range: {square}"));
    }
    let file = (square % 8) as usize;
    let rank = (square / 8) as usize;
    Ok(format!("{}{}", FILES[file] as char, RANKS[rank] as char))
}

pub fn square_index(name: &str) -> Result<u32, String> {
    let bytes = name.as_bytes();
    if bytes.len() != 2 {
        return Err(format!("invalid square name: {name}"));
    }
    let file = FILES.iter().position(|value| *value == bytes[0]);
    let rank = RANKS.iter().position(|value| *value == bytes[1]);
    match (file, rank) {
        (Some(file), Some(rank)) => Ok(file as u32 + 8 * rank as u32),
        _ => Err(format!("invalid square name: {name}")),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum TraceOp {
    Nop = 0,
    WriteSq = 1,
    ReadSq = 2,
    WriteReg = 3,
    PushStack = 4,
    PopStack = 5,
    WriteCastle = 6,
    WriteEp = 7,
    WriteClock = 8,
    WriteHash = 9,
    Candidate = 10,
    LegalSet = 11,
    ScoreSet = 12,
    SampleSet = 13,
    CommitMove = 14,
    TerminalSet = 15,
    HaltGame = 16,
    ProgramHalt = 17,
}

impl TryFrom<u32> for TraceOp {
    type Error = String;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Nop),
            1 => Ok(Self::WriteSq),
            2 => Ok(Self::ReadSq),
            3 => Ok(Self::WriteReg),
            4 => Ok(Self::PushStack),
            5 => Ok(Self::PopStack),
            6 => Ok(Self::WriteCastle),
            7 => Ok(Self::WriteEp),
            8 => Ok(Self::WriteClock),
            9 => Ok(Self::WriteHash),
            10 => Ok(Self::Candidate),
            11 => Ok(Self::LegalSet),
            12 => Ok(Self::ScoreSet),
            13 => Ok(Self::SampleSet),
            14 => Ok(Self::CommitMove),
            15 => Ok(Self::TerminalSet),
            16 => Ok(Self::HaltGame),
            17 => Ok(Self::ProgramHalt),
            _ => Err(format!("trace op out of range: {value}")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum TraceTag {
    None = 0,
    Board = 1,
    State = 2,
    Move = 3,
    Legal = 4,
    Terminal = 5,
}

impl TryFrom<u32> for TraceTag {
    type Error = String;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::None),
            1 => Ok(Self::Board),
            2 => Ok(Self::State),
            3 => Ok(Self::Move),
            4 => Ok(Self::Legal),
            5 => Ok(Self::Terminal),
            _ => Err(format!("trace tag out of range: {value}")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TracePacket {
    pub op: TraceOp,
    pub a0: u32,
    pub a1: u32,
    pub a2: u32,
    pub a3: u32,
    pub tag: TraceTag,
    pub commit: u32,
}

impl TracePacket {
    pub const WIDTH: usize = 7;

    pub fn new(
        op: TraceOp,
        a0: u32,
        a1: u32,
        a2: u32,
        a3: u32,
        tag: TraceTag,
        commit: u32,
    ) -> Self {
        Self {
            op,
            a0,
            a1,
            a2,
            a3,
            tag,
            commit,
        }
    }

    pub fn to_tokens(self) -> [u32; Self::WIDTH] {
        [
            self.op as u32,
            self.a0,
            self.a1,
            self.a2,
            self.a3,
            self.tag as u32,
            self.commit,
        ]
    }

    pub fn from_tokens(values: &[i64]) -> Result<Self, String> {
        if values.len() != Self::WIDTH {
            return Err(format!(
                "TracePacket requires {} fields, got {}",
                Self::WIDTH,
                values.len()
            ));
        }
        let mut converted = [0u32; Self::WIDTH];
        for (index, value) in values.iter().enumerate() {
            converted[index] = u32::try_from(*value)
                .map_err(|_| format!("TracePacket field out of range: {value}"))?;
        }
        Ok(Self::new(
            TraceOp::try_from(converted[0])?,
            converted[1],
            converted[2],
            converted[3],
            converted[4],
            TraceTag::try_from(converted[5])?,
            converted[6],
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        move_flag, square_index, square_name, MovePacket, Promo, TraceOp, TracePacket, TraceTag,
    };

    #[test]
    fn square_codec_round_trips_python_layout() {
        assert_eq!(square_index("a1").unwrap(), 0);
        assert_eq!(square_index("h8").unwrap(), 63);
        assert_eq!(square_name(0).unwrap(), "a1");
        assert_eq!(square_name(63).unwrap(), "h8");
    }

    #[test]
    fn move_packet_tuple_uci_and_id_round_trip_matches_python_codec() {
        let packet = MovePacket::from_uci("a7a8q", move_flag::CAPTURE).unwrap();
        assert_eq!(packet.promo, Promo::Queen);
        assert_eq!(packet.flags & move_flag::PROMOTION, move_flag::PROMOTION);
        assert_eq!(packet.flags & move_flag::CAPTURE, move_flag::CAPTURE);
        assert_eq!(packet.to_uci(), "a7a8q");
        assert_eq!(MovePacket::from_tuple(&packet.to_tuple()).unwrap(), packet);
        assert_eq!(
            MovePacket::from_move_id(packet.move_id(), packet.flags).unwrap(),
            packet
        );
    }

    #[test]
    fn move_packet_rejects_invalid_values() {
        assert!(MovePacket::new(64, 0, Promo::None, move_flag::QUIET).is_err());
        assert!(MovePacket::from_uci("e2e9", move_flag::QUIET).is_err());
        assert!(MovePacket::from_tuple(&[0, 1, 2]).is_err());
    }

    #[test]
    fn trace_packet_fixed_width_round_trip_matches_python_codec() {
        let packet = TracePacket::new(TraceOp::Candidate, 100, 12, 28, 4, TraceTag::Move, 8);
        let tokens = [10, 100, 12, 28, 4, 3, 8];
        assert_eq!(packet.to_tokens(), tokens);
        assert_eq!(
            TracePacket::from_tokens(&tokens.map(i64::from)).unwrap(),
            packet
        );
    }

    #[test]
    fn trace_packet_rejects_bad_width_negative_and_bad_enums() {
        assert!(TracePacket::from_tokens(&[1, 2, 3]).is_err());
        assert!(TracePacket::from_tokens(&[1, -1, 0, 0, 0, 0, 0]).is_err());
        assert!(TracePacket::from_tokens(&[99, 0, 0, 0, 0, 0, 0]).is_err());
        assert!(TracePacket::from_tokens(&[1, 0, 0, 0, 0, 99, 0]).is_err());
    }
}
