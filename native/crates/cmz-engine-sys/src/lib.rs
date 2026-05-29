use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int};
use std::ptr::NonNull;

pub mod packets;
pub use packets::{
    move_flag, square_index, square_name, MovePacket, Promo, TraceOp, TracePacket, TraceTag,
};

#[repr(C)]
struct CmzEngineOpaque {
    _private: [u8; 0],
}

extern "C" {
    fn cmz_engine_create(out: *mut *mut CmzEngineOpaque) -> c_int;
    fn cmz_engine_destroy(engine: *mut CmzEngineOpaque);
    fn cmz_engine_last_error(engine: *const CmzEngineOpaque) -> *const c_char;
    fn cmz_engine_runtime_mode(
        engine: *const CmzEngineOpaque,
        out: *mut c_char,
        out_len: usize,
    ) -> c_int;
    fn cmz_engine_percepta_contract_json(
        engine: *const CmzEngineOpaque,
        out: *mut c_char,
        out_len: usize,
        written: *mut usize,
    ) -> c_int;
    fn cmz_engine_frozen_rule_graph_json(
        engine: *const CmzEngineOpaque,
        out: *mut c_char,
        out_len: usize,
        written: *mut usize,
    ) -> c_int;
    fn cmz_engine_frozen_layer_step_count(engine: *const CmzEngineOpaque) -> usize;
    fn cmz_engine_frozen_attack_mask(
        engine: *mut CmzEngineOpaque,
        piece_token: u32,
        from_square: u32,
        out_mask: *mut u64,
    ) -> c_int;
    fn cmz_engine_frozen_ray_scan_mask(
        engine: *mut CmzEngineOpaque,
        from_square: u32,
        delta_file: c_int,
        delta_rank: c_int,
        occupancy_mask: u64,
        out_mask: *mut u64,
    ) -> c_int;
    fn cmz_engine_frozen_candidate_target_mask(
        engine: *mut CmzEngineOpaque,
        piece_token: u32,
        from_square: u32,
        friendly_mask: u64,
        enemy_mask: u64,
        occupancy_mask: u64,
        ep_square: u32,
        out_mask: *mut u64,
    ) -> c_int;
    fn cmz_engine_frozen_castle_target_mask(
        engine: *mut CmzEngineOpaque,
        fen: *const c_char,
        white: c_int,
        out_mask: *mut u64,
    ) -> c_int;
    fn cmz_engine_frozen_move_legal(
        engine: *mut CmzEngineOpaque,
        fen: *const c_char,
        uci: *const c_char,
        out_legal: *mut c_int,
    ) -> c_int;
    fn cmz_engine_frozen_terminal_status(
        engine: *mut CmzEngineOpaque,
        fen: *const c_char,
        repetition_count: u32,
        adjudication_cap_reached: c_int,
        out_result: *mut u32,
        out_reason: *mut u32,
    ) -> c_int;
    fn cmz_engine_attention_decode_count(engine: *const CmzEngineOpaque) -> usize;
    fn cmz_engine_cuda_trace_emit_attention_count(engine: *const CmzEngineOpaque) -> usize;
    fn cmz_engine_cuda_trace_select_count(engine: *const CmzEngineOpaque) -> usize;
    fn cmz_engine_cutlass_hardmax2d_count(engine: *const CmzEngineOpaque) -> usize;
    fn cmz_engine_cuda_board_projection_count(engine: *const CmzEngineOpaque) -> usize;
    fn cmz_engine_cuda_attack_table_attention_count(engine: *const CmzEngineOpaque) -> usize;
    fn cmz_engine_cuda_candidate_table_attention_count(engine: *const CmzEngineOpaque) -> usize;
    fn cmz_engine_cuda_candidate_move_attention_count(engine: *const CmzEngineOpaque) -> usize;
    fn cmz_engine_cuda_candidate_move_layer_count(engine: *const CmzEngineOpaque) -> usize;
    fn cmz_engine_cuda_resolve_move_attention_count(engine: *const CmzEngineOpaque) -> usize;
    fn cmz_engine_cuda_ray_scan_attention_count(engine: *const CmzEngineOpaque) -> usize;
    fn cmz_engine_cuda_legal_filter_attention_count(engine: *const CmzEngineOpaque) -> usize;
    fn cmz_engine_cuda_legal_filter_batch_attention_count(engine: *const CmzEngineOpaque) -> usize;
    fn cmz_engine_cuda_legal_filter_v2_attention_count(engine: *const CmzEngineOpaque) -> usize;
    fn cmz_engine_cuda_legal_filter_v2_layer_count(engine: *const CmzEngineOpaque) -> usize;
    fn cmz_engine_cuda_legal_filter_v2_batch_attention_count(
        engine: *const CmzEngineOpaque,
    ) -> usize;
    fn cmz_engine_cuda_legal_filter_v2_batch_layer_count(engine: *const CmzEngineOpaque) -> usize;
    fn cmz_engine_cuda_castle_attention_count(engine: *const CmzEngineOpaque) -> usize;
    fn cmz_engine_cuda_make_move_board_attention_count(engine: *const CmzEngineOpaque) -> usize;
    fn cmz_engine_cuda_make_move_metadata_attention_count(engine: *const CmzEngineOpaque) -> usize;
    fn cmz_engine_cuda_terminal_status_attention_count(engine: *const CmzEngineOpaque) -> usize;
    fn cmz_engine_cuda_terminal_status_layer_count(engine: *const CmzEngineOpaque) -> usize;
    fn cmz_engine_cuda_available(engine: *const CmzEngineOpaque) -> c_int;
    fn cmz_engine_cuda_device_name(
        engine: *const CmzEngineOpaque,
        out: *mut c_char,
        out_len: usize,
    ) -> c_int;
    fn cmz_engine_hull_hardmax_2d(
        engine: *mut CmzEngineOpaque,
        keys_xy: *const c_float,
        key_count: usize,
        query_x: c_float,
        query_y: c_float,
        out_index: *mut u32,
        out_score: *mut c_float,
    ) -> c_int;
    fn cmz_engine_nested_hull_topk_2d(
        engine: *mut CmzEngineOpaque,
        keys_xy: *const c_float,
        key_count: usize,
        k: usize,
        query_x: c_float,
        query_y: c_float,
        out_indices: *mut u32,
    ) -> c_int;
    fn cmz_engine_legal_moves_uci(
        engine: *mut CmzEngineOpaque,
        fen: *const c_char,
        out: *mut c_char,
        out_len: usize,
        written: *mut usize,
    ) -> c_int;
    fn cmz_engine_legal_trace_packets(
        engine: *mut CmzEngineOpaque,
        fen: *const c_char,
        out_tokens: *mut u32,
        packet_capacity: usize,
        packet_count: *mut usize,
    ) -> c_int;
    fn cmz_engine_frozen_legal_trace_attention_packets(
        engine: *mut CmzEngineOpaque,
        fen: *const c_char,
        out_tokens: *mut u32,
        packet_capacity: usize,
        packet_count: *mut usize,
    ) -> c_int;
    fn cmz_engine_legal_trace_begin(
        engine: *mut CmzEngineOpaque,
        fen: *const c_char,
        packet_count: *mut usize,
    ) -> c_int;
    fn cmz_engine_legal_trace_next(
        engine: *mut CmzEngineOpaque,
        out_packet_tokens: *mut u32,
    ) -> c_int;
    fn cmz_engine_make_move_trace_packets(
        engine: *mut CmzEngineOpaque,
        fen: *const c_char,
        uci: *const c_char,
        ply: u32,
        repetition_count: u32,
        adjudication_cap_reached: c_int,
        out_tokens: *mut u32,
        packet_capacity: usize,
        packet_count: *mut usize,
    ) -> c_int;
    fn cmz_engine_frozen_make_move_trace_attention_packets(
        engine: *mut CmzEngineOpaque,
        fen: *const c_char,
        uci: *const c_char,
        ply: u32,
        repetition_count: u32,
        adjudication_cap_reached: c_int,
        out_tokens: *mut u32,
        packet_capacity: usize,
        packet_count: *mut usize,
    ) -> c_int;
    fn cmz_engine_project_board_trace(
        engine: *mut CmzEngineOpaque,
        trace_tokens: *const u32,
        packet_count: usize,
        out_square_piece_tokens: *mut u32,
        square_capacity: usize,
        out_side_to_move: *mut u32,
    ) -> c_int;
    fn cmz_engine_decoder_forward(
        engine: *mut CmzEngineOpaque,
        square_piece_tokens: *const u32,
        square_count: usize,
        side_to_move: u32,
        out_command_logits: *mut c_float,
        command_capacity: usize,
    ) -> c_int;
    fn cmz_engine_decoder_policy_gradient_step(
        engine: *mut CmzEngineOpaque,
        square_piece_tokens: *const u32,
        square_count: usize,
        side_to_move: u32,
        selected_command: u32,
        reward: c_float,
        learning_rate: c_float,
        out_loss: *mut c_float,
    ) -> c_int;
    fn cmz_engine_policy_select_move(
        engine: *mut CmzEngineOpaque,
        fen: *const c_char,
        out_move_id: *mut u32,
        out_legal_count: *mut u32,
        out_selected_index: *mut u32,
        out_score: *mut c_float,
    ) -> c_int;
    fn cmz_engine_cuda_probe_double(
        engine: *mut CmzEngineOpaque,
        input: *const u32,
        len: usize,
        output: *mut u32,
    ) -> c_int;
}

pub struct Engine {
    raw: NonNull<CmzEngineOpaque>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HullHardmax2DLookup {
    pub index: usize,
    pub score: f32,
    pub used_dense_scan: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoardHiddenProjection {
    pub square_piece_tokens: [u32; 64],
    pub side_to_move: u32,
}

impl BoardHiddenProjection {
    pub fn square_piece(&self, square: u32) -> u32 {
        self.square_piece_tokens[square as usize]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecoderForwardOutput {
    pub command_logits: Vec<f32>,
    pub command_names: Vec<&'static str>,
    pub attention_head_dim: usize,
    pub tracepacket_backprop: bool,
    pub value_head_enabled: bool,
    pub critic_head_enabled: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PolicyMoveSelection {
    pub move_packet: MovePacket,
    pub legal_count: usize,
    pub selected_index: usize,
    pub score: f32,
    pub policy_decoder_used: bool,
    pub trace_verified_legal: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TerminalStatus {
    pub result: u32,
    pub reason: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PerceptaDecoderScaffold {
    d_model: usize,
    n_heads: usize,
    workspace_slots: usize,
}

impl PerceptaDecoderScaffold {
    pub const COMMAND_COUNT: usize = 6;

    pub fn new(d_model: usize, n_heads: usize, workspace_slots: usize) -> Result<Self, String> {
        if n_heads == 0 {
            return Err("Percepta decoder requires at least one 2D head".to_string());
        }
        if !d_model.is_multiple_of(n_heads) {
            return Err("Percepta decoder d_model must be divisible by n_heads".to_string());
        }
        let head_dim = d_model / n_heads;
        if head_dim != 2 {
            return Err(format!(
                "Percepta decoder requires 2D attention heads, got head_dim={head_dim}"
            ));
        }
        if workspace_slots == 0 {
            return Err("Percepta decoder requires HullKV-compatible workspace slots".to_string());
        }
        Ok(Self {
            d_model,
            n_heads,
            workspace_slots,
        })
    }

    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }

    pub fn shared_for_white_black(&self) -> bool {
        true
    }

    pub fn command_count(&self) -> usize {
        Self::COMMAND_COUNT
    }

    pub fn workspace_slots(&self) -> usize {
        self.workspace_slots
    }

    pub fn value_head_enabled(&self) -> bool {
        false
    }

    pub fn externally_prescribed_critic(&self) -> bool {
        false
    }
}

impl Engine {
    pub fn new() -> Result<Self, String> {
        let mut raw = std::ptr::null_mut();
        let status = unsafe { cmz_engine_create(&mut raw) };
        let raw = NonNull::new(raw)
            .ok_or_else(|| format!("cmz_engine_create returned null, status={status}"))?;
        if status != 0 {
            return Err(last_error(raw));
        }
        Ok(Self { raw })
    }

    pub fn cuda_available(&self) -> bool {
        unsafe { cmz_engine_cuda_available(self.raw.as_ptr()) != 0 }
    }

    pub fn runtime_mode(&self) -> Result<String, String> {
        let mut buffer = vec![0i8; 128];
        let status = unsafe {
            cmz_engine_runtime_mode(self.raw.as_ptr(), buffer.as_mut_ptr(), buffer.len())
        };
        if status != 0 {
            return Err(last_error(self.raw));
        }
        Ok(unsafe { CStr::from_ptr(buffer.as_ptr()) }
            .to_string_lossy()
            .into_owned())
    }

    pub fn percepta_contract_json(&self) -> Result<String, String> {
        let mut written = 0usize;
        let status = unsafe {
            cmz_engine_percepta_contract_json(
                self.raw.as_ptr(),
                std::ptr::null_mut(),
                0,
                &mut written,
            )
        };
        if status != 0 {
            return Err(last_error(self.raw));
        }
        let mut buffer = vec![0i8; written.max(1)];
        let status = unsafe {
            cmz_engine_percepta_contract_json(
                self.raw.as_ptr(),
                buffer.as_mut_ptr(),
                buffer.len(),
                &mut written,
            )
        };
        if status != 0 {
            return Err(last_error(self.raw));
        }
        Ok(unsafe { CStr::from_ptr(buffer.as_ptr()) }
            .to_string_lossy()
            .into_owned())
    }

    pub fn frozen_rule_graph_json(&self) -> Result<String, String> {
        let mut written = 0usize;
        let status = unsafe {
            cmz_engine_frozen_rule_graph_json(
                self.raw.as_ptr(),
                std::ptr::null_mut(),
                0,
                &mut written,
            )
        };
        if status != 0 {
            return Err(last_error(self.raw));
        }
        let mut buffer = vec![0i8; written.max(1)];
        let status = unsafe {
            cmz_engine_frozen_rule_graph_json(
                self.raw.as_ptr(),
                buffer.as_mut_ptr(),
                buffer.len(),
                &mut written,
            )
        };
        if status != 0 {
            return Err(last_error(self.raw));
        }
        Ok(unsafe { CStr::from_ptr(buffer.as_ptr()) }
            .to_string_lossy()
            .into_owned())
    }

    pub fn attention_decode_count(&self) -> usize {
        unsafe { cmz_engine_attention_decode_count(self.raw.as_ptr()) }
    }

    pub fn cuda_trace_emit_attention_count(&self) -> usize {
        unsafe { cmz_engine_cuda_trace_emit_attention_count(self.raw.as_ptr()) }
    }

    pub fn cuda_trace_select_count(&self) -> usize {
        unsafe { cmz_engine_cuda_trace_select_count(self.raw.as_ptr()) }
    }

    pub fn cutlass_hardmax2d_count(&self) -> usize {
        unsafe { cmz_engine_cutlass_hardmax2d_count(self.raw.as_ptr()) }
    }

    pub fn cuda_board_projection_count(&self) -> usize {
        unsafe { cmz_engine_cuda_board_projection_count(self.raw.as_ptr()) }
    }

    pub fn cuda_attack_table_attention_count(&self) -> usize {
        unsafe { cmz_engine_cuda_attack_table_attention_count(self.raw.as_ptr()) }
    }

    pub fn cuda_candidate_table_attention_count(&self) -> usize {
        unsafe { cmz_engine_cuda_candidate_table_attention_count(self.raw.as_ptr()) }
    }

    pub fn cuda_candidate_move_attention_count(&self) -> usize {
        unsafe { cmz_engine_cuda_candidate_move_attention_count(self.raw.as_ptr()) }
    }

    pub fn cuda_candidate_move_layer_count(&self) -> usize {
        unsafe { cmz_engine_cuda_candidate_move_layer_count(self.raw.as_ptr()) }
    }

    pub fn cuda_resolve_move_attention_count(&self) -> usize {
        unsafe { cmz_engine_cuda_resolve_move_attention_count(self.raw.as_ptr()) }
    }

    pub fn cuda_ray_scan_attention_count(&self) -> usize {
        unsafe { cmz_engine_cuda_ray_scan_attention_count(self.raw.as_ptr()) }
    }

    pub fn cuda_legal_filter_attention_count(&self) -> usize {
        unsafe { cmz_engine_cuda_legal_filter_attention_count(self.raw.as_ptr()) }
    }

    pub fn cuda_legal_filter_batch_attention_count(&self) -> usize {
        unsafe { cmz_engine_cuda_legal_filter_batch_attention_count(self.raw.as_ptr()) }
    }

    pub fn cuda_legal_filter_v2_attention_count(&self) -> usize {
        unsafe { cmz_engine_cuda_legal_filter_v2_attention_count(self.raw.as_ptr()) }
    }

    pub fn cuda_legal_filter_v2_layer_count(&self) -> usize {
        unsafe { cmz_engine_cuda_legal_filter_v2_layer_count(self.raw.as_ptr()) }
    }

    pub fn cuda_legal_filter_v2_batch_attention_count(&self) -> usize {
        unsafe { cmz_engine_cuda_legal_filter_v2_batch_attention_count(self.raw.as_ptr()) }
    }

    pub fn cuda_legal_filter_v2_batch_layer_count(&self) -> usize {
        unsafe { cmz_engine_cuda_legal_filter_v2_batch_layer_count(self.raw.as_ptr()) }
    }

    pub fn cuda_castle_attention_count(&self) -> usize {
        unsafe { cmz_engine_cuda_castle_attention_count(self.raw.as_ptr()) }
    }

    pub fn cuda_make_move_board_attention_count(&self) -> usize {
        unsafe { cmz_engine_cuda_make_move_board_attention_count(self.raw.as_ptr()) }
    }

    pub fn cuda_make_move_metadata_attention_count(&self) -> usize {
        unsafe { cmz_engine_cuda_make_move_metadata_attention_count(self.raw.as_ptr()) }
    }

    pub fn cuda_terminal_status_attention_count(&self) -> usize {
        unsafe { cmz_engine_cuda_terminal_status_attention_count(self.raw.as_ptr()) }
    }

    pub fn cuda_terminal_status_layer_count(&self) -> usize {
        unsafe { cmz_engine_cuda_terminal_status_layer_count(self.raw.as_ptr()) }
    }

    pub fn frozen_layer_step_count(&self) -> usize {
        unsafe { cmz_engine_frozen_layer_step_count(self.raw.as_ptr()) }
    }

    pub fn frozen_attack_mask(
        &mut self,
        piece_token: u32,
        from_square: u32,
    ) -> Result<u64, String> {
        let mut mask = 0u64;
        let status = unsafe {
            cmz_engine_frozen_attack_mask(self.raw.as_ptr(), piece_token, from_square, &mut mask)
        };
        if status != 0 {
            return Err(last_error(self.raw));
        }
        Ok(mask)
    }

    pub fn frozen_ray_scan_mask(
        &mut self,
        from_square: u32,
        delta_file: i32,
        delta_rank: i32,
        occupancy_mask: u64,
    ) -> Result<u64, String> {
        let mut mask = 0u64;
        let status = unsafe {
            cmz_engine_frozen_ray_scan_mask(
                self.raw.as_ptr(),
                from_square,
                delta_file,
                delta_rank,
                occupancy_mask,
                &mut mask,
            )
        };
        if status != 0 {
            return Err(last_error(self.raw));
        }
        Ok(mask)
    }

    pub fn frozen_candidate_target_mask(
        &mut self,
        piece_token: u32,
        from_square: u32,
        friendly_mask: u64,
        enemy_mask: u64,
        occupancy_mask: u64,
        ep_square: u32,
    ) -> Result<u64, String> {
        let mut mask = 0u64;
        let status = unsafe {
            cmz_engine_frozen_candidate_target_mask(
                self.raw.as_ptr(),
                piece_token,
                from_square,
                friendly_mask,
                enemy_mask,
                occupancy_mask,
                ep_square,
                &mut mask,
            )
        };
        if status != 0 {
            return Err(last_error(self.raw));
        }
        Ok(mask)
    }

    pub fn frozen_castle_target_mask(&mut self, fen: &str, white: bool) -> Result<u64, String> {
        let fen = CString::new(fen).map_err(|_| "FEN contains interior nul byte".to_string())?;
        let mut mask = 0u64;
        let status = unsafe {
            cmz_engine_frozen_castle_target_mask(
                self.raw.as_ptr(),
                fen.as_ptr(),
                if white { 1 } else { 0 },
                &mut mask,
            )
        };
        if status != 0 {
            return Err(last_error(self.raw));
        }
        Ok(mask)
    }

    pub fn frozen_move_legal(&mut self, fen: &str, uci: &str) -> Result<bool, String> {
        let fen = CString::new(fen).map_err(|_| "FEN contains interior nul byte".to_string())?;
        let uci = CString::new(uci).map_err(|_| "UCI contains interior nul byte".to_string())?;
        let mut legal = 0i32;
        let status = unsafe {
            cmz_engine_frozen_move_legal(self.raw.as_ptr(), fen.as_ptr(), uci.as_ptr(), &mut legal)
        };
        if status != 0 {
            return Err(last_error(self.raw));
        }
        Ok(legal != 0)
    }

    pub fn frozen_terminal_status(
        &mut self,
        fen: &str,
        repetition_count: u32,
        adjudication_cap_reached: bool,
    ) -> Result<TerminalStatus, String> {
        let fen = CString::new(fen).map_err(|_| "FEN contains interior nul byte".to_string())?;
        let mut result = 0u32;
        let mut reason = 0u32;
        let status = unsafe {
            cmz_engine_frozen_terminal_status(
                self.raw.as_ptr(),
                fen.as_ptr(),
                repetition_count,
                i32::from(adjudication_cap_reached),
                &mut result,
                &mut reason,
            )
        };
        if status != 0 {
            return Err(last_error(self.raw));
        }
        Ok(TerminalStatus { result, reason })
    }

    pub fn cuda_device_name(&self) -> Result<String, String> {
        let mut buffer = vec![0i8; 256];
        let status = unsafe {
            cmz_engine_cuda_device_name(self.raw.as_ptr(), buffer.as_mut_ptr(), buffer.len())
        };
        if status != 0 {
            return Err(last_error(self.raw));
        }
        Ok(unsafe { CStr::from_ptr(buffer.as_ptr()) }
            .to_string_lossy()
            .into_owned())
    }

    pub fn hull_hardmax_2d(
        &mut self,
        query: (f32, f32),
        keys: &[(f32, f32)],
    ) -> Result<HullHardmax2DLookup, String> {
        if keys.is_empty() {
            return Err("HullHardmax2D requires at least one key".to_string());
        }
        let flat_keys = flatten_keys(keys);
        let mut index = 0u32;
        let mut score = 0.0f32;
        let status = unsafe {
            cmz_engine_hull_hardmax_2d(
                self.raw.as_ptr(),
                flat_keys.as_ptr(),
                keys.len(),
                query.0,
                query.1,
                &mut index,
                &mut score,
            )
        };
        if status != 0 {
            return Err(last_error(self.raw));
        }
        Ok(HullHardmax2DLookup {
            index: index as usize,
            score,
            used_dense_scan: false,
        })
    }

    pub fn nested_hull_topk_2d(
        &mut self,
        query: (f32, f32),
        keys: &[(f32, f32)],
        k: usize,
    ) -> Result<Vec<usize>, String> {
        if k > keys.len() {
            return Err("NestedHullTopK2D k exceeds key count".to_string());
        }
        let flat_keys = flatten_keys(keys);
        let mut indices = vec![0u32; k];
        let status = unsafe {
            cmz_engine_nested_hull_topk_2d(
                self.raw.as_ptr(),
                flat_keys.as_ptr(),
                keys.len(),
                k,
                query.0,
                query.1,
                indices.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(last_error(self.raw));
        }
        Ok(indices.into_iter().map(|index| index as usize).collect())
    }

    pub fn legal_moves_uci(&mut self, fen: &str) -> Result<Vec<String>, String> {
        let fen = CString::new(fen).map_err(|_| "FEN contains interior nul byte".to_string())?;
        let mut buffer = vec![0i8; 8192];
        let mut written = 0usize;
        let status = unsafe {
            cmz_engine_legal_moves_uci(
                self.raw.as_ptr(),
                fen.as_ptr(),
                buffer.as_mut_ptr(),
                buffer.len(),
                &mut written,
            )
        };
        if status != 0 {
            return Err(last_error(self.raw));
        }
        let text = unsafe { CStr::from_ptr(buffer.as_ptr()) }.to_string_lossy();
        Ok(text.lines().map(str::to_owned).collect())
    }

    pub fn legal_trace_packets(&mut self, fen: &str) -> Result<Vec<TracePacket>, String> {
        let fen = CString::new(fen).map_err(|_| "FEN contains interior nul byte".to_string())?;
        let mut packet_count = 0usize;
        let status = unsafe {
            cmz_engine_legal_trace_packets(
                self.raw.as_ptr(),
                fen.as_ptr(),
                std::ptr::null_mut(),
                0,
                &mut packet_count,
            )
        };
        if status != 0 {
            return Err(last_error(self.raw));
        }

        let mut tokens = vec![0u32; packet_count * TracePacket::WIDTH];
        let status = unsafe {
            cmz_engine_legal_trace_packets(
                self.raw.as_ptr(),
                fen.as_ptr(),
                tokens.as_mut_ptr(),
                packet_count,
                &mut packet_count,
            )
        };
        if status != 0 {
            return Err(last_error(self.raw));
        }
        tokens
            .chunks_exact(TracePacket::WIDTH)
            .map(trace_packet_from_u32_tokens)
            .collect()
    }

    pub fn frozen_legal_trace_attention_packets(
        &mut self,
        fen: &str,
    ) -> Result<Vec<TracePacket>, String> {
        let fen = CString::new(fen).map_err(|_| "FEN contains interior nul byte".to_string())?;
        let mut packet_count = 0usize;
        let status = unsafe {
            cmz_engine_frozen_legal_trace_attention_packets(
                self.raw.as_ptr(),
                fen.as_ptr(),
                std::ptr::null_mut(),
                0,
                &mut packet_count,
            )
        };
        if status != 0 {
            return Err(last_error(self.raw));
        }

        let mut tokens = vec![0u32; packet_count * TracePacket::WIDTH];
        let status = unsafe {
            cmz_engine_frozen_legal_trace_attention_packets(
                self.raw.as_ptr(),
                fen.as_ptr(),
                tokens.as_mut_ptr(),
                packet_count,
                &mut packet_count,
            )
        };
        if status != 0 {
            return Err(last_error(self.raw));
        }
        tokens
            .chunks_exact(TracePacket::WIDTH)
            .map(trace_packet_from_u32_tokens)
            .collect()
    }

    pub fn begin_legal_trace_stream(&mut self, fen: &str) -> Result<usize, String> {
        let fen = CString::new(fen).map_err(|_| "FEN contains interior nul byte".to_string())?;
        let mut packet_count = 0usize;
        let status = unsafe {
            cmz_engine_legal_trace_begin(self.raw.as_ptr(), fen.as_ptr(), &mut packet_count)
        };
        if status != 0 {
            return Err(last_error(self.raw));
        }
        Ok(packet_count)
    }

    pub fn decode_next_legal_trace_packet(&mut self) -> Result<TracePacket, String> {
        let mut tokens = [0u32; TracePacket::WIDTH];
        let status = unsafe { cmz_engine_legal_trace_next(self.raw.as_ptr(), tokens.as_mut_ptr()) };
        if status != 0 {
            return Err(last_error(self.raw));
        }
        trace_packet_from_u32_tokens(&tokens)
    }

    pub fn make_move_trace_packets(
        &mut self,
        fen: &str,
        uci: &str,
        ply: u32,
        repetition_count: u32,
        adjudication_cap_reached: bool,
    ) -> Result<Vec<TracePacket>, String> {
        let fen = CString::new(fen).map_err(|_| "FEN contains interior nul byte".to_string())?;
        let uci = CString::new(uci).map_err(|_| "UCI contains interior nul byte".to_string())?;
        let mut packet_count = 0usize;
        let status = unsafe {
            cmz_engine_make_move_trace_packets(
                self.raw.as_ptr(),
                fen.as_ptr(),
                uci.as_ptr(),
                ply,
                repetition_count,
                i32::from(adjudication_cap_reached),
                std::ptr::null_mut(),
                0,
                &mut packet_count,
            )
        };
        if status != 0 {
            return Err(last_error(self.raw));
        }

        let mut tokens = vec![0u32; packet_count * TracePacket::WIDTH];
        let status = unsafe {
            cmz_engine_make_move_trace_packets(
                self.raw.as_ptr(),
                fen.as_ptr(),
                uci.as_ptr(),
                ply,
                repetition_count,
                i32::from(adjudication_cap_reached),
                tokens.as_mut_ptr(),
                packet_count,
                &mut packet_count,
            )
        };
        if status != 0 {
            return Err(last_error(self.raw));
        }
        tokens
            .chunks_exact(TracePacket::WIDTH)
            .map(trace_packet_from_u32_tokens)
            .collect()
    }

    pub fn frozen_make_move_trace_attention_packets(
        &mut self,
        fen: &str,
        uci: &str,
        ply: u32,
        repetition_count: u32,
        adjudication_cap_reached: bool,
    ) -> Result<Vec<TracePacket>, String> {
        let fen = CString::new(fen).map_err(|_| "FEN contains interior nul byte".to_string())?;
        let uci = CString::new(uci).map_err(|_| "UCI contains interior nul byte".to_string())?;
        let mut packet_count = 0usize;
        let status = unsafe {
            cmz_engine_frozen_make_move_trace_attention_packets(
                self.raw.as_ptr(),
                fen.as_ptr(),
                uci.as_ptr(),
                ply,
                repetition_count,
                i32::from(adjudication_cap_reached),
                std::ptr::null_mut(),
                0,
                &mut packet_count,
            )
        };
        if status != 0 {
            return Err(last_error(self.raw));
        }

        let mut tokens = vec![0u32; packet_count * TracePacket::WIDTH];
        let status = unsafe {
            cmz_engine_frozen_make_move_trace_attention_packets(
                self.raw.as_ptr(),
                fen.as_ptr(),
                uci.as_ptr(),
                ply,
                repetition_count,
                i32::from(adjudication_cap_reached),
                tokens.as_mut_ptr(),
                packet_count,
                &mut packet_count,
            )
        };
        if status != 0 {
            return Err(last_error(self.raw));
        }
        tokens
            .chunks_exact(TracePacket::WIDTH)
            .map(trace_packet_from_u32_tokens)
            .collect()
    }

    pub fn project_board_trace(
        &mut self,
        trace: &[TracePacket],
    ) -> Result<BoardHiddenProjection, String> {
        let tokens = trace
            .iter()
            .flat_map(|packet| packet.to_tokens())
            .collect::<Vec<_>>();
        let mut square_piece_tokens = [0u32; 64];
        let mut side_to_move = 0u32;
        let status = unsafe {
            cmz_engine_project_board_trace(
                self.raw.as_ptr(),
                tokens.as_ptr(),
                trace.len(),
                square_piece_tokens.as_mut_ptr(),
                square_piece_tokens.len(),
                &mut side_to_move,
            )
        };
        if status != 0 {
            return Err(last_error(self.raw));
        }
        Ok(BoardHiddenProjection {
            square_piece_tokens,
            side_to_move,
        })
    }

    pub fn decoder_forward(
        &mut self,
        hidden: &BoardHiddenProjection,
    ) -> Result<DecoderForwardOutput, String> {
        let mut logits = vec![0.0f32; PerceptaDecoderScaffold::COMMAND_COUNT];
        let status = unsafe {
            cmz_engine_decoder_forward(
                self.raw.as_ptr(),
                hidden.square_piece_tokens.as_ptr(),
                hidden.square_piece_tokens.len(),
                hidden.side_to_move,
                logits.as_mut_ptr(),
                logits.len(),
            )
        };
        if status != 0 {
            return Err(last_error(self.raw));
        }
        Ok(DecoderForwardOutput {
            command_logits: logits,
            command_names: decoder_command_names(),
            attention_head_dim: 2,
            tracepacket_backprop: false,
            value_head_enabled: false,
            critic_head_enabled: false,
        })
    }

    pub fn decoder_policy_gradient_step(
        &mut self,
        hidden: &BoardHiddenProjection,
        selected_command: u32,
        reward: f32,
        learning_rate: f32,
    ) -> Result<f32, String> {
        let mut loss = 0.0f32;
        let status = unsafe {
            cmz_engine_decoder_policy_gradient_step(
                self.raw.as_ptr(),
                hidden.square_piece_tokens.as_ptr(),
                hidden.square_piece_tokens.len(),
                hidden.side_to_move,
                selected_command,
                reward,
                learning_rate,
                &mut loss,
            )
        };
        if status != 0 {
            return Err(last_error(self.raw));
        }
        Ok(loss)
    }

    pub fn policy_select_move(&mut self, fen: &str) -> Result<PolicyMoveSelection, String> {
        let fen = CString::new(fen).map_err(|_| "FEN contains interior NUL".to_string())?;
        let mut move_id = 0u32;
        let mut legal_count = 0u32;
        let mut selected_index = 0u32;
        let mut score = 0.0f32;
        let status = unsafe {
            cmz_engine_policy_select_move(
                self.raw.as_ptr(),
                fen.as_ptr(),
                &mut move_id,
                &mut legal_count,
                &mut selected_index,
                &mut score,
            )
        };
        if status != 0 {
            return Err(last_error(self.raw));
        }
        let move_packet = MovePacket::from_move_id(move_id, 0)?;
        let legal_moves = self.legal_moves_uci(fen.to_str().unwrap())?;
        Ok(PolicyMoveSelection {
            move_packet,
            legal_count: legal_count as usize,
            selected_index: selected_index as usize,
            score,
            policy_decoder_used: true,
            trace_verified_legal: legal_moves.contains(&move_packet.to_uci()),
        })
    }

    pub fn cuda_probe_double(&mut self, input: &[u32]) -> Result<Vec<u32>, String> {
        let mut output = vec![0u32; input.len()];
        let status = unsafe {
            cmz_engine_cuda_probe_double(
                self.raw.as_ptr(),
                input.as_ptr(),
                input.len(),
                output.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(last_error(self.raw));
        }
        Ok(output)
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe { cmz_engine_destroy(self.raw.as_ptr()) }
    }
}

fn last_error(raw: NonNull<CmzEngineOpaque>) -> String {
    let ptr = unsafe { cmz_engine_last_error(raw.as_ptr()) };
    if ptr.is_null() {
        return "unknown native engine error".to_string();
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

fn trace_packet_from_u32_tokens(values: &[u32]) -> Result<TracePacket, String> {
    if values.len() != TracePacket::WIDTH {
        return Err(format!(
            "TracePacket requires {} fields, got {}",
            TracePacket::WIDTH,
            values.len()
        ));
    }
    Ok(TracePacket::new(
        TraceOp::try_from(values[0])?,
        values[1],
        values[2],
        values[3],
        values[4],
        TraceTag::try_from(values[5])?,
        values[6],
    ))
}

fn flatten_keys(keys: &[(f32, f32)]) -> Vec<f32> {
    keys.iter()
        .flat_map(|key| [key.0, key.1])
        .collect::<Vec<_>>()
}

fn decoder_command_names() -> Vec<&'static str> {
    vec![
        "QUERY_RULES",
        "READ_WORKSPACE",
        "WRITE_WORKSPACE",
        "SELECT_WORKSPACE",
        "COMMIT_MOVE",
        "NOOP",
    ]
}

#[cfg(test)]
mod tests {
    use super::{
        square_index, Engine, MovePacket, PerceptaDecoderScaffold, TerminalStatus, TraceOp,
        TracePacket,
    };

    const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    fn repo_root() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
    }

    fn cuda_source() -> String {
        std::fs::read_to_string(repo_root().join("native/cpp/src/cmz_cuda_kernels.cu")).unwrap()
    }

    fn cuda_function_body<'a>(source: &'a str, symbol: &str) -> &'a str {
        let mut search_start = 0usize;
        let symbol_start = loop {
            let relative_start = source[search_start..]
                .find(symbol)
                .unwrap_or_else(|| panic!("missing CUDA symbol definition: {symbol}"));
            let candidate_start = search_start + relative_start;
            let after_symbol = &source[candidate_start..];
            let brace_offset = after_symbol
                .find('{')
                .unwrap_or_else(|| panic!("missing CUDA function body: {symbol}"));
            let semicolon_offset = after_symbol.find(';');
            if match semicolon_offset {
                Some(offset) => brace_offset < offset,
                None => true,
            } {
                break candidate_start;
            }
            search_start = candidate_start + symbol.len();
        };
        let after_symbol = &source[symbol_start..];
        let brace_offset = after_symbol
            .find('{')
            .unwrap_or_else(|| panic!("missing CUDA function body: {symbol}"));
        let body_start = symbol_start + brace_offset;
        let mut depth = 0usize;
        for (offset, byte) in source[body_start..].bytes().enumerate() {
            match byte {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        return &source[body_start..=body_start + offset];
                    }
                }
                _ => {}
            }
        }
        panic!("unterminated CUDA function body: {symbol}");
    }

    fn assert_body_contains_any(body: &str, symbol: &str, patterns: &[&str]) {
        assert!(
            patterns.iter().any(|pattern| body.contains(pattern)),
            "CUDA symbol {symbol} must expose concrete semantic source evidence, patterns={patterns:?}"
        );
    }

    fn legal_uci_from_trace(trace: &[TracePacket]) -> Vec<String> {
        trace
            .iter()
            .filter(|packet| packet.op == TraceOp::LegalSet && packet.a1 == 1)
            .map(|packet| MovePacket::from_move_id(packet.a0, 0).unwrap().to_uci())
            .collect()
    }

    #[test]
    fn start_position_has_exact_20_legal_moves() {
        let mut engine = Engine::new().unwrap();
        assert_eq!(engine.cuda_candidate_move_attention_count(), 0);
        assert_eq!(engine.cuda_candidate_move_layer_count(), 0);
        let moves = engine.legal_moves_uci(START).unwrap();
        assert_eq!(engine.cuda_candidate_move_attention_count(), 1);
        assert!(engine.cuda_candidate_move_layer_count() >= 7);
        assert_eq!(
            moves,
            vec![
                "a2a3", "a2a4", "b1a3", "b1c3", "b2b3", "b2b4", "c2c3", "c2c4", "d2d3", "d2d4",
                "e2e3", "e2e4", "f2f3", "f2f4", "g1f3", "g1h3", "g2g3", "g2g4", "h2h3", "h2h4",
            ]
        );
        let graph = engine.frozen_rule_graph_json().unwrap();
        assert!(graph
            .contains("\"pseudo_legal_moves_backend\":\"cuda_candidate_moves_layered_attention\""));
        assert!(graph.contains("\"pseudo_legal_cpp_control_flow_remaining\":false"));
        assert!(graph.contains(
            "\"candidate_moves_layers\":\"context_select,piece_dispatch,target_mask_select,castle_merge,promotion_expand,record_emit,prefix_rank_select,record_order_select\""
        ));
    }

    #[test]
    fn special_rules_include_castling_en_passant_and_promotion() {
        let mut engine = Engine::new().unwrap();

        let castle = engine
            .legal_moves_uci("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
            .unwrap();
        assert!(castle.contains(&"e1g1".to_string()));
        assert!(castle.contains(&"e1c1".to_string()));

        let ep = engine
            .legal_moves_uci("7k/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
            .unwrap();
        assert!(ep.contains(&"e5d6".to_string()));

        let promo = engine
            .legal_moves_uci("7k/P7/8/8/8/8/6K1/8 w - - 0 1")
            .unwrap();
        assert!(promo.contains(&"a7a8q".to_string()));
        assert!(promo.contains(&"a7a8n".to_string()));
    }

    #[test]
    fn legal_filter_rejects_self_check_en_passant_and_castling_through_attack() {
        let mut engine = Engine::new().unwrap();

        let pinned_ep = engine
            .legal_moves_uci("4r2k/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
            .unwrap();
        assert!(!pinned_ep.contains(&"e5d6".to_string()));

        let castle_through_attack = engine
            .legal_moves_uci("r3k2r/8/8/8/8/5r2/8/R3K2R w KQkq - 0 1")
            .unwrap();
        assert!(!castle_through_attack.contains(&"e1g1".to_string()));
        assert!(castle_through_attack.contains(&"e1c1".to_string()));
    }

    #[test]
    fn side_to_move_controls_black_generation_and_check_escape_filter() {
        let mut engine = Engine::new().unwrap();

        let black_start_reply = engine
            .legal_moves_uci("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
            .unwrap();
        assert_eq!(black_start_reply.len(), 20);
        assert!(black_start_reply.contains(&"e7e5".to_string()));
        assert!(black_start_reply.contains(&"g8f6".to_string()));
        assert!(!black_start_reply.contains(&"e2e4".to_string()));

        let in_check = engine
            .legal_moves_uci("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1")
            .unwrap();
        assert_eq!(in_check, vec!["e1d1", "e1e2", "e1f1"]);
    }

    #[test]
    fn native_legal_trace_emits_candidate_legal_pairs_and_halt() {
        let mut engine = Engine::new().unwrap();
        let trace = engine.legal_trace_packets(START).unwrap();
        let candidate_count = trace
            .iter()
            .filter(|packet| packet.op == TraceOp::Candidate)
            .count();
        let legal_count = trace
            .iter()
            .filter(|packet| packet.op == TraceOp::LegalSet)
            .count();

        assert_eq!(candidate_count, 20);
        assert_eq!(legal_count, 20);
        assert_eq!(trace.last().unwrap().op, TraceOp::ProgramHalt);
        let trace_move_ids = trace
            .iter()
            .filter(|packet| packet.op == TraceOp::Candidate)
            .map(|packet| packet.a0)
            .collect::<Vec<_>>();
        let mut sorted_trace_move_ids = trace_move_ids.clone();
        sorted_trace_move_ids.sort_unstable();
        assert_eq!(trace_move_ids, sorted_trace_move_ids);

        let mut trace_uci = legal_uci_from_trace(&trace);
        trace_uci.sort();
        assert_eq!(trace_uci, engine.legal_moves_uci(START).unwrap());
    }

    #[test]
    fn native_legal_trace_preserves_illegal_pseudo_candidate_rejection() {
        let mut engine = Engine::new().unwrap();
        let trace = engine
            .legal_trace_packets("4r2k/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
            .unwrap();
        let ep_move_id = MovePacket::from_uci("e5d6", 0).unwrap().move_id();
        let candidate = trace
            .iter()
            .find(|packet| packet.op == TraceOp::Candidate && packet.a0 == ep_move_id)
            .unwrap();
        let legal = trace
            .iter()
            .find(|packet| packet.op == TraceOp::LegalSet && packet.a0 == ep_move_id)
            .unwrap();

        assert_eq!(
            candidate.commit,
            super::move_flag::CAPTURE | super::move_flag::EP
        );
        assert_eq!(legal.a1, 0);
        assert!(!legal_uci_from_trace(&trace).contains(&"e5d6".to_string()));
    }

    #[test]
    fn native_legal_trace_stream_decodes_one_packet_per_call() {
        let mut engine = Engine::new().unwrap();
        assert_eq!(
            engine.runtime_mode().unwrap(),
            "native_cuda_trace_select_decoder"
        );
        let full_trace = engine.legal_trace_packets(START).unwrap();
        let packet_count = engine.begin_legal_trace_stream(START).unwrap();
        let mut streamed = Vec::new();

        for _ in 0..packet_count {
            let packet = engine.decode_next_legal_trace_packet().unwrap();
            streamed.push(packet);
            if packet.op == TraceOp::ProgramHalt {
                break;
            }
        }

        assert_eq!(streamed, full_trace);
        assert_eq!(streamed.len(), packet_count);
        assert_eq!(engine.attention_decode_count(), packet_count);
    }

    #[test]
    fn native_legal_trace_stream_uses_cuda_trace_select_without_cpu_fallback() {
        let mut engine = Engine::new().unwrap();
        assert_eq!(engine.cuda_trace_select_count(), 0);
        let before_hullkv = engine.cutlass_hardmax2d_count();

        let full_trace = engine.legal_trace_packets(START).unwrap();
        let packet_count = engine.begin_legal_trace_stream(START).unwrap();
        let mut streamed = Vec::new();
        for _ in 0..packet_count {
            streamed.push(engine.decode_next_legal_trace_packet().unwrap());
        }

        assert_eq!(streamed, full_trace);
        assert_eq!(engine.cuda_trace_select_count(), packet_count);
        assert_eq!(engine.attention_decode_count(), packet_count);
        assert_eq!(
            engine.cutlass_hardmax2d_count() - before_hullkv,
            packet_count
        );
        assert!(engine
            .frozen_rule_graph_json()
            .unwrap()
            .contains("\"trace_select_backend\":\"cuda_trace_select_packet\""));
        assert!(engine
            .frozen_rule_graph_json()
            .unwrap()
            .contains("\"trace_select_long_context_cache\":\"HullKVCache\""));
    }

    #[test]
    fn native_make_move_trace_emits_commit_board_writes_state_terminal_and_halt() {
        let mut engine = Engine::new().unwrap();
        let trace = engine
            .make_move_trace_packets(START, "e2e4", 0, 1, false)
            .unwrap();
        let commit = trace
            .iter()
            .find(|packet| packet.op == TraceOp::CommitMove)
            .unwrap();
        let e2 = square_index("e2").unwrap();
        let e3 = square_index("e3").unwrap();
        let e4 = square_index("e4").unwrap();

        assert_eq!(
            commit.a0,
            MovePacket::from_uci("e2e4", 0).unwrap().move_id()
        );
        assert_eq!(commit.a1, e2);
        assert_eq!(commit.a2, e4);
        assert!(trace.iter().any(|packet| {
            packet.op == TraceOp::WriteSq && packet.a0 == e2 && packet.a1 == 0 && packet.commit == 1
        }));
        assert!(trace.iter().any(|packet| {
            packet.op == TraceOp::WriteSq && packet.a0 == e4 && packet.a1 == 1 && packet.commit == 1
        }));
        assert!(trace.iter().any(|packet| {
            packet.op == TraceOp::WriteReg && packet.a0 == 1 && packet.a1 == 1 && packet.a2 == 1
        }));
        assert!(trace
            .iter()
            .any(|packet| packet.op == TraceOp::WriteCastle && packet.a0 == 15 && packet.a1 == 1));
        assert!(trace
            .iter()
            .any(|packet| packet.op == TraceOp::WriteEp && packet.a0 == e3 && packet.a1 == 1));
        assert!(trace
            .iter()
            .any(|packet| packet.op == TraceOp::WriteClock && packet.a0 == 0 && packet.a1 == 1));
        assert!(trace.iter().any(|packet| {
            packet.op == TraceOp::TerminalSet
                && packet.a0 == 0
                && packet.a1 == 0
                && packet.commit == 0
        }));
        assert_eq!(trace.last().unwrap().op, TraceOp::ProgramHalt);
    }

    #[test]
    fn cuda_probe_doubles_exact_values_when_gpu_is_available() {
        let mut engine = Engine::new().unwrap();
        assert!(engine.cuda_available());
        assert_eq!(
            engine.cuda_probe_double(&[1, 2, 7, 11]).unwrap(),
            vec![2, 4, 14, 22]
        );
        assert!(!engine.cuda_device_name().unwrap().is_empty());
    }

    #[test]
    fn native_percepta_contract_requires_hullkv_2d_attention_and_no_fallbacks() {
        let engine = Engine::new().unwrap();
        let contract = engine.percepta_contract_json().unwrap();

        assert!(contract.contains("\"executor_head_dim\":2"));
        assert!(contract.contains("\"rule_attention_backend\":\"hull_hardmax_2d\""));
        assert!(contract.contains("\"topk_backend\":\"nested_hull_topk_2d\""));
        assert!(contract.contains("\"long_context_cache\":\"HullKVCache\""));
        assert!(contract.contains("\"trace_streaming\":true"));
        assert!(contract.contains("\"simple_kv_cache\":false"));
        assert!(contract.contains("\"python_hot_path\":false"));
        assert!(contract.contains("\"fallback_allowed\":false"));
        assert!(contract.contains("\"decoder_backend\":\"libtorch_cuda_policy_only_v1\""));
        assert!(contract.contains("\"learning_method\":\"self_play_policy_gradient\""));
        assert!(contract.contains("\"actor_critic\":false"));
        assert!(contract.contains("\"critic_head_enabled\":false"));
        assert!(contract.contains("\"value_head_enabled\":false"));
        assert!(contract.contains("\"externally_prescribed_critic\":false"));
    }

    #[test]
    fn native_hull_hardmax_2d_matches_dense_argmax() {
        let mut engine = Engine::new().unwrap();
        let keys = [
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (2.0, 2.0),
            (-2.0, 3.0),
            (3.0, -1.0),
        ];
        let queries = [(1.0, 2.0), (-3.0, 1.0), (2.0, -1.0), (0.25, 0.75)];

        for query in queries {
            let result = engine.hull_hardmax_2d(query, &keys).unwrap();
            let expected = dense_hardmax_2d(query, &keys);
            assert_eq!(result.index, expected.0);
            assert_eq!(result.score, expected.1);
            assert!(!result.used_dense_scan);
        }
    }

    #[test]
    fn native_hull_hardmax_2d_uses_cutlass_gemm_frozen_attention_backend() {
        let mut engine = Engine::new().unwrap();
        let keys = [
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (2.0, 2.0),
            (-2.0, 3.0),
            (3.0, -1.0),
        ];

        assert_eq!(engine.cutlass_hardmax2d_count(), 0);
        let result = engine.hull_hardmax_2d((1.0, 2.0), &keys).unwrap();

        assert_eq!(result.index, dense_hardmax_2d((1.0, 2.0), &keys).0);
        assert_eq!(engine.cutlass_hardmax2d_count(), 1);
        assert!(engine
            .percepta_contract_json()
            .unwrap()
            .contains("\"hull_score_backend\":\"cutlass_gemm_2d\""));
        let contract = engine.percepta_contract_json().unwrap();
        assert!(contract.contains("\"hull_select_backend\":\"cuda_hardmax_select\""));
        assert!(contract.contains("\"hull_host_argmax\":false"));
        let graph = engine.frozen_rule_graph_json().unwrap();
        assert!(graph.contains("\"hull_lookup_backend\":\"cutlass_gemm_2d\""));
        assert!(graph.contains("\"hull_hardmax_select_backend\":\"cuda_hardmax_select\""));
        assert!(graph.contains("\"hull_hardmax_host_argmax\":false"));
    }

    #[test]
    fn native_nested_hull_topk_2d_matches_dense_topk_order() {
        let mut engine = Engine::new().unwrap();
        let keys = [
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (2.0, 2.0),
            (-2.0, 3.0),
            (3.0, -1.0),
        ];
        let before_cutlass = engine.cutlass_hardmax2d_count();

        assert_eq!(
            engine.nested_hull_topk_2d((1.0, 2.0), &keys, 3).unwrap(),
            dense_topk_2d((1.0, 2.0), &keys, 3)
        );
        assert_eq!(
            engine.nested_hull_topk_2d((-3.0, 1.0), &keys, 4).unwrap(),
            dense_topk_2d((-3.0, 1.0), &keys, 4)
        );
        assert_eq!(engine.cutlass_hardmax2d_count() - before_cutlass, 2);
        let graph = engine.frozen_rule_graph_json().unwrap();
        assert!(graph.contains("\"nested_hull_topk_backend\":\"cutlass_qk_cuda_topk_select\""));
        assert!(graph.contains("\"nested_hull_topk_cpu\":false"));
    }

    #[test]
    fn native_board_trace_projection_reconstructs_board_hidden_piece_tokens() {
        let mut engine = Engine::new().unwrap();
        let trace = engine
            .make_move_trace_packets(START, "e2e4", 0, 1, false)
            .unwrap();
        let hidden = engine.project_board_trace(&trace).unwrap();

        assert_eq!(hidden.square_piece(square_index("e2").unwrap()), 0);
        assert_eq!(hidden.square_piece(square_index("e4").unwrap()), 1);
        assert_eq!(hidden.side_to_move, 1);
    }

    #[test]
    fn native_decoder_scaffold_requires_2d_heads_and_shared_side_input() {
        assert!(PerceptaDecoderScaffold::new(16, 8, 4).is_ok());
        let decoder = PerceptaDecoderScaffold::new(16, 8, 4).unwrap();

        assert_eq!(decoder.head_dim(), 2);
        assert!(decoder.shared_for_white_black());
        assert_eq!(decoder.command_count(), 6);
        assert!(!decoder.value_head_enabled());
        assert!(!decoder.externally_prescribed_critic());
        assert!(PerceptaDecoderScaffold::new(12, 3, 4).is_err());
    }

    #[test]
    fn native_libtorch_decoder_forward_uses_2d_attention_and_generic_commands() {
        let mut engine = Engine::new().unwrap();
        let trace = engine
            .make_move_trace_packets(START, "e2e4", 0, 1, false)
            .unwrap();
        let hidden = engine.project_board_trace(&trace).unwrap();
        let output = engine.decoder_forward(&hidden).unwrap();

        assert_eq!(output.command_logits.len(), 6);
        assert_eq!(
            output.command_names,
            vec![
                "QUERY_RULES",
                "READ_WORKSPACE",
                "WRITE_WORKSPACE",
                "SELECT_WORKSPACE",
                "COMMIT_MOVE",
                "NOOP",
            ]
        );
        assert_eq!(output.attention_head_dim, 2);
        assert!(!output.tracepacket_backprop);
        assert!(!output.value_head_enabled);
        assert!(!output.critic_head_enabled);
    }

    #[test]
    fn native_libtorch_policy_gradient_step_changes_decoder_weights_without_trace_backprop() {
        let mut engine = Engine::new().unwrap();
        let trace = engine
            .make_move_trace_packets(START, "e2e4", 0, 1, false)
            .unwrap();
        let hidden = engine.project_board_trace(&trace).unwrap();
        let before = engine.decoder_forward(&hidden).unwrap();
        let loss = engine
            .decoder_policy_gradient_step(&hidden, 4, 1.0, 0.05)
            .unwrap();
        let after = engine.decoder_forward(&hidden).unwrap();

        assert!(loss.is_finite());
        assert_ne!(before.command_logits, after.command_logits);
        assert!(!after.tracepacket_backprop);
    }

    #[test]
    fn native_policy_decoder_selects_trace_legal_dashboard_move() {
        let mut engine = Engine::new().unwrap();
        let selection = engine.policy_select_move(START).unwrap();
        let legal = engine.legal_moves_uci(START).unwrap();

        assert_eq!(selection.legal_count, legal.len());
        assert!(selection.selected_index < selection.legal_count);
        assert!(selection.policy_decoder_used);
        assert!(selection.trace_verified_legal);
        assert!(legal.contains(&selection.move_packet.to_uci()));
        let graph = engine.frozen_rule_graph_json().unwrap();
        assert!(graph.contains("\"dashboard_policy_decoder\":true"));
        assert!(graph
            .contains("\"dashboard_policy_selection_backend\":\"native_libtorch_policy_decoder\""));
    }

    #[test]
    fn native_frozen_rule_graph_declares_full_attention_only_contract() {
        let engine = Engine::new().unwrap();
        let graph = engine.frozen_rule_graph_json().unwrap();

        assert!(graph.contains("\"graph_type\":\"frozen_attention_layer_stack\""));
        assert!(graph.contains("\"board_projection\":\"latest_write_hardmax_2d\""));
        assert!(graph.contains("\"board_projection_backend\":\"cuda_latest_write_projection\""));
        assert!(graph.contains("\"trace_select\":\"cursor_hardmax_2d\""));
        assert!(graph.contains("\"trace_select_backend\":\"cuda_trace_select_packet\""));
        assert!(graph.contains("\"trace_streaming_backend\":\"incremental_packet_attention\""));
        assert!(graph.contains("\"trace_streaming_buffered\":false"));
        assert!(graph.contains("\"trace_streaming_full_trace_precompute\":false"));
        assert!(graph.contains("\"piece_dispatch\":\"frozen_table_attention\""));
        assert!(graph.contains("\"attack_masks\":\"static_attack_mask_table_attention\""));
        assert!(graph.contains("\"attack_path_lowered\":\"pawn_knight_king_slider_ray_scan\""));
        assert!(graph.contains("\"ray_scan\":\"blocker_aware_ray_scan_attention\""));
        assert!(graph.contains("\"candidate_targets\":\"target_mask_attention\""));
        assert!(graph.contains("\"castling_targets\":\"castle_path_attention\""));
        assert!(graph.contains("\"legal_filter\":\"king_safety_attention\""));
        assert!(graph.contains("\"make_move\":\"board_write_attention\""));
        assert!(graph.contains("\"terminal_predicates\":\"terminal_status_attention\""));
        assert!(graph.contains("\"move_record_expansion\":\"move_record_attention\""));
        assert!(graph.contains("\"promotion_expansion\":\"promotion_attention\""));
        assert!(graph.contains("\"trace_emission\":\"trace_packet_attention\""));
        assert!(graph.contains("\"make_move_trace_emission\":\"trace_packet_attention\""));
        assert!(graph.contains("\"attention_only_rule_substrate\":true"));
        assert!(graph.contains("\"tensor_layer_substrate\":false"));
        assert!(graph.contains("\"all_rules_must_lower_to_frozen_2d_self_attention\":true"));
        assert!(graph.contains("\"target_full_frozen_attention_only\":true"));
        assert!(graph.contains("\"current_full_frozen_2d_self_attention_only\":false"));
        assert!(graph.contains("\"full_frozen_attention_only\":false"));
        assert!(graph.contains("\"full_rule_lowering_complete\":false"));
        assert!(graph.contains("\"semantic_attention_purity\":false"));
        assert!(graph.contains("\"contract_overclaim_fixed\":true"));
        assert!(graph.contains("\"cpp_control_flow_rule_vm_remaining\":false"));
        assert!(graph.contains(
            "\"resolve_move_backend\":\"cuda_resolve_move_qk_hardmax_legal_set_attention\""
        ));
        assert!(graph.contains("\"resolve_move_scan\":false"));
        assert!(graph.contains("\"resolve_move_qk_hardmax_2d\":true"));
        assert!(graph.contains("\"hullkv_rule_hot_path\":true"));
        assert!(graph.contains("\"trace_select_long_context_cache\":\"HullKVCache\""));
        assert!(graph.contains("\"nested_hull_topk_backend\":\"cutlass_qk_cuda_topk_select\""));
        assert!(graph.contains("\"nested_hull_topk_cpu\":false"));
        assert!(graph.contains("\"dashboard_policy_decoder\":true"));
        assert!(graph
            .contains("\"dashboard_policy_selection_backend\":\"native_libtorch_policy_decoder\""));
        assert!(graph.contains("\"semantic_source_audit\":\"rust_cuda_body_scan_v1\""));
        assert!(graph.contains("\"metadata_only_tests_remaining\":false"));
        assert!(!graph.contains("candidate_pawn_target_mask_control_flow"));
        assert!(!graph.contains("candidate_pawn_slot_condition_control_flow"));
        assert!(!graph.contains("candidate_pawn_push_condition_control_flow"));
        assert!(!graph.contains("candidate_pawn_double_push_condition_control_flow"));
        assert!(!graph.contains("candidate_pawn_capture_ep_condition_control_flow"));
        assert!(!graph.contains("candidate_single_offset_bounds_control_flow"));
        assert!(!graph.contains("candidate_single_offset_bounds_slot_control_flow"));
        assert!(!graph.contains("candidate_single_offset_coordinate_slot_control_flow"));
        assert!(!graph.contains("candidate_single_offset_coordinate_table_control_flow"));
        assert!(graph.contains(
            "\"candidate_single_offset_coordinate_backend\":\"qk_coordinate_slot_lookup\""
        ));
        assert!(graph.contains(
            "\"candidate_single_offset_coordinate_table_backend\":\"qk_coordinate_table_slots\""
        ));
        assert!(!graph.contains("candidate_slider_target_mask_control_flow"));
        assert!(!graph.contains("candidate_slider_ray_slot_control_flow"));
        assert!(!graph.contains("candidate_slider_ray_step_condition_control_flow"));
        assert!(graph.contains(
            "\"candidate_slider_targets_backend\":\"qk_explicit_slider_ray_slot_writes\""
        ));
        assert!(graph
            .contains("\"candidate_slider_ray_backend\":\"qk_explicit_7_step_ray_slot_writes\""));
        assert!(graph
            .contains("\"candidate_offset_targets_backend\":\"qk_explicit_offset_slot_writes\""));
        assert!(!graph.contains("candidate_offset_target_mask_control_flow"));
        assert!(graph
            .contains("\"candidate_target_dispatch_backend\":\"qk_hardmax_piece_family_select\""));
        assert!(!graph.contains("candidate_target_mask_chess_control_flow"));
        assert!(graph
            .contains("\"candidate_record_emit_backend\":\"qk_candidate_slot_write_attention\""));
        assert!(graph.contains(
            "\"candidate_record_compaction_backend\":\"qk_prefix_rank_slot_write_attention\""
        ));
        assert!(!graph.contains("candidate_record_emit_serial_loop"));
        assert!(!graph.contains("candidate_record_slot_compaction_control_flow"));
        assert!(!graph.contains("candidate_record_prefix_rank_control_flow"));
        assert!(!graph.contains("terminal_legal_presence_chess_search"));
        assert!(!graph.contains("terminal_legal_presence_candidate_legal_control_flow"));
        assert!(!graph.contains("terminal_material_counting_control_flow"));
        assert!(
            graph.contains("\"terminal_material_backend\":\"qk_material_class_bitmask_attention\"")
        );
        assert!(graph.contains("terminal_check_state_king_scan"));
        assert!(graph.contains("castle_target_chess_control_flow"));
        assert!(graph.contains("legal_filter_batch_attack_chess_control_flow"));
        assert!(graph.contains("legal_filter_batch_ray_scan_control_flow"));
        assert!(!graph.contains("tests_assert_metadata_not_semantics"));
        assert!(graph.contains("\"monolithic_custom_cuda_rule_kernels_allowed\":false"));
        assert!(graph.contains("\"monolithic_custom_cuda_rule_kernels_remaining\":true"));
        assert!(graph.contains("\"legal_filter_v1_monolithic_cuda_kernel_deprecated\":true"));
        assert!(graph
            .contains("\"legal_filter_v2_target\":\"stack_of_frozen_2d_self_attention_layers\""));
        assert!(graph.contains(
            "\"legal_filter_v2_required_backend\":\"cutlass_qk_scores_hardmax_v_write\""
        ));
    }

    #[test]
    fn native_frozen_rule_graph_declares_known_source_audit_gaps() {
        let engine = Engine::new().unwrap();
        let graph = engine.frozen_rule_graph_json().unwrap();
        let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let cuda_source =
            std::fs::read_to_string(manifest_dir.join("../../cpp/src/cmz_cuda_kernels.cu"))
                .unwrap();
        let cpp_source =
            std::fs::read_to_string(manifest_dir.join("../../cpp/src/cmz_engine.cpp")).unwrap();
        let native_dashboard =
            std::fs::read_to_string(manifest_dir.join("../cmz-dashboard/src/lib.rs")).unwrap();
        let docker_dashboard = std::fs::read_to_string(
            manifest_dir.join("../../../docker/native/start_dashboard.ps1"),
        )
        .unwrap();
        let repo_root = manifest_dir.join("../../..");

        assert!(!cuda_source.contains("cmz_candidate_target_mask_value"));
        assert!(!cuda_source.contains("cmz_candidate_record_emit_attention_kernel"));
        assert!(cuda_source.contains("cmz_candidate_target_mask_qk_hardmax_v_attention_value"));
        assert!(!cuda_source.contains("cmz_candidate_record_emit_qk_hardmax_v_write_kernel"));
        assert!(cuda_source.contains("cmz_candidate_record_slot_validity_qk_write_kernel"));
        assert!(cuda_source.contains("cmz_candidate_record_slot_rank_write_attention_kernel"));

        assert!(!cuda_source.contains("cmz_any_legal_candidate_move"));
        assert!(!cuda_source.contains("cmz_insufficient_material_value"));
        assert!(!cuda_source.contains("cmz_terminal_candidate_move_legal_attention_value"));
        assert!(!cuda_source.contains("cmz_terminal_legal_presence_qk_hardmax_select_value"));
        assert!(cuda_source.contains("cmz_terminal_legal_presence_from_batch_attention_kernel"));
        assert!(!cuda_source.contains("cmz_terminal_material_qk_hardmax_select_value"));
        assert!(cuda_source.contains("cmz_terminal_material_square_class_attention_kernel"));
        assert!(cuda_source.contains("cmz_terminal_material_status_from_masks_attention_value"));
        assert!(!cuda_source.contains("cmz_resolve_move_attention_kernel"));
        assert!(cuda_source.contains("cmz_resolve_move_qk_hardmax_legal_set_attention_kernel"));
        assert!(!cpp_source.contains("legal_trace_stream_tokens = legal_trace_tokens(board)"));
        assert!(!cpp_source.contains("std::vector<uint32_t> legal_trace_stream_tokens"));
        assert!(native_dashboard.contains("policy_select_move"));
        assert!(native_dashboard.contains("policy_decoder_used"));
        assert!(docker_dashboard.contains("cargo run -p cmz-dashboard"));
        assert!(!docker_dashboard.contains("chess_machine_zero.dashboard.server"));
        for legacy_path in [
            "src/chess_machine_zero/model/ranker.py",
            "src/chess_machine_zero/model/baseline.py",
            "src/chess_machine_zero/model/analytic_machine.py",
            "src/chess_machine_zero/model/weight_compiled_machine.py",
            "src/chess_machine_zero/selfplay/actor.py",
            "src/chess_machine_zero/train/losses.py",
            "src/chess_machine_zero/vm/lookahead.py",
            "src/chess_machine_zero/vm/decision_program.py",
        ] {
            assert!(
                !repo_root.join(legacy_path).exists(),
                "legacy strategy/search/eval module must be removed from production source: {legacy_path}"
            );
        }
        for python_runtime_path in [
            "src/chess_machine_zero/model/percepta_attention_rule_kernels.py",
            "src/chess_machine_zero/model/percepta_attention_block_stack.py",
            "src/chess_machine_zero/model/percepta_matrix_attention_runtime.py",
            "src/chess_machine_zero/model/percepta_tensor_trace_runtime.py",
            "src/chess_machine_zero/model/percepta_rule_layer_graph.py",
            "src/chess_machine_zero/model/percepta_frozen_attention_vm.py",
            "src/chess_machine_zero/model/percepta_parametric_selfplay.py",
            "src/chess_machine_zero/dashboard/server.py",
            "src/chess_machine_zero/dashboard/state.py",
        ] {
            assert!(
                !repo_root.join(python_runtime_path).exists(),
                "Python/PyTorch attention runtime must be removed from production source: {python_runtime_path}"
            );
        }

        assert!(graph.contains("\"semantic_source_audit\":\"rust_cuda_body_scan_v1\""));
        assert!(graph.contains("\"metadata_only_tests_remaining\":false"));
        assert!(!graph.contains("candidate_pawn_target_mask_control_flow"));
        assert!(!graph.contains("candidate_pawn_slot_condition_control_flow"));
        assert!(!graph.contains("candidate_pawn_push_condition_control_flow"));
        assert!(!graph.contains("candidate_pawn_double_push_condition_control_flow"));
        assert!(!graph.contains("candidate_pawn_capture_ep_condition_control_flow"));
        assert!(!graph.contains("candidate_single_offset_bounds_control_flow"));
        assert!(!graph.contains("candidate_single_offset_bounds_slot_control_flow"));
        assert!(!graph.contains("candidate_single_offset_coordinate_slot_control_flow"));
        assert!(!graph.contains("candidate_single_offset_coordinate_table_control_flow"));
        assert!(graph.contains(
            "\"candidate_single_offset_coordinate_backend\":\"qk_coordinate_slot_lookup\""
        ));
        assert!(graph.contains(
            "\"candidate_single_offset_coordinate_table_backend\":\"qk_coordinate_table_slots\""
        ));
        assert!(!graph.contains("candidate_slider_target_mask_control_flow"));
        assert!(!graph.contains("candidate_slider_ray_slot_control_flow"));
        assert!(!graph.contains("candidate_slider_ray_step_condition_control_flow"));
        assert!(graph.contains(
            "\"candidate_slider_targets_backend\":\"qk_explicit_slider_ray_slot_writes\""
        ));
        assert!(graph
            .contains("\"candidate_slider_ray_backend\":\"qk_explicit_7_step_ray_slot_writes\""));
        assert!(graph
            .contains("\"candidate_offset_targets_backend\":\"qk_explicit_offset_slot_writes\""));
        assert!(!graph.contains("candidate_offset_target_mask_control_flow"));
        assert!(graph
            .contains("\"candidate_target_dispatch_backend\":\"qk_hardmax_piece_family_select\""));
        assert!(!graph.contains("candidate_target_mask_chess_control_flow"));
        assert!(graph
            .contains("\"candidate_record_emit_backend\":\"qk_candidate_slot_write_attention\""));
        assert!(graph.contains(
            "\"candidate_record_compaction_backend\":\"qk_prefix_rank_slot_write_attention\""
        ));
        assert!(!graph.contains("candidate_record_emit_serial_loop"));
        assert!(!graph.contains("candidate_record_slot_compaction_control_flow"));
        assert!(!graph.contains("candidate_record_prefix_rank_control_flow"));
        assert!(!graph.contains("terminal_legal_presence_chess_search"));
        assert!(!graph.contains("terminal_legal_presence_candidate_legal_control_flow"));
        assert!(!graph.contains("terminal_material_counting_control_flow"));
        assert!(
            graph.contains("\"terminal_material_backend\":\"qk_material_class_bitmask_attention\"")
        );
        assert!(graph.contains("terminal_check_state_king_scan"));
        assert!(graph.contains("castle_target_chess_control_flow"));
        assert!(graph.contains("legal_filter_batch_attack_chess_control_flow"));
        assert!(graph.contains("legal_filter_batch_ray_scan_control_flow"));
        assert!(!graph.contains("tests_assert_metadata_not_semantics"));
        assert!(!graph.contains("legacy_strategy_modules"));
        assert!(!graph.contains("python_attention_runtime_not_cuda_cutlass"));
        assert!(graph.contains("\"trace_streaming_buffered\":false"));
    }

    #[test]
    fn native_semantic_source_audit_names_concrete_rule_control_flow_offenders() {
        let engine = Engine::new().unwrap();
        let graph = engine.frozen_rule_graph_json().unwrap();
        let source = cuda_source();
        let offenders = [
            (
                "terminal_check_state_king_scan",
                "cmz_terminal_check_state_select_attention_kernel",
                &["for (uint32_t square = 0", "king_square"][..],
            ),
            (
                "castle_target_chess_control_flow",
                "cmz_castle_target_mask_value",
                &["castling_rights", "cmz_tokens_attacked_by"][..],
            ),
            (
                "legal_filter_batch_attack_chess_control_flow",
                "cmz_legal_filter_v2_batch_attack_source_select_attention_kernel",
                &["else if ((token == 2U", "cmz_token_is_side"][..],
            ),
            (
                "legal_filter_batch_ray_scan_control_flow",
                "cmz_legal_filter_v2_batch_ray_blocker_select_attention_kernel",
                &["for (int step = 1", "break"][..],
            ),
        ];

        for (gap, symbol, patterns) in offenders {
            let body = cuda_function_body(&source, symbol);
            assert_body_contains_any(body, symbol, patterns);
            assert!(
                graph.contains(gap),
                "frozen rule graph must declare semantic audit gap {gap} for CUDA symbol {symbol}"
            );
        }

        assert!(graph.contains("\"semantic_source_audit\":\"rust_cuda_body_scan_v1\""));
        assert!(graph.contains("\"metadata_only_tests_remaining\":false"));
        assert!(!graph.contains("terminal_legal_presence_candidate_legal_control_flow"));
        assert!(!source.contains("cmz_terminal_candidate_move_legal_attention_value"));
        assert!(!source.contains("cmz_terminal_legal_presence_qk_hardmax_select_value"));
        assert!(!graph.contains("terminal_material_counting_control_flow"));
        assert!(!source.contains("cmz_terminal_material_qk_hardmax_select_value"));
        assert!(!graph.contains("tests_assert_metadata_not_semantics"));
    }

    #[test]
    fn candidate_target_mask_top_level_uses_qk_piece_family_dispatch() {
        let engine = Engine::new().unwrap();
        let graph = engine.frozen_rule_graph_json().unwrap();
        let source = cuda_source();
        let body = cuda_function_body(
            &source,
            "cmz_candidate_target_mask_qk_hardmax_v_attention_value",
        );

        assert!(body.contains("cmz_qk2_hardmax_select_u64"));
        assert!(body.contains("cmz_candidate_pawn_target_mask_attention_value"));
        assert!(body.contains("cmz_candidate_offset_target_mask_attention_value"));
        assert!(body.contains("cmz_candidate_slider_target_mask_attention_value"));
        assert!(!body.contains("if (token == 1U"));
        assert!(!body.contains("else if (token"));
        assert!(!body.contains("bishop || queen"));

        assert!(graph
            .contains("\"candidate_target_dispatch_backend\":\"qk_hardmax_piece_family_select\""));
        assert!(!graph.contains("candidate_target_mask_chess_control_flow"));
        assert!(!graph.contains("candidate_pawn_target_mask_control_flow"));
        assert!(!graph.contains("candidate_pawn_slot_condition_control_flow"));
        assert!(!graph.contains("candidate_pawn_push_condition_control_flow"));
        assert!(!graph.contains("candidate_pawn_double_push_condition_control_flow"));
        assert!(!graph.contains("candidate_pawn_capture_ep_condition_control_flow"));
        assert!(!graph.contains("candidate_single_offset_bounds_control_flow"));
        assert!(!graph.contains("candidate_single_offset_bounds_slot_control_flow"));
        assert!(!graph.contains("candidate_single_offset_coordinate_slot_control_flow"));
        assert!(!graph.contains("candidate_single_offset_coordinate_table_control_flow"));
        assert!(!graph.contains("candidate_slider_target_mask_control_flow"));
        assert!(!graph.contains("candidate_slider_ray_slot_control_flow"));
        assert!(!graph.contains("candidate_slider_ray_step_condition_control_flow"));
    }

    #[test]
    fn candidate_pawn_target_mask_uses_explicit_qk_slot_writes() {
        let engine = Engine::new().unwrap();
        let graph = engine.frozen_rule_graph_json().unwrap();
        let source = cuda_source();
        let body = cuda_function_body(&source, "cmz_candidate_pawn_target_mask_attention_value");

        assert!(body.contains("cmz_candidate_pawn_single_push_slot_attention_value"));
        assert!(body.contains("cmz_candidate_pawn_double_push_slot_attention_value"));
        assert!(body.contains("cmz_candidate_pawn_capture_slot_attention_value"));
        assert!(body.contains("cmz_qk2_select_or_write_u64"));
        assert!(!body.contains("for (int delta_file"));

        assert!(
            graph.contains("\"candidate_pawn_targets_backend\":\"qk_explicit_pawn_slot_writes\"")
        );
        assert!(!graph.contains("candidate_pawn_target_mask_control_flow"));
        assert!(!graph.contains("candidate_pawn_slot_condition_control_flow"));
        assert!(!graph.contains("candidate_pawn_push_condition_control_flow"));
        assert!(!graph.contains("candidate_pawn_double_push_condition_control_flow"));
        assert!(!graph.contains("candidate_pawn_capture_ep_condition_control_flow"));
    }

    #[test]
    fn candidate_pawn_slot_helpers_delegate_to_qk_condition_layers() {
        let engine = Engine::new().unwrap();
        let graph = engine.frozen_rule_graph_json().unwrap();
        let source = cuda_source();
        let single_body = cuda_function_body(
            &source,
            "cmz_candidate_pawn_single_push_slot_attention_value",
        );
        let double_body = cuda_function_body(
            &source,
            "cmz_candidate_pawn_double_push_slot_attention_value",
        );
        let capture_body =
            cuda_function_body(&source, "cmz_candidate_pawn_capture_slot_attention_value");

        assert!(single_body.contains("cmz_candidate_pawn_forward_target_slot_attention_value"));
        assert!(single_body.contains("cmz_candidate_pawn_push_empty_condition_attention_value"));
        assert!(!single_body.contains("cmz_on_board"));
        assert!(!single_body.contains("if ("));

        assert!(double_body.contains("cmz_candidate_pawn_forward_target_slot_attention_value"));
        assert!(double_body.contains("cmz_candidate_pawn_double_push_condition_attention_value"));
        assert!(!double_body.contains("cmz_on_board"));
        assert!(!double_body.contains("if ("));

        assert!(capture_body.contains("cmz_candidate_pawn_capture_target_slot_attention_value"));
        assert!(capture_body.contains("cmz_candidate_pawn_capture_enemy_condition_attention_value"));
        assert!(capture_body.contains("cmz_candidate_pawn_capture_ep_condition_attention_value"));
        assert!(!capture_body.contains("cmz_on_board"));
        assert!(!capture_body.contains("if ("));

        assert!(!graph.contains("candidate_pawn_slot_condition_control_flow"));
        assert!(!graph.contains("candidate_pawn_push_condition_control_flow"));
        assert!(!graph.contains("candidate_pawn_double_push_condition_control_flow"));
        assert!(!graph.contains("candidate_pawn_capture_ep_condition_control_flow"));
    }

    #[test]
    fn candidate_pawn_push_condition_uses_explicit_qk_square_entries() {
        let engine = Engine::new().unwrap();
        let graph = engine.frozen_rule_graph_json().unwrap();
        let source = cuda_source();
        let body = cuda_function_body(
            &source,
            "cmz_candidate_pawn_push_empty_condition_attention_value",
        );

        assert!(source.contains("#define CMZ_PAWN_PUSH_EMPTY_ATTEND_SQUARE"));
        assert!(body.contains("CMZ_PAWN_PUSH_EMPTY_ATTEND_ROW"));
        assert!(body.contains("selected_value"));
        assert!(!body.contains("target_exists"));
        assert!(!body.contains("target_empty"));
        assert!(!body.contains("occupancy_mask & target_slot"));

        assert!(!graph.contains("candidate_pawn_push_condition_control_flow"));
        assert!(!graph.contains("candidate_pawn_double_push_condition_control_flow"));
        assert!(!graph.contains("candidate_pawn_capture_ep_condition_control_flow"));
    }

    #[test]
    fn candidate_pawn_double_push_condition_uses_explicit_qk_entries() {
        let engine = Engine::new().unwrap();
        let graph = engine.frozen_rule_graph_json().unwrap();
        let source = cuda_source();
        let condition_body = cuda_function_body(
            &source,
            "cmz_candidate_pawn_double_push_condition_attention_value",
        );
        let rank_body = cuda_function_body(
            &source,
            "cmz_candidate_pawn_start_rank_match_attention_value",
        );
        let first_step_body = cuda_function_body(
            &source,
            "cmz_candidate_pawn_single_push_nonzero_attention_value",
        );

        assert!(source.contains("#define CMZ_PAWN_DOUBLE_PUSH_ATTEND_SQUARE"));
        assert!(source.contains("#define CMZ_PAWN_START_RANK_ATTEND_ROW"));
        assert!(source.contains("#define CMZ_PAWN_SINGLE_PUSH_NONZERO_ATTEND_ROW"));
        assert!(source.contains("CMZ_PAWN_START_RANK_ATTEND_ENTRY"));
        assert!(source.contains("cmz_qk2_hardmax_select_u32"));
        assert!(condition_body.contains("CMZ_PAWN_DOUBLE_PUSH_ATTEND_ROW"));
        assert!(condition_body.contains("selected_value"));
        assert!(!condition_body.contains("from_rank"));
        assert!(!condition_body.contains("start_rank"));
        assert!(!condition_body.contains("single_push_mask"));
        assert!(!condition_body.contains("target_exists"));
        assert!(!condition_body.contains("target_empty"));
        assert!(!condition_body.contains("occupancy_mask & target_slot"));
        assert!(rank_body.contains("CMZ_PAWN_START_RANK_ATTEND_ROW"));
        assert!(rank_body.contains("selected_value"));
        assert!(!rank_body.contains("from_rank == start_rank"));
        assert!(first_step_body.contains("CMZ_PAWN_SINGLE_PUSH_NONZERO_ATTEND_ROW"));
        assert!(!first_step_body.contains("single_push_mask == 0ULL"));

        assert!(!graph.contains("candidate_pawn_double_push_condition_control_flow"));
        assert!(!graph.contains("candidate_pawn_capture_ep_condition_control_flow"));
    }

    #[test]
    fn candidate_pawn_capture_ep_condition_uses_explicit_qk_entries() {
        let engine = Engine::new().unwrap();
        let graph = engine.frozen_rule_graph_json().unwrap();
        let source = cuda_source();
        let condition_body = cuda_function_body(
            &source,
            "cmz_candidate_pawn_capture_ep_condition_attention_value",
        );
        let target_body = cuda_function_body(
            &source,
            "cmz_candidate_pawn_ep_target_match_attention_value",
        );
        let captured_body = cuda_function_body(
            &source,
            "cmz_candidate_pawn_ep_captured_slot_attention_value",
        );
        let enemy_body = cuda_function_body(
            &source,
            "cmz_candidate_pawn_ep_captured_enemy_attention_value",
        );

        assert!(source.contains("#define CMZ_PAWN_EP_TARGET_ATTEND_SQUARE"));
        assert!(source.contains("#define CMZ_PAWN_EP_CAPTURED_SLOT_ATTEND_ENTRY"));
        assert!(source.contains("#define CMZ_PAWN_EP_CAPTURED_ENEMY_ATTEND_SQUARE"));
        assert!(condition_body.contains("cmz_candidate_pawn_ep_target_match_attention_value"));
        assert!(condition_body.contains("cmz_candidate_pawn_ep_captured_slot_attention_value"));
        assert!(condition_body.contains("cmz_candidate_pawn_ep_captured_enemy_attention_value"));
        assert!(condition_body.contains("cmz_qk2_select_or_write_u64"));
        assert!(!condition_body.contains("ep_square < 64U"));
        assert!(!condition_body.contains("ep_target_slot"));
        assert!(!condition_body.contains("captured_valid"));
        assert!(!condition_body.contains("captured_enemy ="));
        assert!(!condition_body.contains("static_cast<int>(ep_square)"));
        assert!(!condition_body.contains("enemy_mask & captured_slot"));
        assert!(target_body.contains("CMZ_PAWN_EP_TARGET_ATTEND_ROW"));
        assert!(!target_body.contains("ep_square < 64U"));
        assert!(captured_body.contains("CMZ_PAWN_EP_CAPTURED_SLOT_ATTEND_ROW"));
        assert!(source.contains("cmz_qk2_hardmax_select_u64"));
        assert!(captured_body.contains("selected_value"));
        assert!(!captured_body.contains("static_cast<int>(ep_square)"));
        assert!(enemy_body.contains("CMZ_PAWN_EP_CAPTURED_ENEMY_ATTEND_ROW"));
        assert!(!enemy_body.contains("enemy_mask & captured_slot"));

        assert!(!graph.contains("candidate_pawn_capture_ep_condition_control_flow"));
    }

    #[test]
    fn candidate_offset_target_mask_uses_explicit_qk_slot_writes() {
        let engine = Engine::new().unwrap();
        let graph = engine.frozen_rule_graph_json().unwrap();
        let source = cuda_source();
        let body = cuda_function_body(&source, "cmz_candidate_offset_target_mask_attention_value");

        assert!(body.contains("cmz_qk2_select_or_write_u64"));
        assert!(body.contains("cmz_candidate_single_offset_target_mask_attention_value"));
        assert!(!body.contains("cmz_add_offset_targets"));
        assert!(!body.contains("for ("));

        assert!(graph
            .contains("\"candidate_offset_targets_backend\":\"qk_explicit_offset_slot_writes\""));
        assert!(!graph.contains("candidate_offset_target_mask_control_flow"));
        assert!(!graph.contains("candidate_single_offset_bounds_control_flow"));
        assert!(!graph.contains("candidate_single_offset_bounds_slot_control_flow"));
        assert!(!graph.contains("candidate_single_offset_coordinate_slot_control_flow"));
        assert!(!graph.contains("candidate_single_offset_coordinate_table_control_flow"));
    }

    #[test]
    fn candidate_single_offset_uses_bounds_slot_helper() {
        let engine = Engine::new().unwrap();
        let graph = engine.frozen_rule_graph_json().unwrap();
        let source = cuda_source();
        let body = cuda_function_body(
            &source,
            "cmz_candidate_single_offset_target_mask_attention_value",
        );

        assert!(body.contains("cmz_candidate_single_offset_bounds_slot_attention_value"));
        assert!(body.contains("cmz_qk2_select_or_write_u64"));
        assert!(!body.contains("cmz_on_board"));
        assert!(!body.contains("friendly_mask"));

        assert!(graph
            .contains("\"candidate_single_offset_backend\":\"qk_bounds_slot_friendly_filter\""));
        assert!(!graph.contains("candidate_single_offset_bounds_control_flow"));
        assert!(!graph.contains("candidate_single_offset_bounds_slot_control_flow"));
        assert!(!graph.contains("candidate_single_offset_coordinate_slot_control_flow"));
        assert!(!graph.contains("candidate_single_offset_coordinate_table_control_flow"));
    }

    #[test]
    fn candidate_single_offset_bounds_slot_uses_coordinate_qk_lookup() {
        let engine = Engine::new().unwrap();
        let graph = engine.frozen_rule_graph_json().unwrap();
        let source = cuda_source();
        let body = cuda_function_body(
            &source,
            "cmz_candidate_single_offset_bounds_slot_attention_value",
        );

        assert!(body.contains("cmz_candidate_single_offset_coordinate_slot_attention_value"));
        assert!(body.contains("cmz_qk2_select_or_write_u64"));
        assert!(!body.contains("cmz_on_board"));
        assert!(!body.contains("clamped_file"));

        assert!(graph.contains(
            "\"candidate_single_offset_coordinate_backend\":\"qk_coordinate_slot_lookup\""
        ));
        assert!(graph.contains(
            "\"candidate_single_offset_coordinate_table_backend\":\"qk_coordinate_table_slots\""
        ));
        assert!(!graph.contains("candidate_single_offset_bounds_slot_control_flow"));
        assert!(!graph.contains("candidate_single_offset_coordinate_slot_control_flow"));
        assert!(!graph.contains("candidate_single_offset_coordinate_table_control_flow"));
    }

    #[test]
    fn candidate_single_offset_coordinate_slot_uses_qk_table_lookup() {
        let engine = Engine::new().unwrap();
        let graph = engine.frozen_rule_graph_json().unwrap();
        let source = cuda_source();
        let body = cuda_function_body(
            &source,
            "cmz_candidate_single_offset_coordinate_slot_attention_value",
        );

        assert!(body.contains("cmz_candidate_single_offset_coordinate_table_attention_value"));
        assert!(body.contains("cmz_qk2_select_or_write_u64"));
        assert!(!body.contains("cmz_on_board"));
        assert!(!body.contains("clamped_file"));

        assert!(graph.contains(
            "\"candidate_single_offset_coordinate_table_backend\":\"qk_coordinate_table_slots\""
        ));
        assert!(!graph.contains("candidate_single_offset_coordinate_slot_control_flow"));
        assert!(!graph.contains("candidate_single_offset_coordinate_table_control_flow"));
    }

    #[test]
    fn candidate_single_offset_coordinate_table_uses_explicit_qk_entries() {
        let engine = Engine::new().unwrap();
        let graph = engine.frozen_rule_graph_json().unwrap();
        let source = cuda_source();
        let body = cuda_function_body(
            &source,
            "cmz_candidate_single_offset_coordinate_table_attention_value",
        );

        assert!(source.contains("#define CMZ_COORDINATE_TABLE_ATTEND_ENTRY"));
        assert!(body.contains("CMZ_COORDINATE_TABLE_ATTEND_ROW"));
        assert!(body.contains("selected_value"));
        assert!(!body.contains("cmz_on_board"));
        assert!(!body.contains("clamped_file"));
        assert!(!body.contains("if ("));
        assert!(!body.contains("for ("));

        assert!(graph.contains(
            "\"candidate_single_offset_coordinate_table_backend\":\"qk_coordinate_table_slots\""
        ));
        assert!(!graph.contains("candidate_single_offset_coordinate_slot_control_flow"));
        assert!(!graph.contains("candidate_single_offset_coordinate_table_control_flow"));
    }

    #[test]
    fn candidate_slider_target_mask_uses_explicit_qk_ray_slots() {
        let engine = Engine::new().unwrap();
        let graph = engine.frozen_rule_graph_json().unwrap();
        let source = cuda_source();
        let body = cuda_function_body(&source, "cmz_candidate_slider_target_mask_attention_value");

        assert!(body.contains("cmz_candidate_slider_ray_slot_attention_value"));
        assert!(body.contains("cmz_qk2_select_or_write_u64"));
        assert!(!body.contains("if (bishop || queen)"));
        assert!(!body.contains("if (rook || queen)"));

        assert!(graph.contains(
            "\"candidate_slider_targets_backend\":\"qk_explicit_slider_ray_slot_writes\""
        ));
        assert!(!graph.contains("candidate_slider_target_mask_control_flow"));
        assert!(!graph.contains("candidate_slider_ray_slot_control_flow"));
        assert!(!graph.contains("candidate_slider_ray_step_condition_control_flow"));
    }

    #[test]
    fn candidate_slider_ray_slot_uses_explicit_qk_step_writes() {
        let engine = Engine::new().unwrap();
        let graph = engine.frozen_rule_graph_json().unwrap();
        let source = cuda_source();
        let body = cuda_function_body(&source, "cmz_candidate_slider_ray_slot_attention_value");

        assert!(body.contains("cmz_candidate_slider_ray_step_attention_value"));
        assert!(body.contains("cmz_qk2_select_or_write_u64"));
        assert!(!body.contains("cmz_add_ray_targets"));
        assert!(!body.contains("while ("));

        assert!(graph
            .contains("\"candidate_slider_ray_backend\":\"qk_explicit_7_step_ray_slot_writes\""));
        assert!(!graph.contains("candidate_slider_ray_slot_control_flow"));
        assert!(!graph.contains("candidate_slider_ray_step_condition_control_flow"));
    }

    #[test]
    fn candidate_slider_ray_step_uses_explicit_qk_helpers() {
        let engine = Engine::new().unwrap();
        let graph = engine.frozen_rule_graph_json().unwrap();
        let source = cuda_source();
        let body = cuda_function_body(&source, "cmz_candidate_slider_ray_step_attention_value");
        let target_body = cuda_function_body(
            &source,
            "cmz_candidate_slider_ray_step_target_slot_attention_value",
        );
        let prior_body = cuda_function_body(
            &source,
            "cmz_candidate_slider_ray_prior_blocker_attention_value",
        );
        let occupied_body = cuda_function_body(
            &source,
            "cmz_candidate_slider_square_occupied_attention_value",
        );
        let enabled_body = cuda_function_body(
            &source,
            "cmz_candidate_slider_prior_step_enabled_attention_value",
        );

        assert!(source.contains("#define CMZ_SLIDER_PRIOR_STEP_ENABLED_ATTEND_ENTRY"));
        assert!(source.contains("#define CMZ_SLIDER_OCCUPIED_ATTEND_SQUARE"));
        assert!(body.contains("cmz_candidate_slider_ray_step_target_slot_attention_value"));
        assert!(body.contains("cmz_candidate_slider_ray_prior_blocker_attention_value"));
        assert!(body.contains("cmz_candidate_slider_square_occupied_attention_value"));
        assert!(body.contains("cmz_candidate_slider_slot_nonzero_attention_value"));
        assert!(body.contains("cmz_qk2_select_or_write_u64"));
        assert!(!body.contains("cmz_on_board"));
        assert!(!body.contains("for ("));
        assert!(!body.contains("blocked_before_step"));
        assert!(!body.contains("own_target"));
        assert!(!body.contains("own_occupancy_mask &"));
        assert!(source.contains("#define CMZ_SLIDER_COORDINATE_ATTEND_ROW"));
        assert!(target_body.contains("cmz_candidate_slider_coordinate_table_attention_value"));
        assert!(!target_body.contains("cmz_on_board"));
        assert!(prior_body.contains("cmz_candidate_slider_prior_step_enabled_attention_value"));
        assert!(prior_body.contains("cmz_candidate_slider_square_occupied_attention_value"));
        assert!(!prior_body.contains("for ("));
        assert!(!prior_body.contains("cmz_on_board"));
        assert!(occupied_body.contains("CMZ_SLIDER_OCCUPIED_ATTEND_ROW"));
        assert!(!occupied_body.contains("occupancy_mask & square_slot"));
        assert!(enabled_body.contains("CMZ_SLIDER_PRIOR_STEP_ENABLED_ATTEND_ROW"));
        assert!(enabled_body.contains("selected_value"));

        assert!(!graph.contains("candidate_slider_ray_step_condition_control_flow"));
    }

    #[test]
    fn candidate_record_emit_uses_qk_candidate_slot_writes() {
        let engine = Engine::new().unwrap();
        let graph = engine.frozen_rule_graph_json().unwrap();
        let source = cuda_source();
        let body = cuda_function_body(
            &source,
            "cmz_candidate_record_slot_validity_qk_write_kernel",
        );

        assert!(body.contains("cmz_candidate_record_slot_qk_write_value"));
        assert!(body.contains("cmz_write_candidate_record_fields"));
        assert!(!body.contains("for (uint32_t from = 0"));
        assert!(!body.contains("for (uint32_t to = 0"));
        assert!(!body.contains("for (uint32_t promotion"));

        assert!(graph
            .contains("\"candidate_record_emit_backend\":\"qk_candidate_slot_write_attention\""));
        assert!(!graph.contains("candidate_record_emit_serial_loop"));
        assert!(graph.contains(
            "\"candidate_record_compaction_backend\":\"qk_prefix_rank_slot_write_attention\""
        ));
        assert!(!graph.contains("candidate_record_slot_compaction_control_flow"));
        assert!(!graph.contains("candidate_record_prefix_rank_control_flow"));
    }

    #[test]
    fn candidate_record_compaction_uses_parallel_qk_slot_rank_write() {
        let engine = Engine::new().unwrap();
        let graph = engine.frozen_rule_graph_json().unwrap();
        let source = cuda_source();

        assert!(source.contains("cmz_candidate_record_slot_validity_qk_write_kernel"));
        assert!(source.contains("cmz_candidate_record_slot_rank_write_attention_kernel"));
        assert!(!source.contains("cmz_candidate_record_emit_qk_hardmax_v_write_kernel"));

        assert!(graph.contains(
            "\"candidate_record_compaction_backend\":\"qk_prefix_rank_slot_write_attention\""
        ));
        assert!(!graph.contains("candidate_record_slot_compaction_control_flow"));
        assert!(!graph.contains("candidate_record_prefix_rank_control_flow"));
    }

    #[test]
    fn candidate_record_prefix_rank_uses_qk_rank_value_without_serial_scan() {
        let engine = Engine::new().unwrap();
        let graph = engine.frozen_rule_graph_json().unwrap();
        let source = cuda_source();
        let body = cuda_function_body(
            &source,
            "cmz_candidate_record_slot_rank_write_attention_kernel",
        );

        assert!(source.contains("cmz_candidate_record_prefix_rank_attention_value"));
        assert!(source.contains("cmz_candidate_record_total_count_attention_value"));
        assert!(body.contains("cmz_candidate_record_prefix_rank_attention_value"));
        assert!(body.contains("cmz_candidate_record_total_count_attention_value"));
        assert!(!body.contains("for (uint32_t prior_slot = 0"));
        assert!(!body.contains("atomicMax"));
        assert!(!body.contains("candidate_record_prefix_rank_control_flow"));

        assert!(graph.contains(
            "\"candidate_record_compaction_backend\":\"qk_prefix_rank_slot_write_attention\""
        ));
        assert!(!graph.contains("candidate_record_prefix_rank_control_flow"));
        assert!(!graph.contains(
            "\"remaining_non_attention_paths\":\"candidate_record_prefix_rank_control_flow"
        ));
    }

    #[test]
    fn terminal_legal_presence_uses_batched_candidate_and_legal_filter_attention() {
        let engine = Engine::new().unwrap();
        let graph = engine.frozen_rule_graph_json().unwrap();
        let source = cuda_source();
        let body = cuda_function_body(&source, "cmz_cuda_terminal_status_attention");

        assert!(!source.contains("cmz_terminal_candidate_move_legal_attention_value"));
        assert!(!source.contains("cmz_terminal_legal_presence_accumulate_attention_value"));
        assert!(!source.contains("cmz_terminal_legal_presence_qk_hardmax_select_value"));
        assert!(source.contains("cmz_terminal_legal_presence_from_batch_attention_kernel"));
        assert!(body.contains("cmz_cuda_candidate_moves_attention"));
        assert!(body.contains("cmz_cuda_legal_filter_v2_batch_attention"));
        assert!(body.contains("cmz_terminal_legal_presence_from_batch_attention_kernel"));
        assert!(!body.contains("cmz_candidate_move_is_legal"));

        assert!(!graph.contains("terminal_legal_presence_chess_search"));
        assert!(!graph.contains("terminal_legal_presence_candidate_legal_control_flow"));
    }

    #[test]
    fn terminal_legal_presence_routes_through_batched_attention_without_candidate_legal_helper() {
        let engine = Engine::new().unwrap();
        let graph = engine.frozen_rule_graph_json().unwrap();
        let source = cuda_source();
        let terminal_body = cuda_function_body(&source, "cmz_cuda_terminal_status_attention");

        assert!(!source.contains("cmz_terminal_candidate_move_legal_attention_value"));
        assert!(!source.contains("cmz_terminal_legal_presence_qk_hardmax_select_value"));
        assert!(source.contains("cmz_terminal_legal_presence_from_batch_attention_kernel"));
        assert!(terminal_body.contains("cmz_cuda_candidate_moves_attention"));
        assert!(terminal_body.contains("cmz_cuda_legal_filter_v2_batch_attention"));
        assert!(terminal_body.contains("cmz_terminal_legal_presence_from_batch_attention_kernel"));

        assert!(!graph.contains("terminal_legal_presence_chess_search"));
        assert!(!graph.contains("terminal_legal_presence_candidate_legal_control_flow"));
    }

    #[test]
    fn terminal_material_uses_qk_material_class_masks_without_counting_helper() {
        let engine = Engine::new().unwrap();
        let graph = engine.frozen_rule_graph_json().unwrap();
        let source = cuda_source();
        let terminal_body = cuda_function_body(&source, "cmz_cuda_terminal_status_attention");
        let material_body =
            cuda_function_body(&source, "cmz_terminal_material_select_attention_kernel");

        assert!(!source.contains("cmz_terminal_material_qk_hardmax_select_value"));
        assert!(source.contains("cmz_terminal_material_square_class_attention_value"));
        assert!(source.contains("cmz_terminal_material_square_class_attention_kernel"));
        assert!(source.contains("cmz_terminal_material_status_from_masks_attention_value"));
        assert!(terminal_body.contains("cmz_terminal_material_square_class_attention_kernel"));
        assert!(terminal_body.contains("cmz_terminal_material_select_attention_kernel"));
        assert!(!material_body.contains("++non_king_count"));
        assert!(!material_body.contains("return false"));

        assert!(
            graph.contains("\"terminal_material_backend\":\"qk_material_class_bitmask_attention\"")
        );
        assert!(!graph.contains("terminal_material_counting_control_flow"));
    }

    #[test]
    fn board_projection_and_trace_stream_increment_frozen_layer_steps() {
        let mut engine = Engine::new().unwrap();
        let before = engine.frozen_layer_step_count();
        let trace = engine
            .make_move_trace_packets(START, "e2e4", 0, 1, false)
            .unwrap();
        let _hidden = engine.project_board_trace(&trace).unwrap();
        assert!(engine.frozen_layer_step_count() > before);

        let after_projection = engine.frozen_layer_step_count();
        let before_emit = engine.cuda_trace_emit_attention_count();
        let before_select = engine.cuda_trace_select_count();
        let packet_count = engine.begin_legal_trace_stream(START).unwrap();
        for _ in 0..packet_count {
            let packet = engine.decode_next_legal_trace_packet().unwrap();
            if packet.op == TraceOp::ProgramHalt {
                break;
            }
        }
        assert_eq!(
            engine.cuda_trace_emit_attention_count() - before_emit,
            packet_count
        );
        assert_eq!(
            engine.cuda_trace_select_count() - before_select,
            packet_count
        );
        assert!(engine.frozen_layer_step_count() - after_projection >= packet_count * 2);
    }

    #[test]
    fn board_projection_uses_cuda_latest_write_frozen_attention_without_cpu_fallback() {
        let mut engine = Engine::new().unwrap();
        assert_eq!(engine.cuda_board_projection_count(), 0);

        let trace = engine
            .make_move_trace_packets(START, "e2e4", 0, 1, false)
            .unwrap();
        let hidden = engine.project_board_trace(&trace).unwrap();

        assert_eq!(
            hidden.square_piece_tokens[square_index("e2").unwrap() as usize],
            0
        );
        assert_eq!(
            hidden.square_piece_tokens[square_index("e4").unwrap() as usize],
            1
        );
        assert_eq!(hidden.side_to_move, 1);
        assert_eq!(engine.cuda_board_projection_count(), 1);
        assert!(engine
            .frozen_rule_graph_json()
            .unwrap()
            .contains("\"board_projection_backend\":\"cuda_latest_write_projection\""));
    }

    #[test]
    fn frozen_attack_masks_match_static_chess_geometry() {
        let mut engine = Engine::new().unwrap();

        assert_eq!(
            engine
                .frozen_attack_mask(2, square_index("b1").unwrap())
                .unwrap(),
            bitboard(&["a3", "c3", "d2"])
        );
        assert_eq!(
            engine
                .frozen_attack_mask(1, square_index("e4").unwrap())
                .unwrap(),
            bitboard(&["d5", "f5"])
        );
        assert_eq!(
            engine
                .frozen_attack_mask(7, square_index("e4").unwrap())
                .unwrap(),
            bitboard(&["d3", "f3"])
        );
        assert_eq!(
            engine
                .frozen_attack_mask(6, square_index("e4").unwrap())
                .unwrap(),
            bitboard(&["d3", "e3", "f3", "d4", "f4", "d5", "e5", "f5"])
        );
        assert_eq!(
            engine
                .frozen_attack_mask(3, square_index("d4").unwrap())
                .unwrap(),
            bitboard(&[
                "a1", "b2", "c3", "e5", "f6", "g7", "h8", "a7", "b6", "c5", "e3", "f2", "g1"
            ])
        );
        assert_eq!(
            engine
                .frozen_attack_mask(4, square_index("d4").unwrap())
                .unwrap(),
            bitboard(&[
                "d1", "d2", "d3", "d5", "d6", "d7", "d8", "a4", "b4", "c4", "e4", "f4", "g4", "h4"
            ])
        );
    }

    #[test]
    fn frozen_attack_mask_api_increments_layer_counter_and_rejects_invalid_piece() {
        let mut engine = Engine::new().unwrap();
        let before = engine.frozen_layer_step_count();
        let _ = engine
            .frozen_attack_mask(5, square_index("d4").unwrap())
            .unwrap();

        assert_eq!(engine.frozen_layer_step_count(), before + 1);
        assert!(engine
            .frozen_attack_mask(99, square_index("d4").unwrap())
            .is_err());
    }

    #[test]
    fn frozen_attack_mask_uses_cuda_qk_hardmax_v_table_attention_without_cpu_fallback() {
        let mut engine = Engine::new().unwrap();
        assert_eq!(engine.cuda_attack_table_attention_count(), 0);

        assert_eq!(
            engine
                .frozen_attack_mask(2, square_index("b1").unwrap())
                .unwrap(),
            bitboard(&["a3", "c3", "d2"])
        );

        assert_eq!(engine.cuda_attack_table_attention_count(), 1);
        let graph = engine.frozen_rule_graph_json().unwrap();
        assert!(graph.contains("\"attack_masks_backend\":\"cuda_qk_hardmax_v_table_lookup\""));
        assert!(graph.contains("\"table_lookup_semantics\":\"qk_hardmax_v\""));
    }

    #[test]
    fn frozen_ray_scan_mask_matches_blocker_aware_slider_geometry() {
        let mut engine = Engine::new().unwrap();
        let d4 = square_index("d4").unwrap();

        assert_eq!(
            engine.frozen_ray_scan_mask(d4, 1, 0, 0).unwrap(),
            bitboard(&["e4", "f4", "g4", "h4"])
        );
        assert_eq!(
            engine.frozen_ray_scan_mask(d4, 1, 1, 0).unwrap(),
            bitboard(&["e5", "f6", "g7", "h8"])
        );
        assert_eq!(
            engine
                .frozen_ray_scan_mask(d4, 1, 0, bitboard(&["f4", "h4"]))
                .unwrap(),
            bitboard(&["e4", "f4"])
        );
        assert_eq!(
            engine
                .frozen_ray_scan_mask(d4, -1, 0, bitboard(&["b4"]))
                .unwrap(),
            bitboard(&["c4", "b4"])
        );
    }

    #[test]
    fn frozen_ray_scan_mask_increments_counter_and_rejects_invalid_direction() {
        let mut engine = Engine::new().unwrap();
        let before = engine.frozen_layer_step_count();
        let _ = engine
            .frozen_ray_scan_mask(square_index("d4").unwrap(), 0, 1, 0)
            .unwrap();

        assert_eq!(engine.frozen_layer_step_count(), before + 1);
        assert!(engine
            .frozen_ray_scan_mask(square_index("d4").unwrap(), 0, 0, 0)
            .is_err());
        assert!(engine
            .frozen_ray_scan_mask(square_index("d4").unwrap(), 2, 1, 0)
            .is_err());
    }

    #[test]
    fn frozen_ray_scan_uses_cuda_nearest_blocker_attention_without_cpu_fallback() {
        let mut engine = Engine::new().unwrap();
        let d4 = square_index("d4").unwrap();

        assert_eq!(engine.cuda_ray_scan_attention_count(), 0);
        assert_eq!(
            engine
                .frozen_ray_scan_mask(d4, 1, 0, bitboard(&["f4", "h4"]))
                .unwrap(),
            bitboard(&["e4", "f4"])
        );
        assert_eq!(engine.cuda_ray_scan_attention_count(), 1);

        let graph = engine.frozen_rule_graph_json().unwrap();
        assert!(graph.contains("\"ray_scan_backend\":\"cuda_nearest_blocker_attention\""));
        assert!(graph.contains("\"ray_scan_semantics\":\"qk_hardmax_v_nearest_blocker\""));
    }

    #[test]
    fn frozen_candidate_target_mask_filters_friendly_and_keeps_enemy_targets() {
        let mut engine = Engine::new().unwrap();
        let d4 = square_index("d4").unwrap();
        let friendly = bitboard(&["f5", "d6"]);
        let enemy = bitboard(&["f3", "b4"]);
        let occupancy = friendly | enemy;

        assert_eq!(
            engine
                .frozen_candidate_target_mask(2, d4, friendly, enemy, occupancy, 64)
                .unwrap(),
            bitboard(&["b3", "b5", "c2", "c6", "e2", "e6", "f3"])
        );
        assert_eq!(
            engine
                .frozen_candidate_target_mask(4, d4, friendly, enemy, occupancy, 64)
                .unwrap(),
            bitboard(&["d1", "d2", "d3", "d5", "b4", "c4", "e4", "f4", "g4", "h4"])
        );
    }

    #[test]
    fn frozen_candidate_target_mask_handles_pawn_push_capture_and_ep_targets() {
        let mut engine = Engine::new().unwrap();
        let e2 = square_index("e2").unwrap();
        let e4 = square_index("e4").unwrap();
        let white_enemy = bitboard(&["d3"]);
        let black_enemy = bitboard(&["d4", "f3"]);

        assert_eq!(
            engine
                .frozen_candidate_target_mask(1, e2, 0, white_enemy, white_enemy, 64)
                .unwrap(),
            bitboard(&["d3", "e3", "e4"])
        );
        assert_eq!(
            engine
                .frozen_candidate_target_mask(
                    1,
                    e2,
                    bitboard(&["e3"]),
                    white_enemy,
                    bitboard(&["e3"]) | white_enemy,
                    64,
                )
                .unwrap(),
            bitboard(&["d3"])
        );
        assert_eq!(
            engine
                .frozen_candidate_target_mask(
                    7,
                    e4,
                    0,
                    black_enemy,
                    black_enemy,
                    square_index("d3").unwrap(),
                )
                .unwrap(),
            bitboard(&["d3", "e3", "f3"])
        );
    }

    #[test]
    fn frozen_candidate_target_mask_increments_counter_and_rejects_invalid_piece() {
        let mut engine = Engine::new().unwrap();
        let before = engine.frozen_layer_step_count();
        let _ = engine
            .frozen_candidate_target_mask(6, square_index("e4").unwrap(), 0, 0, 0, 64)
            .unwrap();

        assert_eq!(engine.frozen_layer_step_count(), before + 1);
        assert!(engine
            .frozen_candidate_target_mask(99, square_index("e4").unwrap(), 0, 0, 0, 64)
            .is_err());
    }

    #[test]
    fn frozen_candidate_targets_use_cuda_qk_hardmax_v_attention_without_cpu_fallback() {
        let mut engine = Engine::new().unwrap();
        let d4 = square_index("d4").unwrap();
        let friendly = bitboard(&["f5", "d6"]);
        let enemy = bitboard(&["f3", "b4"]);
        let occupancy = friendly | enemy;

        assert_eq!(engine.cuda_candidate_table_attention_count(), 0);
        assert_eq!(
            engine
                .frozen_candidate_target_mask(2, d4, friendly, enemy, occupancy, 64)
                .unwrap(),
            bitboard(&["b3", "b5", "c2", "c6", "e2", "e6", "f3"])
        );
        assert_eq!(engine.cuda_candidate_table_attention_count(), 1);

        let graph = engine.frozen_rule_graph_json().unwrap();
        assert!(graph.contains("\"candidate_targets_backend\":\"cuda_qk_hardmax_v_target_lookup\""));
        assert!(graph.contains("\"candidate_filter_backend\":\"cuda_dynamic_mask_attention\""));
    }

    #[test]
    fn frozen_castle_targets_use_cuda_castle_path_attention_without_cpu_fallback() {
        let mut engine = Engine::new().unwrap();

        assert_eq!(engine.cuda_castle_attention_count(), 0);
        assert_eq!(
            engine
                .frozen_castle_target_mask("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", true)
                .unwrap(),
            bitboard(&["c1", "g1"])
        );
        assert_eq!(engine.cuda_castle_attention_count(), 1);

        let graph = engine.frozen_rule_graph_json().unwrap();
        assert!(graph.contains("\"castling_targets_backend\":\"cuda_castle_path_attention\""));
    }

    #[test]
    fn frozen_move_legal_layer_matches_king_safety_filter_cases() {
        let mut engine = Engine::new().unwrap();

        assert!(engine.frozen_move_legal(START, "e2e4").unwrap());
        assert!(!engine.frozen_move_legal(START, "e2e5").unwrap());
        assert!(!engine
            .frozen_move_legal("4r2k/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6")
            .unwrap());
        assert!(!engine
            .frozen_move_legal("r3k2r/8/8/8/8/5r2/8/R3K2R w KQkq - 0 1", "e1g1")
            .unwrap());
        assert!(engine
            .frozen_move_legal("r3k2r/8/8/8/8/5r2/8/R3K2R w KQkq - 0 1", "e1c1")
            .unwrap());
        assert!(engine
            .frozen_move_legal("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1", "e1d1")
            .unwrap());
        assert!(!engine
            .frozen_move_legal("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1", "e1d2")
            .unwrap());
    }

    #[test]
    fn frozen_move_legal_layer_increments_counter_and_rejects_bad_input() {
        let mut engine = Engine::new().unwrap();
        let before = engine.frozen_layer_step_count();

        assert!(engine.frozen_move_legal(START, "g1f3").unwrap());
        assert_eq!(engine.frozen_layer_step_count(), before + 1);
        assert!(engine.frozen_move_legal("not a fen", "g1f3").is_err());
    }

    #[test]
    fn frozen_move_legal_uses_v2_layered_self_attention_without_v1_kernel() {
        let mut engine = Engine::new().unwrap();

        assert_eq!(engine.cuda_legal_filter_attention_count(), 0);
        assert_eq!(engine.cuda_legal_filter_v2_attention_count(), 0);
        assert_eq!(engine.cuda_legal_filter_v2_layer_count(), 0);
        assert!(engine.frozen_move_legal(START, "e2e4").unwrap());
        assert_eq!(engine.cuda_legal_filter_attention_count(), 0);
        assert_eq!(engine.cuda_legal_filter_v2_attention_count(), 1);
        assert!(engine.cuda_legal_filter_v2_layer_count() >= 9);
        assert!(!engine
            .frozen_move_legal("4r2k/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6")
            .unwrap());
        assert_eq!(engine.cuda_legal_filter_attention_count(), 0);
        assert_eq!(engine.cuda_legal_filter_v2_attention_count(), 2);
        assert!(engine.cuda_legal_filter_v2_layer_count() >= 18);

        let graph = engine.frozen_rule_graph_json().unwrap();
        assert!(graph
            .contains("\"legal_filter_backend\":\"cuda_legal_filter_v2_layered_self_attention\""));
        assert!(graph
            .contains("\"legal_filter_v2_current_backend\":\"cuda_qk_hardmax_v_write_layers\""));
        assert!(graph.contains(
            "\"legal_filter_v2_layers_complete\":\"move_type_select,board_write_select,en_passant_capture_select,castle_rook_write_select,promotion_select,king_square_select,attack_source_select,ray_blocker_select,final_legal_select\""
        ));
        assert!(graph.contains("\"legal_filter_v2_inner_select\":\"qk_hardmax_2d_helpers\""));
        assert!(graph.contains("\"legal_filter_v2_inner_select_plain_cuda_loops_remaining\":false"));
        assert!(graph.contains("\"legal_filter_v1_single_kernel_remaining\":false"));
        assert!(graph.contains("\"legacy_legal_filter_cuda_symbols_present\":false"));
        assert!(graph.contains("\"make_move_backend\":\"cuda_board_write_attention\""));
    }

    #[test]
    fn frozen_legal_trace_uses_batched_v2_layered_self_attention_without_v1_batch_kernel() {
        let mut engine = Engine::new().unwrap();

        assert_eq!(engine.cuda_legal_filter_attention_count(), 0);
        assert_eq!(engine.cuda_legal_filter_batch_attention_count(), 0);
        assert_eq!(engine.cuda_legal_filter_v2_batch_attention_count(), 0);
        assert_eq!(engine.cuda_legal_filter_v2_batch_layer_count(), 0);

        let trace = engine.frozen_legal_trace_attention_packets(START).unwrap();

        assert_eq!(legal_uci_from_trace(&trace).len(), 20);
        assert_eq!(engine.cuda_legal_filter_attention_count(), 0);
        assert_eq!(engine.cuda_legal_filter_batch_attention_count(), 0);
        assert_eq!(engine.cuda_legal_filter_v2_batch_attention_count(), 1);
        assert!(engine.cuda_legal_filter_v2_batch_layer_count() >= 9);

        let graph = engine.frozen_rule_graph_json().unwrap();
        assert!(graph.contains(
            "\"legal_filter_batch_backend\":\"cuda_legal_filter_v2_batched_layered_self_attention\""
        ));
        assert!(graph.contains(
            "\"legal_filter_v2_layers_complete\":\"move_type_select,board_write_select,en_passant_capture_select,castle_rook_write_select,promotion_select,king_square_select,attack_source_select,ray_blocker_select,final_legal_select\""
        ));
        assert!(graph.contains("\"legal_filter_v2_inner_select\":\"qk_hardmax_2d_helpers\""));
        assert!(graph.contains("\"legal_filter_v2_inner_select_plain_cuda_loops_remaining\":false"));
        assert!(graph.contains("\"legal_filter_batch_v1_kernel_remaining\":false"));
        assert!(graph.contains("\"legacy_legal_filter_cuda_symbols_present\":false"));
        assert!(graph.contains("\"small_launch_fusion\":\"legal_filter_batch_v2\""));
    }

    #[test]
    fn frozen_terminal_status_layer_matches_terminal_rule_cases() {
        let mut engine = Engine::new().unwrap();

        assert_eq!(
            engine.frozen_terminal_status(START, 1, false).unwrap(),
            TerminalStatus {
                result: 0,
                reason: 0
            }
        );
        assert_eq!(
            engine.frozen_terminal_status(START, 3, false).unwrap(),
            TerminalStatus {
                result: 3,
                reason: 4
            }
        );
        assert_eq!(
            engine
                .frozen_terminal_status("8/8/8/8/8/8/8/K6k w - - 100 1", 1, false)
                .unwrap(),
            TerminalStatus {
                result: 3,
                reason: 3
            }
        );
        assert_eq!(
            engine
                .frozen_terminal_status("8/8/8/8/8/8/8/K6k w - - 0 1", 1, false)
                .unwrap(),
            TerminalStatus {
                result: 3,
                reason: 5
            }
        );
        assert_eq!(
            engine
                .frozen_terminal_status("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", 1, false)
                .unwrap(),
            TerminalStatus {
                result: 1,
                reason: 1
            }
        );
        assert_eq!(
            engine
                .frozen_terminal_status("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", 1, false)
                .unwrap(),
            TerminalStatus {
                result: 3,
                reason: 2
            }
        );
    }

    #[test]
    fn frozen_terminal_status_layer_increments_counter_and_rejects_bad_input() {
        let mut engine = Engine::new().unwrap();
        let before = engine.frozen_layer_step_count();
        assert_eq!(engine.cuda_terminal_status_attention_count(), 0);
        assert_eq!(engine.cuda_terminal_status_layer_count(), 0);

        let _ = engine.frozen_terminal_status(START, 1, false).unwrap();
        assert!(engine.frozen_layer_step_count() >= before + 5);
        assert_eq!(engine.cuda_terminal_status_attention_count(), 1);
        assert!(engine.cuda_terminal_status_layer_count() >= 5);
        let graph = engine.frozen_rule_graph_json().unwrap();
        assert!(graph.contains(
            "\"terminal_predicates_backend\":\"cuda_terminal_status_layered_attention\""
        ));
        assert!(graph.contains("\"terminal_cpp_logic_remaining\":false"));
        assert!(graph.contains(
            "\"terminal_status_layers\":\"draw_rule_select,legal_presence_select,check_state_select,material_class_select,material_status_select,final_status_select\""
        ));
        assert!(engine
            .frozen_terminal_status("not a fen", 1, false)
            .is_err());
    }

    #[test]
    fn frozen_legal_trace_attention_packets_match_legacy_native_trace_and_emit_promotions() {
        let mut engine = Engine::new().unwrap();
        let legacy = engine.legal_trace_packets(START).unwrap();
        let frozen = engine.frozen_legal_trace_attention_packets(START).unwrap();

        assert_eq!(frozen, legacy);
        assert_eq!(legal_uci_from_trace(&frozen).len(), 20);

        let promotion_trace = engine
            .frozen_legal_trace_attention_packets("7k/P7/8/8/8/8/6K1/8 w - - 0 1")
            .unwrap();
        let promotion_moves = legal_uci_from_trace(&promotion_trace);

        assert!(promotion_moves.contains(&"a7a8q".to_string()));
        assert!(promotion_moves.contains(&"a7a8r".to_string()));
        assert!(promotion_moves.contains(&"a7a8b".to_string()));
        assert!(promotion_moves.contains(&"a7a8n".to_string()));
    }

    #[test]
    fn frozen_legal_trace_attention_packets_increment_layer_counter() {
        let mut engine = Engine::new().unwrap();
        let before = engine.frozen_layer_step_count();
        assert_eq!(engine.cuda_trace_emit_attention_count(), 0);
        let trace = engine.frozen_legal_trace_attention_packets(START).unwrap();

        assert!(!trace.is_empty());
        assert!(engine.frozen_layer_step_count() >= before + trace.len());
        assert_eq!(engine.cuda_trace_emit_attention_count(), trace.len());
        let graph = engine.frozen_rule_graph_json().unwrap();
        assert!(graph.contains("\"trace_append_backend\":\"cuda_trace_packet_emit_attention\""));
        assert!(graph.contains("\"trace_append_cpp_loop_remaining\":false"));
    }

    #[test]
    fn frozen_make_move_trace_attention_packets_match_native_make_move_trace() {
        let mut engine = Engine::new().unwrap();
        assert_eq!(engine.cuda_make_move_board_attention_count(), 0);
        assert_eq!(engine.cuda_make_move_metadata_attention_count(), 0);
        assert_eq!(engine.cuda_resolve_move_attention_count(), 0);
        let native = engine
            .make_move_trace_packets(START, "e2e4", 0, 1, false)
            .unwrap();
        assert_eq!(engine.cuda_resolve_move_attention_count(), 1);
        assert_eq!(engine.cuda_make_move_board_attention_count(), 1);
        assert_eq!(engine.cuda_make_move_metadata_attention_count(), 1);
        let frozen = engine
            .frozen_make_move_trace_attention_packets(START, "e2e4", 0, 1, false)
            .unwrap();
        assert_eq!(engine.cuda_resolve_move_attention_count(), 2);
        assert_eq!(engine.cuda_make_move_board_attention_count(), 2);
        assert_eq!(engine.cuda_make_move_metadata_attention_count(), 2);

        assert_eq!(frozen, native);
        assert!(frozen.iter().any(|packet| packet.op == TraceOp::CommitMove));
        assert!(frozen.iter().any(|packet| packet.op == TraceOp::WriteSq));
        assert!(frozen
            .iter()
            .any(|packet| packet.op == TraceOp::TerminalSet));
        assert_eq!(frozen.last().unwrap().op, TraceOp::ProgramHalt);
        let graph = engine.frozen_rule_graph_json().unwrap();
        assert!(graph
            .contains("\"make_move_board_squares_backend\":\"cuda_make_move_board_attention\""));
        assert!(graph.contains(
            "\"make_move_board_metadata_backend\":\"cuda_make_move_metadata_attention\""
        ));
        assert!(graph.contains(
            "\"resolve_move_backend\":\"cuda_resolve_move_qk_hardmax_legal_set_attention\""
        ));
        assert!(graph.contains("\"resolve_move_cpp_loop_remaining\":false"));
        assert!(graph.contains("\"resolve_move_scan\":false"));
        assert!(graph.contains("\"resolve_move_qk_hardmax_2d\":true"));
    }

    #[test]
    fn frozen_make_move_trace_attention_packets_increment_layer_counter() {
        let mut engine = Engine::new().unwrap();
        let before = engine.frozen_layer_step_count();
        let trace = engine
            .frozen_make_move_trace_attention_packets(START, "e2e4", 0, 1, false)
            .unwrap();

        assert!(!trace.is_empty());
        assert!(engine.frozen_layer_step_count() >= before + trace.len());
    }

    fn dense_hardmax_2d(query: (f32, f32), keys: &[(f32, f32)]) -> (usize, f32) {
        keys.iter()
            .enumerate()
            .map(|(index, key)| (index, query.0 * key.0 + query.1 * key.1))
            .max_by(|left, right| {
                left.1
                    .partial_cmp(&right.1)
                    .unwrap()
                    .then_with(|| right.0.cmp(&left.0))
            })
            .unwrap()
    }

    fn dense_topk_2d(query: (f32, f32), keys: &[(f32, f32)], k: usize) -> Vec<usize> {
        let mut scored = keys
            .iter()
            .enumerate()
            .map(|(index, key)| (index, query.0 * key.0 + query.1 * key.1))
            .collect::<Vec<_>>();
        scored.sort_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap()
                .then_with(|| left.0.cmp(&right.0))
        });
        scored.into_iter().take(k).map(|entry| entry.0).collect()
    }

    fn bitboard(squares: &[&str]) -> u64 {
        squares.iter().fold(0u64, |mask, square| {
            mask | (1u64 << square_index(square).unwrap())
        })
    }
}
