#include "cmz_vm2/knight_circuit.h"
#include "cmz_vm2/attention.h"
namespace cmz::vm2 {
torch::Tensor run_knight_circuit(const KnightCircuit& c,const torch::Tensor& input){auto state=input;for(std::int64_t stage=0;stage<kKnightStages;++stage){const auto a=self_attention(state,c.wq[stage],c.wk[stage],c.wv[stage],c.masks[stage]);state=state+torch::matmul(torch::matmul(c.row_writes[stage],a.output-state),c.feature_writes[stage]);}return state;}
torch::Tensor select_knight_legal(const KnightCircuit& c,const torch::Tensor& state){return torch::matmul(torch::matmul(c.legal_row,state),c.legal_feature).to(torch::kInt64);}
torch::Tensor select_knight_board(const KnightCircuit& c,const torch::Tensor& state){return torch::matmul(torch::matmul(c.board_rows,state),c.board_features);}
torch::Tensor select_knight_side(const KnightCircuit& c,const torch::Tensor& state){return torch::matmul(torch::matmul(c.side_row,state),c.side_features);}
} // namespace cmz::vm2
