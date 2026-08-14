#include <stdexcept>
#include "cmz_vm2/knight_circuit.h"
#include "cmz_vm2/move_policy.h"

using namespace cmz::vm2;
void require(bool ok,const char* msg){if(!ok)throw std::runtime_error(msg);}
KnightBoard base(KnightSide side,std::int64_t source,KnightPiece piece){KnightBoard b{};b.side=side;b.squares[source]=piece;return b;}
void check(const KnightCircuit& c,std::int64_t source,std::int64_t target,KnightBoard board,bool legal,KnightPiece target_piece,KnightSide next){auto input=bind_knight_input(c,board,source,target);auto state=run_knight_circuit(c,input);require(select_knight_legal(c,state).item<std::int64_t>()==legal,"legality mismatch");auto out=select_knight_board(c,state).argmax(1);for(std::int64_t sq=0;sq<64;++sq){auto expected=board.squares[sq];if(legal&&sq==source)expected=KnightPiece::Empty;if(legal&&sq==target)expected=target_piece;auto actual=out[sq].item<std::int64_t>();if(actual!=static_cast<std::int64_t>(expected))throw std::runtime_error("board write mismatch source="+std::to_string(source)+" target="+std::to_string(target)+" square="+std::to_string(sq)+" expected="+std::to_string(static_cast<std::int64_t>(expected))+" actual="+std::to_string(actual));}auto side=select_knight_side(c,state).argmax(1).item<std::int64_t>();require(side==static_cast<std::int64_t>(next),"side mismatch");}
int main(){
 const auto c=compile_knight_circuit();
 const std::array<std::pair<std::int64_t,std::int64_t>,12> basis{{{0,10},{0,17},{7,13},{7,22},{56,41},{56,50},{63,46},{63,53},{27,10},{27,12},{27,42},{27,44}}};
 for(const auto [s,t]:basis){auto df=std::abs(s%8-t%8),dr=std::abs(s/8-t/8);bool geom=(df==1&&dr==2)||(df==2&&dr==1);auto b=base(KnightSide::White,s,KnightPiece::WhiteKnight);check(c,s,t,b,geom,KnightPiece::WhiteKnight,geom?KnightSide::Black:KnightSide::White);}
 {auto b=base(KnightSide::White,1,KnightPiece::WhiteKnight);b.squares[18]=KnightPiece::BlackOther;check(c,1,18,b,true,KnightPiece::WhiteKnight,KnightSide::Black);}
 {auto b=base(KnightSide::White,1,KnightPiece::WhiteKnight);b.squares[18]=KnightPiece::WhiteOther;check(c,1,18,b,false,KnightPiece::WhiteOther,KnightSide::White);}
 check(c,1,18,base(KnightSide::White,1,KnightPiece::BlackKnight),false,KnightPiece::BlackKnight,KnightSide::White);
 check(c,1,18,base(KnightSide::Black,1,KnightPiece::BlackKnight),true,KnightPiece::BlackKnight,KnightSide::White);
 auto a=bind_knight_input(c,base(KnightSide::White,1,KnightPiece::WhiteKnight),1,18);auto b=bind_knight_input(c,base(KnightSide::White,1,KnightPiece::WhiteKnight),1,18);require(torch::equal(a,b),"binding must be deterministic");
 auto other=bind_knight_input(c,base(KnightSide::White,1,KnightPiece::WhiteKnight),6,21);require(!torch::equal(a,other),"desired move must live in input tokens");
 const auto candidate_inputs=torch::stack({
     bind_knight_input(c,base(KnightSide::White,1,KnightPiece::WhiteKnight),1,2),
     bind_knight_input(c,base(KnightSide::White,1,KnightPiece::WhiteKnight),1,18)});
 const auto candidate_states=run_knight_circuit(c,candidate_inputs);
 const auto legality=select_knight_legality_tokens(c,candidate_states).squeeze(1);
 const auto policy=compile_first_legal_policy(2);
 const auto choice=run_move_policy(policy,torch::zeros({2,2},torch::kFloat64),legality);
 require(torch::equal(choice.selection,torch::tensor({0.,1.},torch::kFloat64)),"policy must select the first rule-circuit LEGAL token without host chess logic");
 for(int st=0;st<kKnightStages;++st)require(!c.wq[st].requires_grad()&&!c.wk[st].requires_grad()&&!c.wv[st].requires_grad()&&!c.masks[st].requires_grad()&&!c.row_writes[st].requires_grad()&&!c.feature_writes[st].requires_grad(),"rule tensors must be frozen");
 return 0;
}
