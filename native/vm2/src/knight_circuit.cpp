#include "cmz_vm2/knight_circuit.h"

#include <limits>
#include <stdexcept>

namespace cmz::vm2 {
namespace {
constexpr std::int64_t D=208,Q=0,FILE_REL=1,RANK_REL=65,GEOM_REL=129,SQUARE=132,LEGAL_REL=196,SIDE_REL=296,SRC_WRITE=300,DST_WRITE=301,OUT=302,OUT_SIDE=366,N=367;
constexpr std::int64_t SF=0,TF=8,SR=16,TR=24,DF=32,DR=41,GEOM=50,ONE=52,ADDR=53,TADDR=117,PIECE=181,SRC_PIECE=186,DST_PIECE=191,SIDE=196,LEGAL=198,OUT_PIECE=200,OUT_SIDE_F=205;
torch::Tensor freeze(torch::Tensor x){return x.detach().set_requires_grad(false);}
void copy(torch::Tensor& w,std::int64_t a,std::int64_t b,std::int64_t n,double scale=1){for(std::int64_t i=0;i<n;++i)w.index_put_({a+i,b+i},scale);}
torch::Tensor mask(std::int64_t begin,std::int64_t count,std::int64_t query=Q){auto m=torch::full({N,N},-std::numeric_limits<double>::infinity(),torch::kFloat64);for(std::int64_t r=0;r<N;++r)if(r!=query)m.index_put_({r,r},0.);for(std::int64_t k=begin;k<begin+count;++k)m.index_put_({query,k},0.);return freeze(m);}
torch::Tensor row(std::int64_t begin,std::int64_t count){auto r=torch::zeros({N,N},torch::kFloat64);for(std::int64_t i=begin;i<begin+count;++i)r.index_put_({i,i},1.);return freeze(r);}
torch::Tensor feat(std::int64_t begin,std::int64_t count){auto c=torch::zeros({D,D},torch::kFloat64);for(std::int64_t i=begin;i<begin+count;++i)c.index_put_({i,i},1.);return freeze(c);}
bool allowed(std::int64_t side,std::int64_t source,std::int64_t target){return (side==0&&source==1&&(target==0||target==2||target==4))||(side==1&&source==2&&(target==0||target==1||target==3));}
}

KnightCircuit compile_knight_circuit(std::int64_t source,std::int64_t target){
 if(source<0||source>=64||target<0||target>=64)throw std::out_of_range("knight square outside board");
 auto x=torch::zeros({N,D},torch::kFloat64);x.index_put_({Q,ONE},1.);x.index_put_({Q,SF+source%8},1.);x.index_put_({Q,TF+target%8},1.);x.index_put_({Q,SR+source/8},1.);x.index_put_({Q,TR+target/8},1.);x.index_put_({Q,ADDR+source},1.);x.index_put_({Q,TADDR+target},1.);
 for(std::int64_t a=0;a<8;++a)for(std::int64_t b=0;b<8;++b){auto f=FILE_REL+a*8+b;x.index_put_({f,SF+a},1.);x.index_put_({f,TF+b},1.);x.index_put_({f,DF+std::abs(a-b)},1.);auto r=RANK_REL+a*8+b;x.index_put_({r,SR+a},1.);x.index_put_({r,TR+b},1.);x.index_put_({r,DR+std::abs(a-b)},1.);}
 x.index_put_({GEOM_REL,ONE},1.);x.index_put_({GEOM_REL,GEOM},1.);x.index_put_({GEOM_REL+1,DF+1},1.);x.index_put_({GEOM_REL+1,DR+2},1.);x.index_put_({GEOM_REL+1,GEOM+1},1.);x.index_put_({GEOM_REL+2,DF+2},1.);x.index_put_({GEOM_REL+2,DR+1},1.);x.index_put_({GEOM_REL+2,GEOM+1},1.);
 for(std::int64_t sq=0;sq<64;++sq){x.index_put_({SQUARE+sq,ADDR+sq},1.);x.index_put_({OUT+sq,ADDR+sq},1.);x.index_put_({OUT+sq,ONE},1.);}
 for(std::int64_t g=0;g<2;++g)for(std::int64_t s=0;s<2;++s)for(std::int64_t p=0;p<5;++p)for(std::int64_t t=0;t<5;++t){auto r=LEGAL_REL+(((g*2+s)*5+p)*5+t);x.index_put_({r,GEOM+g},1.);x.index_put_({r,SIDE+s},1.);x.index_put_({r,SRC_PIECE+p},1.);x.index_put_({r,DST_PIECE+t},1.);x.index_put_({r,LEGAL+(g==1&&allowed(s,p,t)?1:0)},1.);}
 for(std::int64_t s=0;s<2;++s)for(std::int64_t l=0;l<2;++l){auto r=SIDE_REL+s*2+l;x.index_put_({r,SIDE+s},1.);x.index_put_({r,LEGAL+l},1.);x.index_put_({r,OUT_SIDE_F+(l?1-s:s)},1.);}
 std::array<torch::Tensor,kKnightStages>wq,wk,wv,masks,rows,features;for(int st=0;st<kKnightStages;++st){wq[st]=torch::zeros({D,D},torch::kFloat64);wk[st]=torch::zeros({D,D},torch::kFloat64);wv[st]=torch::eye(D,torch::kFloat64);masks[st]=mask(1,1);rows[st]=row(Q,1);features[st]=feat(0,0);}
 copy(wq[0],SF,SF,16);copy(wk[0],SF,SF,16);masks[0]=mask(FILE_REL,64);features[0]=feat(DF,9);
 copy(wq[1],SR,SR,16);copy(wk[1],SR,SR,16);masks[1]=mask(RANK_REL,64);features[1]=feat(DR,9);
 copy(wq[2],DF,DF,18);copy(wk[2],DF,DF,18);wq[2].index_put_({ONE,ONE},1.);wk[2].index_put_({ONE,ONE},1.5);masks[2]=mask(GEOM_REL,3);features[2]=feat(GEOM,2);
 copy(wq[3],ADDR,ADDR,64);copy(wk[3],ADDR,ADDR,64);copy(wv[3],PIECE,SRC_PIECE,5);masks[3]=mask(SQUARE,64);features[3]=feat(SRC_PIECE,5);
 copy(wq[4],TADDR,ADDR,64);copy(wk[4],ADDR,ADDR,64);copy(wv[4],PIECE,DST_PIECE,5);masks[4]=mask(SQUARE,64);features[4]=feat(DST_PIECE,5);
 copy(wq[5],GEOM,GEOM,2);copy(wk[5],GEOM,GEOM,2);copy(wq[5],SIDE,SIDE,2);copy(wk[5],SIDE,SIDE,2);copy(wq[5],SRC_PIECE,SRC_PIECE,5);copy(wk[5],SRC_PIECE,SRC_PIECE,5);copy(wq[5],DST_PIECE,DST_PIECE,5);copy(wk[5],DST_PIECE,DST_PIECE,5);copy(wv[5],LEGAL,LEGAL,2);masks[5]=mask(LEGAL_REL,100);features[5]=feat(LEGAL,2);
 copy(wv[6],ADDR,ADDR,64);wv[6].index_put_({ONE,PIECE},1.);copy(wv[6],LEGAL,LEGAL,2);masks[6]=mask(Q,1,SRC_WRITE);rows[6]=row(SRC_WRITE,1);features[6]=feat(ADDR,64)+feat(PIECE,5)+feat(LEGAL,2);
 copy(wv[7],TADDR,ADDR,64);copy(wv[7],SRC_PIECE,PIECE,5);copy(wv[7],LEGAL,LEGAL,2);masks[7]=mask(Q,1,DST_WRITE);rows[7]=row(DST_WRITE,1);features[7]=feat(ADDR,64)+feat(PIECE,5)+feat(LEGAL,2);
 copy(wq[8],ADDR,ADDR,64);copy(wk[8],ADDR,ADDR,64);wq[8].index_put_({ONE,LEGAL+1},.5);wk[8].index_put_({LEGAL+1,LEGAL+1},1.);copy(wv[8],PIECE,OUT_PIECE,5);for(std::int64_t sq=0;sq<64;++sq){masks[8].index_put_({OUT+sq,OUT+sq},-std::numeric_limits<double>::infinity());masks[8].index_put_({OUT+sq,SQUARE+sq},0.);masks[8].index_put_({OUT+sq,SRC_WRITE},0.);masks[8].index_put_({OUT+sq,DST_WRITE},0.);}rows[8]=row(OUT,64);features[8]=feat(OUT_PIECE,5);
 copy(wq[9],SIDE,SIDE,2);copy(wk[9],SIDE,SIDE,2);copy(wq[9],LEGAL,LEGAL,2);copy(wk[9],LEGAL,LEGAL,2);copy(wv[9],OUT_SIDE_F,OUT_SIDE_F,2);masks[9]=mask(SIDE_REL,4);rows[9]=row(Q,1);features[9]=feat(OUT_SIDE_F,2);
 for(int st=0;st<kKnightStages;++st){wq[st]=freeze(wq[st]);wk[st]=freeze(wk[st]);wv[st]=freeze(wv[st]);masks[st]=freeze(masks[st]);rows[st]=freeze(rows[st]);features[st]=freeze(features[st]);}
 auto lr=torch::zeros({1,N},torch::kFloat64);lr.index_put_({0,Q},1.);auto lf=torch::zeros({D,1},torch::kFloat64);lf.index_put_({LEGAL+1,0},1.);auto br=torch::zeros({64,N},torch::kFloat64);for(std::int64_t i=0;i<64;++i)br.index_put_({i,OUT+i},1.);auto bf=torch::zeros({D,5},torch::kFloat64);for(int i=0;i<5;++i)bf.index_put_({OUT_PIECE+i,i},1.);auto sr=torch::zeros({1,N},torch::kFloat64);sr.index_put_({0,Q},1.);auto sf=torch::zeros({D,2},torch::kFloat64);sf.index_put_({OUT_SIDE_F,0},1.);sf.index_put_({OUT_SIDE_F+1,1},1.);
 return KnightCircuit{freeze(x),wq,wk,wv,masks,rows,features,freeze(lr),freeze(lf),freeze(br),freeze(bf),freeze(sr),freeze(sf)};
}

torch::Tensor bind_knight_board(const KnightCircuit& c,const KnightBoard& b){auto x=c.tokens.clone();x.index_put_({Q,torch::indexing::Slice(SIDE,SIDE+2)},0.);x.index_put_({Q,SIDE+static_cast<std::int64_t>(b.side)},1.);for(std::int64_t sq=0;sq<64;++sq){x.index_put_({SQUARE+sq,torch::indexing::Slice(PIECE,PIECE+5)},0.);x.index_put_({SQUARE+sq,PIECE+static_cast<std::int64_t>(b.squares[sq])},1.);}return freeze(x);}
} // namespace cmz::vm2
