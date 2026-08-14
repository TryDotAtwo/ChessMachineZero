#include <cmz_vm3/program_image.h>

#include <array>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace cmz::vm3 {
namespace {

class Reader {
 public:
  explicit Reader(const std::filesystem::path& path) : in_(path, std::ios::binary) {
    if (!in_) fail("cannot open program image");
  }
  template <typename T> T value() {
    T out{};
    in_.read(reinterpret_cast<char*>(&out), sizeof(out));
    if (!in_) fail("truncated program image");
    return out;
  }
  std::string string() {
    const auto n = value<std::uint32_t>();
    if (n > (1u << 20)) fail("oversized string");
    std::string out(n, '\0');
    in_.read(out.data(), n);
    if (!in_) fail("truncated program image");
    return out;
  }
  std::vector<std::uint8_t> bytes(std::uint64_t n) {
    if (n > (1ull << 34)) fail("oversized payload");
    std::vector<std::uint8_t> out(static_cast<std::size_t>(n));
    in_.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(n));
    if (!in_) fail("truncated program image");
    return out;
  }
  static void fail(const std::string& message) { throw std::runtime_error("VM3 loader: " + message); }
 private:
  std::ifstream in_;
};

constexpr std::array<std::uint32_t, 64> kShaK{
  0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
  0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
  0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
  0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
  0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
  0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
  0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
  0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
std::uint32_t rotr(std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
std::array<std::uint8_t,32> sha256(std::vector<std::uint8_t> data) {
  const auto bits = static_cast<std::uint64_t>(data.size()) * 8;
  data.push_back(0x80);
  while ((data.size() % 64) != 56) data.push_back(0);
  for (int i=7;i>=0;--i) data.push_back(static_cast<std::uint8_t>(bits >> (i*8)));
  std::array<std::uint32_t,8> h{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
  for (std::size_t off=0; off<data.size(); off+=64) {
    std::uint32_t w[64]{};
    for(int i=0;i<16;++i) w[i]=(data[off+4*i]<<24)|(data[off+4*i+1]<<16)|(data[off+4*i+2]<<8)|data[off+4*i+3];
    for(int i=16;i<64;++i){auto s0=rotr(w[i-15],7)^rotr(w[i-15],18)^(w[i-15]>>3);auto s1=rotr(w[i-2],17)^rotr(w[i-2],19)^(w[i-2]>>10);w[i]=w[i-16]+s0+w[i-7]+s1;}
    auto a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],z=h[7];
    for(int i=0;i<64;++i){auto s1=rotr(e,6)^rotr(e,11)^rotr(e,25);auto ch=(e&f)^((~e)&g);auto t1=z+s1+ch+kShaK[i]+w[i];auto s0=rotr(a,2)^rotr(a,13)^rotr(a,22);auto maj=(a&b)^(a&c)^(b&c);auto t2=s0+maj;z=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;}
    h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=z;
  }
  std::array<std::uint8_t,32> out{}; for(int i=0;i<8;++i) for(int j=0;j<4;++j) out[4*i+j]=h[i]>>(24-8*j); return out;
}
std::string hex(const std::array<std::uint8_t,32>& v){std::ostringstream s;s<<std::hex<<std::setfill('0');for(auto x:v)s<<std::setw(2)<<int(x);return s.str();}

struct Bounds { double dl, du, el, eu; };
bool close(double a,double b){return std::abs(a-b)<=16*std::numeric_limits<double>::epsilon()*std::max({1.0,std::abs(a),std::abs(b)});}
double lower(long double value){double out=static_cast<double>(value);return static_cast<long double>(out)>value?std::nextafter(out,-std::numeric_limits<double>::infinity()):out;}
double upper(long double value){double out=static_cast<double>(value);return static_cast<long double>(out)<value?std::nextafter(out,std::numeric_limits<double>::infinity()):out;}

}  // namespace

FrozenChessProgram load_frozen_program(const std::filesystem::path& path, const torch::Device& device) {
  Reader r(path); FrozenChessProgram p;
  const std::array<char,8> magic{'C','M','Z','V','M','3','\0','\0'};
  for(char expected:magic) if(r.value<char>()!=expected) Reader::fail("bad magic");
  p.schema_version=r.value<std::uint32_t>(); if(p.schema_version!=1) Reader::fail("unsupported schema");
  const auto nt=r.value<std::uint32_t>(), nn=r.value<std::uint32_t>(), nc=r.value<std::uint32_t>();
  const auto npre=r.value<std::uint32_t>(), npost=r.value<std::uint32_t>();
  if(nt>100000||nn>100000||nc>100000||npre>10000||npost>10000) Reader::fail("count limit");
  for(std::uint32_t i=0;i<nt;++i){
    ProgramTensorManifest m; m.name=r.string(); if(m.name.empty()||p.tensors.count(m.name)) Reader::fail("duplicate tensor name");
    const auto dtype=r.value<std::uint32_t>(); if(dtype!=1) Reader::fail("unsupported dtype"); m.dtype=torch::kFloat64;
    const auto rank=r.value<std::uint32_t>(); if(rank>8) Reader::fail("tensor rank"); std::int64_t numel=1;
    for(std::uint32_t j=0;j<rank;++j){auto d=r.value<std::int64_t>();if(d<0||d>100000000)Reader::fail("tensor shape");m.shape.push_back(d);numel*=d;}
    const auto bytes=r.value<std::uint64_t>(); std::array<std::uint8_t,32> wanted{}; for(auto& x:wanted)x=r.value<std::uint8_t>(); auto payload=r.bytes(bytes);
    if(bytes!=static_cast<std::uint64_t>(numel)*sizeof(double)) Reader::fail("tensor payload size"); if(sha256(payload)!=wanted) Reader::fail("SHA-256 mismatch");
    m.sha256=wanted; auto options=torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
    auto tensor=torch::from_blob(payload.data(), m.shape, options).clone(); tensor.set_requires_grad(false);
    p.manifest.push_back(m); p.tensors.emplace(m.name,std::move(tensor));
  }
  for(std::uint32_t i=0;i<nn;++i){IntervalProofNode n; auto op=r.value<std::uint32_t>();if(op>6)Reader::fail("interval opcode");n.op=static_cast<IntervalOp>(op);auto ni=r.value<std::uint32_t>();if(ni>nn)Reader::fail("interval inputs");for(std::uint32_t j=0;j<ni;++j)n.inputs.push_back(r.value<std::uint32_t>());n.tensor_name=r.string();n.claimed_lower=r.value<double>();n.claimed_upper=r.value<double>();if(!std::isfinite(n.claimed_lower)||!std::isfinite(n.claimed_upper)||n.claimed_lower>n.claimed_upper)Reader::fail("interval proof claim");p.interval_proof.push_back(std::move(n));}
  std::vector<int> color(nn); std::function<void(std::uint32_t)> visit=[&](std::uint32_t i){if(i>=nn)Reader::fail("interval proof input");if(color[i]==1)Reader::fail("interval proof cycle");if(color[i]==2)return;color[i]=1;for(auto j:p.interval_proof[i].inputs)visit(j);color[i]=2;};for(std::uint32_t i=0;i<nn;++i)visit(i);
  std::vector<Bounds> bounds; bounds.reserve(nn);
  for(std::uint32_t i=0;i<nn;++i){const auto& n=p.interval_proof[i];Bounds b{};
    if(n.op==IntervalOp::BoundedPolicyProbability){if(!n.inputs.empty())Reader::fail("bounded probability inputs");b={0,1,0,1};}
    else if(n.op==IntervalOp::CanonicalInput){if(n.tensor_name=="unit_interval")b={0,1,0,1};else if(n.tensor_name=="signed_unit")b={-1,1,-1,1};else Reader::fail("canonical input domain");}
    else if(n.op==IntervalOp::EligibilityMargin){if(n.inputs.size()!=1||!p.tensors.count(n.tensor_name))Reader::fail("eligibility margin proof");auto base=bounds.at(n.inputs[0]);auto t=p.tensors.at(n.tensor_name);if(t.numel()!=1)Reader::fail("margin tensor");double m=t.item<double>();if(!std::isfinite(m))Reader::fail("margin tensor");b={base.dl,base.du,lower(static_cast<long double>(base.el)+m),upper(static_cast<long double>(base.eu)+m)};}
    else if(n.op==IntervalOp::Add){if(n.inputs.empty())Reader::fail("add proof");b={0,0,0,0};for(auto j:n.inputs){auto x=bounds.at(j);b={lower(static_cast<long double>(b.dl)+x.dl),upper(static_cast<long double>(b.du)+x.du),lower(static_cast<long double>(b.el)+x.el),upper(static_cast<long double>(b.eu)+x.eu)};}}
    else if(n.op==IntervalOp::Multiply){if(n.inputs.size()!=2)Reader::fail("multiply proof");auto x=bounds.at(n.inputs[0]),y=bounds.at(n.inputs[1]);auto mul=[](double a,double b,double c,double d){std::array<long double,4> q{static_cast<long double>(a)*c,static_cast<long double>(a)*d,static_cast<long double>(b)*c,static_cast<long double>(b)*d};return std::pair<double,double>{lower(*std::min_element(q.begin(),q.end())),upper(*std::max_element(q.begin(),q.end()))};};auto d=mul(x.dl,x.du,y.dl,y.du),e=mul(x.el,x.eu,y.el,y.eu);b={d.first,d.second,e.first,e.second};}
    else if(n.op==IntervalOp::ReduceMax){if(n.inputs.empty())Reader::fail("reduce max proof");b=bounds.at(n.inputs[0]);for(std::size_t j=1;j<n.inputs.size();++j){auto x=bounds.at(n.inputs[j]);b={std::max(b.dl,x.dl),std::max(b.du,x.du),std::max(b.el,x.el),std::max(b.eu,x.eu)};}}
    else if(n.op==IntervalOp::FrozenMatmul){if(n.inputs.size()!=1||!p.tensors.count(n.tensor_name))Reader::fail("frozen matmul proof");auto w=p.tensors.at(n.tensor_name);if(w.dim()!=2)Reader::fail("frozen matmul tensor");auto x=bounds.at(n.inputs[0]);auto domain=[&](double lo,double hi){double out_lo=std::numeric_limits<double>::infinity(),out_hi=-std::numeric_limits<double>::infinity();auto a=w.accessor<double,2>();for(std::int64_t row=0;row<w.size(0);++row){long double l=0,u=0;for(std::int64_t col=0;col<w.size(1);++col){double c=a[row][col];l+=static_cast<long double>(c)*(c>=0?lo:hi);u+=static_cast<long double>(c)*(c>=0?hi:lo);}out_lo=std::min(out_lo,lower(l));out_hi=std::max(out_hi,upper(u));}return std::pair<double,double>{out_lo,out_hi};};auto d=domain(x.dl,x.du),e=domain(x.el,x.eu);b={d.first,d.second,e.first,e.second};}
    else Reader::fail("unsupported interval proof");
    double lo=std::min(b.dl,b.el),hi=std::max(b.du,b.eu);if(!close(lo,n.claimed_lower)||!close(hi,n.claimed_upper))Reader::fail("interval proof claim mismatch");bounds.push_back(b);
  }
  for(std::uint32_t i=0;i<nc;++i){SelectorScoreCertificate c;c.selector_name=r.string();c.proof_root=r.value<std::uint32_t>();c.enabled_score_lower=r.value<double>();c.disabled_score_upper=r.value<double>();c.sentinel_row=r.value<std::int64_t>();if(c.enabled_score_lower<=c.disabled_score_upper)Reader::fail("enabled score must dominate disabled score");if(c.proof_root>=bounds.size())Reader::fail("selector certificate root");auto b=bounds[c.proof_root];if(!close(c.enabled_score_lower,b.el)||!close(c.disabled_score_upper,b.du))Reader::fail("selector certificate does not match proof");if(c.sentinel_row<0)Reader::fail("sentinel row invalid");p.selector_certificates.push_back(std::move(c));}
  if(!device.is_cpu())for(auto& [name,t]:p.tensors){t=t.to(device);t.set_requires_grad(false);}
  auto tensor_ref=[&](){auto name=r.string();auto it=p.tensors.find(name);if(it==p.tensors.end())Reader::fail("stage tensor reference");return it->second;};
  auto read_block=[&](){Attention2dBlock b;b.query_router=tensor_ref();b.key_router=tensor_ref();b.output_router=tensor_ref();b.fixed_mask=tensor_ref();b.fixed_eligibility=tensor_ref();b.global_key_ids=tensor_ref();return b;};
  auto read_stage=[&](){FrozenStage s;s.wq=tensor_ref();s.wk=tensor_ref();s.wv=tensor_ref();s.fixed_mask=tensor_ref();s.row_router=tensor_ref();s.feature_router=tensor_ref();auto count=r.value<std::uint32_t>();if(count>100000)Reader::fail("attention block count");for(std::uint32_t i=0;i<count;++i)s.attention_plan.blocks.push_back(read_block());auto offsets=r.value<std::uint32_t>();if(offsets<2||offsets>count+1)Reader::fail("query group offsets");for(std::uint32_t i=0;i<offsets;++i)s.attention_plan.query_group_offsets.push_back(r.value<std::int64_t>());if(s.attention_plan.query_group_offsets.front()!=0||s.attention_plan.query_group_offsets.back()!=static_cast<std::int64_t>(count))Reader::fail("query group offset coverage");for(std::size_t i=1;i<s.attention_plan.query_group_offsets.size();++i)if(s.attention_plan.query_group_offsets[i]<=s.attention_plan.query_group_offsets[i-1])Reader::fail("query group offset order");return s;};
  for(std::uint32_t i=0;i<npre;++i)p.pre_policy_stages.push_back(read_stage());
  for(std::uint32_t i=0;i<npost;++i)p.post_policy_stages.push_back(read_stage());
  return p;
}

}  // namespace cmz::vm3
