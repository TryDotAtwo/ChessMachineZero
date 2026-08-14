#include <cmz_vm3/rule_compiler.h>

#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
  const auto program = cmz::vm3::compile_minimal_rule_program();
  if (argc == 2) cmz::vm3::save_frozen_program(program, argv[1]);
  std::cout << "{\"schema\":" << program.schema_version << ",\"tensors\":[";
  bool first = true;
  for (const auto& entry : program.manifest) {
    if (!first) std::cout << ',';
    first = false;
    std::cout << "{\"name\":\"" << entry.name << "\",\"sha256\":\"";
    for (const auto byte : entry.sha256)
      std::cout << std::hex << std::setfill('0') << std::setw(2)
                << static_cast<int>(byte);
    std::cout << "\"}";
  }
  std::cout << "]}\n";
  return 0;
}
