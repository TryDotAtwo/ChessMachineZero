# Fresh FP4 artifact evidence

- RED Python: artifact tests failed because `vm_compiler.artifact` did not exist.
- RED C++: native test failed because `cmz/artifact.h` did not exist.
- GREEN Python: exact E2M1 nibble packing, binary round-trip, payload/scale shape validation, and unknown-opcode rejection pass.
- GREEN C++: the independent native loader compiled, linked, and parsed the literal Python-produced artifact with exit code `0`.
- Container format uses magic `CMZVM001`, version `1`, SHA-256 body digest, strict tensor metadata, packed FP4 bytes, block scales, and explicit operator records.
- Supported generic opcodes are token projection, position add, 2D Hull attention, residual add, gated FFN, output projection, row route, and hardmax STE.
