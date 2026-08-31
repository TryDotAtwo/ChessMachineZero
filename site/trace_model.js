(function traceModelModule() {
  const PIECES = {
    95: "empty", 96: "white pawn", 97: "white knight", 98: "white bishop", 99: "white rook", 100: "white queen", 101: "white king",
    102: "black pawn", 103: "black knight", 104: "black bishop", 105: "black rook", 106: "black queen", 107: "black king",
  };
  const expectedFixture = [[52, 54, 96], [47, 45, 102], [54, 45, 96]];

  // Published export schema, not a chess interpreter. Topology is fixed separately
  // from the generic arithmetic checks below. A new export schema must update it.
  const FROZEN_SHAPES = [[128,2],[64,2],[64,128],[64,128],[400,128],[1,128],
    [128,5],[128,5],[128,5],[400,5],[5,128],[5,128],[5,128],[5,128],
    [128,29],[128,29],[128,29],[128,29],[128,29],[128,29],[400,29],[29,128],[29,128],[2064,2]];
  const FROZEN_NAMES = ["latest_square_key_projection", "latest_square_queries", "initial_square_tokens",
    "initial_piece_state", "empty_source_events", "padding_row", "castle_match_slot_0", "castle_match_slot_1",
    "castle_match_slot_2", "castle_no_match_bias", "castle_rook_source_targets", "castle_rook_source_values",
    "castle_rook_destination_targets", "castle_rook_destination_values", "en_passant_match_slot_0",
    "en_passant_match_slot_1", "en_passant_match_slot_2", "en_passant_match_slot_3", "en_passant_match_slot_4",
    "en_passant_match_slot_5", "en_passant_no_match_bias", "en_passant_clear_targets", "en_passant_clear_values",
    "latest_event_time_bias"];
  const OP_PLAN = [
    ["ROW_ROUTE",[0],null,[3,1203,3]], ["ROW_ROUTE",[0],null,[4,1204,3]], ["ROW_ROUTE",[0],null,[5,1205,3]],
    ["FROZEN_EXPAND",[0],2], ["TOKEN_PROJECT",[1],6], ["TOKEN_PROJECT",[2],7], ["TOKEN_PROJECT",[3],8],
    ["RESIDUAL_ADD",[5,6]], ["RESIDUAL_ADD",[8,7]], ["POSITION_ADD",[9],9], ["HARDMAX_STE",[10]],
    ["TOKEN_PROJECT",[11],10], ["TOKEN_PROJECT",[11],11], ["TOKEN_PROJECT",[11],12], ["TOKEN_PROJECT",[11],13],
    ["ROW_ROUTE",[0],null,[3,1200,3]], ["ROW_ROUTE",[0],null,[4,1201,3]], ["ROW_ROUTE",[0],null,[5,1202,3]],
    ["FROZEN_EXPAND",[0],5], ["ROW_CONCAT",[19,16]], ["ROW_CONCAT",[19,17]], ["ROW_CONCAT",[19,18]],
    ["TOKEN_PROJECT",[1],14], ["TOKEN_PROJECT",[2],15], ["TOKEN_PROJECT",[3],16],
    ["TOKEN_PROJECT",[20],17], ["TOKEN_PROJECT",[21],18], ["TOKEN_PROJECT",[22],19],
    ["RESIDUAL_ADD",[23,24]], ["RESIDUAL_ADD",[29,25]], ["RESIDUAL_ADD",[30,26]],
    ["RESIDUAL_ADD",[31,27]], ["RESIDUAL_ADD",[32,28]], ["POSITION_ADD",[33],20], ["HARDMAX_STE",[34]],
    ["TOKEN_PROJECT",[35],21], ["TOKEN_PROJECT",[35],22], ["ROW_CONCAT",[4,1,2,12,14,36]],
    ["TOKEN_PROJECT",[38],0], ["POSITION_ADD",[39],23], ["FROZEN_EXPAND",[0],1],
    ["FROZEN_EXPAND",[0],3], ["FROZEN_EXPAND",[0],4], ["ROW_CONCAT",[42,43,3,13,15,37]],
    ["HULL_ATTN_2D",[41,40,44]],
  ];

  function requireCondition(condition, message) { if (!condition) throw new Error(message); }
  function object(value) { return value !== null && typeof value === "object" && !Array.isArray(value); }
  function same(actual, expected) {
    if (Array.isArray(expected)) return Array.isArray(actual) && actual.length === expected.length && expected.every((v,i) => same(actual[i],v));
    if (object(expected)) return object(actual) && same(Object.keys(actual).sort(), Object.keys(expected).sort()) && Object.keys(expected).every(k => same(actual[k],expected[k]));
    return actual === expected;
  }
  function requireKeys(value, keys, label) {
    requireCondition(object(value) && same(Object.keys(value).sort(), keys.slice().sort()), label + ": missing or unexpected fields");
  }
  function text(value, label) {
    requireCondition(typeof value === "string" && value.trim().length > 0, label + ": missing text");
  }

  function attributes(opcode, frozen, route) {
    if (opcode === "ROW_ROUTE") return {start:route[0], end:route[1], stride:route[2], axis:"row"};
    if (opcode === "TOKEN_PROJECT") return {frozen, axis:"last_dimension"};
    if (opcode === "POSITION_ADD") return {frozen, axis:"elementwise"};
    if (opcode === "FROZEN_EXPAND") return {frozen, axis:"batch"};
    if (opcode === "ROW_CONCAT") return {axis:"row"};
    if (opcode === "HARDMAX_STE") return {temperature:0.5, axis:"last_dimension", forward:"lowest-index argmax tie"};
    if (opcode === "HULL_ATTN_2D") return {top_k:8, temperature:0.5, candidate_count:2064,
      candidate_first:0, candidate_last:2063, forward:"hard argmax value read", backward:"selected-top-k softmax surrogate for Q, K and V"};
    return {axis:"elementwise"};
  }
  function equation(op) {
    const lhs = op.output + " = "; const a = op.attributes_info; const x = op.inputs;
    switch (op.opcode) {
      case "ROW_ROUTE": return lhs + x[0] + "[:, " + a.start + ":" + a.end + ":" + a.stride + ", :]";
      case "TOKEN_PROJECT": return lhs + x[0] + " @ " + a.frozen;
      case "POSITION_ADD": return lhs + x[0] + " + " + a.frozen;
      case "RESIDUAL_ADD": return lhs + x[0] + " + " + x[1];
      case "FROZEN_EXPAND": return lhs + "expand_batch(" + a.frozen + ")";
      case "ROW_CONCAT": return lhs + "concat_rows(" + x.join(", ") + ")";
      case "HARDMAX_STE": return lhs + "one_hot(argmax(" + x[0] + "))";
      case "HULL_ATTN_2D": return lhs + "hardmax(" + x[0] + " @ transpose(" + x[1] + ")) @ " + x[2];
      default: throw new Error("unknown operation");
    }
  }
  function validateTopology(trace) {
    requireCondition(Array.isArray(trace.operations) && trace.operations.length === OP_PLAN.length, "operation coverage mismatch");
    const shapes = {v0:[1,2048,128]};
    OP_PLAN.forEach(([opcode, inputs, weight, route], i) => {
      const op = trace.operations[i]; const frozen = Number.isInteger(weight) ? "w" + weight : null;
      requireKeys(op, ["index","opcode","equation","inputs","frozen","derived","output","attributes_info","sample"], "operation " + (i+1));
      const expected = {index:i+1, opcode, inputs:inputs.map(v => "v" + v), frozen:frozen ? [frozen] : [],
        derived:opcode === "HULL_ATTN_2D" ? ["op45_scores","op45_attention"] : [], output:"v" + (i+1),
        attributes_info:attributes(opcode,frozen,route)};
      Object.keys(expected).forEach(k => requireCondition(same(op[k],expected[k]), "operation " + (i+1) + ": invalid " + k));
      requireCondition(op.equation === equation(op), "operation equation mismatch");
      let shape = shapes[op.inputs[0]].slice();
      if (opcode === "ROW_ROUTE") shape[1] = Math.ceil((route[1]-route[0])/route[2]);
      if (opcode === "FROZEN_EXPAND") shape = [1,...FROZEN_SHAPES[weight]];
      if (opcode === "TOKEN_PROJECT") shape[2] = FROZEN_SHAPES[weight][1];
      if (opcode === "ROW_CONCAT") shape[1] = op.inputs.reduce((sum,id) => sum + shapes[id][1],0);
      if (opcode === "HULL_ATTN_2D") shape = [1,shape[1],shapes[op.inputs[2]][2]];
      shapes[op.output] = shape;
    });
    return shapes;
  }

  function readMatrix(matrix, shape, label, frozenName) {
    requireKeys(matrix, frozenName ? ["name","shape","layout","nnz","entries"] : ["shape","layout","nnz","entries"], label);
    requireCondition(same(matrix.shape,shape), label + ": shape mismatch");
    if (frozenName) requireCondition(matrix.name === frozenName, label + ": frozen name mismatch");
    const size = shape.reduce((n,d) => n*d,1);
    requireCondition(matrix.layout === "COO" && Array.isArray(matrix.entries) && Number.isInteger(matrix.nnz) &&
      matrix.nnz === matrix.entries.length && matrix.nnz <= size, label + ": COO metadata mismatch");
    const data = new Float32Array(size); const occupied = new Uint8Array(size);
    for (const entry of matrix.entries) {
      requireCondition(Array.isArray(entry) && entry.length === shape.length+1, label + ": COO rank mismatch");
      let offset = 0;
      for (let axis = 0; axis < shape.length; axis += 1) {
        requireCondition(Number.isInteger(entry[axis]) && entry[axis] >= 0 && entry[axis] < shape[axis], label + ": COO coordinate out of bounds");
        offset = offset*shape[axis]+entry[axis];
      }
      const value = entry[shape.length];
      requireCondition(typeof value === "number" && Number.isFinite(value) && value !== 0 && Math.fround(value) === value, label + ": COO must contain finite nonzero FP32 values");
      requireCondition(occupied[offset] === 0, label + ": duplicate COO coordinate");
      occupied[offset] = 1; data[offset] = value;
    }
    return {shape, data};
  }
  function compareData(actual, expected, label) {
    requireCondition(actual.length === expected.length, label + ": extent mismatch");
    for (let i = 0; i < actual.length; i += 1) requireCondition(actual[i] === expected[i], label + ": arithmetic mismatch at " + i);
  }
  function validateSemantics(trace) {
    const sem = trace.semantics;
    requireKeys(sem,["tensors","values","derived","event_streams","patterns"],"semantics");
    for (const group of ["tensors","values","derived"]) {
      requireKeys(sem[group],Object.keys(trace[group]),"semantics." + group);
      for (const id of Object.keys(trace[group])) {
        const record = sem[group][id];
        requireKeys(record, group === "values" ? ["technical_id","en","ru","producer"] : ["technical_id","en","ru"], "semantic " + id);
        requireCondition(record.technical_id === id,"semantic identifier mismatch");
        for (const lang of ["en","ru"]) {
          requireKeys(record[lang],["name","purpose","rows","columns"],id + "." + lang);
          for (const field of ["name","purpose","rows","columns"]) text(record[lang][field],id + "." + lang + "." + field);
        }
        if (group === "values") {
          const op = trace.operations[Number(id.slice(1))-1];
          const expected = id === "v0" ? {kind:"input"} : {operation:op.index, opcode:op.opcode, inputs:op.inputs};
          requireCondition(same(record.producer,expected),id + ": semantic producer mismatch");
        }
      }
    }
    const boundaries = [0,64,464,864,1264,1664,2064];
    requireCondition(Array.isArray(sem.event_streams) && sem.event_streams.length === 6,"event stream metadata missing");
    sem.event_streams.forEach((stream,i) => {
      requireKeys(stream,["start","end","en","ru"],"event stream");
      requireCondition(stream.start === boundaries[i] && stream.end === boundaries[i+1],"event stream boundary mismatch");
      text(stream.en,"event stream en"); text(stream.ru,"event stream ru");
    });
    requireKeys(sem.patterns,["castle","en_passant"],"patterns");
    for (const [kind,count] of [["castle",5],["en_passant",29]]) {
      requireCondition(Array.isArray(sem.patterns[kind]) && sem.patterns[kind].length === count,"pattern coverage mismatch");
      sem.patterns[kind].forEach((pattern,i) => {
        requireKeys(pattern,["column","en","ru"],"pattern");
        requireCondition(pattern.column === i,"pattern column mismatch"); text(pattern.en,"pattern en"); text(pattern.ru,"pattern ru");
      });
    }
  }

  function validateSample(op, matrices, winners) {
    const sample = op.sample; const output = matrices[op.output];
    requireCondition(object(sample),"sample missing");
    const coords = sample.output_index;
    requireCondition(Array.isArray(coords) && coords.length === output.shape.length &&
      coords.every((x,i) => Number.isInteger(x) && x >= 0 && x < output.shape[i]),"sample coordinate mismatch");
    const row = coords[1], column = coords[2], offset = row*output.shape[2]+column;
    const expected = {output_index:coords, output_value:output.data[offset]};
    const x = matrices[op.inputs[0]], a = op.attributes_info;
    if (op.opcode === "TOKEN_PROJECT") {
      const w = matrices[a.frozen]; const terms = [];
      for (let k = 0; k < x.shape[2]; k += 1) {
        const left = x.data[row*x.shape[2]+k], right = w.data[k*w.shape[1]+column];
        const product = Math.fround(left*right);
        if (product !== 0) terms.push([k,left,right,product]);
      }
      Object.assign(expected,{identity:"sum_k X[...,k] * W[k,j]",terms});
    } else if (op.opcode === "POSITION_ADD" || op.opcode === "RESIDUAL_ADD") {
      const y = matrices[a.frozen || op.inputs[1]];
      Object.assign(expected,{identity:"left + right",terms:[x.data[offset],y.data[offset % y.data.length]]});
    } else if (op.opcode === "HULL_ATTN_2D") {
      const keys = matrices[op.inputs[1]], winner = winners[row];
      Object.assign(expected,{identity:"argmax_j(sum_d Q[row,d] * K[j,d]); output = V[winner,channel]",
        winner, score_terms:[0,1].map(d => [d,x.data[row*2+d],keys.data[winner*2+d]]),
        winner_score:matrices.op45_scores.data[row*keys.shape[1]+winner]});
    }
    requireCondition(same(sample,expected),"operation " + op.index + ": sample provenance mismatch");
  }
  function validateArithmetic(trace, matrices) {
    // Exact comparisons are intentional for this fixed binary/integer FP32
    // fixture: fround after each product/add, with ZERO numerical tolerance.
    // In particular no relative tolerance may hide the 2^-21 chronology gap.
    for (const op of trace.operations) {
      const x = matrices[op.inputs[0]], actual = matrices[op.output], a = op.attributes_info;
      const expected = new Float32Array(actual.data.length); const width = actual.shape[2];
      let winners;
      switch (op.opcode) {
        case "ROW_ROUTE":
          for (let row = 0; row < actual.shape[1]; row += 1) expected.set(x.data.subarray((a.start+row*a.stride)*width,(a.start+row*a.stride+1)*width),row*width);
          break;
        case "FROZEN_EXPAND": expected.set(matrices[a.frozen].data); break;
        case "ROW_CONCAT": {
          let offset = 0;
          op.inputs.forEach(id => { expected.set(matrices[id].data,offset); offset += matrices[id].data.length; });
          break;
        }
        case "TOKEN_PROJECT": {
          const w = matrices[a.frozen], inner = x.shape[2];
          for (let row = 0; row < x.shape[1]; row += 1) for (let k = 0; k < inner; k += 1) {
            const left = x.data[row*inner+k];
            if (left === 0) continue;
            for (let col = 0; col < width; col += 1) expected[row*width+col] = Math.fround(expected[row*width+col] + Math.fround(left*w.data[k*width+col]));
          }
          break;
        }
        case "POSITION_ADD": case "RESIDUAL_ADD": {
          const y = matrices[a.frozen || op.inputs[1]];
          for (let i = 0; i < expected.length; i += 1) expected[i] = Math.fround(x.data[i]+y.data[i % y.data.length]);
          break;
        }
        case "HARDMAX_STE":
          for (let row = 0; row < x.shape[1]; row += 1) {
            let winner = 0;
            for (let col = 1; col < width; col += 1) if (x.data[row*width+col] > x.data[row*width+winner]) winner = col;
            expected[row*width+winner] = 1;
          }
          break;
        case "HULL_ATTN_2D": {
          const k = matrices[op.inputs[1]], v = matrices[op.inputs[2]], count = k.shape[1];
          const scores = new Float32Array(x.shape[1]*count), attention = new Float32Array(scores.length);
          winners = [];
          for (let row = 0; row < x.shape[1]; row += 1) {
            let winner = 0;
            for (let col = 0; col < count; col += 1) {
              const score = Math.fround(Math.fround(x.data[row*2]*k.data[col*2]) + Math.fround(x.data[row*2+1]*k.data[col*2+1]));
              requireCondition(Number.isFinite(score),"nonfinite QK product"); scores[row*count+col] = score;
              if (score > scores[row*count+winner]) winner = col;
            }
            attention[row*count+winner] = 1; winners.push(winner);
            expected.set(v.data.subarray(winner*width,(winner+1)*width),row*width);
          }
          compareData(matrices.op45_scores.data,scores,"QK scores");
          compareData(matrices.op45_attention.data,attention,"hard attention");
          break;
        }
        default: throw new Error("unsupported arithmetic opcode");
      }
      compareData(actual.data,expected,op.output);
      validateSample(op,matrices,winners);
    }
  }

  function square(token) {
    if (!Number.isInteger(token) || token < 11 || token > 88 || token % 10 === 0 || token % 10 === 9) return null;
    return `${String.fromCharCode(96 + Math.floor(token / 10))}${token % 10}`;
  }

  function fixtureMove(move) {
    if (!Array.isArray(move) || move.length !== 3) return null;
    const from = square(move[0]); const to = square(move[1]);
    return from && to && PIECES[move[2]] ? {tokens: move.slice(), uci: `${from}${to}`, from, to, piece: PIECES[move[2]]} : null;
  }

  function decodeFixture(trace) {
    if (!trace || !trace.fixture || !Array.isArray(trace.fixture.moves)) return [];
    return trace.fixture.moves.map(fixtureMove).filter(Boolean);
  }

  function matrixValue(matrix, coordinates) {
    if (!matrix || !Array.isArray(matrix.entries)) return 0;
    const key = coordinates.join(",");
    const entry = matrix.entries.find((item) => item.slice(0, -1).join(",") === key);
    return entry ? entry.at(-1) : 0;
  }

  function decodeBoard(trace) {
    const output = trace && trace.values && trace.values.v45;
    const matrix = readMatrix(output,[1,64,128],"v45 board");
    const board = {};
    for (let compact = 0; compact < 64; compact += 1) {
      const squareName = `${String.fromCharCode(97 + Math.floor(compact / 8))}${compact % 8 + 1}`;
      let token = null;
      for (let channel = 0; channel < 128; channel += 1) {
        const value = matrix.data[compact*128+channel];
        if (value === 0) continue;
        requireCondition(value === 1 && token === null && Object.hasOwn(PIECES,channel),"board row is not a single piece/EMPTY token");
        token = channel;
      }
      requireCondition(token !== null,"board row is empty");
      board[squareName] = PIECES[token];
    }
    return board;
  }

  function validateTrace(trace) {
    // No identity cache: callers may mutate a previously accepted object.
    // SHA fields are checked for schema only, not cryptographic authenticity.
    try {
      requireKeys(trace,["artifact","operation_count","runtime_opcodes","final_output","fixture","format","provenance","semantics","tensors","values","derived","operations"],"trace");
      requireCondition(trace.artifact === "position_latest_event_v1" && trace.operation_count === 45,"trace schema mismatch");
      requireCondition(same(trace.runtime_opcodes,["FROZEN_EXPAND","HARDMAX_STE","HULL_ATTN_2D","POSITION_ADD","RESIDUAL_ADD","ROW_CONCAT","ROW_ROUTE","TOKEN_PROJECT"]),"opcode coverage mismatch");
      requireCondition(same(trace.final_output,{value:45,shape:["B",64,128]}),"final output metadata mismatch");
      requireCondition(trace.format === "Exact nonzero entries in zero-based COO; omitted entries are exactly zero.","COO format metadata mismatch");
      requireCondition(same(trace.fixture,{moves:expectedFixture,encoding:"one-hot"}),"trace fixture is inconsistent");
      const provenance = trace.provenance;
      requireKeys(provenance,["executor","dtype","artifact_sha256","source_sha256","source_identifiers"],"provenance");
      requireCondition(provenance.executor === "vm_compiler.reference_executor" && provenance.dtype === "float32","export executor/precision mismatch");
      for (const key of ["artifact_sha256","source_sha256"]) requireCondition(typeof provenance[key] === "string" && /^[0-9a-f]{64}$/.test(provenance[key]),"invalid provenance digest");
      requireCondition(same(provenance.source_identifiers,["vm_compiler.compiler","vm_compiler.state_circuit","vm_compiler.reference_executor","vm_compiler.site_trace","vm_compiler.site_semantics"]),"source identifiers mismatch");
      const shapes = validateTopology(trace), matrices = {};
      requireKeys(trace.tensors,FROZEN_SHAPES.map((_,i) => "w" + i),"frozen tensors");
      requireKeys(trace.values,Object.keys(shapes),"SSA values");
      requireKeys(trace.derived,["op45_scores","op45_attention"],"derived attention");
      FROZEN_SHAPES.forEach((shape,i) => { matrices["w" + i] = readMatrix(trace.tensors["w" + i],shape,"w" + i,FROZEN_NAMES[i]); });
      Object.keys(shapes).forEach(id => { matrices[id] = readMatrix(trace.values[id],shapes[id],id); });
      for (const id of ["op45_scores","op45_attention"]) matrices[id] = readMatrix(trace.derived[id],[1,64,2064],id);
      validateSemantics(trace);
      const source = new Float32Array(2048*128), tokens = expectedFixture.flat();
      for (let row = 0; row < 2048; row += 1) source[row*128+(row >= 3 && row < 12 ? tokens[row-3] : 0)] = 1;
      compareData(matrices.v0.data,source,"fixture input");
      validateArithmetic(trace,matrices);
      decodeBoard(trace);
      return {ok:true};
    } catch (error) {
      return {ok:false,error:error instanceof Error ? error.message : String(error)};
    }
  }

  const api = {matrixValue, decodeFixture, decodeBoard, validateTrace, square};
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (typeof window !== "undefined") window.TraceModel = api;
}());
