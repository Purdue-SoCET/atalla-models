# Finding: instruction / data layout mismatch in the flat-RAM conversion path

**Status:** verified from code on this checkout (functional_sim `f159f0d`, compiler `351b5fab`), 2026-11-04.
**Scope:** the `.in` test-file format that the functional simulator consumes, and
`functional_sim/merged_in_to_ram_hex.py`, the converter that flattens a `.in` into a single
hex image for an RTL / dummy-RAM preload.
**Owner questions** are collected at the end; nothing below that is marked *question* is asserted as fact.

## 1. What the `.in` format is (verified)

`functional_sim/src/misc/memory.py:1-68` — class `Memory`:

* Two **separate** dictionaries: `instr_mem` (160-bit words) and `data_mem` (32-bit words)
  (`memory.py:20-21`). This is a **Harvard** model: an instruction address and a data address
  can be numerically equal and never collide.
* Lines before `.data` go to `instr_mem`, lines after go to `data_mem` (`memory.py:40-42, 61-68`).
* Each line is `<addr>: <hex>`; spaces and underscores in the hex are removed before parsing
  (`memory.py:51`), so `00000000: 00000400AD 0000000031 0000000031 0000000031` becomes **one** 160-bit
  integer whose most-significant 40 bits are the first text word.
* Instruction words are rejected above 160 bits (`memory.py:62`), data words above 32 bits (`memory.py:66`).

`functional_sim/build.py`:

* `INSTR_BYTE_WIDTH = 5`, `REAL_PACKET_SIZE = 4`, `INSTR_ADDR_STRIDE = 20` (`build.py:27-31`): one packet is
  4 x 40-bit = **160 bits = 20 bytes**, and consecutive packets are 20 bytes apart.
* `encode_instruction` returns a 10-hex-char (40-bit) string (`build.py:64, 250`).
* `emit_test_format` (`build.py:1171-1197`) writes packets starting at address `0`, pads short packets with
  `nop.s` to 4 slots, and advances `addr += INSTR_ADDR_STRIDE`.
* `DRAMWriter.render_data_mem` (`build.py:1338-1341`) emits `addr: u32` lines from `to_u32_words(stride=2)`:
  by default a 32-bit word is emitted at **every even byte address**, so consecutive lines overlap by two bytes
  (`_word_addrs`, "Emit overlapping 32-bit words at even-byte boundaries"). Words are little-endian
  (`DRAMWriter.endian = "little"`, `build.py:1229`).
* `render_testfile` (`build.py:1343-1353`) = instruction lines, blank, `.data`, blank, data lines.

Simulator fetch (`functional_sim/src/functional_sim.py`, `src/components/decode.py`):

* `run.py:43` starts at `pc = 0x00000000`; `run()` uses `pc_increment = packet_length * 5` = 20 bytes
  (`functional_sim.py:218`) and fetches with `mem.read_instr(pc)` (`functional_sim.py:271`).
* `decode_packet` (`decode.py:177-185`) extracts slot `i` from bits `[(3-i)*40 +: 40]`: **slot 0 is the
  most-significant 40 bits**, i.e. the first text word on the line.
* Branch immediates count 40-bit instructions, not bytes or packets (`build.py:417-418`, `pc_relative_words`).

Graph pipeline (atalla-graph):

* `atalla-graph/kernels/common.py:5`: `ADDR_TABLE = 60` (= `0x3C`). Every emitted kernel writes its config
  words at `0x3C, 0x40, 0x44, ...` (`codegen/c_emitter.py:207-217, 444-445, 523-530, 581-586`) and places
  tensors from `0x1000` upward (`c_emitter.py:332, 394, 440, 518, 572`).
* `run_graph.py:_run_emulator` (`run_graph.py:102-131`) does **not** go through `run.py`; it builds `Memory(in_file)`
  in-process, sets the stack pointer `x2 = ((max_data_addr + 0x1000) & ~0xFFF) + 0x1000` and
  `x33 = x2 + 0x1000` (`run_graph.py:113-117`), then calls `run(..., pc=0, packet_length=4, ...)`.
  **The stack pointer is therefore not part of the `.in` file**; `run.py` as a CLI leaves `x2 = 0`.

## 2. What the flat-RAM converter does (verified)

`functional_sim/merged_in_to_ram_hex.py`:

* Docstring (`:2-7`): "Instruction packet words are stored at their byte addresses (160 bits = 20 bytes,
  big-endian). Data words are 32-bit little-endian ... instructions typically begin at 0x00000000; DRAM
  follows cfg addresses."
* `merged_in_to_flat_bytes` (`:23-51`) allocates one `bytearray` spanning both sections, writes each
  instruction packet **big-endian** (byte `a+0` = MSB of slot 0, `:32-36`), then writes each data word
  **little-endian** at its address (`:37-43`). Data is written **after** instructions, so wherever the two
  address ranges overlap, **data silently overwrites instruction bytes**.
* Output is rows of `--bytes-per-line` (default 32) bytes, lowercase hex, no addresses (`:72-79`).

## 3. The mismatch, measured on a real pipeline `.in`

Command (this session):

```bash
cd atalla-graph
/Users/robert.zhang/.venvs/atalla/bin/python run_graph.py --model basic --mode validate --out-dir out/docs_probe --quiet
# probe: load out/docs_probe/add.in with Memory() and merged_in_to_flat_bytes()
```

Result for `out/docs_probe/add.in` (the `add` kernel of BasicModule):

| Quantity | Value |
|---|---|
| Instruction packets | 99, addresses `0x0 .. 0x7A8`, byte span `0 .. 0x7BB` |
| First data word | `0x3C` (ADDR_TABLE) |
| Data words inside the instruction byte span | **960** of 6146 (`0x3C, 0x3E, 0x40, ...`) |
| Packet 0 intact after flattening | yes |
| Packet at `0x3C` (`sw.s ...`) intact after flattening | **no** (overwritten by config words) |

So on the simulator's Harvard `Memory` the image runs correctly (cos = 1.000000 for the full model,
`out/docs_probe/m.json`), but the flat image produced by `merged_in_to_ram_hex.py` is **not** a faithful copy:
every kernel the graph emits is longer than 3 packets (60 bytes), and every kernel's config table starts
at byte 60, so the converter clobbers packets 3..N with the ADDR_TABLE and tensor bytes. The converter's own
docstring assumption ("DRAM follows cfg addresses", i.e. no overlap) does not hold for graph output.

Secondary issues found in the same path:

1. **Overlapping data lines.** `render_data_mem(stride=2)` emits a 32-bit word at every even byte address.
   `Memory` tolerates this (later lines overwrite the overlapping half consistently, and `read_bf16_le` /
   `write_bf16_le` at `memory.py:93-122` reassemble halves from the 4-byte-aligned word). A loader that
   treats each `.data` line as an independent 32-bit word at a 4-byte slot (e.g. an RTL `$readmemh`-style
   preload keyed by word index) would see `0x3E` as a separate word and double-count. The flat converter
   handles it correctly because it writes bytes.
2. **Endianness differs by section.** Instructions are written big-endian per 20-byte packet, data
   little-endian per 4-byte word (`merged_in_to_ram_hex.py:32-43`). Both choices are internally consistent
   with `decode_packet` and `DRAMWriter`, but they are two conventions in one flat image; whether RTL fetches
   a packet as "byte 0 = slot 0 MSB" is not established by anything in this checkout.
3. **Packet width.** Everything in this checkout is 4 x 40 = **160 bits** (`build.py:27-31`,
   `memory.py:62`, `decode.py`). A `PACKET_W = 192` figure was raised for the RTL (Srivalli, Discord,
   2025-10-25); **no file in functional_sim or aihw-ppci-compiler mentions 192** (`grep -rn 192 ...` over
   `*.py *.md *.tex` returns only unrelated hits). If RTL packets are 192 bits (e.g. 4 x 48, or 160 + 32 bits
   of packet header/padding), the 20-byte stride, PC increment, and branch-offset units in the `.in` all
   change, and the flat converter cannot be used as-is.
4. **Start PC / stack not encoded.** The `.in` has no header: PC is assumed 0 by both `run.py:43` and
   `run_graph.py:_run_emulator`; the stack pointer is set by Python before the run and differs between
   the two entry points (0 in `run.py`, above the data image in `run_graph.py`). An RTL preload needs
   both from somewhere else.
5. **`.data` stripping in the graph path.** `codegen/c_emitter.py:compile_and_assemble` runs
   `build_compiler.compile_asm` and then strips the `.data` section it returns, and `render_in_file`
   re-attaches the graph's own `DRAMWriter` data (`c_emitter.py` lines around the `compile_and_assemble`
   and `render_in_file` definitions; `pipeline_reference.md` 6.2). This is fine for the simulator but means a
   `.in` from the graph never contains anything `build_compiler` might have placed in `.data`.

## 4. Can a `.in` instruction section be dumped straight into a `.hex`?

Only if the consumer is also Harvard (separate instruction ROM) **or** instructions and data are placed in
disjoint address ranges. With the current conventions (code at 0, config at 0x3C, tensors at 0x1000+) the
instruction section of every real kernel overlaps the data section, so a single flat image requires one of:

* relocating the config table and tensors above the code (`ADDR_TABLE` is one constant in
  `atalla-graph/kernels/common.py` and `functional_sim/build.py`, but the `0x1000` tensor bases are
  hard-coded per emitter in `c_emitter.py`, and the compiler's `atalla_layout.mmap` would need to agree), or
* a code-offset for the instruction ROM (e.g. load instructions at a separate base and start PC there), or
* keeping two images (instruction hex + data hex) and letting the RTL testbench load them into separate memories.

Which of these matches the RTL is a question for the simulator/RTL owners, not something this repo can decide.

## 5. Precise questions for the simulator / RTL owners

1. **Memory model.** Does the RTL have a single unified RAM for instructions and data, or separate
   instruction and data memories? If unified, what address range is reserved for code, and where must the
   ADDR_TABLE / tensor region start? (Today: code at 0, cfg at 0x3C, tensors at 0x1000 -> overlap.)
2. **Packet width.** Is a fetched packet 160 bits (4 x 40, as `build.py` / `memory.py` / `decode.py` assume)
   or 192 bits? If 192: what occupies the extra 32 bits (header, alignment, 48-bit slots?), and what is
   the PC increment per packet and the unit of branch immediates?
3. **Byte order within a packet.** When the RTL reads 20 (or 24) bytes at packet address A, is slot 0
   `bytes[A..A+4]` MSB-first (what `merged_in_to_ram_hex.py` writes) or is it the low-order 40 bits of a
   little-endian word (what a little-endian RAM dump of `Memory.instr_mem` would give)?
4. **Data word order.** Are 32-bit data words little-endian with BF16 halves at `+0` (low) / `+2` (high),
   as `DRAMWriter` and `Memory.read_bf16_le` implement?
5. **Start PC and stack.** Is reset PC 0? Who initialises `x2` (stack) and `x33` (vector spill base) in
   RTL, since neither is encoded in the `.in`?
6. **Hex file format.** `$readmemh`-style: one word per line, width = ? (bytes per line), addressed from
   0 with no gaps (what `bytes_to_hex_file` writes), or `@addr` directives permitted?
7. **Overlapping `.data` lines.** Is a loader expected to accept the stride-2 overlapping word lines that
   `render_data_mem` produces, or should the graph emit `stride=4` (4-byte-aligned, non-overlapping) lines
   for RTL hand-off? (`render_data_mem(stride=4)` is a one-argument change.)

## 6. Suggested next step (not done)

Add a `--code-base` / `--data-base` pair to `merged_in_to_ram_hex.py` that errors instead of silently
overwriting when ranges overlap, and a `--split` mode that emits `instr.hex` and `data.hex` separately.
Not implemented here because the file is in the functional_sim submodule, which this branch pins and does
not modify; the owners' answers to questions 1-3 decide which of the layouts in section 4 is right.
