+++
title = "Solving the Prospero challenge"
description = "How I wrote a small JIT compiler from scratch, in Rust."
date = 2025-04-25
+++

Performance optimization challenges are extremely compelling to me. You have a well defined problem with a deceptively simple solution, and your job is to use any tricks you can to make it run as quickly as possible. You start from a high level idea of how to architect your solution to be performant, and before long you're looking at assembly code and processor performance counters to squeeze the last few drops of performance you can. It's all very engaging.

A few weeks ago a colleague shared Matt Keeter's [Prospero Challenge](https://www.mattkeeter.com/projects/prospero/) on the office Slack channel, and I immediately knew it would be up my alley.

I've enjoyed hacking away at it even more than I thought I would, so much so that I decided to not just put my solution up on GitHub, but to walk through it step by step in this blog post.

(Hence why this is pretty late compared to when the challenge was posted. Whoops.)

<!-- more -->

## The challenge
You are given a linear stream of instructions for a simple virtual machine
with only a handful of operations and no control flow. All intermediate values are floating-point numbers.

```
# Text of a monologue from The Tempest
_0 const 2.95
_1 var-x
_2 const 8.13008
_3 mul _1 _2
_4 add _0 _3
_5 const 3.675
_6 add _5 _3
_7 neg _6
_8 max _4 _7
[... another 7000 or so more instructions ...]
_1eb5 sub _1eb3 _15
_1eb6 max _1eb4 _1eb5
_1eb7 min _1eaf _1eb6
_1eb8 max _1ea9 _1eb7
_1eb9 min _1ea3 _1eb8
```

This "program" is really just a single function, which accepts two parameters (`var-x` and `var-y`) and whose return value is the last instruction in the stream.
What this function represents is a 2D implicit surface over the `[-1, +1]` square, with the X and Y positions bound to the `var-x` and `var-y` parameters.

By evaluating this surface on a square grid of pixels, then thresholding the result so that only pixels with a value < 0.0 are white, you should get the following image:

![Text image of a passage from Shakespeare's The Tempest, Act 5, Scene 1](prospero.png)

Of course, the higher the resolution you render the image at, the longer it will take.

The goal of the challenge is to optimize the evaluation of this function so it renders as fast as possible.
The only constraint is no pre-computation: the solution should work with any instruction stream, not just the example one given in the challenge.
Other than that, anything goes!

## My approach: writing a compiler
The fact that the same stream of instructions is run for each pixel in the image immediately stuck out to me.

This is the perfect use case for _compiling_ the VM instructions into actual machine code that our processor can run directly.
We'd have to do this at runtime (remember, we can't pre-compute the instruction stream) and this would incur some overhead.
But, that overhead is fixed and occurs only once, whereas the VM instructions can be run up to millions of times at higher resolutions!
It stands to reason then that spending some time making the VM instructions as fast as possible would be well worth it, and beat even the most optimized of interpreters.

To really get the most bang for our buck in terms of performance, we can generate [SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data) instructions that will evaluate multiple pixels at once, cutting our execution time down to a fraction of what it would be with serial instructions.

Now, we could (and a production-ready implementation probably _should_) use a ready-made compiler framework
like [LLVM](https://llvm.org/) or [Cranelift](https://cranelift.dev/) as our compiler back-end.
But for this challenge, I instead opted for a bespoke solution, for a few reasons:
- LLVM generates very optimized code, but is very slow at doing so. [Other challenge submissions](https://github.com/yolhan83/ProsperoChal) that use LLVM have reported compilation taking up to 15 seconds.
  That would completely obliterate any time savings that could possibly be achieved by more efficient code.
- The instruction stream is already in [SSA form](https://en.wikipedia.org/wiki/Static_single-assignment_form),
and instructions are already sorted such that an instruction always comes after its inputs.
  Therefore, there's no need for complex instruction transformations: we can simply do a linear scan over the instruction stream and convert it to machine code directly.
- I just wanted to learn how the sausage is made! Despite working on a JIT compiler at [my day job](https://www.graalvm.org),
  I never actually got to go down in the weeds of instruction encoding and register allocation, and this seemed like the perfect opportunity to learn.

So the plan was set: write a small JIT compiler from scratch.
I picked Rust as my implementation language, because ~I'm a Rust fanboy~ it's very well suited for this kind of performance work.

Knowing full well that a project like this could ruin my life if I let it, I immediately set down some limitations in scope:
- Only support one single processor architecture and OS, specifically my laptop's (x86-64 on Linux, and no AVX-512 instructions).
- No GPU compute. As much as I would have liked to learn more about GPU compilation, I don't have access to an Nvidia GPU for CUDA.
- The use of external libraries should be kept to a minimum, or it would defeat the point of the learning exercise. I thought it fine to use a library for writing the image data to a PNG file, however.

Finally, a couple rules of thumb for the rest of this blog post:
- I'll be assuming you have some knowledge of Rust syntax and know the basics of computer architecture.
- Code examples will usually be simplified compared to their final optimized versions. If you want to see the actual final version of the code, head over to [the GitHub repository](https://github.com/CRefice/prospero.vm).

Okay, enough foreplay. Let's get started!

## Instruction representation
First up: we need a way to represent VM instructions in our compiler.
I've gone back and forth on what exact representation should be used, and this the shape it ended up taking by the end:

```rust
/// A single VM instruction.
enum Instr {
    VarX,
    VarY,
    Const(f32),
    Unary {
        op: UnaryOpcode,
        operand: VarId,
    },
    Binary {
        op: BinaryOpcode,
        lhs: VarId,
        rhs: VarId,
    },
}

/// Index type for accessing an instruction from a Vec of instructions.
struct VarId(u32);

/// Operation types for instructions with a single operand.
enum UnaryOpcode {
    Neg,
    Square,
    Sqrt,
}

/// Operation types for instructions with two operands.
enum BinaryOpcode {
    Add,
    Sub,
    Mul,
    Max,
    Min,
}
```

Nothing too surprising here, though there are a few decisions worth explaining:
- Instructions are parsed into a single flat `Vec`, and variable handles (`VarId`s) are just newtype'd indices into this `Vec`.
  This is the classic Rust trick of representing graphs as flattened arrays instead of separate heap-allocated objects with pointers between them,
  which is much more cache-friendly, and also helps appease the almighty borrow checker.
- The `Instr` enum is not flattened (with one variant per opcode), instead instructions are grouped into unary (e.g. `square x`) and binary (e.g. `add x y`).
  This is much more ergonomic when doing pretty much anything with those instructions other than parsing them, and I'd honestly recommend this pattern for any kind of compiler-like project.
- You might be tempted to reduce the number of unique opcodes for the sake of implementation simplicity.
  For example, `square x` is essentially equivalent to `mul x x`, and `neg x` is equivalent to `sub (const 0) x`.
  However, when it comes time to apply some optimizations to the VM code, you'll need all the information
  you can get to drive optimization decisions. So resist that temptation!

Actually parsing these instructions from the input text file is really not that interesting, so we'll skip over it here. You can check out [the relevant source code](https://github.com/CRefice/prospero.jit/blob/fd05c64402bdccaf3bf4168ac256626fb09226ed/src/lib.rs#L47) if you want to see how that's done.

## Emitting CPU instructions

Now we have the VM instructions in a format we can work with. Can we convert them to actual processor instructions already?
Not so fast! We first need to go on a bit of a tangent about __instruction encoding__.

Every processor architecture has its own set of instructions, and its own special way to encode these instructions into bytes.
Our chosen architecture, x86, is really old, being first introduced in 1978 with Intel's first 16-bit processor.
It has evolved incrementally over the years, with new instructions being essentially bolted on top of the old ones, instead of ever reworking the instruction set from scratch.

As a result, the instruction encoding scheme x86 uses is really quite complicated and unintuitive.
To try and keep things simpler, we'll focus on SIMD instruction encoding only, specifically __AVX__,
the most advanced type of SIMD instructions that my processor supports.
With its 256-bit wide registers, AVX allows us to work with a whopping 8 32-bit floats at once!

An encoded AVX instruction is made up of multiple parts:
- A __`VEX` prefix__, which can be two or three bytes long and serves both as a marker for "this is an AVX instruction",
  and also encodes which registers the instruction will read its operands from. Well, part of this information anyway. More on that in a bit.
- An __opcode__ byte, which denotes which operation this instruction performs (addition, subtraction, etc.)
- A __`ModR/M`__ byte, which contains the rest of the information on input/output registers not encoded by the VEX prefix.
- Further immediate operands such as memory address displacements, constants, etc, depending on the instruction.

You might have noticed some repetition there. Why is operand information split between the `VEX` prefix and the `ModR/M` byte?

Well, up until fairly recently x86 processors featured only 8 general-purpose registers.
This meant one register could be specified with only 3 bits. So you could specify two registers in one byte (say, one for input and one for output)
and still have two bits left over to specify an addressing mode. This is exactly what the `ModR/M` byte does.

Later on, x86_64 came along and extended the number of registers to 16.
But, there's no more space available in the `ModR/M` byte! So the engineers at AMD had two options:
- Completely change the way register operands are encoded (while still supporting the old way for backwards compatibility).
- Finding a couple bits of space somewhere else.

Unsurprisingly, they chose the latter, and used part of the instruction prefix to store those fourth bits.
This approach stuck, and is thus used also for AVX instructions.
I told you it was a messy history!

Let's drill further into the `VEX` prefix, which is probably the most complicated part to understand.
The two-byte version serves only as a shorthand for a three-byte prefix with certain properties, so we'll only look at the three-byte version.
This is what those three bytes look like, bit by bit:
```
00-07: 11000100 (0xC4, fixed)
08-15: RXBMMMMM
16-24: WVVVVLPP
```

What do does letters mean?
- `R` and `B` are those fourth bits of the first and second operand registers we just talked about.
- `X` is yet another fourth bit for specifying a register, though it is only useful with an addressing mode we won't be using, so we can safely ignore it.
- `MMMMM` is an "opcode map", which is really just additional bits to specify an opcode. This extends the number of possible AVX operations that can be encoded by the instruction set from 256 to 8192.
- `VVVV` is an optional third operand register, usually the right-hand side of a binary operation. All four bits are stored together this time, hooray!
- `L` specifies the vector width of this instruction: 0 for 128-bit, 1 for 256-bit. Since we want the widest vector available for performance, we'll always set this to 1.
- `W` and `PP`: these are entirely determined by the instruction in question on a case-by-case basis.

For some unknown reason, the encoding of `R`, `X`, `B` and `VVVV` is flipped compared to the number they represent.
So for example, in order to select register 0 as a third operand, the `VVVV` bits need to be set to 1111. `¯\_(ツ)_/¯`

This is a lot to take in, so let's take the `vaddps` instruction as an example.
This adds the contents of two registers together, and stores the result in a third register.
Let's take a look at the instruction definition from [Felix cloutier's x86 instruction listings](https://www.felixcloutier.com/x86/addps).
In this case, the third row in the table is the 256-bit version of the instruction, which is the one we're interested in.

We'll ignore the operands for now and just focus on the instruction and the prefix:
```
VEX.256.0F.WIG 58 /r VADDPS [...]
```
Let's break down what this means, part by part:
- `VEX`: this instruction requires a `VEX` prefix.
- `256`: this is the 256-bit version, so `L` will be set to 1.
- `0F`: this maps to the `MMMMM` bits, but confusingly, doest not represent the _contents_ of those bits. There's only three values available here:
  `0F` which corresponds to map 1, `0F38` to 2, and `0F3A` corresponds to 3. Other maps are so far unused, though could be used for future instruction set extensions. So in this case, `MMMMM = 00001`.
- `WIG`: this means the W bit will be ignored and can thus be set to 0 or 1. If this was `W0` or `W1`, the W bit would have to be set accordingly.
- `58`: this is the instruction opcode byte (in hexadecimal).
- There's no reference to what the `PP` bits should be set to, so they are set to zero.

Let's do another example to really drive the point home, this time with a broadcast instruction, which takes a single 32-bit floating-point number and copies it to all eight slots in a 256-bit vector register.

```
VEX.256.66.0F38.W0 18 /r VBROADCASTSS [...]
```

- `VEX.256`: same as before.
- `66`: careful here: this isn't a value for the opcode map, but actually specifies the `PP` bits. Once again this is not the value of those bits, but a mapping: `66` for 1, `F3` for 2, and `F2` for 3. No value implies `PP = 0`, like in the previous example.
- `0F38`: as we saw before, this implies an opcode map of 02, so `MMMMM = 00010`
- `W0`: `W` must be set to 0.
- `1A`: the instruction opcode.

It's a little convoluted, but you quickly start seeing the patterns once you get going.
The ultimate reference for how to read this type of instruction encoding-encoding (text encoding for a binary encoding) is [Intel's Software Developer Manual](https://software.intel.com/en-us/download/intel-64-and-ia-32-architectures-sdm-combined-volumes-1-2a-2b-2c-2d-3a-3b-3c-3d-and-4), Volume 2, Chapter 3.1.

The last thing left to discuss is operand encoding with the `ModR/M` byte.

I hand-waved it a bit earlier, but the inputs and outputs to an instruction don't always have to be registers.
They can also be values stored in memory, at addresses which are _specified_ by registers in different ways.
The collective term for ways to specify these addresses is _addressing modes_, and they're encoded in the top 2 bits of the `ModR/M` byte.

Being able to read values from memory will be very useful for us soon, so we will make use of two addressing modes:
- Mode 3, _Register-direct_: the operand value is the contents of the register in question.
- Mode 2, _Register-indirect_: the operand is read from memory, at an address specified by a base register, plus a 32-bit displacement.
  The displacement is encoded directly following the `ModR/M` byte, as a four-byte number in little-endian byte order.
  Essentially, it's like reading the value from an array, where the pointer to that array is stored in a register,
  and the index is constant and is encoded in the instruction itself.

Alright, we now know enough to actually write some code to write encoded instructions.
We'll start by defining an `enum` for instruction operands, since we'll want to generate instructions independently of the addressing mode used.

```rust
#[derive(Debug, Clone, Copy)]
pub enum Operand {
    /// Register-direct: the operand is the value of the register in question
    Register(u8),
    /// Register-indirect: the operand is the value pointed to by the base register + a 32-bit displacement.
    Memory { base: u8, disp: u32 },
}

impl Operand {
    /// Gets the underlying register that this operand uses to refer to its value.
    /// For register-direct mode, this is the operand register itself.
    /// For register-indirect mode, this is the base regsiter.
    fn register(&self) -> u8 {
        match self {
            Operand::Register(x) => *x,
            Operand::Memory{ base, .. } => *base,
        }
    }
}
```

We'll implement the bulk of our encoding logic as member functions of the `CodeBuffer` struct,
which just pushes bytes into a `Vec<u8>`.

```rust
#[derive(Default)]
pub struct CodeBuffer {
    code: Vec<u8>,
    constants: Vec<f32>,
}
```

We'll make helper functions for writing `VEX` prefixes and `ModR/M` bytes from their constituent parts:

```rust
impl CodeBuffer {
    /// Emit the VEX prefix with the given parameters
    fn vex(&mut self, reg: u8, vvvv: u8, r_m: u8, pp: u8, map: u8) {
        self.code.push(0xc4);

        {
            let r_bar = ((!reg) & 0b1000) >> 3;
            let x_bar = 1;
            let b_bar = ((!r_m) & 0b1000) >> 3;
            let mmmmm = map & 0b11111;
            self.code.push((r_bar << 7) | (x_bar << 6) | (b_bar << 5) | mmmmm);
        }

        {
            let w = 0;
            let vvvv_bar = (!vvvv) & 0b1111;
            let l = 1;
            let pp = pp & 0b11;
            self.code.push((w << 7) | (vvvv_bar << 3) | (l << 2) | pp);
        }
    }

    /// Emit the ModR/M byte with the given parameters.
    fn mod_r_m(&mut self, r#mod: u8, reg: u8, r_m: u8) {
        let reg = reg & 0b111;
        let r_m = r_m & 0b111;
        self.append((r#mod << 6) | (reg << 3) | r_m);
    }
}
```

as well as a function to turn our generic operand into the appropriate `ModR/M`, addressing mode and all:

```rust
impl CodeBuffer {
    /// Emit an appropriate ModR/M byte based on the right-hand operand.
    fn operands(&mut self, reg: u8, r_m: Operand) {
        match r_m {
            Operand::Reg(r_m) => self.mod_r_m(0b11, reg, r_m),
            Operand::Memory { base, disp } => {
                self.mod_r_m(0b10, reg, base);
                self.code.extend_from_slice(&disp.to_le_bytes());
            }
        }
    }
}
```

Finally, we'll use these helpers and information from the instruction listings to encode actual instructions.
We don't know yet _which_ operands we will use for those instructions, so they'll be passed in as arguments:
```rust
impl CodeBuffer {
    // Note that most VEX instructions only support register-indirect mode for the right-hand operand.
    // The left-hand side and output must always be a register, so we represent them directly as `u8`.

    fn add(&mut self, dest: u8, x: u8, y: Operand) {
        self.vex(dest, x, y.register(), 0, 1);
        self.code.push(0x58); // opcode
        self.operands(dest, y);
    }

    fn sub(&mut self, dest: u8, x: u8, y: Operand) {
        self.vex(dest, x, y.register(), 0, 1);
        self.code.push(0x5c); // opcode
        self.operands(dest, y);
    }

    fn broadcast(&mut self, dest: u8, source: Operand) {
        // Broacast only supports memory operands
        let Operand::Memory { base, disp } = source else {
            unreachable!("Cannot broadcast register value: {:?}", source)
        };

        self.vex(dest, 0, base, 1, 2);
        self.code.push(0x18); // opcode
        self.operands(dest, Operand::Memory { base, disp });
    }

    // And so on...
}
```

## Register allocation

There are two key issues we have been ignoring thus far:
- How do we pick which registers to use for our instructions?
- We only have 16 registers available to use in our machine instructions, yet we have upwards of 7000 variables in our VM instructions,
  any of which can be used as an operand for any other. How can we avoid ovewriting a register that contains a value we need to use later?

This, of course, is just an instance of the problem of [Register Allocation](https://en.wikipedia.org/wiki/Register_allocation).
We have `M` "virtual" registers that we want to map onto `N` physical registers (where `M` >> `N`) without losing data in the process.
There are many register allocation algorithms to choose from, each with their strengths and weaknesses.

No matter what approach we choose to juggle our values between registers, we will inevitably come to a point where we don't have any registers left
to hold our values, so we'll need to write some of them to memory. This is known in the literature as register __spilling__.
Spilling incurs a performance penalty, so an optimal register allocation solution would have as few spills as possible.

Unfortunately, optimal register allocation is known to be an NP-complete problem, so we cannot expect to achieve optimal spills
without spending a __lot__ of time computing the allocation in the first place.

There is therefore a tradeoff to be made between code that runs faster, and code that is faster to compile.
In a JIT compiler like ours, where compilation also happens at runtime, it is generally a good idea to skew a little more towards the latter compared to an ahead-of-time compiler.

It just so happens, there is an algorithm that does a pretty good job at allocating registers with only a linear (or, well, close to linear) time complexity:
[Linear Scan Register Allocation](https://en.wikipedia.org/wiki/Register_allocation#Linear_scan).

The idea behind it is relatively simple: we first figure out the interval in the instruction stream during which each value is _live_, meaning the interval between when the value is produced by an instruction and when it is last used as an input to another instruction:

```
_0 const 2.95     ───────────────────┐
_1 var-x          ───┐               │
_2 var-y        ─┐  1-3             0-3
_3 mul _1 _2   <─┘ <─┘  ────┐        │
_4 add _0 _3               3-6 ─┐ <──┘
_5 const 3.675  ─┐          │   │
_6 add _5 _3   <─┘ ──┐   <──┘  4-8
_7 neg _6      <─────┘ ──┐      │
_8 max _4 _7   <─────────┘  <───┘
```

Then, we walk through the instructions, and assign them to any register that is not currently holding another value, in a greedy manner.
Once a value is no longer live, we free up its register so it can be used by another instruction.

When we run out of registers and need to spill something, we choose the register that holds the value that lives _the longest in the future_.
This gives us the most bang for the buck, freeing up a register for the longest time possible.

Before we get to implementing the algorithm, we'll define the data structures involved.
Turns out there are quite a few of them that need to be kept in sync as we go, so we'll bundle them into a `struct`.
We'll get to why each data structure was picked in a second, for now we'll start with the data contained within them.

```rust
/// Interval in the instruction stream during which a variable is live.
#[derive(Default, Debug, Clone, Copy, PartialOrd, Ord, PartialEq, Eq)]
struct LiveInterval {
    /// Index of the last instruction that uses this variable as an operand.
    end: VarId,
    /// Index of the variable this interval refers to.
    start: VarId,
}

struct RegisterAllocator {
    /// Tracks which register or memory location currently contains which variable.
    assigned: Vec<Operand>,
    /// Binary search tree of currently live intervals, sorted by end time.
    active: BTreeSet<LiveInterval>,
    /// Stack of registers that are not currently holding a live variable.
    available_regs: Vec<u8>,
    /// How many variables have we spilled so far.
    spill_count: u32,
}
```

Note how the `LiveInterval` struct is defined with the `end` field first, and then `start`.
This makes it so the default implementation of `Ord` specified by `#[derive(PartialOrd, Ord)]`
will sort intervals in order of increasing end time first (and then start time, in case of a tie).
This is exactly the order that the `active` set will store the currently live intervals in.

At the beginning, the active set is empty, and all registers are available and free to be used by instructions. But, although AVX provides 16 vector registers (known as `ymm` registers) to play with, we can't make use of all of them for storing variables.
For one, we need to reserve two registers for the `x` and `y` parameters, which will be passed in from the calling code.

We also need to save one register for use a _scratch_ register.
This is necessary for operations like negation, which can't be expressed as a single instruction (there is no `negps` AVX instruction), but must be performed by zeroing out a register, and then subtracting the operand from it. Using a scratch register in these cases saves us from having to spill two registers instead of just one, simplifying the implementation.

```rust
impl RegisterAllocator {
    fn new(instrs: &[Instr]) -> Self {
        Self {
            assigned: Vec::with_capacity(instrs.len()),
            active: BTreeSet::new(),
            // ymm0 and ymm1 are occupied by params,
            // keep ymm15 as scratch register for spilled values
            available_regs: (2..15).rev().collect(),
            spill_count: 0,
        }
    }
}
```

Figuring out the live intervals is straightforward.
Since we'll keep the intervals for all values in a `Vec` (indexed by the corresponding `VarId`),
the start of an interval is just its index in the `Vec`. To avoid redundancy we'll just store the interval ends, like so:
```rust
impl RegisterAllocator {
    fn compute_last_usage(instrs: &[Instr]) -> Vec<VarId> {
        let mut uses: Vec<VarId> = vec![VarId(0); instrs.len()];
        for (id, instr) in instrs.iter().enumerate() {
            let id = VarId(id as u32);
            match instr {
                Instr::Unary { operand, .. } => {
                    uses[operand.0 as usize] = id;
                }
                Instr::Binary { lhs, rhs, .. } => {
                    uses[lhs.0 as usize] = id;
                    uses[rhs.0 as usize] = id;
                }
                _ => (),
            }
        }
        uses
    }
}
```

We can now sketch out our full register allocation algorithm.
Once we have allocated a register for a VM instruction, we'll have all the information we need to convert it to machine code.
Therefore, we will let the register allocator also drive code generation:

```rust
impl RegisterAllocator {
    /// Free up registers for variables that won't be used again after `cur`.
    fn free_dead_registers(&mut self, cur: VarId) {
        todo!();
    }

    /// Allocate a register for `instr`, assuming it is live during `interval`, and write the corresponding instruction into `buf`.
    fn allocate_instr(&mut self, buf: &mut CodeBuffer, instr: &Instr, interval: LiveInterval) {
        todo!();
    }

    /// Write code for all instructions in `instrs` into `buf`.
    fn generate_code(&mut self, buf: &mut CodeBuffer, instrs: &[Instr]) {
        let ends = Self::compute_last_usage(instrs);

        for (i, instr) in instrs.iter().enumerate() {
            let interval = LiveInterval {
                end: ends[i],
                start: VarId(i as u32),
            };

            self.free_dead_registers(interval.start);
            self.allocate_instr(buf, instr, interval);
        }
    }
}
```

To free up registers for dead variables, we'll make use of the ordering of our active set.
The first variable in the set is the one that "dies" the soonest.
So if that first variable is not dead yet, none of the remaining ones in the set will be either,
and we can exit our loop early.
Otherwise, we use the `assigned` vector to find the register that is currently holding it, and push it into `available_regs`:

```rust
impl RegisterAllocator {
    /// Free up registers for variables that won't be used again after `cur`.
    fn free_dead_registers(&mut self, cur: VarId) {
        while let Some(i) = self.active.first().copied() {
            // Bail only if strictly greater, as we free registers
            // before allocating a register.
            if i.end > cur {
                break;
            }
            self.active.pop_first();
            match self.assigned[i.start] {
                Operand::Reg(reg) => self.available_regs.push(*reg),
                _ => unreachable!("Only registers should be active"),
            }
        }
    }
}
```

The actual process of allocating a register is roughly as I described before: if there is a register available, we use it. Otherwise, we have to spill.
```rust
impl RegisterAllocator {
    /// Mark tregister `reg` as containing the value that `interval.start` refers to.
    fn assign_register(&mut self, reg: u8, interval: LiveInterval) {
        self.assigned.push(Operand::Reg(reg));
        self.active.insert(interval);
    }

    /// Generate code for `instr` using `reg` as the destination register, and write it into `buf`.
    fn generate_instruction(&mut self, buf: &mut CodeBuffer, instr: &Instr, reg: u8) {
        todo!();
    }

    /// Spill a register to make room for storing the value computed by `instr`, assuming it is live during `interval`.
    /// This also generates code for `instr`.
    fn spill(&mut self, buf: &mut CodeBuffer, instr: &Instr, interval: LiveInterval) {
        todo!()
    }

    /// Allocate a register for `instr`, assuming it is live during `interval`, and write the corresponding code into `buf`.
    fn allocate_instr(&mut self, buf: &mut CodeBuffer, instr: &Instr, interval: LiveInterval) {
        if let Instr::Var(reg) = instr {
            // No need to generate anything, arguments are already stored in registers
            // at the beginning of the function
            self.assign_register(*reg as u8, interval);
        } else if let Some(reg) = self.available_regs.pop() {
            self.generate_instruction(buf, instr, reg);
            self.assign_register(reg, interval);
        } else {
            self.spill(buf, instr, interval);
        }
    }
}
```

Earlier I mentioned that, when we need to spill something, we should always pick the interval that lives the longest into the future.

Remember how our active set is ordered by interval end time? We already used that ordering to discard dead intervals,
and we can use it again to pick the longest-lived one: all we have to do is simply pick the greatest element in the set!
This, by the way, is why our active set is a binary search tree and not simply a heap: we need to pop values from it from _both_ ends.

One small complication to our algorithm is that we should also consider the current variable as a spilling candidate.
If that instruction lives longer than the longest-lived one in the active set, there's no point in spilling another value; we should just write the result directly into memory.
Unfortunately AVX does not support writing to memory as an output destination directly, so we'll use our scratch register as a temporary holding ground.

```rust
impl RegisterAllocator {
    /// Create a new memory location for values to be spilled to.
    fn new_spill_location(&mut self) -> Operand {
        todo!();
    }

    /// Spill a register to make room for storing the value computed by `instr`, assuming it is live during `interval`.
    /// This also generates code for `instr`.
    fn spill(&mut self, buf: &mut CodeBuffer, instr: &Instr, interval: LiveInterval) {
        // Get the longest-lived active interval
        let candidate = self
            .active
            .last()
            .copied()
            .expect("There's no live value to spill");

        if candidate.end > interval.end {
            // Candidate lives the longest, so spill it.
            self.active.pop_last();

            let Operand::Reg(reg) = self.assigned[candidate.start] else {
                panic!("Cannot spill a memory location: {:?}", self.assigned[candidate.start]);
            };
            let new_loc = self.new_spill_location();
            self.assigned[candidate.start] = new_loc;

            // Write register contents to memory
            buf.mov(new_loc, Operand::Reg(reg));
            self.generate_instruction(buf, instr, reg);
            self.assign_register(reg, interval);
        } else {
            // Write result to scratch register
            let scratch = 15;
            self.generate_instruction(buf, instr, scratch);
            let mem = self.new_spill_location();
            // Write scratch register to memory
            buf.mov(mem, Operand::Reg(scratch));
            self.assigned.push(mem);
        }
    }
}
```

One thing I haven't touched on yet: what memory should we spill our values into?
Typically, in most compiled languages, local variables are spilled to the __stack__, a big chunk of preallocated scratch memory that a function can use (mostly) as it sees fit. All it needs to do is allocate and deallocate memory by decrementing or incrementing the __stack pointer__ register at the stard and end of the function respectively.

In our case however, I found that this allocation and deallocation has a significant performance overhead,
and it also increased the complexity of the implementation, as the number of spilled variables needs to be determined before generating any code.
So instead, we'll just spill to a heap-allocated buffer that we will allocate once after the code is compiled and then reuse for all runs of our compiled function.

We'll keep the base pointer of our spill buffer in a fixed register (say, `rcx`), and encode a value's index into this buffer as a constant displacement,
since once we have determined the spill location of a given instruction it won't need to change when evaluating different pixels:
```rust
impl RegisterAllocator {
    /// Create a new memory location for values to be spilled to.
    fn new_spill_location(&mut self) -> Operand {
        let disp = {
            let slot = self.stack_size;
            self.stack_size += 1;
            slot * CodeBuffer::VALUE_SIZE
        };
        Operand::Memory {
            base: CodeBuffer::RCX,
            disp,
        }
    }
}
```

We'll also use a heap-allocated buffer to hold our constants, this time keeping its base pointer in `rax`.
This way we can use the register-indirect addressing mode for all memory addresses in our program:
```rust
impl CodeBuffer {
    fn constant(&mut self, cnst: f32) -> Operand {
        let slot = self.constants.len() as u32;
        self.constants.push(cnst);
        Operand::Memory {
            base: Self::RAX,
            disp: slot * std::mem::size_of::<f32>() as u32,
        }
    }
}
```

And with that final piece of the puzzle, we can generate the actual code for all our instructions:
```rust
impl RegisterAllocator {
    fn generate_instruction(&mut self, buf: &mut CodeBuffer, instr: &Instr, dest: u8) {
        match instr {
            Instr::Var(_) => (),
            Instr::Const(cnst) => {
                let cnst = buf.constant(*cnst);
                buf.broadcast(dest, cnst);
            }
            Instr::Unary { op, operand } => {
                let x: Operand = self.assigned[*operand];
                match op {
                    UnaryOpcode::Neg => {
                        let scratch = CodeBuffer::SCRATCH;
                        buf.xor(scratch, scratch, Operand::Reg(scratch));
                        buf.sub(dest, scratch, x);
                    }
                    UnaryOpcode::Sqrt => buf.sqrt(dest, x),
                    /* ... */
                }
            }
            Instr::Binary {op, lhs,rhs} => {
                let x: Operand  = self.assigned[*lhs];
                let y: Operand  = self.assigned[*rhs];

                // Left-hand operand must be a register,
                // so load it into a scratch register if it's in memory
                let x: u8 = match x {
                    Operand::Reg(reg) => reg,
                    mem @ Operand::Memory { .. } => {
                        buf.mov(Operand::Reg(CodeBuffer::SCRATCH), mem);
                        CodeBuffer::SCRATCH
                    }
                };

                match op {
                    BinaryOpcode::Add => buf.add(dest, x, y),
                    /* and so on ... */
                }
            }
        }
    }
}
```

## Running the code

We can pat ourselves on the back here: the hardest part's done! We have successfully converted our VM instruction into actual processor instructions.

Now comes the most exciting part: actually running the code we have generated!
But how can we do that, when all we have is a `Vec<u8>`?

On POSIX platforms at least, this is how:
1. Allocate one or more _pages_ of memory, enough to fit all the code we generated.
2. Copy the code to this newly-allocated memory buffer.
3. Set the correct _permissions_ so that the buffer's contents can be executed as code.
4. Transfer control to the code with an indirect call to the buffer's address.

The first three steps are pretty straightforward to implement with calls to `libc`.
We'll hold the code buffer in its own struct so that it can be cleanly deallocated when dropped:

```rust
/// A buffer of finished, immutable generated machine code that can be executed.
pub struct InstalledCode {
    code_buf: *const u8,
    constants: Vec<f32>,
    stack_size: usize,
    layout: Layout,
}

impl CodeBuffer {
    /// Install the code generated so far for execution.
    pub fn install(self) -> InstalledCode {
        use libc::{_SC_PAGESIZE, sysconf};
        let page_size = unsafe { sysconf(_SC_PAGESIZE) } as usize;
        let num_pages = usize::max(1, self.code.len().div_ceil(page_size));
        let layout =
            Layout::from_size_align(page_size * num_pages, page_size).expect("invalid layout");

        unsafe {
            let ptr = alloc::alloc(layout);
            if ptr.is_null() {
                alloc::handle_alloc_error(layout);
            }
            // Fill with RET instructions for safety, in case something goes wrong
            // we'll at least return from the function.
            ptr.write_bytes(0xc3, layout.size());
            // Copy code from readable buffer to executable buffer
            ptr.copy_from_nonoverlapping(self.code.as_ptr(), self.code.len());

            // Make memory executable (but not writable)
            libc::mprotect(
                ptr as *mut libc::c_void,
                layout.size(),
                libc::PROT_EXEC | libc::PROT_READ,
            );

            InstalledCode {
                code_buf: ptr,
                _code_size: self.code.len(),
                stack_size: self.stack_size as usize,
                constants: self.constants,
                layout,
            }
        }
    }
}

/// Deallocate the buffer when it falls out of scope.
impl Drop for InstalledCode {
    fn drop(&mut self) {
        use std::alloc;
        unsafe {
            // Don't forget to restore the original protections!
            libc::mprotect(
                self.code_buf as *mut libc::c_void,
                self.layout.size(),
                libc::PROT_READ | libc::PROT_WRITE,
            );
            alloc::dealloc(self.code_buf as *mut u8, self.layout);
        }
    }
}
```

Actually transferring control to the code requires a bit more effort.
It would be nice if we could cast a pointer to the code buffer to a function pointer and call it directly, like so:
```rust
/// 256-bit wide floating-point vector
pub type Ymm = std::arch::x86_64::__m256;

impl InstalledCode {
    pub fn invoke(&self, x: Ymm, y: Ymm, temp_buf: &mut [Ymm]) -> Ymm {
        unsafe {
            let fn_ptr: extern "sysv64" fn(*const f32, *const Ymm, Ymm, Ymm) -> Ymm = unsafe {
                std::mem::transmute(self.code_buf)
            };

            fn_ptr(self.constants.as_ptr(), temp_buf.as_mut_ptr(), x, y)
        }
    }
```

There are a couple problems with this approach:
- We'd need to adjust our generated code to match the used _calling convention_.
In the snippet above I marked the function as using the [System V AMD64](https://wiki.osdev.org/System_V_ABI) calling convention, where the first two arguments are passed in registers `rdi` and `rsi`.
That would make it difficult to port this code to platforms that only support other calling conventions.
- Other calling conventions on e.g. Windows involve writing arguments to the stack before the call and then the function reading them inside. This is much slower than a direct call.
- The correlation between the used calling convention and the order of the registers used is implicit, and requires looking it up on an external reference.

So instead, I found it much easier, clearer and more performant to write a bit of _inline assembly_, which Rust offers very good support for.

We can bind Rust variables to specific registers in the assembly code both as inputs and outputs, and even define which registers the code we're calling will overwrite.
The compiler will then preserve any clobbered registers automatically.

This is what that looks like, lightly annotated:
```rust
impl InstalledCode {
    pub fn invoke(&self, x: Ymm, y: Ymm, temp_buf: &mut [Ymm]) -> Ymm {
        unsafe {
            let fn_ptr = self.code_buf;
            let result: Ymm;
            std::arch::asm!(
                "call {}",
                // Bind fn_ptr to an arbitrary free register
                in(reg) fn_ptr,
                // Bind the constants buffer base pointer to rax
                in("rax") self.constants.as_ptr(),
                // Bind the spill buffer base pointer to rcx
                in("rcx") temp_buf.as_mut_ptr(),
                // ymm0 is used both to pass the x argument to the function, and to retrieve its return value
                inout("ymm0") x => result,
                // ymm1 is used as an argument and can be overwritten by the generated code
                inout("ymm1") y => _,
                // Our generated code can overwrite all remaining ymm registers, so we mark them all as clobbered
                out("ymm2") _,
                out("ymm3") _,
                out("ymm4") _,
                out("ymm5") _,
                out("ymm6")  _,
                out("ymm7")  _,
                out("ymm8")  _,
                out("ymm9")  _,
                out("ymm10") _,
                out("ymm11") _,
                out("ymm12") _,
                out("ymm13") _,
                out("ymm14") _,
                out("ymm15") _,
                // Let the compiler know that we don't modify the stack in our generated code
                options(nostack),
            );
            result
        }
    }
}
```

And that's it! The rest of our program is mostly just boilerplate to allocate an image, call the code repeatedly with the right parameters,
and then convert the result to bytes and store it into the image.

Actually, that last part deserves some explanation.
As we mentioned, the code will evaluate 8 pixels at once, each represented by a 32-bit floating point number.
In order to save those pixels to a black-and-white image, we need to convert those 32-bit floats to 8-bit bytes.

We could simply extract each of them from the vector one by one and write them into the image individually, but it's much more performant to write all of them at once, in parallel.

To do so, we convert the vector of floating-point values to a vector of integers, and then shuffle their bytes around with a couple of `pack` instructions:
```rust
/// Convert the 8 floating-point values stored in a ymm register to 8 pixel values to be written into an image buffer.
fn to_image_bytes(x: Ymm) -> [u8; 8] {
    unsafe {
        // Thresholding: convert 8 32-bit floats to 32-bit integers with values of 255 if below 0, and 0 otherwise.
        let mask = _mm256_cmp_ps::<_CMP_LT_OQ>(x, _mm256_setzero_ps());
        let mask: __m256i = std::mem::transmute(mask);
        let ones = _mm256_set1_epi32(255);
        let result = _mm256_and_si256(mask, ones);

        // Pack down the least significant bytes of each integer so that they are stored in the first 8 bytes of the vector.
        let result = _mm256_packus_epi32(result, result);
        let result = _mm256_packus_epi16(result, result);
        let result =
            _mm256_permutevar8x32_epi32(result, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));

        // Extract the first 8 bytes (64 bits) from the vector
        _mm256_extract_epi64::<0>(result).to_le_bytes()
    }
}
```

## Debugging

Nothing ever goes right the first time, so it's very likely you'll need to debug the runtime-generated code.
`gdb` gives us some pretty nice utilities to do that.

The first is setting breakpoints at arbitrary memory addresses, with a nice shorthand syntax for addresses held in variables:
```
# Stop at the function which transfers control to the generated code
break prospero::codegen::InstalledCode::invoke
cont
# Breakpoint hit
# We are now at the start of the call stub.
# Set a breakpoint at the actual compiled code address
break *fn_ptr
cont
# We are now at the beginning of our runtime-compiled code
```

We can use the `display` command to track the instructions we're executing and the contents of our registers as we step through our generated code:

```
# Disassemble the next 5 instructions following the one we're currently stopped at
display/5i $pc
# Display the contents of ymmm0 and ymm1 as 8 32-bit floats
display $ymm0.v8_float
display $ymm1.v8_float
# Step through the code instruction by instruction
si
si
...
```

## Intermediate results

Success! We now have a working solution that generates the correct image using runtime-compiled code.

So, how fast is it?
I used [hyperfine](https://github.com/sharkdp/hyperfine) to measure my implementation's runtime across multiple runs.
I originally wanted to compare it against the baseline numpy implementation provided by Matt, but for the life of me I could not get it to work at sizes higher than 256x256 without running out of memory. So, I also wrote an extremely simple interpreter as an additional point of comparison.

The following table shows the results for different image sizes:
<div class="table-wrapper">

| Image size | Numpy<br>reference impl. | Interpreter | Compiler       |
|------------|--------------------------|-------------|----------------|
| 256        | 1.564 s                  | 2.773 s     | 13.7 ms        |
| 512        | N/A                      | 10.014 s    | 44.2 ms        |
| 1024       | N/A                      | 40.682 s    | 162.1 ms       |
| 2048       | N/A                      | 164.528 s   | 637.6 ms       |
| 4096       | N/A                      | 693.809 s   | 2.546 s        |

</div>

(Experiments were run on an AMD Ryzen 7 6800HS CPU on an 2022 ASUS Zephyrus G14 laptop. Your mileage may vary.)

On average, the compiler is a whopping 250 times faster than the interpreter, and that's all on a single thread!

These results include time taken to parse the instructions from a file and to compile the code,
both of which are negligible: compilation takes on average 700 _nanoseconds_, and parsing code takes around 850 microseconds
when the instructions file has been cached by the OS after a couple of warm-up runs.


## Optimization

So of course we handily beat the baseline, that's hardly surprising.
Still, more than 2.5 seconds to render a single 4K frame doesn't exactly scream _blazing fast_ to me.

We have squeezed pretty much all that we could from taking the instructions as they are and turning them into code.
Can we go faster by optimizing the stream of instructions themselves, before they are even compiled?

I initially thought of applying the usual set of generic compiler optimization techniques:
folding constants, removing unused instructions, that kind of thing.
However, some exploratory analysis showed that there weren't that many opportunities for such optimizations.

I suppose that makes sense: if the entire program can be optimized as a whole, it would make more sense to save it back to file in an optimized form, rather than doing this optimization every time the program is run.
In fact, I presume the instruction stream given by the challenge was already optimized as such.
So, back to the drawing board.

The actual breakthrough came by thinking about what all these mathematical expressions are actually computing.
The program encodes a bunch of implicit shape definitions for all the letters shown on the image, that we then evaluate on a grid of pixels.

But, we are always running the same instructions for every pixel of the image.
This means that when we're rendering, say, the start of the first row of text, we're still doing all the work of evaluating the shape of _every single character_ in the entire image, only to throw it away after!
There's no way that characters in the bottom-right part of the image are going to have any effect on the pixels in the top left, so we're just wasting work by even considering them!

If we could __split__ the program into a bunch of smaller programs that only evaluate a __chunk__ of the image, there would be much less duplicate work involved.

But, if all we have is just a linear chunk of instructions, how do we determine which ones are necessary to evaluate a given chunk and which aren't?

Well, we know that the only inputs to the function we are evaluating are the x and y positions of the pixel in image space.
Every other instruction in the program is a pure mathematical function, and therefore depends only on the parameter values.

If we consider one individual chunk of the image, we also know what __range__ the x and y positions will be in for that particular chunk, based on the chunk's boundaries:

{{ svg_embed(path="chunks.svg") }}

The consequence of these two facts is that, by starting from the ranges of the x and y parameters, we can determine the range every value in the program can take when evaluated within a given chunk.

```
_0 const 2.95       range: [2.95, 2.95] (constant range)
_1 var-x            range: [-0.5, 0.0]  (range of input parameter)
_2 const 8.13008    range: [8.13008, 8.13008] (constant range)
_3 mul _1 _2        range: [−4.06504, 0.0] (obtained from input ranges)
_4 add _0 _3        range: [−1.11504, 2.95] 
[... etc ...]
```

Where this becomes especially useful is with the `min` and `max` instructions.
If the ranges of the two variables being compared do not overlap, we know at compile time that one of these two variables will always be greater than the other.

```
# assuming _a has range: [-4.0, 2.0]
# and  _b has range: [5.0, 7.0]
# so _b will always be greater than _a
_c max _a _b
# Can be simplified simply to
_c _b
```

So in cases like these, the instruction can be completely replaced with one of its two operands. Not only that, but if the other operand isn't used by any other instruction, we can outright remove it from our chunk-specific program.

In our implementation, we'll do this work in multiple passes.

First, we'll propagate the input parameter ranges through the instructions,
and collect all instruction replacements into a hash map:

```rust
#[derive(Default, Debug, Clone, Copy)]
pub struct Range {
    pub min: f32,
    pub max: f32,
}

/// Determines the range of the elements in the given slice.
fn range_of(elems: &[f32]) -> Range {
    let min = elems.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max = elems.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    Range { min, max }
}

/// Computes the range of all variables in `instrs` based on the input
/// parameter ranges `x_range` and `y_range`, and compute a mapping
/// of instruction replacements.
fn compute_replacements(instrs: &[Instr], x_range: Range, y_range: Range) -> HashMap<VarId, VarId> {
    let mut ranges: Vec<Range> = vec![Default::default(); instrs.len()];
    let mut replacements: HashMap<VarId, VarId> = HashMap::new();
    for (i, instr) in instrs.iter().enumerate() {
        let range = match instr {
            Instr::VarX => x_range,
            Instr::VarY => y_range,
            Instr::Const(c) => Range { min: *c, max: *c },
            Instr::Unary { op, operand } => {
                let range = &ranges[operand.0 as usize];
                use UnaryOpcode::*;
                match op {
                    Neg => Range {
                        max: -range.min,
                        min: -range.max,
                    },
                    // We assume that sqrt won't ever be applied to negative inputs,
                    // so we constrain the input range to be non-negative.
                    Sqrt => Range {
                        min: range.min.max(0.0).sqrt(),
                        max: range.max.sqrt(),
                    },
                    Square => range_of(&[range.min * range.min, range.max * range.max),
                }
            }
            Instr::Binary { op, lhs, rhs } => {
                let xr = &ranges[lhs.0 as usize];
                let yr = &ranges[rhs.0 as usize];
                use BinaryOpcode::*;
                match op {
                    Add => Range {
                        min: xr.min + yr.min,
                        max: xr.max + yr.max,
                    },
                    Sub => Range {
                        min: xr.min - yr.max,
                        max: xr.max - yr.min,
                    },
                    Mul => range_of(&[
                        xr.min * yr.min,
                        xr.min * yr.max,
                        xr.max * yr.min,
                        xr.max * yr.max,
                    ]),
                    Max => {
                        if xr.min > yr.max {
                            // max(x, y) == x
                            let repl = replacements.get(lhs).unwrap_or(lhs);
                            replacements.insert(VarId(i as u32), repl);
                            *xr
                        } else if xr.max < yr.min {
                            // max(x, y) == y
                            let repl = replacements.get(rhs).unwrap_or(rhs);
                            replacements.insert(VarId(i as u32), repl);
                            *yr
                        } else {
                            Range {
                                min: xr.min.max(yr.min),
                                max: xr.max.max(yr.max),
                            }
                        }
                    }
                    Min => {
                        if xr.min > yr.max {
                            // min(x, y) == y
                            let repl = replacements.get(rhs).unwrap_or(rhs);
                            replacements.insert(VarId(i as u32), repl);
                            *yr
                        } else if xr.max < yr.min {
                            // min(x, y) == x
                            let repl = replacements.get(lhs).unwrap_or(lhs);
                            replacements.insert(VarId(i as u32), repl);
                            *xr
                        } else {
                            Range {
                                min: xr.min.min(yr.min),
                                max: xr.max.min(yr.max),
                            }
                        }
                    }
                }
            }
        };
        ranges[i] = range;
    }
    replacements
}
```

With the replacements in hand, we can then apply them to the instructions with a second pass.
We can't replace the instruction directly, so we'll instead modify any instructions that
use the replaced instruction as an input, changing the input to its replacement.

```rust
/// Modifies `instrs` so that for any (lhs, rhs) pair in `replacements`,
/// all instructions with `lhs` as an operand will now have `rhs` as its operand instead.
fn apply_replacements(instrs: &mut [Instr], replacements: &HashMap<VarId, VarId>) {
    for instr in instrs.iter_mut() {
        match instr {
            Instr::Unary { operand, .. } => {
                *operand = replacements.get(operand).unwrap_or(operand);
            }
            Instr::Binary { lhs, rhs, .. } => {
                *lhs = replacements.get(lhs).unwrap_or(lhs);
                *rhs = replacements.get(rhs).unwrap_or(rhs);
            }
            _ => (),
        }
    }
}
```

With this we have optimized each instruction individually, but the instruction stream is still left with a bunch of unused instructions.
We will clean them up in one more pass (or well, technically two passes).

Since in our implementation variable IDs are just their indices in a `Vec`, removing
an instruction from that `Vec` will also involve adjusting the IDs of all following instructions:

```rust
/// Returns a boolean vector where each element represents
/// where the corresponding instruction in `instrs` is used
/// by any other instruction in `instrs`.
fn compute_instr_usage(instrs: &[Instr]) -> Vec<bool> {
    // We could also do this with a bitset, but this is less complex
    let mut is_used = vec![false; instrs.len()];
    // Return value is always used
    *is_used.last_mut().unwrap() = true;
    // Propagate is_used from last to first
    for (i, instr) in instrs.iter().enumerate().rev() {
        if is_used[i] {
            match instr {
                Instr::Unary { operand, .. } => {
                    is_used[operand.0 as usize] = true;
                }
                Instr::Binary { lhs, rhs, .. } => {
                    is_used[lhs.0 as usize] = true;
                    is_used[rhs.0 as usize] = true;
                }
                _ => (),
            }
        }
    }
    is_used
}

fn cleanup_unused(mut instrs: Vec<Instr>) -> Vec<Instr> {
    // First pass: track which instructions are used in a `Vec<bool>`.
    let is_used = compute_instr_usage(&instrs);

    // Second pass: remove any instructions for which is_used[i] == false

    // Removing an instructions from the stream will change the index of
    // all following instructions, so we maintain a mapping from old ID to new ID
    let mut id_mapping = Vec::new();

    // How many instructions have we kept so far.
    let mut retained = 0u32;

    instrs
        .into_iter()
        .zip(is_used)
        .filter_map(|mut instr, is_used| {
            id_mapping.push(VarId(retained));
            if !is_used {
                return None;
            }
            match &mut instr {
                Instr::Unary { operand, .. } => {
                    *operand = id_mapping[operand.0 as usize];
                }
                Instr::Binary { lhs, rhs, .. } => {
                    *lhs = id_mapping[lhs.0 as usize];
                    *rhs = id_mapping[rhs.0 as usize];
                }
                _ => (),
            };
            Some(instr)
        })
        .collect()
}
```

And with that, we have our whole optimization pipeline to obtain a version of the function specialized to a single chunk:
```rust
/// Computes an optimized stream of instructions for the image chunk defined by `x_range` and `y_range`.
pub fn specialize(mut instrs: Vec<Instr>, x_range: Range, y_range: Range) -> Vec<Instr> {
    let replacements = compute_ranges(&instrs, x_range, y_range);
    // If the returned instruction has a replacement, we can directly
    // truncate the array so that its replacement is the new last instruction,
    // as all instructions in between will for sure be unused.
    let ret = VarId(instrs.len() - 1);
    if let Some(ret) = replacements.get(ret) {
        instrs.truncate(ret.0 as usize + 1);
    }
    apply_replacements(&mut instrs, &replacements);
    cleanup_unused(instrs)
}
```

Feel free to read [the source code on GitHub](https://github.com/CRefice/prospero.jit/blob/e8677025e7abdfcb55642097d84a1ddd65c7f82e/src/main.rs#L191) to see how this type of specialization is used in the final version of the code.

## Multithreading and borrow checker issues

Initially, I didn't want to talk about multithreading in this blog post as it's really not that interesting in the scope of this challenge:
image rendering is embarrassingly parallel, so distributing the work across N processor cores will in theory achieve
an almost N-fold speedup over doing it serially.

I did however want to implement multi-threading anyway, for the sake of making the final results look as good as possible.
In the process, I ran into a Rust borrow-checker issue I thought was worth talking about.

Instead of bringing in `rayon` as a dependency, I thought to use the recently stabilized feature of [scoped threads](https://doc.rust-lang.org/std/thread/fn.scope.html).
For the sake of simplicity we'll use it in a pretty "dumb" way, spawning one thread per image chunk.
This looks roughly as follows:

```rust
let mut image = vec![0u8; image_size * image_size];
let chunk_size = image_size / num_splits;
std::thread::scope(|s| {
    for (y, row) in specialized.into_iter().enumerate() {
        for (x, code) in row.into_iter().enumerate() {
            // Spawn a separate thread to process each chunk of the image
            s.spawn(move || {
                let start_y = y * chunk_size;
                let end_y = start_y + chunk_size;
                let start_x = x * chunk_size;
                let end_x = start_x + chunk_size;
                // process image[start_y..end_y][start_x..end_x]
            });
        }
    }
});
```

The problem is, as soon as you try taking a mutable reference to `image`, the borrow checker will yell at you:
```
error[E0499]: cannot borrow `image` as mutable more than once at a time
   --> src/main.rs:97:21
    |
92  |       std::thread::scope(|s| {
    |                           - has type `&'1 Scope<'1, '_>`
...
97  |               s.spawn(|| {
    |               -       ^^ `image` was mutably borrowed here in the previous iteration of the loop
    |  _____________|
    | |
    | |                     <snip>
...   |
137 | |             });
    | |______________- argument requires that `image` is borrowed for `'1`

```

In theory, what we're doing here should be perfectly sound.
There is no data race, nor any need for synchronization; each thread is independently writing to a separate chunk of the image.

The problem is, the Rust compiler cannot _prove_ that we're actually writing to separate chunks. All it sees is the image being borrowed mutably across multiple threads, and throws up its hands.

If we were dealing with a 1D array, `Vec::chunks_mut` would be a good way to solve this:
```rust
let image: Vec<u8> = /*...*/;
std::thread::scope(|s| {
    for chunk in image.into_iter().chunks_mut(chunk_size) {
        s.spawn(move || {
            // write to chunk
        });
    }
});
```

But with a 2D array, each thread isn't accessing a contiguous slice of memory, so things get a bit harder.
The simplest way I found to get around this is with some `unsafe`, pulling the wool over the compiler's eyes by converting the `image` slice to a pointer, and wrapping that in a struct which implements `Send` and can convert the pointer back to a slice:
```rust
/// Allows sharing a `&mut [u8]` between threads.
struct Smuggle(*mut u8);
unsafe impl Send for Smuggle {}
impl Smuggle {
    unsafe fn as_slice(&mut self, len: usize) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.0, len) }
    }
}

/* [...] */

let chunk_size = image_size / num_splits;
std::thread::scope(|s| {
    for (y, row) in specialized.into_iter().enumerate() {
        for (x, code) in row.into_iter().enumerate() {
            // Smuggle the image past the borrow checker
            let mut image = Smuggle(image.as_mut_ptr());
            s.spawn(move || {
                let start_y = y * chunk_size;
                let end_y = start_y + chunk_size;
                let start_x = x * chunk_size;
                let end_x = start_x + chunk_size;
                let image = unsafe { image.as_slice(image_size * image_size) };
                // process image[start_y..end_y][start_x..end_x]
            });
        }
    }
});
```

## Final results

By splitting up the image into chunks, and running a specialized version of the program for each chunk,
we will perform a lot less duplicate work, and the function will be evaluated a lot faster.

But, there is still a trade-off as to how many chunks we split the image into.
Too few chunks, and we'll still be doing a lot of duplicate work when running the compiled program.
Too many, and we'll spend more time compiling all the different specialized versions of the program than actually running them.

The following graph shows the total runtime of running the example program at the same image size of 4096x4096 but with different chunk sizes,
broken up into compilation time and evaluation time.

{{ svg_embed(path="performance-graph.svg") }}

Cutting up the image into more chunks does result in better evaluation performance (at least up to 64 subdivisions), but the time taken up by compiling the code for all those chunks quickly starts to outweigh those savings. 16x16 seems like the optimal number of chunks to split the image into, resulting in the overall lowest runtime.

With that, here are our final results, compared to our previous intermediate results:

<div class="table-wrapper">

| Image size | Unoptimized | Optimized (16x16 chunks) | Speedup |
|------------|-------------|--------------------------|---------|
| 256        | 13.7 ms     | 14.6 ms                  | 0.94x   |
| 512        | 44.2 ms     | 13.6 ms                  | 3.25x   |
| 1024       | 162.1 ms    | 14.6 ms                  | 11.10x  |
| 2048       | 637.6 ms    | 17.9 ms                  | 35.62x  |
| 4096       | 2.546 s     | 31.4 ms                  | 81.08x  |

</div>

Now these are some impressive numbers if I do say so myself!
We can evaluate a 4K image in less than a thirtieth of a second, meaning we could sustain a framerate of 30fps if we were to use this as an interactive visualizer. Not bad for a CPU-only implementation!

## Conclusion

There are a few more optimizations that I glossed over, such as specializing the program by recursively subdividing it into chunks
rather than iteratively, and writing the installed code of different chunks into a single buffer.
Once again, feel free to check out [the actual source code](https://github.com/CRefice/prospero.jit/tree/main/src) to see the version of the code I obtained the results with.

Looking at other people's submissions, it seems like I approached this problem somewhat "backwards".
I started off with the idea of writing a compiler, then stumbled upon the idea of optimizing the program by chunks.
Had I discovered the chunking idea first, writing a compiler would probably have been unnecessary in terms of performance.

Still, I learned a ton during this project and had a lot of fun, and I hope you gained something from this as well.
