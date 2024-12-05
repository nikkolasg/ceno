use anyhow::Result;
use ceno_emul::{ByteAddr, CENO_PLATFORM, EmuContext, StepRecord, VMState};

// TODO(Matthias): much of this is copied from `test_elf.rs` in Ceno.  These are generally useful
// functions, so we should make them available for importing from the library, instead of copying
// them here.
//
// So in the end, this file should just have a really simple main.
// See how sproll-evm does it with SP1.

fn run(state: &mut VMState) -> Result<Vec<StepRecord>> {
    let steps = state.iter_until_halt().collect::<Result<Vec<_>>>()?;
    eprintln!("Emulator ran for {} steps.", steps.len());
    Ok(steps)
}

const WORD_SIZE: usize = 4;
const INFO_OUT_ADDR: u32 = 0xC000_0000;

fn read_all_messages(state: &VMState) -> Vec<Vec<u8>> {
    let mut all_messages = Vec::new();
    let mut word_offset = 0;
    loop {
        let out = read_message(state, word_offset);
        if out.is_empty() {
            break;
        }
        word_offset += out.len().div_ceil(WORD_SIZE) as u32 + 1;
        all_messages.push(out);
    }
    all_messages
}

fn read_message(state: &VMState, word_offset: u32) -> Vec<u8> {
    let out_addr = ByteAddr(INFO_OUT_ADDR).waddr() + word_offset;
    let byte_len = state.peek_memory(out_addr);
    let word_len_up = byte_len.div_ceil(4);

    let mut info_out = Vec::with_capacity(WORD_SIZE * word_len_up as usize);
    for i in 1..1 + word_len_up {
        let value = state.peek_memory(out_addr + i);
        info_out.extend_from_slice(&value.to_le_bytes());
    }
    info_out.truncate(byte_len as usize);
    info_out
}

fn main() {
    let mut state = VMState::new_from_elf(CENO_PLATFORM, elf::ELF).expect("Failed to load ELF");
    let steps = run(&mut state).expect("Failed to run the program");
    println!("Ran for {} steps.", steps.len());
    let all_messages = read_all_messages(&state);
    for msg in &all_messages {
        print!("{}", String::from_utf8_lossy(msg));
    }
}
