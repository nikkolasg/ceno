use std::time::Instant;

use ceno_zkvm::{
    instructions::riscv::arith::AddInstruction,
    scheme::prover::ZKVMProver,
    tables::{ProgramTableCircuit, RegTableCircuit},
};
use clap::Parser;
use const_env::from_env;

use ceno_emul::{
    ByteAddr, EmuContext, InsnKind::ADD, StepRecord, VMState, WordAddr, CENO_PLATFORM,
};
use ceno_zkvm::{
    scheme::{constants::MAX_NUM_VARIABLES, verifier::ZKVMVerifier},
    structs::{ZKVMConstraintSystem, ZKVMFixedTraces, ZKVMWitnesses},
    tables::U16TableCircuit,
};
use goldilocks::GoldilocksExt2;
use itertools::Itertools;
use mpcs::{Basefold, BasefoldRSParams, PolynomialCommitmentScheme};
use rand_chacha::ChaCha8Rng;
use sumcheck::util::is_power_of_2;
use tracing_flame::FlameLayer;
use tracing_subscriber::{fmt, layer::SubscriberExt, EnvFilter, Registry};
use transcript::Transcript;

#[from_env]
const RAYON_NUM_THREADS: usize = 8;

const PROGRAM_SIZE: usize = 512;
// For now, we assume registers
//  - x0 is not touched,
//  - x1 is initialized to 1,
//  - x2 is initialized to -1,
//  - x3 is initialized to loop bound.
// we use x4 to hold the acc_sum.
#[allow(clippy::unusual_byte_groupings)]
const ECALL_HALT: u32 = 0b_000000000000_00000_000_00000_1110011;
#[allow(clippy::unusual_byte_groupings)]
const PROGRAM_ADD_LOOP: [u32; PROGRAM_SIZE] = {
    let mut program: [u32; PROGRAM_SIZE] = [ECALL_HALT; PROGRAM_SIZE];
    (program[0], program[1], program[2]) = (
        // func7   rs2   rs1   f3  rd    opcode
        0b_0000000_00100_00001_000_00100_0110011, // add x4, x4, x1 <=> addi x4, x4, 1,
        0b_0000000_00011_00010_000_00011_0110011, // add x3, x3, x2 <=> addi x3, x3, -1
        0b_1_111111_00000_00011_001_1100_1_1100011, // bne x3, x0, -8
    );
    program
};
type ExampleProgramTableCircuit<E> = ProgramTableCircuit<E, PROGRAM_SIZE>;

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// start round
    #[arg(short, long, default_value_t = 8)]
    start: u8,

    /// end round
    #[arg(short, long, default_value_t = 22)]
    end: u8,
}

fn main() {
    let args = Args::parse();
    type E = GoldilocksExt2;
    type Pcs = Basefold<GoldilocksExt2, BasefoldRSParams, ChaCha8Rng>;

    let max_threads = {
        if !is_power_of_2(RAYON_NUM_THREADS) {
            #[cfg(not(feature = "non_pow2_rayon_thread"))]
            {
                panic!(
                    "add --features non_pow2_rayon_thread to enable unsafe feature which support non pow of 2 rayon thread pool"
                );
            }

            #[cfg(feature = "non_pow2_rayon_thread")]
            {
                use sumcheck::{local_thread_pool::create_local_pool_once, util::ceil_log2};
                let max_thread_id = 1 << ceil_log2(RAYON_NUM_THREADS);
                create_local_pool_once(1 << ceil_log2(RAYON_NUM_THREADS), true);
                max_thread_id
            }
        } else {
            RAYON_NUM_THREADS
        }
    };

    let (flame_layer, _guard) = FlameLayer::with_file("./tracing.folded").unwrap();
    let subscriber = Registry::default()
        .with(
            fmt::layer()
                .compact()
                .with_thread_ids(false)
                .with_thread_names(false),
        )
        .with(EnvFilter::from_default_env())
        .with(flame_layer.with_threads_collapsed(true));
    tracing::subscriber::set_global_default(subscriber).unwrap();

    // keygen
    let pcs_param = Pcs::setup(1 << MAX_NUM_VARIABLES).expect("Basefold PCS setup");
    let (pp, vp) = Pcs::trim(&pcs_param, 1 << MAX_NUM_VARIABLES).expect("Basefold trim");
    let mut zkvm_cs = ZKVMConstraintSystem::default();
    let add_config = zkvm_cs.register_opcode_circuit::<AddInstruction<E>>();
    let range_config = zkvm_cs.register_table_circuit::<U16TableCircuit<E>>();
    let reg_config = zkvm_cs.register_table_circuit::<RegTableCircuit<E>>();
    let prog_config = zkvm_cs.register_table_circuit::<ExampleProgramTableCircuit<E>>();

    for instance_num_vars in args.start..args.end {
        let step_loop = 1 << (instance_num_vars - 1); // 1 step in loop contribute to 2 add instance

        let mut zkvm_fixed_traces = ZKVMFixedTraces::default();
        zkvm_fixed_traces.register_opcode_circuit::<AddInstruction<E>>(&zkvm_cs);
        zkvm_fixed_traces.register_table_circuit::<U16TableCircuit<E>>(
            &zkvm_cs,
            range_config.clone(),
            &(),
        );
        // init vm.x1 = 1, vm.x2 = -1, vm.x3 = step_loop
        // vm.x4 += vm.x1
        zkvm_fixed_traces.register_table_circuit::<RegTableCircuit<E>>(
            &zkvm_cs,
            reg_config.clone(),
            &Some(
                vec![
                    0,         // x0
                    1,         // x1
                    u32::MAX,  // x2
                    step_loop, // x3
                ]
                .into_iter()
                .chain(std::iter::repeat(0u32))
                .take(32)
                .collect_vec(),
            ),
        );
        zkvm_fixed_traces.register_table_circuit::<ExampleProgramTableCircuit<E>>(
            &zkvm_cs,
            prog_config.clone(),
            &PROGRAM_ADD_LOOP,
        );

        let pk = zkvm_cs
            .clone()
            .key_gen::<Pcs>(pp.clone(), vp.clone(), zkvm_fixed_traces)
            .expect("keygen failed");
        let vk = pk.get_vk();

        // proving
        let prover = ZKVMProver::new(pk);
        let verifier = ZKVMVerifier::new(vk);

        let mut vm = VMState::new(CENO_PLATFORM);
        let pc_start = ByteAddr(CENO_PLATFORM.pc_start()).waddr();

        vm.init_register_unsafe(1usize, 1);
        vm.init_register_unsafe(2usize, u32::MAX); // -1 in two's complement
        vm.init_register_unsafe(3usize, step_loop);
        for (i, inst) in PROGRAM_ADD_LOOP.iter().enumerate() {
            vm.init_memory(pc_start + i, *inst);
        }
        let records = vm
            .iter_until_success()
            .collect::<Result<Vec<StepRecord>, _>>()
            .expect("vm exec failed")
            .into_iter()
            .filter(|record| record.insn().kind().1 == ADD)
            .collect::<Vec<_>>();
        tracing::info!("tracer generated {} ADD records", records.len());

        let mut zkvm_witness = ZKVMWitnesses::default();
        // assign opcode circuits
        zkvm_witness
            .assign_opcode_circuit::<AddInstruction<E>>(&zkvm_cs, &add_config, records)
            .unwrap();
        zkvm_witness.finalize_lk_multiplicities();
        // assign table circuits
        zkvm_witness
            .assign_table_circuit::<U16TableCircuit<E>>(&zkvm_cs, &range_config, &())
            .unwrap();
        // assign cpu register circuit
        let final_access = vm.tracer().final_accesses();
        zkvm_witness
            .assign_table_circuit::<RegTableCircuit<E>>(
                &zkvm_cs,
                &reg_config,
                &(0..32)
                    .map(|reg_id| {
                        let vma: WordAddr = CENO_PLATFORM.register_vma(reg_id).into();
                        (
                            vm.peek_register(reg_id),                     // final value
                            *final_access.get(&vma).unwrap_or(&0) as u32, // final cycle
                        )
                    })
                    .unzip(),
            )
            .unwrap();
        zkvm_witness
            .assign_table_circuit::<ExampleProgramTableCircuit<E>>(
                &zkvm_cs,
                &prog_config,
                &PROGRAM_ADD_LOOP.len(),
            )
            .unwrap();

        let timer = Instant::now();

        let transcript = Transcript::new(b"riscv");
        let zkvm_proof = prover
            .create_proof(zkvm_witness, max_threads, transcript)
            .expect("create_proof failed");

        println!(
            "AddInstruction::create_proof, instance_num_vars = {}, time = {}",
            instance_num_vars,
            timer.elapsed().as_secs_f64()
        );

        let transcript = Transcript::new(b"riscv");
        assert!(
            verifier
                .verify_proof(zkvm_proof, transcript)
                .expect("verify proof return with error"),
        );
    }
}
