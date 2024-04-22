use ff::Field;
use gkr::structs::{Circuit, LayerWitness};
use goldilocks::SmallField;
use paste::paste;
use revm_interpreter::Record;
use simple_frontend::structs::{CircuitBuilder, MixedCell};
use singer_utils::{
    chip_handler::{
        BytecodeChipOperations, CalldataChipOperations, GlobalStateChipOperations, OAMOperations,
        ROMOperations, RangeChipOperations, StackChipOperations,
    },
    constants::OpcodeType,
    copy_clock_from_record, copy_operand_timestamp_from_record, copy_operand_u64_from_record,
    copy_pc_add_from_record, copy_pc_from_record, copy_stack_memory_ts_add_from_record,
    copy_stack_top_from_record, copy_stack_ts_add_from_record, copy_stack_ts_from_record,
    copy_stack_ts_lt_from_record, register_witness,
    structs::{PCUInt, RAMHandler, ROMHandler, StackUInt, TSUInt, UInt64},
    uint::{u2fvec, UIntAddSub, UIntCmp},
};
use std::sync::Arc;

use crate::{error::ZKVMError, CircuitWiresIn};

use super::{ChipChallenges, InstCircuit, InstCircuitLayout, Instruction, InstructionGraph};

impl<F: SmallField> InstructionGraph<F> for CalldataloadInstruction {
    type InstType = Self;
}

pub struct CalldataloadInstruction;

register_witness!(
    CalldataloadInstruction,
    phase0 {
        pc => PCUInt::N_OPRAND_CELLS,
        stack_ts => TSUInt::N_OPRAND_CELLS,
        memory_ts => TSUInt::N_OPRAND_CELLS,
        ts => TSUInt::N_OPRAND_CELLS,
        stack_top => 1,
        clk => 1,

        pc_add => UIntAddSub::<PCUInt>::N_NO_OVERFLOW_WITNESS_UNSAFE_CELLS,
        stack_ts_add => UIntAddSub::<TSUInt>::N_NO_OVERFLOW_WITNESS_CELLS,

        data => StackUInt::N_OPRAND_CELLS,
        offset => UInt64::N_OPRAND_CELLS,
        old_stack_ts => TSUInt::N_OPRAND_CELLS,
        old_stack_ts_lt => UIntCmp::<TSUInt>::N_WITNESS_CELLS
    }
);

impl<F: SmallField> Instruction<F> for CalldataloadInstruction {
    const OPCODE: OpcodeType = OpcodeType::CALLDATALOAD;
    const NAME: &'static str = "CALLDATALOAD";
    fn construct_circuit(challenges: ChipChallenges) -> Result<InstCircuit<F>, ZKVMError> {
        let mut circuit_builder = CircuitBuilder::new();
        let (phase0_wire_id, phase0) = circuit_builder.create_witness_in(Self::phase0_size());
        let mut ram_handler = RAMHandler::new(&challenges);
        let mut rom_handler = ROMHandler::new(&challenges);

        // State update
        let pc = PCUInt::try_from(&phase0[Self::phase0_pc()])?;
        let stack_ts = TSUInt::try_from(&phase0[Self::phase0_stack_ts()])?;
        let memory_ts = &phase0[Self::phase0_memory_ts()];
        let stack_top = phase0[Self::phase0_stack_top().start];
        let stack_top_expr = MixedCell::Cell(stack_top);
        let clk = phase0[Self::phase0_clk().start];
        let clk_expr = MixedCell::Cell(clk);
        ram_handler.state_in(
            &mut circuit_builder,
            pc.values(),
            stack_ts.values(),
            &memory_ts,
            stack_top,
            clk,
        );

        let next_pc =
            ROMHandler::add_pc_const(&mut circuit_builder, &pc, 1, &phase0[Self::phase0_pc_add()])?;
        let next_stack_ts = rom_handler.add_ts_with_const(
            &mut circuit_builder,
            &stack_ts,
            1,
            &phase0[Self::phase0_stack_ts_add()],
        )?;

        ram_handler.state_out(
            &mut circuit_builder,
            next_pc.values(),
            next_stack_ts.values(),
            &memory_ts,
            stack_top_expr,
            clk_expr.add(F::BaseField::ONE),
        );

        // Range check for stack top
        rom_handler.range_check_stack_top(
            &mut circuit_builder,
            stack_top_expr.sub(F::BaseField::from(1)),
        )?;

        // Stack pop offset from the stack.
        let old_stack_ts = TSUInt::try_from(&phase0[Self::phase0_old_stack_ts()])?;
        let offset = &phase0[Self::phase0_offset()];
        ram_handler.stack_pop(
            &mut circuit_builder,
            stack_top_expr.sub(F::BaseField::ONE),
            old_stack_ts.values(),
            offset,
        );
        UIntCmp::<TSUInt>::assert_lt(
            &mut circuit_builder,
            &mut rom_handler,
            &old_stack_ts,
            &stack_ts,
            &phase0[Self::phase0_old_stack_ts_lt()],
        )?;

        // CallDataLoad check (offset, data)
        let data = &phase0[Self::phase0_data()];
        rom_handler.calldataload(&mut circuit_builder, offset, data);

        // Stack push data to the stack.
        ram_handler.stack_push(
            &mut circuit_builder,
            stack_top_expr.sub(F::BaseField::ONE),
            stack_ts.values(),
            data,
        );

        // Bytecode table (pc, CalldataLoad)
        rom_handler.bytecode_with_pc_opcode(
            &mut circuit_builder,
            pc.values(),
            <Self as Instruction<F>>::OPCODE,
        );

        let (ram_load_id, ram_store_id) = ram_handler.finalize(&mut circuit_builder);
        let rom_id = rom_handler.finalize(&mut circuit_builder);
        circuit_builder.configure();

        let outputs_wire_id = [ram_load_id, ram_store_id, rom_id];

        Ok(InstCircuit {
            circuit: Arc::new(Circuit::new(&circuit_builder)),
            layout: InstCircuitLayout {
                chip_check_wire_id: outputs_wire_id,
                phases_wire_id: vec![phase0_wire_id],
                ..Default::default()
            },
        })
    }

    fn generate_wires_in(record: &Record) -> CircuitWiresIn<F> {
        let mut wire_values = vec![F::ZERO; Self::phase0_size()];
        copy_pc_from_record!(wire_values, record);
        copy_stack_ts_from_record!(wire_values, record);
        copy_stack_top_from_record!(wire_values, record);
        copy_clock_from_record!(wire_values, record);
        copy_pc_add_from_record!(wire_values, record);
        copy_stack_ts_add_from_record!(wire_values, record);

        // The operand offset is assumed to be 64 bit, although stored in a U256
        copy_operand_u64_from_record!(wire_values, record, phase0_offset, 0);
        copy_stack_ts_lt_from_record!(wire_values, record);

        vec![LayerWitness {
            instances: vec![wire_values],
        }]
    }
}

#[cfg(test)]
mod test {
    use ark_std::test_rng;
    use core::ops::Range;
    use ff::Field;
    use gkr::structs::LayerWitness;
    use goldilocks::{Goldilocks, GoldilocksExt2, SmallField};
    use itertools::Itertools;
    use revm_interpreter::Interpreter;
    use simple_frontend::structs::CellId;
    use singer_utils::constants::{OpcodeType, RANGE_CHIP_BIT_WIDTH};
    use singer_utils::structs::TSUInt;
    use std::collections::BTreeMap;
    use std::time::Instant;
    use transcript::Transcript;

    use crate::instructions::{
        CalldataloadInstruction, ChipChallenges, Instruction, InstructionGraph,
        SingerCircuitBuilder,
    };
    use crate::scheme::GKRGraphProverState;
    use crate::test::{
        get_uint_params, test_opcode_circuit, test_opcode_circuit_with_witness_vector, u2vec,
    };
    use crate::{CircuitWiresIn, SingerGraphBuilder, SingerParams};

    impl CalldataloadInstruction {
        #[inline]
        fn phase0_idxes_map() -> BTreeMap<String, Range<CellId>> {
            let mut map = BTreeMap::new();

            map.insert("phase0_pc".to_string(), Self::phase0_pc());
            map.insert("phase0_stack_ts".to_string(), Self::phase0_stack_ts());
            map.insert("phase0_memory_ts".to_string(), Self::phase0_memory_ts());
            map.insert("phase0_ts".to_string(), Self::phase0_ts());
            map.insert("phase0_stack_top".to_string(), Self::phase0_stack_top());
            map.insert("phase0_clk".to_string(), Self::phase0_clk());
            map.insert("phase0_pc_add".to_string(), Self::phase0_pc_add());
            map.insert(
                "phase0_stack_ts_add".to_string(),
                Self::phase0_stack_ts_add(),
            );
            map.insert("phase0_data".to_string(), Self::phase0_data());
            map.insert("phase0_offset".to_string(), Self::phase0_offset());
            map.insert(
                "phase0_old_stack_ts".to_string(),
                Self::phase0_old_stack_ts(),
            );
            map.insert(
                "phase0_old_stack_ts_lt".to_string(),
                Self::phase0_old_stack_ts_lt(),
            );

            map
        }
    }

    #[test]
    fn test_calldataload_construct_circuit() {
        let challenges = ChipChallenges::default();

        let phase0_idx_map = CalldataloadInstruction::phase0_idxes_map();
        let phase0_witness_size = CalldataloadInstruction::phase0_size();

        #[cfg(feature = "witness-count")]
        {
            println!("CALLDATALOAD: {:?}", &phase0_idx_map);
            println!("CALLDATALOAD witness_size: {:?}", phase0_witness_size);
        }

        // initialize general test inputs associated with push1
        let inst_circuit = CalldataloadInstruction::construct_circuit(challenges).unwrap();

        #[cfg(feature = "test-dbg")]
        println!("{:?}", inst_circuit);

        let mut phase0_values_map = BTreeMap::<String, Vec<Goldilocks>>::new();
        phase0_values_map.insert("phase0_pc".to_string(), vec![Goldilocks::from(1u64)]);
        phase0_values_map.insert("phase0_ts".to_string(), vec![Goldilocks::from(1u64)]);
        phase0_values_map.insert("phase0_stack_ts".to_string(), vec![Goldilocks::from(3u64)]);
        phase0_values_map.insert("phase0_memory_ts".to_string(), vec![Goldilocks::from(1u64)]);
        phase0_values_map.insert(
            "phase0_stack_top".to_string(),
            vec![Goldilocks::from(100u64)],
        );
        phase0_values_map.insert("phase0_clk".to_string(), vec![Goldilocks::from(1u64)]);
        phase0_values_map.insert(
            "phase0_pc_add".to_string(),
            vec![], // carry is 0, may test carry using larger values in PCUInt
        );
        phase0_values_map.insert(
            "phase0_stack_ts_add".to_string(),
            vec![
                Goldilocks::from(4u64), // first TSUInt::N_RANGE_CHECK_CELLS = 1*(56/16) = 4 cells are range values, stack_ts + 1 = 4
                Goldilocks::from(0u64),
                Goldilocks::from(0u64),
                Goldilocks::from(0u64),
                // no place for carry
            ],
        );
        phase0_values_map.insert(
            "phase0_old_stack_ts".to_string(),
            vec![Goldilocks::from(2u64)],
        );
        let m: u64 = (1 << get_uint_params::<TSUInt>().1) - 1;
        let range_values = u2vec::<{ TSUInt::N_RANGE_CHECK_CELLS }, RANGE_CHIP_BIT_WIDTH>(m);
        phase0_values_map.insert(
            "phase0_old_stack_ts_lt".to_string(),
            vec![
                Goldilocks::from(range_values[0]),
                Goldilocks::from(range_values[1]),
                Goldilocks::from(range_values[2]),
                Goldilocks::from(range_values[3]),
                Goldilocks::from(1u64), // borrow
            ],
        );
        phase0_values_map.insert(
            "phase0_data".to_string(),
            vec![
                Goldilocks::from(7u64),
                Goldilocks::from(6u64),
                Goldilocks::from(5u64),
                Goldilocks::from(4u64),
                Goldilocks::from(3u64),
                Goldilocks::from(2u64),
                Goldilocks::from(1u64),
                Goldilocks::from(0u64),
            ],
        );
        phase0_values_map.insert("phase0_offset".to_string(), vec![Goldilocks::from(1u64)]);

        let circuit_witness_challenges = vec![
            Goldilocks::from(2),
            Goldilocks::from(2),
            Goldilocks::from(2),
        ];

        let _circuit_witness = test_opcode_circuit(
            &inst_circuit,
            &phase0_idx_map,
            phase0_witness_size,
            &phase0_values_map,
            circuit_witness_challenges,
        );
    }

    /// Test the correctness of the witness generated by the interpreter
    /// for the calldataload instruction circuit.
    #[test]
    fn test_interpreter_for_calldataload_circuit() {
        let challenges = ChipChallenges::default();
        let bytecode = [
            OpcodeType::PUSH1 as u8,
            0,
            OpcodeType::CALLDATALOAD as u8,
            OpcodeType::PUSH1 as u8,
            45,
            OpcodeType::ADD as u8,
        ];
        let records = Interpreter::<Goldilocks>::execute(&bytecode, &[1, 2]);
        // The interpreter has ensured the last instruction must be stop, so here
        // we actually executed five instructions.
        assert_eq!(records.len(), 5);
        let circuit_wires_in: Vec<Vec<Goldilocks>> =
            CalldataloadInstruction::generate_wires_in(&records[1])[0]
                .instances
                .clone();
        assert_eq!(circuit_wires_in.len(), 1);
        assert_eq!(
            circuit_wires_in[0].len(),
            CalldataloadInstruction::phase0_size()
        );

        #[cfg(feature = "witness-count")]
        {
            println!("Calldataload: {:?}", &phase0_idx_map);
            println!("Calldataload witness_size: {:?}", phase0_witness_size);
        }

        // initialize general test inputs associated with push1
        let inst_circuit = CalldataloadInstruction::construct_circuit(challenges).unwrap();

        #[cfg(feature = "test-dbg")]
        println!("{:?}", inst_circuit);

        // The actual challenges used is:
        // challenges
        //  { ChallengeConst { challenge: 1, exp: i }: [Goldilocks(c^i)] }
        let c: u64 = 6;
        let circuit_witness_challenges = vec![
            Goldilocks::from(c),
            Goldilocks::from(c),
            Goldilocks::from(c),
        ];

        let circuit_witness = test_opcode_circuit_with_witness_vector(
            &inst_circuit,
            circuit_wires_in,
            circuit_witness_challenges,
        );

        // check the correctness of add operation
        // stack_push = RLC([stack_ts=3, RAMType::Stack=0, stack_top=98, result=0,1,0,0,0,0,0,0, len=11])
        //            = 3 (stack_ts) + c^2 * 98 (stack_top) + c^4 * 1 + c^11
        // let add_stack_push_wire_id = inst_circuit.layout.chip_check_wire_id[1].unwrap().0;
        // let add_stack_push =
        //     &circuit_witness.witness_out_ref()[add_stack_push_wire_id as usize].instances[0][1];
        // let add_stack_push_value: u64 = 3 + c.pow(2_u32) * 98 + c.pow(4u32) * 1 + c.pow(11_u32);
        // assert_eq!(*add_stack_push, Goldilocks::from(add_stack_push_value));
    }

    fn bench_calldataload_instruction_helper<F: SmallField>(instance_num_vars: usize) {
        let chip_challenges = ChipChallenges::default();
        let circuit_builder =
            SingerCircuitBuilder::<F>::new(chip_challenges).expect("circuit builder failed");
        let mut singer_builder = SingerGraphBuilder::<F>::new();

        let mut rng = test_rng();
        let size = CalldataloadInstruction::phase0_size();
        let phase0: CircuitWiresIn<F::BaseField> = vec![LayerWitness {
            instances: (0..(1 << instance_num_vars))
                .map(|_| {
                    (0..size)
                        .map(|_| F::BaseField::random(&mut rng))
                        .collect_vec()
                })
                .collect_vec(),
        }];

        let real_challenges = vec![F::random(&mut rng), F::random(&mut rng)];

        let timer = Instant::now();

        let _ = CalldataloadInstruction::construct_graph_and_witness(
            &mut singer_builder.graph_builder,
            &mut singer_builder.chip_builder,
            &circuit_builder.insts_circuits
                [<CalldataloadInstruction as Instruction<F>>::OPCODE as usize],
            vec![phase0],
            &real_challenges,
            1 << instance_num_vars,
            &SingerParams::default(),
        )
        .expect("gkr graph construction failed");

        let (graph, wit) = singer_builder.graph_builder.finalize_graph_and_witness();

        println!(
            "CalldataloadInstruction::construct_graph_and_witness, instance_num_vars = {}, time = {}",
            instance_num_vars,
            timer.elapsed().as_secs_f64()
        );

        let point = vec![F::random(&mut rng), F::random(&mut rng)];
        let target_evals = graph.target_evals(&wit, &point);

        let mut prover_transcript = &mut Transcript::new(b"Singer");

        let timer = Instant::now();
        let _ = GKRGraphProverState::prove(&graph, &wit, &target_evals, &mut prover_transcript)
            .expect("prove failed");
        println!(
            "CalldataloadInstruction::prove, instance_num_vars = {}, time = {}",
            instance_num_vars,
            timer.elapsed().as_secs_f64()
        );
    }

    #[test]
    fn bench_calldataload_instruction() {
        bench_calldataload_instruction_helper::<GoldilocksExt2>(10);
    }
}
