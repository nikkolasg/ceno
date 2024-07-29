use ff::Field;
use ff_ext::ExtensionField;
use gkr::structs::Circuit;
use itertools::izip;
use paste::paste;
use simple_frontend::structs::{CircuitBuilder, MixedCell};
use singer_utils::chip_handler::bytecode::BytecodeChip;
use singer_utils::chip_handler::global_state::GlobalStateChip;
use singer_utils::chip_handler::ram_handler::RAMHandler;
use singer_utils::chip_handler::range::RangeChip;
use singer_utils::chip_handler::rom_handler::ROMHandler;
use singer_utils::chip_handler::stack::StackChip;
use singer_utils::{
    constants::OpcodeType,
    register_witness,
    uint::constants::AddSubConstants,
    structs::{PCUInt, StackUInt, TSUInt},
};
use std::cell::RefCell;
use std::rc::Rc;
use std::{collections::BTreeMap, sync::Arc};

use crate::error::ZKVMError;

use super::{ChipChallenges, InstCircuit, InstCircuitLayout, Instruction, InstructionGraph};

pub struct JumpiInstruction;

impl<E: ExtensionField> InstructionGraph<E> for JumpiInstruction {
    type InstType = Self;
}

register_witness!(
    JumpiInstruction,
    phase0 {
        pc => PCUInt::N_OPERAND_CELLS ,
        stack_ts => TSUInt::N_OPERAND_CELLS,
        memory_ts => TSUInt::N_OPERAND_CELLS,
        stack_top => 1,
        clk => 1,

        old_stack_ts_dest => TSUInt::N_OPERAND_CELLS,
        old_stack_ts_dest_lt => AddSubConstants::<TSUInt>::N_WITNESS_CELLS,
        old_stack_ts_cond => TSUInt::N_OPERAND_CELLS,
        old_stack_ts_cond_lt => AddSubConstants::<TSUInt>::N_WITNESS_CELLS,

        dest_values => StackUInt::N_OPERAND_CELLS,
        cond_values => StackUInt::N_OPERAND_CELLS,
        cond_values_inv => StackUInt::N_OPERAND_CELLS,
        cond_non_zero_or_inv => 1,

        pc_add => AddSubConstants::<PCUInt>::N_NO_OVERFLOW_WITNESS_UNSAFE_CELLS,
        pc_plus_1_opcode => 1
    }
);

impl<E: ExtensionField> Instruction<E> for JumpiInstruction {
    const OPCODE: OpcodeType = OpcodeType::JUMPI;
    const NAME: &'static str = "JUMPI";
    fn construct_circuit(challenges: ChipChallenges) -> Result<InstCircuit<E>, ZKVMError> {
        let mut circuit_builder = CircuitBuilder::new();
        let (phase0_wire_id, phase0) = circuit_builder.create_witness_in(Self::phase0_size());

        let mut rom_handler = Rc::new(RefCell::new(ROMHandler::new(challenges.clone())));
        let mut ram_handler = Rc::new(RefCell::new(RAMHandler::new(challenges.clone())));

        // instantiate chips
        let global_state_chip = GlobalStateChip::new(ram_handler.clone());
        let mut range_chip = RangeChip::new(rom_handler.clone());
        let stack_chip = StackChip::new(ram_handler.clone());
        let bytecode_chip = BytecodeChip::new(rom_handler.clone());

        // State update
        let pc = PCUInt::try_from(&phase0[Self::phase0_pc()])?;
        let stack_ts = TSUInt::try_from(&phase0[Self::phase0_stack_ts()])?;
        let memory_ts = &phase0[Self::phase0_memory_ts()];
        let stack_top = phase0[Self::phase0_stack_top().start];
        let stack_top_expr = MixedCell::Cell(stack_top);
        let clk = phase0[Self::phase0_clk().start];
        let clk_expr = MixedCell::Cell(clk);
        global_state_chip.state_in(
            &mut circuit_builder,
            pc.values(),
            stack_ts.values(),
            &memory_ts,
            stack_top,
            clk,
        );

        // Range check stack_top - 2
        range_chip.range_check_stack_top(
            &mut circuit_builder,
            stack_top_expr.sub(E::BaseField::from(2)),
        )?;

        // Pop the destination pc from stack.
        let dest_values = &phase0[Self::phase0_dest_values()];
        let dest_stack_addr = stack_top_expr.sub(E::BaseField::ONE);

        let old_stack_ts_dest = (&phase0[Self::phase0_old_stack_ts_dest()]).try_into()?;
        TSUInt::assert_lt(
            &mut circuit_builder,
            &mut range_chip,
            &old_stack_ts_dest,
            &stack_ts,
            &phase0[Self::phase0_old_stack_ts_dest_lt()],
        )?;
        stack_chip.pop(
            &mut circuit_builder,
            dest_stack_addr,
            old_stack_ts_dest.values(),
            dest_values,
        );

        // Pop the condition from stack.
        let cond_values = &phase0[Self::phase0_cond_values()];
        let old_stack_ts_cond = (&phase0[Self::phase0_old_stack_ts_cond()]).try_into()?;
        TSUInt::assert_lt(
            &mut circuit_builder,
            &mut range_chip,
            &old_stack_ts_cond,
            &stack_ts,
            &phase0[Self::phase0_old_stack_ts_cond_lt()],
        )?;

        stack_chip.pop(
            &mut circuit_builder,
            stack_top_expr.sub(E::BaseField::from(2)),
            old_stack_ts_cond.values(),
            cond_values,
        );

        // Execution, cond_values_non_zero[i] = [cond_values[i] != 0]
        let cond_values_inv = &phase0[Self::phase0_cond_values_inv()];
        let mut cond_values_non_zero = Vec::new();
        for (val, wit) in izip!(cond_values, cond_values_inv) {
            cond_values_non_zero.push(range_chip.non_zero(&mut circuit_builder, *val, *wit)?);
        }
        // cond_non_zero = [summation of cond_values_non_zero[i] != 0]
        let non_zero_or = circuit_builder.create_cell();
        cond_values_non_zero
            .iter()
            .for_each(|x| circuit_builder.add(non_zero_or, *x, E::BaseField::ONE));
        let cond_non_zero_or_inv = phase0[Self::phase0_cond_non_zero_or_inv().start];
        let cond_non_zero =
            range_chip.non_zero(&mut circuit_builder, non_zero_or, cond_non_zero_or_inv)?;

        // If cond_non_zero, next_pc = dest, otherwise, pc = pc + 1
        let pc_add_1 = &phase0[Self::phase0_pc_add()];
        let pc_plus_1 = RangeChip::add_pc_const(&mut circuit_builder, &pc, 1, pc_add_1)?;
        let pc_plus_1 = pc_plus_1.values();
        let next_pc = circuit_builder.create_cells(PCUInt::N_OPERAND_CELLS);
        for i in 0..PCUInt::N_OPERAND_CELLS {
            circuit_builder.select(next_pc[i], pc_plus_1[i], dest_values[i], cond_non_zero);
        }

        // State out
        global_state_chip.state_out(
            &mut circuit_builder,
            &next_pc,
            stack_ts.values(), // Because there is no stack push.
            memory_ts,
            stack_top_expr.sub(E::BaseField::from(2)),
            clk_expr.add(E::BaseField::ONE),
        );

        // Bytecode check for (pc, jumpi)
        bytecode_chip.bytecode_with_pc_opcode(
            &mut circuit_builder,
            pc.values(),
            <Self as Instruction<E>>::OPCODE,
        );

        // If cond_non_zero, next_opcode = JUMPDEST, otherwise, opcode = pc + 1 opcode
        let pc_plus_1_opcode = phase0[Self::phase0_pc_plus_1_opcode().start];
        let next_opcode = circuit_builder.create_cell();
        circuit_builder.sel_mixed(
            next_opcode,
            pc_plus_1_opcode.into(),
            MixedCell::Constant(E::BaseField::from(OpcodeType::JUMPDEST as u64)),
            cond_non_zero,
        );

        // Bytecode check for (next_pc, next_opcode)
        bytecode_chip.bytecode_with_pc_byte(&mut circuit_builder, &next_pc, next_opcode);

        let (ram_load_id, ram_store_id) = ram_handler.borrow_mut().finalize(&mut circuit_builder);
        let rom_id = rom_handler.borrow_mut().finalize(&mut circuit_builder);
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
}
