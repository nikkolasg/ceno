use goldilocks::SmallField;
use std::collections::{HashMap, HashSet};

use crate::structs::{
    Cell, CellType, CircuitBuilder, ConstantType, GateType, TableData, TableType,
};

impl<F: SmallField> Cell<F> {
    pub fn new() -> Self {
        Self {
            layer: None,
            gates: vec![],
            assert_const: None,
            cell_type: None,
        }
    }
}

impl<F: SmallField> TableData<F> {
    pub fn new(witness_cell_type: CellType) -> Self {
        Self {
            table_items: vec![],
            table_items_const: vec![],
            input_items: vec![],
            challenge: None,
            witness_cell_type,
        }
    }
    pub fn add_table_item(&mut self, cell: usize) {
        self.table_items.push(cell);
    }
    pub fn add_input_item(&mut self, cell: usize) {
        self.input_items.push(cell);
    }
    pub fn add_table_item_const(&mut self, constant: F) {
        self.table_items_const.push(constant);
    }
    pub fn assign_challenge(&mut self, challenge: ConstantType<F>) {
        assert!(matches!(challenge, ConstantType::Challenge(_)));
        assert!(self.challenge.is_none());
        self.challenge = Some(challenge);
    }
}

impl<F> CircuitBuilder<F>
where
    F: SmallField,
{
    pub fn new() -> Self {
        let marked_cells = HashMap::new();
        Self {
            cells: vec![],
            marked_cells,
            tables: HashMap::new(),
            n_layers: None,
        }
    }
    pub fn create_cell(&mut self) -> usize {
        self.cells.push(Cell::new());
        self.cells.len() - 1
    }

    pub fn create_cells(&mut self, num: usize) -> Vec<usize> {
        self.cells.extend((0..num).map(|_| Cell::new()));
        (self.cells.len() - num..self.cells.len()).collect()
    }

    /// This is to mark the cell with special functionality.
    pub fn mark_cell(&mut self, cell_type: CellType, cell: usize) {
        self.cells[cell].cell_type = Some(cell_type);
    }

    /// This is to mark the cells with special functionality.
    pub fn mark_cells(&mut self, cell_type: CellType, cells: &[usize]) {
        cells.iter().for_each(|cell| {
            self.cells[*cell].cell_type = Some(cell_type);
        });
    }

    pub fn add_const(&mut self, out: usize, constant: ConstantType<F>) {
        let out_cell = &mut self.cells[out];
        out_cell.gates.push(GateType::AddC(constant));
    }

    pub fn add(&mut self, out: usize, in_0: usize, scaler: ConstantType<F>) {
        let out_cell = &mut self.cells[out];
        out_cell.gates.push(GateType::Add(in_0, scaler));
    }

    pub fn mul2(&mut self, out: usize, in_0: usize, in_1: usize, scaler: ConstantType<F>) {
        let out_cell = &mut self.cells[out];
        out_cell.gates.push(GateType::Mul2(in_0, in_1, scaler));
    }

    pub fn mul3(
        &mut self,
        out: usize,
        in_0: usize,
        in_1: usize,
        in_2: usize,
        scaler: ConstantType<F>,
    ) {
        let out_cell = &mut self.cells[out];
        out_cell
            .gates
            .push(GateType::Mul3(in_0, in_1, in_2, scaler));
    }

    pub fn assert_const(&mut self, out: usize, constant: F) {
        let out_cell = &mut self.cells[out];
        out_cell.assert_const = Some(constant);
    }

    /// Compute \sum_{i = 0}^{in_0_array.len()} scalers[i] * in_0_array[i] * in_1_array[i].
    pub fn inner_product(
        &mut self,
        out: usize,
        in_0_array: &[usize],
        in_1_array: &[usize],
        scaler_array: &[ConstantType<F>],
    ) {
        assert_eq!(in_0_array.len(), in_1_array.len());
        assert_eq!(in_0_array.len(), scaler_array.len());
        for ((in_0, in_1), scaler) in in_0_array.iter().zip(in_1_array).zip(scaler_array) {
            self.mul2(out, *in_0, *in_1, *scaler);
        }
    }
    pub fn inner_product_const(
        &mut self,
        out: usize,
        in_0_array: &[usize],
        scaler_array: &[ConstantType<F>],
    ) {
        assert_eq!(in_0_array.len(), scaler_array.len());
        for (in_0, scaler) in in_0_array.iter().zip(scaler_array) {
            self.add(out, *in_0, *scaler);
        }
    }
    pub fn product_of_array(&mut self, out: usize, in_array: &[usize]) {
        match in_array.len() {
            0 => {}
            1 => {
                if out != in_array[0] {
                    self.add(out, in_array[0], ConstantType::Field(F::ONE));
                }
            }
            _ => {
                let mut leaves = in_array.to_vec();
                while leaves.len() > 3 {
                    let mut new_leaves = vec![];
                    // Want to have the last slice length 3 instead of length 1.
                    for i in (0..leaves.len() - 1).step_by(2) {
                        if i + 3 != leaves.len() {
                            let new_leaf = self.create_cell();
                            self.mul2(
                                new_leaf,
                                leaves[i],
                                leaves[i + 1],
                                ConstantType::Field(F::ONE),
                            );
                            new_leaves.push(new_leaf);
                        } else {
                            let new_leaf = self.create_cell();
                            self.mul3(
                                new_leaf,
                                leaves[i],
                                leaves[i + 1],
                                leaves[i + 2],
                                ConstantType::Field(F::ONE),
                            );
                            new_leaves.push(new_leaf);
                        }
                    }
                    leaves = new_leaves;
                }
                if leaves.len() == 2 {
                    self.mul2(out, leaves[0], leaves[1], ConstantType::Field(F::ONE));
                } else {
                    self.mul3(
                        out,
                        leaves[0],
                        leaves[1],
                        leaves[2],
                        ConstantType::Field(F::ONE),
                    );
                }
            }
        }
    }

    pub fn frac_addition_of_array(
        &mut self,
        out_den: usize,
        out_num: usize,
        den_array: &[usize],
        num_array: &[usize],
    ) {
        assert!(den_array.len() == num_array.len());
        assert!(den_array.len() > 0);
        match den_array.len() {
            1 => {
                if out_den != den_array[0] {
                    self.add(out_den, den_array[0], ConstantType::Field(F::ONE));
                }
                if out_num != num_array[0] {
                    self.add(out_num, num_array[0], ConstantType::Field(F::ONE));
                }
            }
            _ => {
                let mut leaves_num = num_array.to_vec();
                let mut leaves_den = den_array.to_vec();
                while leaves_den.len() > 3 {
                    let mut new_leaves_den = vec![];
                    let mut new_leaves_num = vec![];
                    // Want to have the last slice length 3 instead of length 1.
                    for i in (0..leaves_den.len() - 1).step_by(2) {
                        if i + 3 != leaves_den.len() {
                            let new_leaf_den = self.create_cell();
                            self.mul2(
                                new_leaf_den,
                                leaves_den[i],
                                leaves_den[i + 1],
                                ConstantType::Field(F::ONE),
                            );
                            let new_leaf_num = self.create_cell();
                            self.mul2(
                                new_leaf_num,
                                leaves_num[i],
                                leaves_den[i + 1],
                                ConstantType::Field(F::ONE),
                            );
                            self.mul2(
                                new_leaf_num,
                                leaves_num[i + 1],
                                leaves_den[i],
                                ConstantType::Field(F::ONE),
                            );
                            new_leaves_num.push(new_leaf_num);
                            new_leaves_den.push(new_leaf_den);
                        } else {
                            let new_leaf_den = self.create_cell();
                            self.mul3(
                                new_leaf_den,
                                leaves_den[i],
                                leaves_den[i + 1],
                                leaves_den[i + 2],
                                ConstantType::Field(F::ONE),
                            );
                            let new_leaf_num = self.create_cell();
                            self.mul3(
                                new_leaf_num,
                                leaves_num[i],
                                leaves_den[i + 1],
                                leaves_den[i + 2],
                                ConstantType::Field(F::ONE),
                            );
                            self.mul3(
                                new_leaf_num,
                                leaves_num[i + 1],
                                leaves_den[i],
                                leaves_den[i + 2],
                                ConstantType::Field(F::ONE),
                            );
                            self.mul3(
                                new_leaf_num,
                                leaves_num[i + 2],
                                leaves_den[i],
                                leaves_den[i + 1],
                                ConstantType::Field(F::ONE),
                            );
                            new_leaves_num.push(new_leaf_num);
                            new_leaves_den.push(new_leaf_den);
                        }
                    }
                    leaves_num = new_leaves_num;
                    leaves_den = new_leaves_den;
                }
                if leaves_den.len() == 2 {
                    self.mul2(
                        out_den,
                        leaves_den[0],
                        leaves_den[1],
                        ConstantType::Field(F::ONE),
                    );
                    self.mul2(
                        out_num,
                        leaves_num[0],
                        leaves_den[1],
                        ConstantType::Field(F::ONE),
                    );
                    self.mul2(
                        out_num,
                        leaves_num[1],
                        leaves_den[0],
                        ConstantType::Field(F::ONE),
                    );
                } else {
                    self.mul3(
                        out_den,
                        leaves_den[0],
                        leaves_den[1],
                        leaves_den[2],
                        ConstantType::Field(F::ONE),
                    );
                    self.mul3(
                        out_num,
                        leaves_num[0],
                        leaves_den[1],
                        leaves_den[2],
                        ConstantType::Field(F::ONE),
                    );
                    self.mul3(
                        out_num,
                        leaves_num[1],
                        leaves_den[0],
                        leaves_den[2],
                        ConstantType::Field(F::ONE),
                    );
                    self.mul3(
                        out_num,
                        leaves_num[2],
                        leaves_den[0],
                        leaves_den[1],
                        ConstantType::Field(F::ONE),
                    );
                }
            }
        }
    }

    pub fn inv_addition_of_array(&mut self, out_den: usize, out_num: usize, den_array: &[usize]) {
        assert!(den_array.len() > 0);
        match den_array.len() {
            1 => {
                if out_den != den_array[0] {
                    self.add(out_den, den_array[0], ConstantType::Field(F::ONE));
                }
                self.add_const(out_num, ConstantType::Field(F::ONE));
            }
            _ => {
                let mut new_leaves_den = vec![];
                let mut new_leaves_num = vec![];
                // Want to have the last slice length 3 instead of length 1.
                for i in (0..den_array.len() - 1).step_by(2) {
                    if i + 3 != den_array.len() {
                        let new_leaf_den = self.create_cell();
                        self.mul2(
                            new_leaf_den,
                            den_array[i],
                            den_array[i + 1],
                            ConstantType::Field(F::ONE),
                        );
                        new_leaves_den.push(new_leaf_den);
                        let new_leaf_num = self.create_cell();
                        self.add(new_leaf_num, den_array[i], ConstantType::Field(F::ONE));
                        self.add(new_leaf_num, den_array[i + 1], ConstantType::Field(F::ONE));
                        new_leaves_num.push(new_leaf_num);
                    } else {
                        let new_leaf_den = self.create_cell();
                        self.mul3(
                            new_leaf_den,
                            den_array[i],
                            den_array[i + 1],
                            den_array[i + 2],
                            ConstantType::Field(F::ONE),
                        );
                        new_leaves_den.push(new_leaf_den);
                        let new_leaf_num = self.create_cell();
                        self.mul2(
                            new_leaf_num,
                            den_array[i + 1],
                            den_array[i + 2],
                            ConstantType::Field(F::ONE),
                        );
                        self.mul2(
                            new_leaf_num,
                            den_array[i],
                            den_array[i + 2],
                            ConstantType::Field(F::ONE),
                        );
                        self.mul2(
                            new_leaf_num,
                            den_array[i],
                            den_array[i + 1],
                            ConstantType::Field(F::ONE),
                        );
                        new_leaves_num.push(new_leaf_num);
                    }
                }
                self.frac_addition_of_array(out_den, out_num, &new_leaves_den, &new_leaves_num)
            }
        }
    }

    /// Input a table type and initialize a table. We can define an enum type to
    /// indicate the table and convert it to usize. This should throw an error
    /// if the type has been defined. Return the index of the witness.
    pub fn define_table_type(&mut self, table_type: TableType, witness_cell_type: CellType) {
        assert!(!self.tables.contains_key(&table_type));
        self.tables
            .insert(table_type, TableData::new(witness_cell_type));
    }

    pub fn add_input_item(&mut self, table_type: TableType, cell: usize) {
        assert!(self.tables.contains_key(&table_type));
        self.tables
            .get_mut(&table_type)
            .unwrap()
            .add_input_item(cell);
    }

    pub fn add_table_item(&mut self, table_type: TableType, cell: usize) {
        assert!(self.tables.contains_key(&table_type));
        self.tables
            .get_mut(&table_type)
            .unwrap()
            .add_table_item(cell);
    }

    pub fn add_table_item_const(&mut self, table_type: TableType, constant: F) {
        assert!(self.tables.contains_key(&table_type));
        self.tables
            .get_mut(&table_type)
            .unwrap()
            .add_table_item_const(constant);
    }

    /// Assign table challenge to the table.
    pub fn assign_table_challenge(&mut self, table_type: TableType, challenge: ConstantType<F>) {
        assert!(self.tables.contains_key(&table_type));
        self.tables
            .get_mut(&table_type)
            .unwrap()
            .assign_challenge(challenge);
    }

    /// Prepare the circuit. This is to build the circuit structure of lookup
    /// tables, and assign the layers and challenge levels to the cells.
    pub fn configure(&mut self) {
        // Build all lookup circuits.
        self.build_lookup_circuits();
        // Assign layers and challenge levels to all cells.
        for (cell_type, cells) in self.marked_cells.iter() {
            match cell_type {
                CellType::WireIn(_) => {
                    for cell in cells.iter() {
                        self.cells[*cell].layer = Some(0);
                    }
                }
                _ => {}
            }
        }

        for i in 0..self.cells.len() {
            if self.cells[i].layer.is_none() {
                let _ = self.assign_layer(i);
            }
            if *self.cells[i].layer.as_ref().unwrap() == 0 {
                assert!(self.cells[i].cell_type.is_some());
            }
        }

        // Compute the number of layers in the circuit.
        let max_layer_id = self
            .cells
            .iter()
            .map(|cell| cell.layer.unwrap())
            .max()
            .unwrap();

        self.n_layers = Some(max_layer_id + 1);

        for i in 0..self.cells.len() {
            if matches!(self.cells[i].cell_type, Some(CellType::WireOut(_))) {
                self.cells[i].layer = Some(max_layer_id);
            }
        }

        self.marked_cells.clear();
        for (i, cell) in self.cells.iter().enumerate() {
            if cell.cell_type.is_some() {
                let cell_type = cell.cell_type.unwrap();
                self.marked_cells
                    .entry(cell_type)
                    .or_insert(HashSet::new())
                    .insert(i);
            }
        }
    }

    /// Recursively assign layers to all cells.
    fn assign_layer(&mut self, id: usize) -> usize {
        if self.cells[id].gates.len() == 0 {
            self.cells[id].layer = Some(0);
            return 0;
        }
        if self.cells[id].layer.is_some() {
            return *self.cells[id].layer.as_ref().unwrap();
        }
        let mut prep_max_layer = 0;

        let cell = self.cells[id].clone();
        for gate in cell.gates.iter() {
            match gate {
                GateType::Add(in_0, _) => {
                    let prep_layer = self.assign_layer(*in_0);
                    prep_max_layer = std::cmp::max(prep_max_layer, prep_layer);
                }
                GateType::AddC(_) => prep_max_layer = std::cmp::max(prep_max_layer, 0),
                GateType::Mul2(in_0, in_1, _) => {
                    let prep_0_layer = self.assign_layer(*in_0);
                    let prep_1_layer = self.assign_layer(*in_1);
                    prep_max_layer =
                        std::cmp::max(prep_max_layer, std::cmp::max(prep_0_layer, prep_1_layer));
                }
                GateType::Mul3(in_0, in_1, in_2, _) => {
                    let prep_0_layer = self.assign_layer(*in_0);
                    let prep_1_layer = self.assign_layer(*in_1);
                    let prep_2_layer = self.assign_layer(*in_2);
                    prep_max_layer = std::cmp::max(
                        prep_max_layer,
                        std::cmp::max(prep_0_layer, std::cmp::max(prep_1_layer, prep_2_layer)),
                    );
                }
            }
        }
        self.cells[id].layer = Some(prep_max_layer + 1);
        prep_max_layer + 1
    }

    /// Build lookup circuit. The current method is [LogUp](https://eprint.iacr.org/2023/1284)
    fn build_lookup_circuits(&mut self) {
        let tables = self.tables.clone();
        for (_, table_data) in tables.iter() {
            assert!(table_data.challenge.is_some());

            let challenge = table_data.challenge.unwrap();
            let counts = self
                .create_cells(table_data.table_items.len() + table_data.table_items_const.len());
            self.mark_cells(table_data.witness_cell_type, &counts);

            // Compute (input_item + challenge)
            let input_items_with_challenge = self.create_cells(table_data.input_items.len());
            for (input_item_with_challenge, input_item) in
                (input_items_with_challenge.iter()).zip(table_data.input_items.iter())
            {
                self.add(
                    *input_item_with_challenge,
                    *input_item,
                    ConstantType::Field(F::ONE),
                );
                self.add_const(*input_item_with_challenge, challenge);
            }

            // Compute (table_item + challenge)
            let table_items_with_challenge = self.create_cells(counts.len());
            for (table_item_with_challenge, table_item) in
                (table_items_with_challenge.iter()).zip(table_data.table_items.iter())
            {
                self.add(
                    *table_item_with_challenge,
                    *table_item,
                    ConstantType::Field(F::ONE),
                );
                self.add_const(*table_item_with_challenge, challenge);
            }

            for (table_item_with_challenge, table_item) in (table_items_with_challenge
                .iter()
                .skip(table_data.table_items.len()))
            .zip(table_data.table_items_const.iter())
            {
                self.add_const(*table_item_with_challenge, ConstantType::Field(*table_item));
                self.add_const(*table_item_with_challenge, challenge);
            }

            // Construct fractional addition circuit.
            let input_inv_sum_den = self.create_cell();
            let input_inv_sum_num = self.create_cell();
            self.inv_addition_of_array(
                input_inv_sum_den,
                input_inv_sum_num,
                &input_items_with_challenge,
            );

            let table_inv_count_den = self.create_cell();
            let table_inv_count_num = self.create_cell();
            self.frac_addition_of_array(
                table_inv_count_den,
                table_inv_count_num,
                &table_items_with_challenge,
                &counts,
            );

            let diff_den = self.create_cell();
            let diff_num = self.create_cell();
            self.add(diff_den, input_inv_sum_den, ConstantType::Field(F::ONE));
            self.add(diff_den, table_inv_count_den, ConstantType::Field(-F::ONE));
            self.add(diff_num, input_inv_sum_num, ConstantType::Field(F::ONE));
            self.add(diff_num, table_inv_count_num, ConstantType::Field(-F::ONE));

            self.assert_const(diff_den, F::ZERO);
            self.assert_const(diff_num, F::ZERO);
        }
    }

    pub fn print_info(&self) {
        println!("The number of layers: {}", self.n_layers.as_ref().unwrap());
        println!("The number of cells: {}", self.cells.len());
        self.marked_cells.iter().for_each(|(ty, set)| {
            if let CellType::WireIn(i) = ty {
                println!("wire_in[{}] size: {}", i, set.len());
            }
        });
        self.marked_cells.iter().for_each(|(ty, set)| {
            if let CellType::WireOut(i) = ty {
                println!("wire_out[{}] size: {}", i, set.len());
            }
        });
        self.marked_cells.iter().for_each(|(ty, set)| {
            if let CellType::OtherInWitness(i) = ty {
                println!("other_in_witness[{}] size: {}", i, set.len());
            }
        });

        for (i, cell) in self.cells.iter().enumerate() {
            println!("Cell {}: {:?}", i, cell);
        }
    }

    pub fn is_wire_in(&self, cell: usize) -> bool {
        self.cells[cell]
            .cell_type
            .map_or(false, |x| matches!(x, CellType::WireIn(_)))
    }
}
