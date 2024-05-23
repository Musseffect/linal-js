import AbstractDenseMatrix from "./abstractDenseMatrix";
import { assert } from "../utils";


class SymmetricMatrix extends AbstractDenseMatrix {
    constructor(data: number[], size: number) {
        super(data, size, size);
        assert(this.data.length * 2 == size * (size + 1), "Incorrect data array size");
    }
    override index(row: number, col: number): number {
        if (this.isRowMajor()) {
            let i = row;
            let j = col;
            if (row > col) {
                j = row;
                i = col;
            }
            return (j - i) + this.numRows() * i - (i * (i - 1)) / 2;
        } else {
            let i = col;
            let j = row;
            if (col > row) {
                i = row;
                j = col;
            }
        }
    }
}
/*
class SymmetricMatrixView extends MatrixView {
    upper: boolean;

}*/