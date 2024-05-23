import Matrix from "../../dense/denseMatrix";
import { PermutationType, PermutationMatrix } from "../../permutationMatrix";
import { assert, SmallestTolerance, SmallTolerance, swap } from "../../utils";
import Vector from "../../dense/vector";
import { TriMatrixType, TriMatrixView } from "../../dense/matrixView";
import { InsufficientRankException } from "./exceptions";

const SolverName = "'PartialPivLU'";

/* 
    LU decomposition with row permutations in the form PA = LU
    for Ax = b square problems
*/
export default class PartialPivLU {
    private lu: Matrix = null;
    private p: PermutationMatrix;
    private A: Matrix;
    constructor(A: Matrix | null = null) {
        this.factorize(A);
    }
    get L(): TriMatrixView {
        return new TriMatrixView(this.lu, TriMatrixType.lower, 1);
    }
    get U(): TriMatrixView {
        return new TriMatrixView(this.lu, TriMatrixType.upper);
    }
    get LU(): Matrix {
        return this.lu;
    }
    get P(): PermutationMatrix {
        return this.p;
    }
    public factorize(A: Matrix | null): void {
        this.A = A;
        this.lu = null;
        this.p = null;
        if (A == null)
            return;
        assert(A.isSquare(), "Non-square matrix");
        this.p = PermutationMatrix.identity(this.A.numRows(), PermutationType.Row);
        let lu: Matrix = this.A.clone();
        // todo (NI): check for rectangular matrices
        for (let step = 0; step + 1 < lu.numRows(); step++) {
            let maxPivotRow = step;
            let maxPivot = lu.get(step, step);
            for (let row = step; row < lu.numRows(); ++row) {
                let value = lu.get(row, step);
                if (Math.abs(value) > Math.abs(maxPivot)) {
                    maxPivotRow = row;
                    maxPivot = value;
                }
            }
            if (Math.abs(maxPivot) == 0) continue;

            this.p.swap(step, maxPivotRow);

            lu.swapRows(step, maxPivotRow);

            for (let row = step + 1; row < lu.numRows(); row++) {
                let ratio = lu.get(row, step) / maxPivot;
                for (let column = step + 1; column < lu.numCols(); column++)
                    lu.set(row, column, lu.get(row, column) - ratio * lu.get(step, column));
                lu.set(row, step, ratio);
            }
        }
        this.lu = lu;
    }
    solveInplace(rhs: Matrix | Vector): Matrix | Vector {
        let size = this.LU.width();
        if (rhs instanceof Matrix) {
            assert(rhs.height() == size, "Incompatible RHS");
            this.p.permuteInplace(rhs);
            for (let column = 0; column < rhs.width(); ++column) {
                for (let row = 0; row < size; ++row) {
                    let value = rhs.get(row, column);
                    for (let col = 0; col < row; ++col)
                        value -= rhs.get(col, column) * this.LU.get(row, col);
                    rhs.set(row, column, value);
                }
                for (let row = size - 1; row >= 0; --row) {
                    let value = rhs.get(row, column);
                    for (let col = row + 1; col < size; ++col)
                        value -= rhs.get(col, column) * this.LU.get(row, col);
                    rhs.set(row, column, value / this.LU.get(row, row));
                }
            }
        } else {
            assert(rhs.size() == size, "Incompatible RHS");
            this.p.permuteInplace(rhs);
            for (let row = 0; row < size; ++row) {
                let value = rhs.get(row);
                for (let col = 0; col < row; ++col)
                    value -= rhs.get(col) * this.LU.get(row, col);
                rhs.set(row, value);
            }
            for (let row = size - 1; row >= 0; --row) {
                let value = rhs.get(row);
                for (let col = row + 1; col < size; ++col)
                    value -= rhs.get(col) * this.LU.get(row, col);
                rhs.set(row, value / this.LU.get(row, row));
            }
            //console.log(`Rhs ${rhs.toString()}`);
        }
        return rhs;
    }
    solve(rhs: Matrix | Vector): Matrix | Vector {
        if (rhs instanceof Matrix)
            return this.solveInplace(rhs.clone());
        else
            return this.solveInplace(rhs.clone());
    }
    static solveMatrix(A: Matrix, B: Matrix): Matrix {
        assert(A.height() == B.height(), "Not determined system");
        assert(A.isSquare(), "Non-square matrix");
        let rank = B.height();

        let LU = A.clone();
        let Rhs = B.clone();

        let permutations = new Array(rank);
        for (let row = 0; row < rank; row++)
            permutations[row] = row;

        for (let step = 0; step < rank; step++) {
            let maxPivotRow = step;
            let maxPivotValue = LU.get(permutations[maxPivotRow], step);
            for (let row = step + 1; row < rank; row++) {
                let value = LU.get(permutations[row], step);
                if (Math.abs(value) > Math.abs(maxPivotValue)) {
                    maxPivotRow = row;
                    maxPivotValue = value;
                }
            }

            if (maxPivotValue == 0) continue;

            swap(permutations, step, maxPivotRow);

            let stepRow = permutations[step];

            for (let row = step + 1; row < rank; row++) {
                let curRow = permutations[row];
                let ratio = LU.get(curRow, step) / maxPivotValue;
                for (let column = step + 1; column < rank; column++)
                    LU.set(curRow, column, LU.get(curRow, column) - ratio * LU.get(stepRow, column));
                for (let bCol = 0; bCol < Rhs.width(); ++bCol)
                    Rhs.set(curRow, bCol, Rhs.get(curRow, bCol) - ratio * Rhs.get(stepRow, bCol));
                LU.set(curRow, step, 0);
            }
        }
        let x = Matrix.empty(rank, Rhs.width());
        for (let bCol = 0; bCol < Rhs.width(); ++bCol) {
            x.set(rank - 1, bCol, Rhs.get(permutations[rank - 1], bCol) / LU.get(permutations[rank - 1], rank - 1));
            for (let row = rank - 2; row > -1; row--) {
                let curRow = permutations[row];
                let k = 0.;
                for (let column = row + 1; column < rank; column++)
                    k = k + LU.get(curRow, column) * x.get(column, bCol);

                x.set(row, bCol, (Rhs.get(curRow, bCol) - k) / LU.get(curRow, row));
            }
        }
        return x;
    }
    isSingular(): boolean { return this.lu == null; }
    determinant(): number {
        if (this.isSingular()) return 0.0;
        let determinant = this.p.determinant();
        for (let i = 0; i < this.lu.width(); ++i)
            determinant *= this.lu.get(i, i);
        return determinant;
    }
    inverse(): Matrix | null {
        if (this.lu == null) return null;
        let result = Matrix.identity(this.lu.width());
        return this.solveInplace(result) as Matrix;
    }
    static solve(A: Matrix, b: Vector, tolerance: number = SmallTolerance): Vector {
        assert(A.width() == b.data.length, "Width of matrix isn't compatible with vector's length");
        assert(A.width() == A.height(), "Non-square matrix");

        let LU = A.clone();
        let rhs = b.clone();

        let rank = rhs.size();
        let permutations = new Array(rank);
        for (let row = 0; row < rank; row++)
            permutations[row] = row;

        for (let step = 0; step < rank; step++) {
            let maxPivotRow = step;
            let maxPivotValue = LU.get(permutations[maxPivotRow], step);
            for (let row = step + 1; row < rank; row++) {
                let value = LU.get(permutations[row], step);
                if (Math.abs(value) > Math.abs(maxPivotValue)) {
                    maxPivotRow = row;
                    maxPivotValue = value;
                }
            }

            swap(permutations, step, maxPivotRow);

            let stepRow = permutations[step];

            for (let row = step + 1; row < rank; row++) {
                let curRow = permutations[row];
                let ratio = LU.get(curRow, step) / maxPivotValue;
                for (let column = step + 1; column < rank; column++)
                    LU.set(curRow, column, LU.get(curRow, column) - ratio * LU.get(stepRow, column));
                rhs.set(curRow, rhs.get(curRow) - ratio * rhs.get(stepRow));
                LU.set(curRow, step, 0);
            }
        }
        let x = Vector.empty(rank);
        x.set(rank - 1, rhs.get(permutations[rank - 1]) / LU.get(permutations[rank - 1], rank - 1));
        for (let row = rank - 2; row > -1; row--) {
            let curRow = permutations[row];
            let k = 0.;
            for (let column = row + 1; column < rank; column++)
                k = k + LU.get(curRow, column) * x.get(column);

            x.set(row, (rhs.get(curRow) - k) / LU.get(curRow, row));
        }
        return x;
    }
}