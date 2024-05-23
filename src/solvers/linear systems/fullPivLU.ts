import Matrix from "../../dense/denseMatrix";
import { PermutationType, PermutationMatrix } from "../../permutationMatrix";
import { TriMatrixType, TriMatrixView } from "../../dense/matrixView";
import { assert, SmallestTolerance, SmallTolerance, swap } from "../../utils";
import Vector from "../../dense/vector";
import { InsufficientRankException } from "./exceptions";

const SolverName = "'fullPivLU'";

/** LU decomposition with row and column permutations in the form PAQ = LU
 * 
 * L - lower unitriangular matrix
 * 
 * U - upper triangular matrix
 */

function findPivot(lu: Matrix, step: number, rowIdx: (i: number) => number, colIdx: (i: number) => number): { pivotValue: number, pivotRow: number, pivotColumn: number } {
    let rank = Math.min(lu.numRows(), lu.numCols());
    let pivotRow = step;
    let pivotColumn = step;
    let pivotValue = lu.get(rowIdx(pivotRow), colIdx(pivotColumn));
    for (let row = step; row < rank; ++row) {
        for (let column = step; column < rank; ++column) {
            let value = lu.get(rowIdx(row), colIdx(column));
            if (Math.abs(value) > Math.abs(pivotValue)) {
                pivotRow = row;
                pivotColumn = column;
                pivotValue = value;
            }
        }
    }
    return { pivotValue, pivotRow, pivotColumn };
}


export default class FullPivLU {
    protected lu: Matrix = null;
    protected p: PermutationMatrix;
    protected q: PermutationMatrix;
    protected A: Matrix;
    protected _rank: number;
    constructor(A: Matrix | null = null) {
        this.factorize(A);
    }
    public rank(): number {
        return this._rank;
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
    get Q(): PermutationMatrix {
        return this.q;
    }
    get P(): PermutationMatrix {
        return this.p;
    }
    isSingular(): boolean { return this.lu == null; }
    determinant(): number {
        if (this.isSingular()) return 0.0;
        let determinant = this.p.determinant() * this.q.determinant();
        for (let i = 0; i < this.lu.width(); ++i)
            determinant *= this.lu.get(i, i);
        return determinant;
    }
    public factorize(A: Matrix | null): void {
        this.A = A;
        this.lu = null;
        this.p = null;
        this.q = null;
        if (A == null)
            return;
        assert(A.isSquare(), "Non-square matrix");
        this.p = PermutationMatrix.identity(this.A.numRows(), PermutationType.Row);
        this.q = PermutationMatrix.identity(this.A.numCols(), PermutationType.Col);
        let lu: Matrix = this.A.clone();
        // todo (NI): check for rectangular matrices
        for (let step = 0; step + 1 < lu.numRows(); step++) {
            let maxPivotRow = step;
            let maxPivotColumn = step;
            let maxPivot = lu.get(step, step);
            for (let row = step; row < lu.numRows(); ++row) {
                for (let column = step; column < lu.numCols(); ++column) {
                    let value = lu.get(row, column);
                    if (Math.abs(value) > Math.abs(maxPivot)) {
                        maxPivotRow = row;
                        maxPivotColumn = column;
                        maxPivot = value;
                    }
                }
            }
            this._rank = step;
            if (Math.abs(maxPivot) == 0) break;

            this.p.swap(step, maxPivotRow);
            this.q.swap(step, maxPivotColumn);

            lu.swapRows(step, maxPivotRow);
            lu.swapColumns(step, maxPivotColumn);
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
        assert(this.A != null, "Factorization is not available");
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
        }
        return this.q.permuteInplace(rhs, PermutationType.Row);
    }
    solve(rhs: Matrix | Vector): Matrix | Vector {
        if (rhs instanceof Matrix)
            return this.solveInplace(rhs.clone());
        else
            return this.solveInplace(rhs.clone());
    }
    inverse(): Matrix | null {
        if (this.lu == null) return null;
        let result = Matrix.identity(this.lu.width());
        return this.solveInplace(result) as Matrix;
    }
    static solveMatrix(A: Matrix, B: Matrix, tolerance: number = SmallTolerance) {
        assert(A.height() == B.height(), "Not determined system");
        assert(A.isSquare(), "Non-square matrix");

        let LU = A.clone();
        let Rhs = B.clone();

        let rank = Rhs.height();
        let rowPermutations = new Array(rank);
        let columnPermutations = new Array(rank);
        for (let i = 0; i < rank; i++) {
            rowPermutations[i] = i;
            columnPermutations[i] = i;
        }
        let rowIdx = (idx: number) => rowPermutations[idx];
        let colIdx = (idx: number) => columnPermutations[idx];
        for (let step = 0; step < rank; step++) {
            let { pivotValue, pivotRow, pivotColumn } = findPivot(LU, step, rowIdx, colIdx);
            if (Math.abs(pivotValue) < tolerance)
                throw new InsufficientRankException(SolverName, step);

            swap(rowPermutations, step, pivotRow);
            swap(columnPermutations, step, pivotColumn);

            //console.log(`Step ${step}`);
            //console.log(`rowPermutations ${rowPermutations}, maxPivotRow ${pivotRow}`);
            //console.log(`columnPermutations ${columnPermutations}, maxPivotColumn ${pivotColumn}`);
            let stepRowIdx = rowIdx(step);

            //console.log(`Initial LU ${LU.toString()}`)
            const rowMat = new PermutationMatrix(rowPermutations, PermutationType.Row).toMatrix();
            const colMat = new PermutationMatrix(columnPermutations, PermutationType.Col).toMatrix();
            //console.log(`Initial permuted LU ${Matrix.mul(Matrix.mul(rowMat, LU), colMat).toString()}`)
            //console.log(`Initial Rhs ${Rhs.toString()}`)
            //console.log(`Initial permuted Rhs ${Matrix.mul(Matrix.mul(rowMat, Rhs), colMat).toString()}`);
            for (let row = step + 1; row < rank; row++) {
                let curRow = rowIdx(row);
                let ratio = LU.get(curRow, colIdx(step)) / pivotValue;
                for (let column = step + 1; column < rank; column++) {
                    let curColIdx = colIdx(column);
                    LU.set(curRow, curColIdx, LU.get(curRow, curColIdx) - ratio * LU.get(stepRowIdx, curColIdx));
                }
                for (let bCol = 0; bCol < Rhs.width(); ++bCol) {
                    Rhs.set(curRow, bCol, Rhs.get(curRow, bCol) - ratio * Rhs.get(stepRowIdx, bCol));
                }
                LU.set(curRow, colIdx(step), 0);
            }
            //console.log(`Result LU ${LU.toString()}`)
            //console.log(`Result permuted LU ${Matrix.mul(Matrix.mul(rowMat, LU), colMat).toString()}`)
            //console.log(`Result Rhs ${Rhs.toString()}`)
            //console.log(`Result permuted Rhs ${Matrix.mul(Matrix.mul(rowMat, Rhs), colMat).toString()}`);
        }

        let x = Matrix.empty(rank, Rhs.width());
        for (let bCol = 0; bCol < Rhs.width(); ++bCol) {
            let bColIdx = rowIdx(bCol);
            x.set(colIdx(rank - 1), bColIdx, Rhs.get(rowIdx(rank - 1), bColIdx) / LU.get(rowIdx(rank - 1), colIdx(rank - 1)));
            for (let row = rank - 2; row > -1; row--) {
                let curRowIdx = rowIdx(row);
                let k = 0.;
                for (let column = row + 1; column < rank; column++)
                    k = k + LU.get(curRowIdx, colIdx(column)) * x.get(colIdx(column), bColIdx);

                x.set(colIdx(row), bColIdx, (Rhs.get(curRowIdx, bColIdx) - k) / LU.get(curRowIdx, colIdx(row)));
            }
        }
        //console.log(`X ${x.toString()}`);
        return x;
    }
    static solve(A: Matrix, b: Vector, tolerance: number = SmallTolerance): Vector {
        assert(A.width() == b.data.length, "Width of matrix isn't compatible with vector's length");
        assert(A.width() == A.height(), "Non-square matrix");

        let LU = A.clone();
        let rhs = b.clone();

        let rank = rhs.size();
        let x = Vector.empty(rank);
        let rowPermutations = new Array(rank);
        let columnPermutations = new Array(rank);
        for (let i = 0; i < rank; i++) {
            rowPermutations[i] = i;
            columnPermutations[i] = i;
        }
        let rowIdx = (idx: number) => rowPermutations[idx];
        let colIdx = (idx: number) => columnPermutations[idx];
        for (let step = 0; step < rank; step++) {
            let { pivotValue, pivotRow, pivotColumn } = findPivot(LU, step, rowIdx, colIdx);
            if (Math.abs(pivotValue) < tolerance)
                throw new InsufficientRankException(SolverName, step);

            swap(rowPermutations, step, pivotRow);
            swap(columnPermutations, step, pivotColumn);

            for (let column = step + 1; column < rank; column++)
                LU.set(rowIdx(step), colIdx(column), LU.get(rowIdx(step), colIdx(column)) / pivotValue);
            rhs.set(rowIdx(step), rhs.get(rowIdx(step)) / pivotValue);
            LU.set(rowIdx(step), colIdx(step), 1);

            for (let row = step + 1; row < rank; row++) {
                let firstValue = LU.get(rowIdx(row), colIdx(step));
                for (let column = step + 1; column < rank; column++)
                    LU.set(rowIdx(row), colIdx(column), LU.get(rowIdx(row), colIdx(column)) - firstValue * LU.get(rowIdx(step), colIdx(column)));
                rhs.set(rowIdx(row), rhs.get(rowIdx(row)) - firstValue * rhs.get(rowIdx(step)));
                LU.set(rowIdx(row), colIdx(step), 0);
            }
        }

        x.set(colIdx(rank - 1), rhs.get(rowIdx(rank - 1)));
        for (let row = rank - 2; row > -1; row--) {
            let value = rhs.get(rowIdx(row));
            for (let column = row + 1; column < rank; column++)
                value -= LU.get(rowIdx(row), colIdx(column)) * x.get(colIdx(column));

            x.set(colIdx(row), value);
        }
        return x;
    }
}