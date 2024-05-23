import Matrix from "../../dense/denseMatrix";
import { TriMatrixType, TriMatrixView } from "../../dense/matrixView";
import { assert, near, SmallestTolerance, SmallTolerance } from "../../utils";
import Vector from "../../dense/vector";
import { NotPositiveDefiniteMatrixException } from "./exceptions";

const SolverName = "'Cholesky'";

export default class LLT {
    llt: Matrix | null = null;
    // todo (NI): replace with TriangularMatrix or SymmetricMatrix
    A: Matrix | null = null;
    _tolerance: number = SmallestTolerance;
    constructor(A: Matrix | null = null, tolerance: number = SmallestTolerance) {
        this._tolerance = tolerance;
        this.factorize(A);
    }
    factorize(A: Matrix | null): void {
        this.A = A;
        this.llt = null;
        if (A == null)
            return;
        assert(A.isSquare(), "Non-square matrix");
        assert(A.isSymmetric(), "Non-symmetric");
        let llt = A.clone();
        let size = A.width();
        for (let row = 0; row < size; ++row) {
            if (llt.get(row, row) <= 0) continue;
            for (let column = 0; column <= row; ++column) {
                let value = llt.get(row, column);
                for (let k = 0; k < column; ++k)
                    value -= llt.get(row, k) * llt.get(column, k);
                if (row == column) {
                    if (value < 0)
                        return;
                    value = Math.sqrt(value);
                    llt.set(row, row, value);
                }
                else {
                    value = value / llt.get(column, column);
                    llt.set(row, column, value);
                    llt.set(column, row, value);
                }
            }
        }
        this.llt = llt;
    };
    get L(): TriMatrixView {
        return new TriMatrixView(this.llt, TriMatrixType.lower);
    }
    get LT(): TriMatrixView {
        return new TriMatrixView(this.llt, TriMatrixType.upper);
    }
    get LLT(): Matrix {
        return this.llt;
    }
    solveInplace(rhs: Matrix | Vector): Matrix | Vector {
        const size = this.llt.width();
        if (rhs instanceof Matrix) {
            assert(rhs.height() == size, "Incompatible RHS");
            for (let column = 0; column < rhs.width(); ++column) {
                for (let row = 0; row < size; ++row) {
                    let value = rhs.get(row, column);
                    for (let col = 0; col < row; ++col)
                        value -= this.llt.get(row, col) * rhs.get(col, column);
                    value /= this.llt.get(row, row);
                    rhs.set(row, column, value);
                }
                for (let row = size - 1; row >= 0; --row) {
                    let value = rhs.get(row, column);
                    for (let col = row + 1; col < size; ++col)
                        value -= this.llt.get(row, col) * rhs.get(col, column);
                    value /= this.llt.get(row, row);
                    rhs.set(row, column, value);
                }
            }
            return rhs;
        } else {
            assert(rhs.size() == size, "Incompatible RHS");
            for (let row = 0; row < size; ++row) {
                let value = rhs.get(row);
                for (let col = 0; col < row; ++col)
                    value -= this.llt.get(row, col) * rhs.get(col);
                value /= this.llt.get(row, row);
                rhs.set(row, value);
            }
            for (let row = size - 1; row >= 0; --row) {
                let value = rhs.get(row);
                for (let col = row + 1; col < size; ++col)
                    value -= this.llt.get(row, col) * rhs.get(col);
                value /= this.llt.get(row, row);
                rhs.set(row, value);
            }
            return rhs;
        }
    }
    solve(rhs: Matrix | Vector): Matrix | Vector {
        if (rhs instanceof Matrix)
            return this.solveInplace(rhs.clone());
        else
            return this.solveInplace(rhs.clone());
    }
    inverse(): Matrix | null {
        if (this.llt == null) return null;
        let result = Matrix.identity(this.llt.width());
        return this.solveInplace(result) as Matrix;
    }
    // Solve inplace
    static solve(A: Matrix, b: Vector) {
        assert(A.width() == b.data.length, "Width of matrix isn't compatible with vector's length");
        assert(A.isSquare(), "Non-square matrix");

        let rank = b.size();
        for (let row = 0; row < rank; ++row) {
            for (let column = 0; column <= row; ++column) {
                let value = A.get(row, column);
                for (let k = 0; k < column; ++k)
                    value -= A.get(row, k) * A.get(column, k);
                if (row == column) {
                    if (value < 0.0) throw new NotPositiveDefiniteMatrixException(SolverName);
                    value = Math.sqrt(value);
                    A.set(row, column, value);
                }
                else {
                    value = value / A.get(column, column);
                    A.set(row, column, value);
                    A.set(column, row, value);
                }
            }
        }
        let y = Vector.empty(rank);
        for (let row = 0; row < rank; ++row) {
            let value = b.get(row);
            for (let column = 0; column < row; ++column)
                value -= A.get(row, column) * y.get(column);
            value /= A.get(row, row);
            y.set(row, value);
        }
        let x = Vector.empty(rank);
        for (let row = rank - 1; row >= 0; --row) {
            let value = y.get(row);
            for (let column = row + 1; column < rank; ++column)
                value -= A.get(row, column) * x.get(column);
            value /= A.get(row, row);
            x.set(row, value);
        }
        return x;
    }
    determinant(): number {
        let result = 1;
        for (let i = 0; i < this.llt.width(); ++i)
            result *= this.llt.get(i, i) * this.llt.get(i, i);
        return result;
    }
}