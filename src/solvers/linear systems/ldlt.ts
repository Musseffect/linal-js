import Matrix from "../../dense/denseMatrix";
import { DiagonalMatrixView, TriMatrixType, TriMatrixView } from "../../dense/matrixView";
import { SmallestTolerance, assert } from "../../utils";
import Vector from "../../dense/vector";


class TriangularMatrix {
    isLower: boolean;
    data: number[];
    size: number;
    constructor(size: number, isLower: boolean = true) {
        this.size = size;
        this.data = new Array(size * (size + 1) / 2).fill(0);
        this.isLower = isLower;
    }
    get(row: number, col: number) {
        let i = this.isLower ? row : col;
        let j = this.isLower ? col : row;
        if (j > i) return 0.0;
        return this.data[(i * (i + 1)) / 2 + j];
    }
    set(row: number, col: number, value: number) {
        let i = this.isLower ? row : col;
        let j = this.isLower ? col : row;
        assert(i >= j, "Invalid index");
        this.data[(i * (i + 1)) / 2 + j] = value;
    }
    width(): number {
        return this.size;
    }
    height(): number {
        return this.size;
    }
    determinant(): number {
        let result = 1;
        for (let i = 0; i < this.size; ++i)
            result *= this.get(i, i);
        return result;
    }
    toMatrix(): Matrix {
        let result = Matrix.empty(this.size, this.size);
        for (let i = 0; i < this.size; ++i) {
            for (let j = 0; j < this.size; ++j) {
                result.set(i, j, this.get(i, j));
            }
        }
        return result;
    }
}

// todo (NI): optimize for large matrices
// todo (NI): pivoting
// todo (NI): block LDLT for indefinite decomposition and LTLT
export default class LDLT {
    ldlt: Matrix = null;
    A: Matrix = null;
    _tolerance: number = SmallestTolerance;
    constructor(A: Matrix | null = null, tolerance: number = SmallestTolerance) {
        this._tolerance = tolerance;
        this.factorize(A);
    }
    factorize(A: Matrix | null) {
        this.A = A;
        if (A == null) return;
        assert(A.isSquare(), "Non-square matrix");
        const size = A.width();
        let ldlt = A.clone();
        for (let row = 0; row < size; ++row) {
            for (let col = 0; col < row; ++col) {
                let value = A.get(row, col);
                for (let i = 0; i < col; ++i) {
                    value -= ldlt.get(row, i) * ldlt.get(col, i) * ldlt.get(i, i);
                }
                ldlt.set(row, col, value / ldlt.get(col, col));
            }
            let value = A.get(row, row);
            for (let i = 0; i < row; ++i)
                value -= ldlt.get(row, i) * ldlt.get(row, i) * ldlt.get(i, i);
            ldlt.set(row, row, value);
        }
        this.ldlt = ldlt;
    }
    solveInplace(rhs: Matrix | Vector): Matrix | Vector {
        const size = this.ldlt.width();
        if (rhs instanceof Matrix) {
            assert(rhs.height() == size, "Incompatible RHS");
            for (let column = 0; column < rhs.width(); ++column) {
                for (let row = 0; row < size; ++row) {
                    let value = rhs.get(row, column);
                    for (let col = 0; col < row; ++col)
                        value -= this.ldlt.get(row, col) * rhs.get(col, column);
                    rhs.set(row, column, value);
                }
                for (let row = 0; row < size; ++row)
                    rhs.set(row, column, rhs.get(row, column) / this.ldlt.get(row, row));
                for (let row = size - 1; row >= 0; --row) {
                    let value = rhs.get(row, column);
                    // pay attention to ldlt.get(col, row) order of indices
                    for (let col = row + 1; col < size; ++col)
                        value -= this.ldlt.get(col, row) * rhs.get(col, column);
                    rhs.set(row, column, value);
                }
            }
        } else {
            assert(rhs.size() == size, "Incompatible RHS");
            for (let row = 0; row < size; ++row) {
                let value = rhs.get(row);
                for (let col = 0; col < row; ++col)
                    value -= this.ldlt.get(row, col) * rhs.get(col);
                rhs.set(row, value);
            }
            for (let row = 0; row < size; ++row)
                rhs.set(row, rhs.get(row) / this.ldlt.get(row, row));
            for (let row = size - 1; row >= 0; --row) {
                let value = rhs.get(row);
                for (let col = row + 1; col < size; ++col)
                    value -= this.ldlt.get(col, row) * rhs.get(col);
                rhs.set(row, value);
            }
        }
        return rhs;
    }
    solve(rhs: Matrix | Vector): Matrix | Vector {
        if (rhs instanceof Matrix)
            return this.solveInplace(rhs.clone());
        else
            return this.solveInplace(rhs.clone());
    }
    inverse(): Matrix | null {
        if (this.LDLT == null) return null;
        let result = Matrix.identity(this.ldlt.width());
        return this.solveInplace(result) as Matrix;
    }
    get LDLT(): Matrix {
        return this.ldlt;
    }
    get L(): TriMatrixView {
        return new TriMatrixView(this.ldlt, TriMatrixType.lower);
    }
    get LT(): TriMatrixView {
        return new TriMatrixView(this.ldlt, TriMatrixType.upper);
    }
    get D(): DiagonalMatrixView {
        return new DiagonalMatrixView(this.ldlt);
    }
    static solve(A: Matrix, rhs: Vector) {
        const rank = A.width();
        let L = new TriangularMatrix(rank, true);
        for (let row = 0; row < rank; ++row) {
            for (let col = 0; col < row; ++col) {
                let value = A.get(row, col);
                for (let i = 0; i < col; ++i) {
                    value -= L.get(row, i) * L.get(col, i) * L.get(i, i);
                }
                L.set(row, col, value / L.get(col, col));
            }
            let value = A.get(row, row);
            for (let i = 0; i < row; ++i)
                value -= L.get(row, i) * L.get(row, i) * L.get(row, row);
            L.set(row, row, value);
        }
        let y = Vector.empty(rank);
        for (let row = 0; row < rank; ++row) {
            let value = rhs.get(row);
            for (let column = 0; column < row; ++column)
                value -= L.get(row, column) * y.get(column);
            y.set(row, value / L.get(row, row));
        }
        let x = Vector.empty(rank);
        for (let row = rank - 1; row >= 0; --row) {
            let value = y.get(row);
            for (let column = row + 1; column < rank; ++column)
                value -= L.get(column, row) * x.get(column);
            x.set(row, value);
        }
        return x;
    }
    determinant(): number {
        let result = 1;
        for (let i = 0; i < this.ldlt.width(); ++i)
            result *= this.ldlt.get(i, i);
        return result;
    }
}