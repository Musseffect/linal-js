import Matrix from "../../../dense/denseMatrix";
import Vector from "../../../dense/vector";
import { PermutationMatrix, PermutationType } from "../../../permutationMatrix";
import { SparseMatrixCSR } from "../../../sparse/sparseMatrix";
import { SmallTolerance, assert } from "../../../utils";


export default class PartialPivLU {
    private lu: SparseMatrixCSR = null;
    private p: PermutationMatrix;
    private diagonalIndices: number[];
    constructor(A: SparseMatrixCSR = null) {
        if (A == null) return;
        this.factorize(A);
    }
    public get LU(): SparseMatrixCSR {
        return this.lu;
    }
    public get P(): PermutationMatrix {
        return this.p;
    }
    public get L(): SparseMatrixCSR {
        assert(this.LU != null, "Null LU");
        let nonZeroElements: number[] = [];
        let outerStarts: number[] = [0];
        let innerIndices: number[] = [];
        for (let row = 0; row < this.LU.numRows(); ++row) {
            for (let i = this.LU.outerStart(row); i < this.diagonalIndices[row]; ++i) {
                nonZeroElements.push(this.LU.nonZeroElement(i));
                innerIndices.push(this.LU.innerIndex(i));
            }
            nonZeroElements.push(1);
            innerIndices.push(row);
            outerStarts.push(innerIndices.length);
        }
        return new SparseMatrixCSR(this.LU.numRows(), this.LU.numCols(), nonZeroElements, innerIndices, outerStarts);
    }
    public get U(): SparseMatrixCSR {
        assert(this.LU != null, "Null LU");
        let nonZeroElements: number[] = [];
        let outerStarts: number[] = [0];
        let innerIndices: number[] = [];
        for (let row = 0; row < this.LU.numRows(); ++row) {
            for (let i = this.diagonalIndices[row]; i < this.LU.outerStart(row + 1); ++i) {
                if (this.LU.innerIndex(i) < row) continue;
                nonZeroElements.push(this.LU.nonZeroElement(i));
                innerIndices.push(this.LU.innerIndex(i));
            }
            outerStarts.push(innerIndices.length);
        }
        return new SparseMatrixCSR(this.LU.numRows(), this.LU.numCols(), nonZeroElements, innerIndices, outerStarts);
    }
    determinant(): number {
        let determinant = 1;
        for (let i of this.diagonalIndices)
            determinant *= this.LU.nonZeroElement(i);
        return this.P.determinant() * determinant;
    }
    factorize(A: SparseMatrixCSR) {
        assert(A.isSquare(), "Square matrix is expected");

        let diagonalIndices = [];
        let permutationMatrix = PermutationMatrix.identity(A.numCols(), PermutationType.Row);
        let rows: number[][] = [];
        let colIndices: number[][] = [];
        for (let row = 0; row < A.numRows(); row++) {
            rows.push([]);
            colIndices.push([]);
            for (let colIt = A.outerStart(row); colIt < A.outerStart(row + 1); ++colIt) {
                rows[row].push(A.nonZeroElement(colIt));
                colIndices[row].push(A.innerIndex(colIt));
            }
        }
        // fill sparse L + U matrix row by row
        let nonZeroElements: number[] = [];
        let innerIndices: number[] = [];
        let outerStarts: number[] = [0];

        for (let step = 0; step + 1 < A.numRows(); step++) {
            let pivot = 0.0;
            let pivotRowIdx = 0;
            diagonalIndices.push(-1);
            for (let row = step; row < A.numRows(); ++row) {
                let permutedRow = permutationMatrix.at(row);
                let curRowElements = rows[permutedRow];
                for (let i = 0; i < curRowElements.length; ++i) {
                    let col = colIndices[permutedRow][i];
                    let value = curRowElements[i];
                    if (col == step && Math.abs(pivot) < Math.abs(value)) {
                        pivot = value;
                        pivotRowIdx = row;
                        diagonalIndices[step] = i;
                    }
                }
            }
            permutationMatrix.swap(step, pivotRowIdx);
            pivotRowIdx = permutationMatrix.at(step);
            diagonalIndices[step] += nonZeroElements.length;
            const pivotRowColIndices = colIndices[pivotRowIdx];
            const pivotRowElements = rows[pivotRowIdx];
            for (let i = 0; i < pivotRowElements.length; ++i) {
                let col = pivotRowColIndices[i];
                nonZeroElements.push(pivotRowElements[i]);
                innerIndices.push(col);
            }
            outerStarts.push(nonZeroElements.length);
            for (let row = step + 1; row < A.numRows(); ++row) {
                let permutedRowIdx = permutationMatrix.at(row);
                let elements: number[] = [];
                let cols: number[] = [];
                let pivotRowIt = diagonalIndices[step];
                let curRowIt = 0;
                const curRowElements = rows[permutedRowIdx];
                const curColIndices = colIndices[permutedRowIdx];
                while (curRowIt < curColIndices.length && curColIndices[curRowIt] < step) {
                    elements.push(curRowElements[curRowIt]);
                    cols.push(curColIndices[curRowIt]);
                    ++curRowIt;
                }
                if (curColIndices[curRowIt] > step)
                    continue;
                let ratio = curRowElements[curRowIt] / pivot;
                elements.push(ratio);
                cols.push(curColIndices[curRowIt]);
                ++curRowIt;
                assert(pivotRowIt < innerIndices.length && innerIndices[pivotRowIt] == step, "Incorrect iterator");
                ++pivotRowIt;
                while (pivotRowIt < innerIndices.length) {
                    const isValidIt2 = curRowIt < curColIndices.length;
                    let col1 = innerIndices[pivotRowIt];
                    let col2 = isValidIt2 ? curColIndices[curRowIt] : A.numRows();
                    let colIdx = 0;
                    let value1 = 0.0;
                    let value2 = 0.0;
                    if (col1 <= col2) {
                        colIdx = col1;
                        value1 = nonZeroElements[pivotRowIt];
                        ++pivotRowIt;
                    }
                    if (col2 <= col1) {
                        colIdx = col2;
                        value2 = curRowElements[curRowIt];
                        ++curRowIt;
                    }
                    let newValue = value2 - value1 * ratio;
                    if (Math.abs(newValue) != 0) {
                        elements.push(newValue);
                        cols.push(colIdx);
                    }
                }
                while (curRowIt < curColIndices.length) {
                    elements.push(curRowElements[curRowIt]);
                    cols.push(curColIndices[curRowIt]);
                    ++curRowIt;
                }
                // update cur row elements and indices
                colIndices[permutedRowIdx] = cols;
                rows[permutedRowIdx] = elements;
            }
            // clear pivot row
            colIndices[pivotRowIdx] = [];
            rows[pivotRowIdx] = [];
        }
        const pivotRow = permutationMatrix.at(A.numRows() - 1);
        const pivotRowColIndices = colIndices[pivotRow];
        const pivotRowElements = rows[pivotRow];
        for (let i = 0; i < pivotRowElements.length; ++i) {
            let col = pivotRowColIndices[i];
            nonZeroElements.push(pivotRowElements[i]);
            if (col == A.numRows() - 1)
                diagonalIndices.push(innerIndices.length);
            innerIndices.push(col);
        }
        outerStarts.push(nonZeroElements.length);
        if (diagonalIndices.length != A.numRows()) return;

        this.lu = new SparseMatrixCSR(A.numRows(), A.numCols(), nonZeroElements, innerIndices, outerStarts);
        this.p = permutationMatrix;
        this.diagonalIndices = diagonalIndices;
    }
    solve(rhs: Vector): Vector {
        assert(rhs.size() == this.LU.numRows(), "Sizes don't match");
        let result = rhs.clone();
        this.P.permuteInplace(result);
        // forward substitution
        for (let row = 0; row < this.LU.numRows(); ++row) {
            let value = result.get(row);
            for (let i = this.LU.outerStart(row); i < this.diagonalIndices[row]; ++i)
                value -= result.get(this.LU.innerIndex(i)) * this.LU.nonZeroElement(i);
            result.set(row, value);
        }
        // backward substitution
        for (let row = this.LU.numRows() - 1; row >= 0; --row) {
            let value = result.get(row);
            for (let i = this.diagonalIndices[row] + 1; i < this.LU.outerStart(row + 1); ++i)
                value -= result.get(this.LU.innerIndex(i)) * this.LU.nonZeroElement(i);
            result.set(row, value / this.LU.nonZeroElement(this.diagonalIndices[row]));
        }
        return result;
    }
    inverse(): Matrix {
        let inverse = Matrix.empty(this.LU.numCols(), this.LU.numRows());
        for (let y = 0; y < this.LU.numCols(); y++) {
            for (let x = 0; x < this.LU.numRows(); ++x) {
                let value = this.P.at(x) == y ? 1 : 0;
                for (let i = this.LU.outerStart(x); i < this.diagonalIndices[x]; ++i)
                    value -= this.LU.nonZeroElement(i) * inverse.get(this.LU.innerIndex(i), y);
                inverse.set(x, y, value);
            }
            for (let x = this.LU.numRows() - 1; x >= 0; --x) {
                let value = inverse.get(x, y);
                for (let i = this.diagonalIndices[x] + 1; i < this.LU.outerStart(x + 1); ++i)
                    value -= this.LU.nonZeroElement(i) * inverse.get(this.LU.innerIndex(i), y);
                inverse.set(x, y, value / this.LU.nonZeroElement(this.diagonalIndices[x]));
            }
        }
        return inverse;
    }
};