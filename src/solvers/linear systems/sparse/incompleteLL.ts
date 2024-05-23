import Vector from "../../../dense/vector";
import { SparseMatrixCSR } from "../../../sparse/sparseMatrix";
import { assert } from "../../../utils";


export default class IncompleteLL {
    private ll: SparseMatrixCSR = null;
    /**
     * Lower triangular sparse matrix with sparsity pattern equivalent to the pattern of lower triangular part of initial matrix
     */
    get L(): SparseMatrixCSR {
        return this.ll;
    }
    constructor(A: SparseMatrixCSR | null) {
        if (A != null)
            this.factorize(A);
    }
    factorize(A: SparseMatrixCSR) {
        assert(A.isSquare(), "Square matrix is expected");
        let nonZeroElements: number[] = [];
        let innerIndices: number[] = [];
        let outerStarts: number[] = [0];
        let curColIndices: number[] = new Array(A.numRows()).fill(0);
        for (let step = 0; step < A.numRows(); ++step) {
            curColIndices[step] = outerStarts[step];
            for (let i = A.outerStart(step); i < A.outerStart(step + 1) && A.innerIndex(i) <= step; ++i) {
                nonZeroElements.push(A.nonZeroElement(i));
                innerIndices.push(A.innerIndex(i));
            }
            outerStarts.push(nonZeroElements.length);
        }
        let ll = new SparseMatrixCSR(A.numRows(), A.numCols(), nonZeroElements, innerIndices, outerStarts);
        for (let step = 0; step < ll.numRows(); ++step) {
            let pivot = 0.0;
            let pivotIt = curColIndices[step];
            if (ll.innerIndex(pivotIt) == step && pivotIt < ll.outerStart(step + 1)) {
                pivot = ll.nonZeroElement(pivotIt);
            }
            // assert(ll.innerIndex(curColIndices[step]) > step || curColIndices[step] == ll.outerStart(step + 1), "DEV: Remove this");
            if (pivot <= 0) return;
            pivot = Math.sqrt(pivot);
            ll.setNonZeroElement(pivotIt, pivot);
            let pivotCol: number[] = new Array(ll.numRows() - step - 1).fill(0);
            for (let row = step + 1; row < ll.numRows(); ++row) {
                let it = curColIndices[row];
                if (ll.innerIndex(it) != step || it >= ll.outerStart(row + 1)) continue;
                ++curColIndices[row];
                // assert(ll.innerIndex(curColIndices[row]) > step || curColIndices[row] == ll.outerStart(row + 1), "DEV: Remove this");
                let value = ll.nonZeroElement(it) / pivot;
                pivotCol[row - step - 1] = value;
                ll.setNonZeroElement(it, value);
            }
            for (let row = step + 1; row < ll.numRows(); ++row) {
                for (let colIt = curColIndices[row]; colIt < ll.outerStart(row + 1); ++colIt) {
                    let col = ll.innerIndex(colIt);
                    ll.setNonZeroElement(colIt, ll.nonZeroElement(colIt) - pivotCol[row - step - 1] * pivotCol[col - step - 1]);
                }
            }
        }
        this.ll = ll;
    }
    solve(rhs: Vector): Vector {
        // forward substitution
        // y = L * rhs
        let result = rhs.clone();
        for (let row = 0; row < this.ll.numRows(); ++row) {
            let value = result.get(row);
            for (let colIt = this.L.outerStart(row); colIt < this.L.outerStart(row + 1) - 1; ++colIt)
                value -= this.ll.nonZeroElement(colIt) * result.get(this.ll.innerIndex(colIt));
            result.set(row, value / this.ll.nonZeroElement(this.L.outerStart(row + 1) - 1));
        }
        // backward substitution
        // x = LT * y
        for (let row = this.ll.numRows() - 1; row >= 0; --row) {
            let value = result.get(row) / this.ll.nonZeroElement(this.L.outerStart(row + 1) - 1);
            result.set(row, value);
            for (let colIt = this.L.outerStart(row); colIt < this.L.outerStart(row + 1) - 1; ++colIt) {
                let col = this.ll.innerIndex(colIt);
                result.set(col, result.get(col) - value * this.ll.nonZeroElement(colIt));
            }
        }
        return result;
    }
}