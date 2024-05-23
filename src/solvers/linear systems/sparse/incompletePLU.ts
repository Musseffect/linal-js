import Vector from "../../../dense/vector";
import { PermutationMatrix, PermutationType } from "../../../permutationMatrix";
import { SparseMatrixCSR } from "../../../sparse/sparseMatrix";
import { SmallTolerance, assert } from "../../../utils";


export default class IncompletePLU {
    LU: SparseMatrixCSR = null;
    private p: PermutationMatrix = null;
    diagonalIndices: number[] = null;
    constructor(A?: SparseMatrixCSR) {
        if (A == null) return;
        this.factorize(A);
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
            //if (this.diagonalIndices[row] < this.LU.outerStart(row) && this.LU.innerIndex(this.diagonalIndices[row]) == rowIdx) 
            {
                nonZeroElements.push(1);
                innerIndices.push(row);
            }
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
    factorize(A: SparseMatrixCSR) {
        assert(A.isSquare(), "Square matrix is expected");
        let lu = A.clone();
        let p = PermutationMatrix.identity(A.numRows(), PermutationType.Row);

        let diagonalIndices: number[] = [];
        for (let row = 0; row < lu.numRows(); row++)
            diagonalIndices.push(lu.outerStart(row));

        for (let step = 0; step + 1 < lu.numRows(); step++) {
            let pivotRowIdx = -1;
            let pivot = 0;
            for (let rowIdx = step; rowIdx < lu.numRows(); rowIdx++) {
                let row = p.at(rowIdx);
                let start = diagonalIndices[row];
                if (lu.innerIndex(start) != step) continue;
                let curValue = lu.nonZeroElement(start);
                if (Math.abs(pivot) < Math.abs(curValue)) {
                    pivotRowIdx = rowIdx;
                    pivot = curValue;
                }
            }
            if (Math.abs(pivot) == 0) continue;
            p.swap(step, pivotRowIdx);
            let pivotRow = p.at(step);
            let start = diagonalIndices[pivotRow];
            let end = lu.outerStart(pivotRow + 1);
            let columnIdxMap: number[] = new Array(lu.numCols() - step - 1).fill(-1);
            for (let it = start + 1; it < end; ++it)
                columnIdxMap[lu.innerIndex(it) - step - 1] = it;
            for (let rowIdx = step + 1; rowIdx < lu.numRows(); rowIdx++) {
                let row = p.at(rowIdx);
                let firstIt = diagonalIndices[row];
                if (firstIt == lu.outerStart(row + 1) || lu.innerIndex(firstIt) != step) continue;
                ++diagonalIndices[row];

                let ratio = lu.nonZeroElement(firstIt) / pivot;
                lu.setNonZeroElement(firstIt, ratio);
                for (let colIt = firstIt + 1; colIt < lu.outerStart(row + 1); ++colIt) {
                    let col = lu.innerIndex(colIt);
                    let pivotRowIdx = columnIdxMap[col - step - 1];
                    if (pivotRowIdx < 0) continue;
                    lu.setNonZeroElement(colIt, lu.nonZeroElement(colIt) - ratio * lu.nonZeroElement(pivotRowIdx));
                }
            }
        }
        this.p = p;
        this.LU = this.P.permuteSparseMatrix(lu);
        this.diagonalIndices = [];
        for (let rowIdx = 0; rowIdx < A.numRows(); ++rowIdx) {
            let unpermutedIdx = this.p.at(rowIdx);
            this.diagonalIndices.push(this.LU.outerStart(rowIdx) + (diagonalIndices[unpermutedIdx] - lu.outerStart(unpermutedIdx)));
        }
    }
    solve(rhs: Vector): Vector {
        assert(rhs.size() == this.LU.numRows(), "Sizes don't match");
        let result = rhs.clone();
        this.p.permuteInplace(result);
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
            let diagIdx = this.diagonalIndices[row];
            for (let i = diagIdx + 1; i < this.LU.outerStart(row + 1); ++i)
                value -= result.get(this.LU.innerIndex(i)) * this.LU.nonZeroElement(i);
            let diagElement = 1;
            if (diagIdx < this.LU.outerStart(row + 1) && this.LU.innerIndex(diagIdx) == row)
                diagElement = this.LU.nonZeroElement(diagIdx);
            result.set(row, value / diagElement);
        }
        return result;
    }
};