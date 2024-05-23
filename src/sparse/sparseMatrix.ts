import AbstractMatrix from "../abstractMatrix";
import Matrix from "../dense/denseMatrix";
import { SparseVector, SparseVectorElement } from "./sparseVector";
import Triplet from "./triplet";
import { assert, SmallTolerance, SmallestTolerance } from "../utils";
import Vector from "../dense/vector";
import { SparseMatrixTriplets } from "./sparseMatrixTriplets";

export class CellRef {
    private matrix: SparseMatrixCSR;
    private idx: number;
    private rowIdx: number;
    private colIdx: number;
    constructor(matrix: SparseMatrixCSR, idx: number, rowIdx: number, colIdx: number) {
        this.matrix = matrix;
        this.idx = idx;
        this.rowIdx = rowIdx;
        this.colIdx = colIdx;
    }
    outerIdx(): number {
        return this.idx;
    }
    row(): number {
        return this.rowIdx;
    }
    column(): number {
        return this.colIdx;
    }
    get(): number {
        return this.matrix.nonZeroElement(this.idx);
    }
    set(value: number): void {
        this.matrix.setNonZeroElement(this.idx, value);
    }
}
// list of lists
// dictionary of keys
// sorted triples

interface RowValuePair {
    value1: number;
    value2: number;
    colIdx: number;
}


export class SparseMatrixTwinRowIterator {
    private m1: SparseMatrixCSR;
    private m2: SparseMatrixCSR;
    private rowIdx: number;
    private it1: number;
    private it2: number;
    constructor(m1: SparseMatrixCSR, m2: SparseMatrixCSR, rowIdx: number) {
        this.m1 = m1;
        this.m2 = m2;
        this.rowIdx = rowIdx;
        this.it1 = this.m1.outerStart(this.rowIdx);
        this.it2 = this.m2.outerStart(this.rowIdx);
    }
    isDone(): boolean {
        const isValidIt1 = this.it1 < this.m1.outerStart(this.rowIdx + 1);
        const isValidIt2 = this.it2 < this.m2.outerStart(this.rowIdx + 1);
        return !isValidIt1 && !isValidIt2;
    }
    advance(): RowValuePair {
        const isValidIt1 = this.it1 < this.m1.outerStart(this.rowIdx + 1);
        const isValidIt2 = this.it2 < this.m2.outerStart(this.rowIdx + 1);
        let col1 = isValidIt1 ? this.m1.innerIndex(this.it1) : this.m1.width();
        let col2 = isValidIt2 ? this.m2.innerIndex(this.it2) : this.m2.width();
        let colIdx = 0;
        let value1 = 0.0;
        let value2 = 0.0;
        if (col1 <= col2) {
            colIdx = col1;
            value1 = this.m1.nonZeroElement(this.it1);
            ++this.it1;
        }
        if (col2 <= col1) {
            colIdx = col2;
            value2 = this.m2.nonZeroElement(this.it2);
            ++this.it2;
        }
        return { value1, value2, colIdx };
    }
}

// iterate values in a row
export class SparseMatrixRowIterator {
    private m: SparseMatrixCSR;
    private rowIdx: number;
    private it: number;
    constructor(m: SparseMatrixCSR, rowIdx: number) {
        this.m = m;
        this.rowIdx = rowIdx;
        this.it = this.m.outerStart(this.rowIdx);
    }
    isDone(): boolean {
        return this.it >= this.m.outerStart(this.rowIdx + 1);
    }
    advance(): { value: number, colIdx: number } {
        let colIdx = this.m.innerIndex(this.it);
        let value = this.m.nonZeroElement(this.it);
        ++this.it;
        return { value, colIdx };
    }
}

// todo (NI): CSC
// todo (NI): track state of nonZeroElements array with state:number variable
export class SparseMatrixCSR extends AbstractMatrix {
    // values sizeof(NNZ)
    protected nonZeroElements: number[];
    // column indices sizeof(NNZ)
    protected innerIndices: number[];
    // row starts
    protected outerStarts: number[];

    constructor(numRows: number, numCols: number, nonZeroElements: number[], innerIndices: number[], outerStarts: number[]) {
        super(numRows, numCols);
        this.nonZeroElements = nonZeroElements;
        this.innerIndices = innerIndices;
        this.outerStarts = outerStarts;
    }
    static empty(numRows: number, numCols: number): SparseMatrixCSR {
        let nonZeroElements: number[] = [];
        let innerIndices: number[] = [];
        let outerStarts = new Array(numRows + 1).fill(0);
        return new SparseMatrixCSR(numRows, numCols, nonZeroElements, innerIndices, outerStarts);
    }
    clone(): SparseMatrixCSR {
        return new SparseMatrixCSR(this.numRows(), this.numCols(), this.nonZeroElements.slice(), this.innerIndices.slice(), this.outerStarts.slice());
    }
    l2Norm(): number {
        let result = 0.0;
        for (let value of this.nonZeroElements)
            result += value * value;
        return Math.sqrt(result);
    }
    lInfNorm(): number {
        let result = 0.0;
        for (let value of this.nonZeroElements)
            result = Math.max(result, Math.abs(value));
        return result;
    }
    static lInfDistance(m1: SparseMatrixCSR, m2: SparseMatrixCSR): number {
        assert(m1.width() == m2.width() && m1.height() == m2.height(), "Incompatible sizes");
        let distance = 0;
        for (let row = 0; row < m1.numRows(); ++row) {
            const UseIterator = false;
            if (UseIterator) {
                let it = new SparseMatrixTwinRowIterator(m1, m2, row);
                while (!it.isDone()) {
                    let value = it.advance();
                    distance = Math.max(Math.abs(value.value1 - value.value2), distance);
                }
            }
            else {
                let it1 = m1.outerStarts[row];
                let it2 = m2.outerStarts[row];
                let isValidIt1 = it1 < m1.outerStarts[row + 1];
                let isValidIt2 = it2 < m2.outerStarts[row + 1];
                while (isValidIt1 || isValidIt2) {
                    let col1 = isValidIt1 ? m1.innerIndices[it1] : m1.width();
                    let col2 = isValidIt2 ? m2.innerIndices[it2] : m2.width();
                    let value1 = 0.0;
                    let value2 = 0.0;
                    if (col1 <= col2) {
                        value1 = m1.nonZeroElements[it1];
                        ++it1;
                        isValidIt1 = it1 < m1.outerStarts[row + 1];
                    }
                    if (col2 <= col1) {
                        value2 = m2.nonZeroElements[it2];
                        ++it2;
                        isValidIt2 = it2 < m2.outerStarts[row + 1];
                    }
                    distance = Math.max(Math.abs(value1 - value2), distance);
                }
            }
        }
        return distance;
    }
    static near(m1: SparseMatrixCSR, m2: SparseMatrixCSR, tolerance: number = SmallTolerance): boolean {
        assert(m1.width() == m2.width() && m1.height() == m2.height(), "Incompatible sizes");
        for (let row = 0; row < m1.numRows(); ++row) {
            const UseIterator = false;
            if (UseIterator) {
                let it = new SparseMatrixTwinRowIterator(m1, m2, row);
                while (!it.isDone()) {
                    let value = it.advance();
                    if (Math.abs(value.value1 - value.value2) > tolerance)
                        return false;
                }
            }
            else {
                let it1 = m1.outerStarts[row];
                let it2 = m2.outerStarts[row];
                let isValidIt1 = it1 < m1.outerStarts[row + 1];
                let isValidIt2 = it2 < m2.outerStarts[row + 1];
                while (isValidIt1 || isValidIt2) {
                    let col1 = isValidIt1 ? m1.innerIndices[it1] : m1.width();
                    let col2 = isValidIt2 ? m2.innerIndices[it2] : m2.width();
                    let value1 = 0.0;
                    let value2 = 0.0;
                    if (col1 <= col2) {
                        value1 = m1.nonZeroElements[it1];
                        ++it1;
                    }
                    if (col2 <= col1) {
                        value2 = m1.nonZeroElements[it2];
                        ++it2;
                    }
                    if (Math.abs(value1 - value2) > tolerance)
                        return false;
                    isValidIt1 = it1 < m1.outerStarts[row + 1];
                    isValidIt2 = it2 < m2.outerStarts[row + 1];
                }
            }
        }
        return true;
    }
    static identity(size: number): SparseMatrixCSR {
        let nonZeroElements: number[] = [];
        let innerIndices: number[] = [];
        let outerStarts: number[] = [];
        for (let row = 0; row < size; ++row) {
            outerStarts.push(nonZeroElements.length);
            innerIndices.push(row);
            nonZeroElements.push(1);
        }
        outerStarts.push(nonZeroElements.length);
        return new SparseMatrixCSR(size, size, nonZeroElements, innerIndices, outerStarts);
    }
    private static combine(m1: SparseMatrixCSR, m2: SparseMatrixCSR, f: (a: number, b: number) => number): SparseMatrixCSR {
        assert(m1.width() == m2.width() && m1.height() == m2.height(), "Incompatible sizes");
        let nonZeroElements: number[] = [];
        let innerIndices: number[] = [];
        let outerStarts: number[] = [];
        for (let row = 0; row < m1.numRows(); ++row) {
            outerStarts.push(nonZeroElements.length);
            let it = new SparseMatrixTwinRowIterator(m1, m2, row);
            while (!it.isDone()) {
                let value = it.advance();
                innerIndices.push(value.colIdx);
                nonZeroElements.push(f(value.value1, value.value2));
            }
        }
        outerStarts.push(nonZeroElements.length);
        return new SparseMatrixCSR(m1.width(), m2.width(), nonZeroElements, innerIndices, outerStarts);
    }
    static add(m1: SparseMatrixCSR, m2: SparseMatrixCSR): SparseMatrixCSR {
        return SparseMatrixCSR.combine(m1, m2, (a: number, b: number) => a + b);
    }
    static sub(m1: SparseMatrixCSR, m2: SparseMatrixCSR): SparseMatrixCSR {
        return SparseMatrixCSR.combine(m1, m2, (a: number, b: number) => a - b);
    }
    static mul(a: SparseMatrixCSR, b: SparseMatrixCSR): SparseMatrixCSR {
        let nonZeroElements: number[] = [];
        let innerIndices: number[] = [];
        let outerStarts: number[] = [];
        // transpose second matrix and sum multiplication of row elements
        let bTransposed = b.transpose();
        for (let row = 0; row < a.numRows(); ++row) {
            outerStarts.push(nonZeroElements.length);
            const aNumElementsInRow = a.outerStarts[row + 1] - a.outerStarts[row];
            if (aNumElementsInRow == 0) continue;
            for (let col = 0; col < b.numCols(); ++col) {
                let aColIt = a.outerStarts[row];
                let value = 0.0;
                for (let bRowIt = bTransposed.outerStarts[col]; bRowIt < bTransposed.outerStarts[col + 1]; ++bRowIt) {
                    const bRowIdx = bTransposed.innerIndices[bRowIt];
                    while (aColIt < a.outerStarts[row + 1] && a.innerIndices[aColIt] < bRowIdx)
                        ++aColIt;
                    if (aColIt == a.outerStarts[row + 1]) break;
                    if (a.innerIndices[aColIt] > bRowIdx) continue;
                    const bRowValue = bTransposed.nonZeroElements[bRowIt];
                    value += bRowValue * a.nonZeroElements[aColIt];
                }
                if (value != 0.0) {
                    nonZeroElements.push(value);
                    innerIndices.push(col);
                }
            }
        }
        outerStarts.push(nonZeroElements.length);
        return new SparseMatrixCSR(a.numRows(), b.numCols(), nonZeroElements, innerIndices, outerStarts);
    }
    static kroneckerProduct(a: SparseMatrixCSR, b: SparseMatrixCSR): SparseMatrixCSR {
        let nonZeroElements: number[] = [];
        let innerIndices: number[] = [];
        let outerStarts: number[] = [];
        for (let aRow = 0, resRow = 0; aRow < a.numRows(); ++aRow) {
            for (let bRow = 0; bRow < b.numRows(); ++bRow, ++resRow) {
                outerStarts.push(nonZeroElements.length);
                for (let aOuterIdxIt = a.outerStarts[aRow]; aOuterIdxIt < a.outerStarts[aRow + 1]; ++aOuterIdxIt) {
                    let aCol = a.innerIndices[aOuterIdxIt];
                    let aVal = a.nonZeroElements[aOuterIdxIt];
                    for (let bOuterIdxIt = b.outerStarts[bRow]; bOuterIdxIt < b.outerStarts[bRow + 1]; ++bOuterIdxIt) {
                        let bCol = b.innerIndices[bOuterIdxIt];
                        let bVal = b.nonZeroElements[bOuterIdxIt];
                        let resCol = aCol * b.numCols() + bCol;
                        innerIndices.push(resCol);
                        nonZeroElements.push(aVal * bVal);
                    }
                }
            }
        }
        outerStarts.push(nonZeroElements.length);
        return new SparseMatrixCSR(a.numRows() * b.numRows(), a.numCols() * b.numCols(), nonZeroElements, innerIndices, outerStarts);
    }
    static entrywiseProduct(a: SparseMatrixCSR, b: SparseMatrixCSR): SparseMatrixCSR {
        assert(a.numRows() == b.numRows() && b.numCols() == b.numCols(), "Sizes don't match");
        let nonZeroElements: number[] = [];
        let innerIndices: number[] = [];
        let outerStarts: number[] = [];
        for (let rowIdx = 0; rowIdx < a.numRows(); ++rowIdx) {
            outerStarts.push(nonZeroElements.length);
            let it = new SparseMatrixTwinRowIterator(a, b, rowIdx);
            while (!it.isDone()) {
                let { value1, value2, colIdx } = it.advance();
                let result = value1 * value2;
                if (result != 0) {
                    nonZeroElements.push(result);
                    innerIndices.push(colIdx);
                }
            }
        }
        outerStarts.push(nonZeroElements.length);
        return new SparseMatrixCSR(a.numRows(), a.numCols(), nonZeroElements, innerIndices, outerStarts);

    }
    static postMulSparse(m: SparseMatrixCSR, v: SparseVector): SparseVector {
        assert(v.size() == m.numCols(), "Sizes don't match");
        let result: SparseVectorElement[] = [];
        for (let rowIdx = 0; rowIdx < m.numRows(); ++rowIdx) {
            let value = 0.0;
            let curVectorIdxIt = 0;
            for (let outerIdxIt = m.outerStarts[rowIdx]; outerIdxIt < m.outerStarts[rowIdx + 1]; ++outerIdxIt) {
                let colIdx = m.innerIndices[outerIdxIt];
                while (curVectorIdxIt < v.elements.length && v.elements[curVectorIdxIt].index < colIdx)
                    ++curVectorIdxIt;
                if (curVectorIdxIt == v.elements.length || v.elements[curVectorIdxIt].index > colIdx) continue;
                value += v.elements[curVectorIdxIt].value * m.nonZeroElements[outerIdxIt];
            }
            if (value != 0.0)
                result.push({ index: rowIdx, value: value });
        }
        return new SparseVector(m.numRows(), result);
    }
    static preMulSparse(v: SparseVector, m: SparseMatrixCSR): SparseVector {
        assert(v.size() == m.numRows(), "Sizes don't match");
        let transposed = m.transpose();
        return SparseMatrixCSR.postMulSparse(transposed, v);
    }
    static postMul(m: SparseMatrixCSR, v: Vector): Vector {
        assert(v.size() == m.numCols(), "Sizes don't match");
        let result: Vector = Vector.empty(m.numRows());
        for (let rowIdx = 0; rowIdx < m.numRows(); ++rowIdx) {
            let value = 0.0;
            for (let outerIdxIt = m.outerStarts[rowIdx]; outerIdxIt < m.outerStarts[rowIdx + 1]; ++outerIdxIt) {
                value += v.get(m.innerIndices[outerIdxIt]) * m.nonZeroElements[outerIdxIt];
            }
            result.set(rowIdx, value);
        }
        return result;
    }
    static preMul(v: Vector, m: SparseMatrixCSR): Vector {
        assert(v.size() == m.numRows(), "Sizes don't match");
        let transposed = m.transpose();
        return SparseMatrixCSR.postMul(transposed, v);
    }
    scale(scalar: number): SparseMatrixCSR {
        for (let i = 0; i < this.nonZeroElements.length; ++i)
            this.nonZeroElements[i] *= scalar;
        return this;
    }
    static scale(m: SparseMatrixCSR, scalar: number): SparseMatrixCSR {
        return m.clone().scale(scalar);
    }
    rowVector(row: number): SparseVector {
        let values: SparseVectorElement[] = [];
        for (let innerIdxIt = this.outerStarts[row]; innerIdxIt < this.outerStarts[row + 1]; ++innerIdxIt)
            values.push({ index: this.innerIndices[innerIdxIt], value: this.nonZeroElements[innerIdxIt] });
        return new SparseVector(this.numCols(), values);
    }
    columnVector(column: number): SparseVector {
        let values: SparseVectorElement[] = [];
        for (let rowIdx = 0; rowIdx < this.numRows(); ++rowIdx) {
            for (let innerIdxIt = this.outerStarts[rowIdx]; innerIdxIt < this.outerStarts[rowIdx + 1]; ++innerIdxIt) {
                let colIdx = this.innerIndices[innerIdxIt];
                let value = this.nonZeroElements[innerIdxIt];
                if (colIdx == column)
                    values.push({ index: rowIdx, value });
            }
        }
        return new SparseVector(this.numCols(), values);
    }
    transpose(): SparseMatrixCSR {
        let nonZeroElements: number[] = Array(this.nonZeroElements.length);
        let innerIndices: number[] = Array(this.nonZeroElements.length);
        let outerStarts: number[] = Array(this.numCols() + 1).fill(0);
        // calc number of entries per column/ row
        for (let i = 0; i < this.nonZeroElements.length; ++i)
            outerStarts[this.innerIndices[i]]++;
        // for outerStarts
        for (let sum = 0, col = 0; col < this.numCols(); ++col) {
            let curValue = outerStarts[col];
            outerStarts[col] = sum;
            sum += curValue;
        }
        // outerStarts tracks innerIdxit for transposed elements
        for (let rowIdx = 0; rowIdx < this.numRows(); ++rowIdx) {
            for (let innerIdxIt = this.outerStarts[rowIdx]; innerIdxIt < this.outerStarts[rowIdx + 1]; ++innerIdxIt) {
                let colIdx = this.innerIndices[innerIdxIt];
                let value = this.nonZeroElements[innerIdxIt];
                let resultInnerIdxIt = outerStarts[colIdx];
                outerStarts[colIdx]++;
                innerIndices[resultInnerIdxIt] = rowIdx;
                nonZeroElements[resultInnerIdxIt] = value;
            }
        }
        // restore proper format for outerStarts
        for (let col = 0, prev = 0; col <= this.numCols(); ++col) {
            let temp = outerStarts[col];
            outerStarts[col] = prev;
            prev = temp;
        }
        return new SparseMatrixCSR(this.numCols(), this.numRows(), nonZeroElements, innerIndices, outerStarts);
    }
    static fromTriplets(numRows: number, numCols: number, triplets: Triplet[], tolerance: number = SmallestTolerance): SparseMatrixCSR {
        // sorted in ascending "row by row" order
        triplets.sort((a: Triplet, b: Triplet) => {
            let rowSign = a.row - b.row;
            if (rowSign != 0) return rowSign;
            return a.column - b.column;
        });

        if (triplets.length == 0) return SparseMatrixCSR.empty(numRows, numCols);
        let nonZeroElements: number[] = [];
        let innerIndices: number[] = [];
        let outerStarts: number[] = [];
        outerStarts.push(0);
        let currentRow = 0;
        for (let i = 0; i < triplets.length; ++i) {
            assert(triplets[i].row < numRows && triplets[i].row >= 0, "Invalid row index");
            assert(triplets[i].column < numCols && triplets[i].column >= 0, "Invalid column index");
            if (Math.abs(triplets[i].value) < tolerance) continue
            if (triplets[i].row != currentRow) {
                for (let row = currentRow + 1; row <= triplets[i].row; ++row)
                    outerStarts.push(nonZeroElements.length);
                currentRow = triplets[i].row;
            } else if (i > 0 && triplets[i].column == triplets[i - 1].column) {
                nonZeroElements[nonZeroElements.length - 1] = triplets[i].value;
            }
            nonZeroElements.push(triplets[i].value);
            innerIndices.push(triplets[i].column);
        }
        for (let row = currentRow + 1; row <= numRows; ++row)
            outerStarts.push(nonZeroElements.length);
        assert(outerStarts.length == numRows + 1, "result.outerStarts.length == numRows + 1");
        assert(outerStarts[outerStarts.length - 1] == nonZeroElements.length, "Incorrect outerStarts");
        return new SparseMatrixCSR(numRows, numCols, nonZeroElements, innerIndices, outerStarts);
    }
    static fromDense(dense: Matrix, tolerance: number): SparseMatrixCSR {
        let nonZeroElements: number[] = [];
        let innerIndices: number[] = [];
        let outerStarts: number[] = [];
        for (let row = 0; row < dense.numRows(); ++row) {
            outerStarts.push(nonZeroElements.length);
            for (let col = 0; col < dense.numCols(); ++col) {
                let value = dense.get(row, col);
                if (Math.abs(value) > tolerance) {
                    nonZeroElements.push(value);
                    innerIndices.push(col);
                }
            }
        }
        outerStarts.push(nonZeroElements.length);
        return new SparseMatrixCSR(dense.numRows(), dense.numCols(), nonZeroElements, innerIndices, outerStarts);
    }
    toDense(): Matrix {
        let result = Matrix.empty(this.numRows(), this.numCols());
        for (let rowIdx = 0; rowIdx < this.numRows(); ++rowIdx) {
            let startIdx = this.outerStarts[rowIdx];
            let endIdx = this.outerStarts[rowIdx + 1];
            for (let j = startIdx; j < endIdx; ++j) {
                let colIdx = this.innerIndices[j];
                let value = this.nonZeroElements[j];
                result.set(rowIdx, colIdx, value);
            }
        }
        return result;
    }
    numNonZeroes(): number {
        return this.nonZeroElements.length;
    }
    coeffRef(row: number, column: number): CellRef {
        let left = this.outerStarts[row];
        let right = this.outerStarts[row + 1];
        while (left != right) {
            let middleIt = Math.floor((left + right) / 2);
            const middleCol = this.innerIndices[middleIt];
            if (middleCol < column) {
                left = middleIt + 1;
            } else if (middleCol > column) {
                right = middleIt;
            } else {
                return new CellRef(this, middleIt, row, column);
            }
        }
        return null;
    }
    erase(cell: CellRef) {
        for (let row = cell.row() + 1; row <= this.numRows(); ++row)
            this.outerStarts[row]--;
        this.innerIndices.splice(cell.outerIdx(), 1);
        this.nonZeroElements.splice(cell.outerIdx(), 1);
    }
    set(row: number, column: number, value: number): void {
        let left = this.outerStarts[row];
        let right = this.outerStarts[row + 1];
        while (left != right) {
            let middleIt = Math.floor((left + right) / 2);
            const middleCol = this.innerIndices[middleIt];
            if (middleCol < column) {
                left = middleIt + 1;
            } else if (middleCol > column) {
                right = middleIt;
            } else {
                this.nonZeroElements[middleIt] = value;
                return;
            }
        }
        if (left != this.outerStarts[row + 1] && this.innerIndices[left] < column)
            ++left;
        this.innerIndices.splice(left, 0, column);
        this.nonZeroElements.splice(left, 0, value);
        for (let rowIdx = row + 1; rowIdx <= this.numRows(); ++rowIdx)
            this.outerStarts[rowIdx]++;
    }
    get(row: number, column: number): number {
        let left = this.outerStarts[row];
        let right = this.outerStarts[row + 1];
        while (left != right) {
            let middleIt = Math.floor((left + right) / 2);
            const middleCol = this.innerIndices[middleIt];
            if (middleCol < column) {
                left = middleIt + 1;
            } else if (middleCol > column) {
                right = middleIt;
            } else {
                return this.nonZeroElements[middleIt];
            }
        }
        return 0.0;
    }
    nonZeroElement(idx: number): number {
        return this.nonZeroElements[idx];
    }
    innerIndex(idx: number): number {
        return this.innerIndices[idx];
    }
    outerStart(row: number): number {
        return this.outerStarts[row];
    }
    setNonZeroElement(idx: number, value: number) {
        return this.nonZeroElements[idx] = value;
    }
    isValid(): boolean {
        let result = true;
        result = result && this.outerStarts.length == this.numRows() + 1;
        result = result && this.innerIndices.length == this.nonZeroElements.length;
        result = result && this.outerStart(this.numRows()) == this.nonZeroElements.length;
        if (!result)
            return false;
        let prevIt = 0;
        for (let row = 0; row < this.numRows(); ++row) {
            let prevCol = -1;
            if (this.outerStart(row + 1) < prevIt)
                return false;
            for (let colIt = this.outerStart(row); colIt < this.outerStart(row + 1); ++colIt) {
                const col = this.innerIndex(colIt);
                if (col <= prevCol || col < 0 || col >= this.numCols() || this.nonZeroElement(colIt) == undefined)
                    return false;
                prevCol = col;
            }
            prevIt = this.outerStarts[row];
        }
        return true;
    }
    isSymmetric(tolerance: number = SmallTolerance): boolean {
        let transpose = this.transpose();
        return SparseMatrixCSR.near(this, transpose, tolerance);
    }
    toTriplets(): SparseMatrixTriplets {
        let triplets: Triplet[];
        for (let rowIdx = 0; rowIdx < this.numRows(); ++rowIdx) {
            let startIdx = this.outerStarts[rowIdx];
            let endIdx = this.outerStarts[rowIdx + 1];
            for (let j = startIdx; j < endIdx; ++j) {
                let colIdx = this.innerIndices[j];
                let value = this.nonZeroElements[j];
                triplets.push({ row: rowIdx, column: colIdx, value });
            }
        }
        return SparseMatrixTriplets.fromTriplets(this.numRows(), this.numCols(), triplets);
    }
    printAsMatrix(): string {
        let result = "[";
        const printValue = (col: number, value: string) => {
            if (col != 0)
                result += ", ";
            result += value;
        };
        for (let row = 0; row < this.numRows(); ++row) {
            if (row != 0)
                result += ",";
            result += "\n\t[";
            let col = 0;
            for (let i = this.outerStarts[row]; i != this.outerStarts[row + 1]; ++i) {
                while (col != this.innerIndex(i)) {
                    printValue(col, "______");
                    ++col;
                }
                ++col;
                printValue(this.innerIndex(i), this.nonZeroElement(i).toFixed(4));
            }
            while (col != this.numCols()) {
                printValue(col, "______");
                ++col;
            }
            result += "]";
        }
        return result + "\n]";
    }
    toString(): string {
        let result = "[\n";
        for (let row = 0; row < this.numRows(); ++row) {
            if (this.outerStarts[row + 1] == this.outerStarts[row]) continue;
            result += `row ${row}: \t[`;
            let id = 0;
            for (let i = this.outerStarts[row]; i != this.outerStarts[row + 1]; ++i) {
                if (id != 0)
                    result += ", ";
                ++id;
                result += `{col: ${this.innerIndex(i)}, value: ${this.nonZeroElement(i)}}`;
            }
            result += "]\n"
        }
        return result + "]";
    }
    static compareSparsity(m1: SparseMatrixCSR, m2: SparseMatrixCSR): boolean {
        assert(m1.numRows() == m2.numRows() && m1.numCols() == m2.numCols(), "Dimensions should match");
        if (m1.numNonZeroes() != m2.numNonZeroes()) return false;
        for (let row = 0; row < m1.numRows(); ++row) {
            let start = m1.outerStart(row);
            let end = m1.outerStart(row + 1);
            if (end != m2.outerStart(row + 1) || start != m1.outerStart(row))
                return false;
            for (let i = start; i < end; ++i)
                if (m1.innerIndex(i) != m2.innerIndex(i)) return false;
        }
        return true;
    }
}

// TODO (NI): sparse matrix class that combines triplets for construction and CSR for usage