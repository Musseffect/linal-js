import AbstractMatrix from "../abstractMatrix";
import Matrix from "../dense/denseMatrix";
import Vector from "../dense/vector";
import { assert } from "../utils";
import { SparseVector } from "./sparseVector";
import Triplet from "./triplet";

export interface TwinIteratorValue {
    row: number;
    column: number;
    value1: number;
    value2: number;
};
export class SparseMatrixTripletsTwinIterator {
    private m1: SparseMatrixTriplets;
    private m2: SparseMatrixTriplets;
    private it1: number;
    private it2: number;
    constructor(m1: SparseMatrixTriplets, m2: SparseMatrixTriplets) {
        this.m1 = m1;
        this.m2 = m2;
        this.it1 = 0;
        this.it2 = 0;
        assert(this.m1.numRows() == this.m2.numRows() && this.m1.numCols() == this.m2.numCols(), "Sizes won't match");
    }
    isDone(): boolean {
        const isValidIt1 = this.it1 < this.m1.numNonZeroElements();
        const isValidIt2 = this.it2 < this.m2.numNonZeroElements();
        return !isValidIt1 && !isValidIt2;
    }
    advance(): TwinIteratorValue {
        const isValidIt1 = this.it1 < this.m1.numNonZeroElements();
        const isValidIt2 = this.it2 < this.m2.numNonZeroElements();
        let element1 = isValidIt1 ? this.m1.nonZeroElement(this.it1) : { row: this.m1.numRows(), column: this.m1.numCols(), value: 0 };
        let element2 = isValidIt2 ? this.m2.nonZeroElement(this.it2) : { row: this.m2.numRows(), column: this.m2.numCols(), value: 0 };
        let idx1 = this.m1.index(element1.row, element1.column);
        let idx2 = this.m2.index(element2.row, element2.column);
        let row = 0;
        let column = 0;
        let value1 = 0.0;
        let value2 = 0.0;
        if (idx1 <= idx2) {
            value1 = element1.value;
            row = element1.row;
            column = element1.column;
            ++this.it1;
        }
        if (idx2 <= idx1) {
            value2 = element2.value;
            row = element2.row;
            column = element2.column;
            ++this.it2;
        }
        return { row, column, value1, value2 };
    }
}

// sorted triplets, row major
export class SparseMatrixTriplets extends AbstractMatrix {
    private triplets: Triplet[];
    constructor(numRows: number, numCols: number) {
        super(numRows, numCols);
        this.triplets = [];
    }
    printAsMatrix(): string {
        let idx = 0;
        let result = "[";
        const printValue = (col: number, row: number, value: string) => {
            if (col == 0) {
                if (row != 0)
                    result += "],";
                result += "\n\t[";
            }
            else {
                result += ", ";
            }
            result += value;
        };
        for (let item of this.triplets) {
            let curIdx = this.index(item.row, item.column);
            while (idx != curIdx) {
                let row = Math.floor(idx / this.numCols());
                let col = idx - row * this.numCols();
                printValue(col, row, "______");
                ++idx;
            }
            printValue(item.column, item.row, item.value.toFixed(4));
            ++idx;
        }
        while (idx != this.numRows() * this.numCols()) {
            let row = Math.floor(idx / this.numCols());
            let col = idx - row * this.numCols();
            printValue(col, row, "______");
            ++idx;
        }
        if (idx != 0)
            result += "]";
        return result + "\n]";
    }
    toString(): string {
        let row = -1;
        let result = "[\n";
        const rowEnding = () => { result += "]\n" };
        for (let item of this.triplets) {
            if (item.row != row) {
                if (row != -1)
                    rowEnding();
                result += `row ${item.row}: \t[`;
                row = item.row;
            } else {
                result += ", ";
            }
            result += `{col: ${item.column}, value: ${item.value}}`;
        }
        if (row != -1)
            rowEnding();
        return result + "]";
    }
    clone(): SparseMatrixTriplets {
        let matrix = new SparseMatrixTriplets(this.numRows(), this.numCols());
        matrix.triplets = this.triplets.slice();
        for (let i = 0; i < matrix.triplets.length; ++i)
            matrix.triplets[i] = matrix.nonZeroElement(i);
        return matrix;
    }
    index(row: number, column: number) {
        return column + this.numCols() * row;
    }
    numNonZeroElements(): number {
        return this.triplets.length;
    }
    nonZeroElement(index: number): Triplet {
        const triplet = this.triplets[index];
        return { row: triplet.row, column: triplet.column, value: triplet.value };
    }
    setNonZeroElement(index: number, value: number): void {
        this.triplets[index].value = value;
    }
    transpose(): SparseMatrixTriplets {
        let triplets: Triplet[] = [];
        for (const { row, column, value } of this.triplets)
            triplets.push({ row: column, column: row, value });
        return SparseMatrixTriplets.fromTriplets(this.numCols(), this.numRows(), triplets);
    }
    toDense(): Matrix {
        return Matrix.fromTriplets(this.numRows(), this.numCols(), this.triplets);
    }
    static mul(m1: SparseMatrixTriplets, m2: SparseMatrixTriplets): SparseMatrixTriplets {
        let result = new SparseMatrixTriplets(m1.numRows(), m2.numCols());
        let transposed = m2.transpose();
        let it1 = 0;
        while (it1 < m1.triplets.length) {
            let curRow = m1.triplets[it1].row;
            let it2 = 0;
            while (it2 < transposed.triplets.length) {
                let value = 0;
                let rowIt1 = it1;
                let curCol = transposed.triplets[it2].row;
                // traverse tranposed matrix row by row
                do {
                    const isValidIt1 = rowIt1 < m1.triplets.length && m1.triplets[rowIt1].row == curRow;
                    let col1 = isValidIt1 ? m1.triplets[rowIt1].column : m1.numCols();
                    let col2 = transposed.triplets[it2].column;
                    if (col1 < col2) {
                        ++rowIt1;
                    } else if (col1 > col2) {
                        ++it2;
                    } else {
                        value += m1.triplets[rowIt1].value * transposed.triplets[it2].value;
                        ++rowIt1;
                        ++it2;
                    }
                } while (it2 < transposed.triplets.length && transposed.triplets[it2].row == curCol)
                if (value != 0)
                    result.triplets.push({ column: curCol, row: curRow, value });
            }
            // go to next non-empty row of first matrix
            do {
                ++it1;
            } while (it1 < m1.triplets.length && m1.triplets[it1].row == curRow)
        }
        return result;
    }
    static postMul(m: SparseMatrixTriplets, v: Vector): Vector {
        assert(m.numCols() == v.size(), "Sizes don't match");
        let result = Vector.empty(m.numRows());
        for (const element of m.triplets)
            result.set(element.row, result.get(element.row) + v.get(element.column) * element.value);
        return result;
    }
    static preMul(v: Vector, m: SparseMatrixTriplets): Vector {
        let tranposed = m.transpose();
        return SparseMatrixTriplets.postMul(tranposed, v);
    }
    static postMulSparse(m: SparseMatrixTriplets, v: SparseVector): SparseVector {
        assert(m.numCols() == v.size(), "Sizes don't match");
        let result = SparseVector.empty(m.numRows());
        let it1 = 0;
        while (it1 < m.triplets.length) {
            let curRow = m.triplets[it1].row;
            let it2 = 0;
            while (it2 < v.elements.length) {
                let value = 0;
                let rowIt1 = it1;
                // traverse tranposed matrix row by row
                do {
                    const isValidIt1 = rowIt1 < m.triplets.length && m.triplets[rowIt1].row == curRow;
                    let col1 = isValidIt1 ? m.triplets[rowIt1].column : m.numCols();
                    let col2 = v.elements[it2].index;
                    if (col1 < col2) {
                        ++rowIt1;
                    } else if (col1 > col2) {
                        ++it2;
                    } else {
                        value += m.triplets[rowIt1].value * v.elements[it2].value;
                        ++rowIt1;
                        ++it2;
                    }
                } while (it2 < v.elements.length)
                if (value != 0)
                    result.elements.push({ index: curRow, value });
            }
            // go to next non-empty row of first matrix
            do {
                ++it1;
            } while (it1 < m.triplets.length && m.triplets[it1].row == curRow)
        }
        return result;
    }
    static preMulSparse(v: SparseVector, m: SparseMatrixTriplets): SparseVector {
        let tranposed = m.transpose();
        return SparseMatrixTriplets.postMulSparse(tranposed, v);
    }
    static fromTriplets(numRows: number, numCols: number, triplets: Triplet[]): SparseMatrixTriplets {
        let matrix = new SparseMatrixTriplets(numRows, numCols);
        matrix.triplets = triplets.slice();
        for (let i = 0; i < triplets.length; ++i)
            matrix.triplets[i] = matrix.nonZeroElement(i);
        matrix.triplets.sort((a: Triplet, b: Triplet) => {
            let aIdx = matrix.index(a.row, a.column);
            let bIdx = matrix.index(b.row, b.column);
            return aIdx - bIdx;
        });
        return matrix;
    }
    private static combine(m1: SparseMatrixTriplets, m2: SparseMatrixTriplets, f: (a: number, b: number) => number): SparseMatrixTriplets {
        let result = new SparseMatrixTriplets(m1.numRows(), m2.numCols());
        let it = new SparseMatrixTripletsTwinIterator(m1, m2);
        while (!it.isDone()) {
            let { row, column, value1, value2 } = it.advance();
            let value = f(value1, value2);
            result.triplets.push({ row, column, value: value });
        }
        return result;
    }
    static near(m1: SparseMatrixTriplets, m2: SparseMatrixTriplets, Tolerance: number): boolean {
        let it = new SparseMatrixTripletsTwinIterator(m1, m2);
        while (!it.isDone()) {
            let { row, column, value1, value2 } = it.advance();
            if (Math.abs(value1 - value2) > Tolerance) return false;
        }
        return true;
    }
    static lInfDistance(m1: SparseMatrixTriplets, m2: SparseMatrixTriplets): number {
        let it = new SparseMatrixTripletsTwinIterator(m1, m2);
        let distance = 0;
        while (!it.isDone()) {
            let { row, column, value1, value2 } = it.advance();
            distance = Math.max(Math.abs(value1 - value2), distance);
        }
        return distance;
    }
    static add(m1: SparseMatrixTriplets, m2: SparseMatrixTriplets): SparseMatrixTriplets {
        return SparseMatrixTriplets.combine(m1, m2, (a: number, b: number) => a + b);
    }
    static sub(m1: SparseMatrixTriplets, m2: SparseMatrixTriplets): SparseMatrixTriplets {
        return SparseMatrixTriplets.combine(m1, m2, (a: number, b: number) => a - b);
    }
    scale(scalar: number): SparseMatrixTriplets {
        for (let triplet of this.triplets)
            triplet.value *= scalar;
        return this;
    }
    static scale(m: SparseMatrixTriplets, scalar: number): SparseMatrixTriplets {
        return m.clone().scale(scalar);
    }
    static fromDense(dense: Matrix, tolerance: number): SparseMatrixTriplets {
        let result = new SparseMatrixTriplets(dense.numRows(), dense.numCols());
        for (let row = 0; row < dense.numRows(); ++row) {
            for (let column = 0; column < dense.numCols(); ++column) {
                let value = dense.get(row, column);
                if (Math.abs(value) > tolerance)
                    result.triplets.push({ row, column, value });
            }
        }
        return result;
    }
    get(row: number, column: number): number {
        let searchIdx = this.index(row, column);
        let leftIt = 0;
        let rightIt = this.triplets.length;
        while (leftIt != rightIt) {
            let middleIt = Math.floor((leftIt + rightIt) / 2);
            const middleElement = this.triplets[middleIt];
            const middleIdx = this.index(middleElement.row, middleElement.column);
            if (middleIdx < searchIdx)
                leftIt = middleIt + 1;
            else if (middleIdx > searchIdx)
                rightIt = middleIt;
            else
                return middleElement.value;
        }
        return 0;
    }
    set(row: number, column: number, value: number) {
        // binary search
        let searchIdx = this.index(row, column);
        let leftIt = 0;
        let rightIt = this.triplets.length;
        while (leftIt != rightIt) {
            let middleIt = Math.floor((leftIt + rightIt) / 2);
            const middleElement = this.triplets[middleIt];
            const middleIdx = this.index(middleElement.row, middleElement.column);
            if (middleIdx < searchIdx) {
                leftIt = middleIt + 1;
            } else if (middleIdx > searchIdx) {
                rightIt = middleIt;
            } else {
                this.triplets[middleIt].value = value;
                return;
            }
        }
        if (this.triplets.length != 0 && this.index(this.triplets[leftIt].row, this.triplets[leftIt].column) < searchIdx)
            ++leftIt;
        this.triplets.splice(leftIt, 0, { row, column, value });
    }
    isValid(): any {
        for (let i = 1; i < this.triplets.length; ++i) {
            let idx1 = this.index(this.triplets[i].row, this.triplets[i].column);
            let idx2 = this.index(this.triplets[i - 1].row, this.triplets[i - 1].column);
            if (idx1 <= idx2) return false;
        }
        return true;
    }
}