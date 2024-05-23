import Matrix from "../dense/denseMatrix";
import vector from "../dense/vector";
import { assert, SmallestTolerance } from "../utils";
import { SparseMatrixCSR } from "./sparseMatrix";

const DefaultTolerance = SmallestTolerance;

export interface SparseVectorElement {
    index: number;
    value: number;
}

interface ValuePair {
    value1: number;
    value2: number;
    index: number;
}

function reduceVector(v: SparseVector, op: (oldValue: number, curValue: number) => number): number {
    let result = 0;
    for (let i = 0; i < v.elements.length; ++i)
        result = op(result, Math.abs(v.elements[i].value));
    return result;
}

function reduceVectorsDiff(v1: SparseVector, v2: SparseVector, op: (oldValue: number, curValue: number) => number): number {
    assert(v1.size() == v2.size(), "Vectors should have the same size");
    if (v1.elements.length == 0) return reduceVector(v2, op);
    if (v2.elements.length == 0) return reduceVector(v1, op);

    let result = 0;
    let it = new SparseVectorTwinIterator(v1, v2);
    while (!it.isDone()) {
        const { value1, value2, index } = it.advance();
        result = op(result, Math.abs(value1 - value2));
    }
    return result;

}

export class SparseVectorTwinIterator {
    private v1: SparseVector;
    private v2: SparseVector;
    private it1: number;
    private it2: number;
    constructor(v1: SparseVector, v2: SparseVector) {
        this.v1 = v1;
        this.v2 = v2;
        this.it1 = 0;
        this.it2 = 0;
        assert(this.v1.length == this.v2.length, "Sizes of vectors should match");
    }
    isDone(): boolean {
        const isValidIt1 = this.it1 < this.v1.elements.length;
        const isValidIt2 = this.it2 < this.v2.elements.length;
        return !isValidIt1 && !isValidIt2;
    }
    advance(): ValuePair {
        const isValidIt1 = this.it1 < this.v1.elements.length;
        const isValidIt2 = this.it2 < this.v2.elements.length;
        let idx1 = isValidIt1 ? this.v1.elements[this.it1].index : this.v1.length;
        let idx2 = isValidIt2 ? this.v2.elements[this.it2].index : this.v2.length;
        let index = 0;
        let value1 = 0.0;
        let value2 = 0.0;
        if (idx1 <= idx2) {
            index = idx1;
            value1 = this.v1.elements[this.it1].value;
            ++this.it1;
        }
        if (idx2 <= idx1) {
            index = idx2;
            value2 = this.v2.elements[this.it2].value;
            ++this.it2;
        }
        return { value1, value2, index };
    }
}

export class SparseVector {
    length: number;
    elements: SparseVectorElement[];
    constructor(length: number, elements: SparseVectorElement[]) {
        this.length = length;
        this.elements = elements;
        elements.sort((a: SparseVectorElement, b: SparseVectorElement): number => {
            return a.index - b.index;
        });
    }
    clone(): SparseVector {
        let result = SparseVector.empty(this.length);
        for (const { index, value } of this.elements)
            result.elements.push({ index, value });
        return result;
    }
    static near(v1: SparseVector, v2: SparseVector, tolerance: number): boolean {
        let it = new SparseVectorTwinIterator(v1, v2);
        while (!it.isDone()) {
            const { value1, value2 } = it.advance();
            if (Math.abs(value1 - value2) > tolerance)
                return false;
        }
        return true;
    }
    static empty(size: number): SparseVector {
        return new SparseVector(size, []);
    }
    static binaryOp(v1: SparseVector, v2: SparseVector, op: (a: number, b: number) => number, tolerance: number = 0): SparseVector {
        assert(v1.size() == v2.size(), "Vectors should have the same size");
        let result = new SparseVector(v1.size(), []);
        let it = new SparseVectorTwinIterator(v1, v2);
        while (!it.isDone()) {
            const { value1, value2, index } = it.advance();
            let value = op(value1, value2);
            if (Math.abs(value) > tolerance)
                result.elements.push({ index, value });
        }
        return result;
    }
    static dot(v1: SparseVector, v2: SparseVector): number {
        let result = 0.0;
        let it = new SparseVectorTwinIterator(v1, v2);
        while (!it.isDone()) {
            const { value1, value2 } = it.advance();
            result += value1 * value2;
        }
        return result;
    }
    /** Compute v1 * v2^T, where both vectors are column vectors */
    static outerProduct(v1: SparseVector, v2: SparseVector): SparseMatrixCSR {
        let nonZeroElements: number[] = [];
        let innerIndices: number[] = [];
        let outerIndices: number[] = [];
        let prevRow = 0;
        for (let i = 0; i < v1.elements.length; ++i) {
            let value1 = v1.elements[i];
            const row = value1.index;
            for (let k = prevRow; k <= row; ++k)
                outerIndices.push(innerIndices.length);
            for (let j = 0; j < v2.elements.length; ++j) {
                const value2 = v2.elements[j];
                const column = value2.index;
                nonZeroElements.push(value1.value * value2.value);
                innerIndices.push(column);
            }
            prevRow = row + 1;
        }
        for (let k = prevRow; k <= v1.size(); ++k)
            outerIndices.push(innerIndices.length);
        return new SparseMatrixCSR(v1.length, v2.length, nonZeroElements, innerIndices, outerIndices);
    }
    static outerProductDense(v1: SparseVector, v2: SparseVector): Matrix {
        let result = Matrix.empty(v1.length, v2.length);
        for (let i = 0; i < v1.elements.length; ++i) {
            let value1 = v1.elements[i];
            const row = value1.index;
            for (let j = 0; j < v2.elements.length; ++j) {
                const value2 = v2.elements[j];
                const column = value2.index;
                result.set(row, column, value1.value * value2.value);
            }
        }
        return result;
    }
    static add(v1: SparseVector, v2: SparseVector, tolerance: number = 0): SparseVector {
        return this.binaryOp(v1, v2, (a: number, b: number) => a + b, tolerance);
    }
    static sub(v1: SparseVector, v2: SparseVector, tolerance: number = 0): SparseVector {
        return this.binaryOp(v1, v2, (a: number, b: number) => a - b, tolerance);
    }
    static mul(v1: SparseVector, v2: SparseVector, tolerance: number = 0): SparseVector {
        return this.binaryOp(v1, v2, (a: number, b: number) => a * b, tolerance);
    }
    static div(v1: SparseVector, v2: SparseVector, tolerance: number = 0): SparseVector {
        return this.binaryOp(v1, v2, (a: number, b: number) => a / b, tolerance);
    }
    /** L1 norm (sum of abs coordinates of vector from a to b) between two vectors */
    static l1Distance(a: SparseVector, b: SparseVector): number {
        return reduceVectorsDiff(a, b, (prev: number, cur: number) => { return prev + cur; });
    }
    /** Euclidian norm (sum of squared coordinates of vector from a to b) between two vectors */
    static l2Distance(a: SparseVector, b: SparseVector): number {
        return Math.sqrt(reduceVectorsDiff(a, b, (prev: number, cur: number) => { return prev + cur * cur; }));
    }
    /** LInf norm (max of absolute coordinates of vector from a to b) between two vectors */
    static lInfDistance(a: SparseVector, b: SparseVector): number {
        return reduceVectorsDiff(a, b, (prev: number, cur: number) => { return Math.max(prev, cur); });
    }
    /** LP norm between two vectors */
    static lpDistance(a: SparseVector, b: SparseVector, p: number): number {
        return Math.pow(reduceVectorsDiff(a, b, (prev: number, cur: number) => { return prev + Math.pow(cur, p); }), 1 / p);
    }
    static negate(v: SparseVector): SparseVector {
        let result = v.clone();
        for (let i = 0; i < v.elements.length; ++i)
            result.elements[i].value = -result.elements[i].value;
        return result;
    }
    static normalize(v: SparseVector): SparseVector {
        let length = v.l2Norm();
        let result = v.clone();
        for (let i = 0; i < v.elements.length; ++i)
            result.elements[i].value /= length;
        return result;
    }
    l1Norm(): number {
        let result = 0;
        for (let element of this.elements)
            result += Math.abs(element.value);
        return result;
    }
    l2Norm(): number {
        return Math.sqrt(this.squaredLength());
    }
    lInfNorm(): number {
        let result = 0;
        for (let element of this.elements)
            result = Math.max(Math.abs(element.value), result);
        return result;
    }
    squaredLength(): number {
        let result = 0;
        for (let element of this.elements)
            result += element.value * element.value;
        return result;
    }
    static fromVector(array: number[], tolerance: number = DefaultTolerance): SparseVector {
        let elements: SparseVectorElement[] = [];
        for (let index = 0; index < array.length; ++index) {
            let value = array[index];
            if (Math.abs(value) > tolerance)
                elements.push({ index, value });
        }
        let sparse = new SparseVector(array.length, elements);
        return sparse;
    }
    size(): number {
        return this.length;
    }
    toDense(): vector {
        let dense = vector.empty(this.size());
        for (let element of this.elements)
            dense.set(element.index, element.value);
        return dense;
    }
    set(index: number, value: number) {
        let l = 0;
        let r = this.elements.length;
        while (l != r) {
            let middle = Math.floor((r + l) / 2);
            if (this.elements[middle].index < index)
                l = middle + 1;
            else if (this.elements[middle].index > index)
                r = middle;
            else if (value != 0) {
                this.elements[middle].value = value;
                return;
            } else {
                this.elements.splice(middle, 1);
                return;
            }
        }
        if (Math.abs(value) != 0)
            this.elements.splice(l, 0, { value, index });
    }
    get(index: number) {
        let l = 0;
        let r = this.elements.length;
        while (l != r) {
            let middle = Math.floor((r + l) / 2);
            if (this.elements[middle].index < index)
                l = middle + 1;
            else if (this.elements[middle].index > index)
                r = middle;
            else
                return this.elements[middle].value;
        }
        return 0.0;
    }
    isIndexPresent(index: number): boolean {
        let l = 0;
        let r = this.elements.length;
        while (l != r) {
            let middle = Math.floor((r + l) / 2);
            if (this.elements[middle].index < index)
                l = middle + 1;
            else if (this.elements[middle].index > index)
                r = middle;
            else
                return true;
        }
        return false;
    }
    isValid(): boolean {
        let prevCol = -1;
        for (let it = 0; it < this.elements.length; ++it) {
            let idx = this.elements[it].index;
            if (idx <= prevCol || idx < 0 || idx > this.length) return false;
            prevCol = idx;
        }
        return true;
    }
    toString(): string {
        let result = `sparse(${this.length})[`;
        for (let i = 0; i < this.elements.length; ++i) {
            result += `${i != 0 ? ", " : ""}${this.elements[i].index}: ${this.elements[i].value}`;
        }
        return result + "]";
    }
}