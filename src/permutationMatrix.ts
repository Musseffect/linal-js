import Matrix from "./dense/denseMatrix";
import RandomNumberGenerator from "./random/generator";
import JSGenerator from "./random/js";
import { SparseMatrixCSR, SparseMatrixRowIterator } from "./sparse/sparseMatrix";
import Triplet from "./sparse/triplet";
import { assert, swap } from "./utils";
import Vector from "./dense/vector";
import { SparseVector } from "./sparse/sparseVector";

export enum PermutationType {
    Row = 1,
    Col = 0
}

// row permutation - pre multiplied, col permutation - post multiplied
export class PermutationMatrix {
    static random(size: number, type: PermutationType, generator: RandomNumberGenerator = new JSGenerator()): PermutationMatrix {
        let indices = [];
        for (let i = 0; i < size; ++i)
            indices.push(i);
        for (let j = 0; j < size; ++j) {
            let r = Math.min(size - j - 1, Math.floor(generator.randomUnit() * (size - j)));
            swap(indices, r, size - j - 1);
        }
        return new PermutationMatrix(indices, type);
    }
    permuteInplace(obj: Matrix | Vector, type?: PermutationType) {
        let elementAtPosition = Array(this.permutations.length);
        let positionOfElement = Array(this.permutations.length);
        for (let i = 0; i < this.permutations.length; ++i) {
            elementAtPosition[i] = i;
            positionOfElement[i] = i;
        }
        if (type === undefined) type = this._type;
        if (obj instanceof Matrix) {
            if (type == PermutationType.Row)
                assert(obj.numRows() == this.permutations.length, "Incompatible sizes");
            else
                assert(obj.numCols() == this.permutations.length, "Incompatible sizes");
        } else {

            assert(obj.size() == this.permutations.length, "Incompatible sizes");
        }
        for (let i = 0; i < this.permutations.length; ++i) {
            let curElementIdx = this.permutations[i];
            let curElementPos = positionOfElement[curElementIdx];
            let otherElementIdx = elementAtPosition[i];
            if (obj instanceof Matrix) {
                if (type)
                    obj.swapRows(i, curElementPos);
                else
                    obj.swapColumns(i, curElementPos);
            }
            else {
                obj.swap(i, curElementPos);
            }
            positionOfElement[otherElementIdx] = curElementPos;
            elementAtPosition[i] = curElementIdx;
            positionOfElement[curElementIdx] = i;
            elementAtPosition[curElementIdx] = otherElementIdx;
        }
        return obj;
    }
    private permutations: number[];
    private _type: PermutationType;
    constructor(permutations: number[], type: PermutationType) {
        this.permutations = permutations.slice();
        this.type = type;
    }
    static identity(size: number, type: PermutationType): PermutationMatrix {
        let data = Array(size);
        for (let i = 0; i < size; ++i)
            data[i] = i;
        return new PermutationMatrix(data, type);
    }
    clone(): PermutationMatrix {
        return new PermutationMatrix(this.permutations.slice(), this._type);
    }
    get type() {
        return this._type;
    }
    set type(value: PermutationType) {
        this._type = value;
    }
    swap(i: number, j: number) {
        swap(this.permutations, i, j);
    }
    size() {
        return this.permutations.length;
    }
    findIndexByValue(value: number) {
        assert(value >= 0 && value < this.permutations.length, "Invalid value");
        let index = this.permutations.findIndex((v) => {
            return v == value;
        });
        assert(index != -1, "Invalid permutation");
        return index;
    }
    determinant(): number {
        // calc number of pairs (i, j) where i < j and p(i) > p(j)
        let d = 0;
        for (let i = 0; i < this.permutations.length; ++i) {
            for (let j = i; j < this.permutations.length; ++j) {
                if (this.permutations[i] > this.permutations[j])
                    d++;
            }
        }
        return d & 1 ? -1 : 1;
    }
    at(idx: number): number {
        return this.permutations[idx];
    }
    isValid(): boolean {
        let values = new Array(this.permutations.length);
        values.fill(0);
        this.permutations.forEach((v) => {
            if (v >= 0 && v < this.permutations.length)
                values[v]++;
        });
        for (const value of values) {
            if (value != 1) return false;
        }
        return true;
    }
    static inverse(permutations: number[]): number[] {
        let values = new Array(permutations.length);
        permutations.forEach((v, i) => {
            values[v] = i;
        });
        return values;
    }
    value(i: number): number {
        return this.permutations[i];
    }
    inverse(): PermutationMatrix {
        return new PermutationMatrix(PermutationMatrix.inverse(this.permutations), this._type);
    }
    array(): number[] {
        return this.permutations;
    }
    permuteVector(v: Vector): Vector {
        let result = v.clone();
        for (let i = 0; i < result.size(); ++i)
            result.set(i, v.get(this.value(i)));
        return result;
    }
    permuteMatrix(m: Matrix): Matrix {
        let result = m.clone();
        for (let i = 0; i < result.numCols(); ++i) {
            for (let j = 0; j < result.numRows(); ++j) {
                let { row, column } = this.unpermuteIndex(j, i);
                result.set(j, i, m.get(row, column));
            }
        }
        return result;
    }
    permuteSparseMatrix(m: SparseMatrixCSR): SparseMatrixCSR {
        let nonZeroElements: number[] = [];
        let innerIndices: number[] = [];
        let outerStarts: number[] = [0];
        if (this.type == PermutationType.Row) {
            assert(this.permutations.length == m.numRows(), "Number of rows doesn't match permutation length");
            for (let i = 0; i < this.permutations.length; ++i) {
                let row = this.permutations[i];
                let it = new SparseMatrixRowIterator(m, row);
                while (!it.isDone()) {
                    const { colIdx, value } = it.advance();
                    innerIndices.push(colIdx);
                    nonZeroElements.push(value);
                }
                outerStarts.push(innerIndices.length);
            }
        } else {
            assert(this.permutations.length == m.numCols(), "Number of cols doesn't match permutation length");
            for (let row = 0; row < m.numRows(); ++row) {
                let start = m.outerStart(row);
                let end = m.outerStart(row + 1);
                if (end == start) {
                    outerStarts.push(innerIndices.length);
                    continue;
                }
                let permutedElements = SparseVector.empty(m.numCols());
                for (let i = start; i < end; ++i)
                    permutedElements.set(this.findIndexByValue(m.innerIndex(i)), m.nonZeroElement(i));
                for (const { index, value } of permutedElements.elements) {
                    innerIndices.push(index);
                    nonZeroElements.push(value);
                }
                outerStarts.push(innerIndices.length);
            }
        }
        return new SparseMatrixCSR(m.numRows(), m.numCols(), nonZeroElements, innerIndices, outerStarts);
    }
    permuteIndex(row: number, column: number) {
        if (this.type == PermutationType.Row)
            row = this.findIndexByValue(row);
        else column = this.findIndexByValue(column);
        return { row, column };
    }
    unpermuteIndex(row: number, column: number) {
        if (this.type == PermutationType.Row)
            row = this.permutations[row];
        else column = this.permutations[column];
        return { row, column };
    }
    get(row: number, column: number): number {
        if (this.type == PermutationType.Row) {
            if (column == this.permutations[row]) return 1;
        } else {
            if (row == this.permutations[column]) return 1;
        }
        return 0;
    }
    toMatrix(): Matrix {
        let m: Matrix = Matrix.empty(this.permutations.length, this.permutations.length);
        for (let i = 0; i < this.permutations.length; ++i) {
            const index = this.permutations[i];
            if (this.type == PermutationType.Row)
                m.set(i, index, 1);
            else
                m.set(index, i, 1)
        }
        return m;
    }
    toTriplets(): Triplet[] {
        let triplets: Triplet[] = [];
        for (let i = 0; i < this.permutations.length; ++i) {
            const index = this.permutations[i];
            if (this.type == PermutationType.Row)
                triplets.push({ row: i, column: index, value: 1 });
            else
                triplets.push({ row: index, column: i, value: 1 });
        }
        return triplets;
    }
    toSparseMatrix(): SparseMatrixCSR {
        return SparseMatrixCSR.fromTriplets(this.permutations.length, this.permutations.length, this.toTriplets());
    }
}