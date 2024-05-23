import AbstractDenseMatrix from "./abstractDenseMatrix";
import Matrix from "./denseMatrix";
import { assert } from "../utils";


export enum TriMatrixType {
    lower = 0,
    upper = 1
}

// todo (NI): add column major storage option
export class TriMatrix {
    private type: TriMatrixType;
    private data: number[];
    private width: number;
    constructor(width: number, type: TriMatrixType) {
        // todo (NI): support rectangular matrices
        let size = width * (width + 1) / 2;
        this.data = new Array(size).fill(0);
        this.width = width;
        throw new Error("Not implemented");
    }
    private fromIndex(index: number): { row: number, column: number } {
        switch (this.type) {
            case TriMatrixType.lower:
                let row = Math.floor((Math.sqrt(1 + 8 * index) - 1) / 2);
                let column = index - row * (row + 1) / 2;
            case TriMatrixType.upper:
                throw new Error("Not implemented");
        }
    }
    private toIndex(row: number, column: number): number {
        switch (this.type) {
            case TriMatrixType.lower:
                if (column < row) return -1;
                return column + row * (row + 1) / 2;
            case TriMatrixType.upper:
                if (column > row) return -1;
            //return row + (this.size - column) * ((this.size - column) + 1) / 2;
        }
    }
    get(row: number, column: number): number {
        let index = this.toIndex(row, column);
        if (index < 0) return 0.0;
        return this.data[index];
    }
    set(row: number, column: number, value: number): void {
        let index = this.toIndex(row, column);
        assert(index < 0, "Invalid index");
        this.data[index] = value;
    }
    toMatrix(): Matrix {
        let result = Matrix.empty(this.width, this.width);
        for (let i = 0; i < this.data.length; ++i) {
            const { row, column } = this.fromIndex(i);
            result.set(row, column, this.data[i]);
        }
        return result;
    }
}

// todo (NI): add shift
export class DiagonalMatrixView {
    private m: AbstractDenseMatrix;
    constructor(m: AbstractDenseMatrix) {
        this.m = m;
    }
    get(row: number, column: number) {
        if (row != column) return 0.0;
        return this.m.get(row, column);
    }
    toMatrix() {
        let result = Matrix.empty(this.m.numRows(), this.m.numCols());
        for (let i = 0; i < Math.min(this.m.numRows(), this.m.numCols()); ++i)
            result.set(i, i, this.m.get(i, i));
        return result;
    }
}

// todo (NI): add BandedMatrixView
export class TriMatrixView {
    private m: AbstractDenseMatrix;
    private type: TriMatrixType;
    private diagonal: number | null;
    constructor(m: AbstractDenseMatrix, type: TriMatrixType, diagonal: number | null = null) {
        this.m = m;
        this.type = type;
        this.diagonal = diagonal;
    }
    get(row: number, column: number): number {
        if (row == column && this.diagonal != null)
            return this.diagonal;
        switch (this.type) {
            case TriMatrixType.lower:
                if (column > row)
                    return 0.0;
                break;
            case TriMatrixType.upper:
                if (column < row)
                    return 0.0;
                break;
        }
        return this.m.get(row, column);
    }
    toMatrix(): Matrix {
        let result = Matrix.empty(this.m.numRows(), this.m.numCols());
        switch (this.type) {
            case TriMatrixType.lower:
                for (let j = 0; j < result.height(); ++j) {
                    for (let i = 0; i <= j; ++i)
                        result.set(j, i, this.get(j, i));
                }
                return result;
            case TriMatrixType.upper:
                for (let j = 0; j < result.height(); ++j) {
                    for (let i = j; i < result.width(); ++i)
                        result.set(j, i, this.get(j, i));
                }
                return result;
        }
    }
    /*toTriangularMatrix(): TriMatrix {
        throw new Error("Not implemented");
    }*/
}