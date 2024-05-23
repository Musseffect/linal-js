import AbstractMatrix from "../abstractMatrix";
import { assert, Tolerance } from "../utils";

abstract class IConstMatrix {
    abstract get(row: number, col: number): number;
    abstract numRows(): number;
    abstract numCols(): number;
    isSquare(): boolean {
        return this.numCols() == this.numRows();
    }
    height(): number {
        return this.numRows();
    }
    width(): number {
        return this.numCols();
    }
}

export default abstract class AbstractDenseMatrix extends AbstractMatrix {
    data: number[];
    _rowMajor: boolean = true;
    index(row: number, col: number) {
        return (this._rowMajor ? row * this.numCols() + col : col * this.numRows() + row);
    }
    isRowMajor(): boolean {
        return this._rowMajor;
    }
    constructor(data: number[], numRows: number, numCols: number) {
        super(numRows, numCols);
        this.data = data;
    }
    get(row: number, col: number): number {
        return this.data[this.index(row, col)];
    }
    set(row: number, col: number, value: number): void {
        this.data[this.index(row, col)] = value;
    }
    values(): number[] {
        return this.data;
    }
    addSelf(a: AbstractDenseMatrix): void {
        const rows = this.numRows();
        const cols = this.numCols();
        assert(a.numRows() == rows && a.numCols() == cols, "Sizes don't match");
        for (let i = 0; i < this.data.length; ++i)
            this.data[i] += a.data[i];
    }
    subSelf(a: AbstractDenseMatrix): void {
        const rows = this.numRows();
        const cols = this.numCols();
        assert(a.numRows() == rows && a.numCols() == cols, "Sizes don't match");
        for (let i = 0; i < this.data.length; ++i)
            this.data[i] -= a.data[i];
    }
    mulSelfPointwise(a: AbstractDenseMatrix): void {
        const rows = this.numRows();
        const cols = this.numCols();
        assert(a.numRows() == rows && a.numCols() == cols, "Sizes don't match");
        for (let i = 0; i < this.data.length; ++i)
            this.data[i] *= a.data[i];
    }
    divSelfPointwise(a: AbstractDenseMatrix): void {
        const rows = this.numRows();
        const cols = this.numCols();
        assert(a.numRows() == rows && a.numCols() == cols, "Sizes don't match");
        for (let i = 0; i < this.data.length; ++i)
            this.data[i] *= a.data[i];
    }
    scaleSelf(scalar: number): void {
        for (let i = 0; i < this.data.length; ++i)
            this.data[i] *= scalar;
    }
    l2SquaredNorm(): number {
        let result = 0.0;
        for (let value of this.data)
            result += value * value;
        return result;
    }
    l2Norm(): number {
        return Math.sqrt(this.l2SquaredNorm());
    }
    lInfNorm(): number {
        let result = 0.0;
        for (let value of this.data)
            result = Math.max(result, Math.abs(value));
        return result;
    }
}