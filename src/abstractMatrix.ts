


export default abstract class AbstractMatrix {
    protected _numRows: number;
    protected _numCols: number;
    constructor(numRows: number, numCols: number) {
        this._numRows = numRows;
        this._numCols = numCols;
    }
    public numRows(): number {
        return this._numRows;
    }
    public numCols(): number {
        return this._numCols;
    }
    public width(): number {
        return this._numCols;
    }
    public height(): number {
        return this._numRows;
    }
    public isSquare(): boolean {
        return this.numCols() == this.numRows();
    }
}