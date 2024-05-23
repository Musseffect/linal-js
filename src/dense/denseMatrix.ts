import AbstractMatrix from "./abstractDenseMatrix";
import FullPivLU from "../solvers/linear systems/fullPivLU";
import PartialPivLU from "../solvers/linear systems/partialPivLU";
import Triplet from "../sparse/triplet";
import { assert, near, SmallTolerance, SmallestTolerance, swap } from "../utils";
import Vector from "./vector";
import { RealSchurDecomposition, SymmetricEigendecomposition } from "../solvers/linear systems/eigenvalues";

export default class Matrix extends AbstractMatrix {
    setFromMatrix(m: Matrix) {
        this.data = m.data.slice();
        this._numCols = m._numCols;
        this._numRows = m._numRows;
    }
    diag(shift: number = 0): Vector {
        let result = Vector.empty(Math.min(this.numRows() - Math.max(shift, 0), this.numCols() - Math.max(-shift, 0)));
        for (let i = 0; i < result.size(); ++i)
            result.set(i, this.get(i + Math.max(shift, 0), i + Math.max(-shift, 0)));
        return result;
    }
    setSubColumn(values: Vector, row: number, col: number): void {
        assert(row + values.size() <= this.numRows(), `Incorrect arguments: ${row} + ${values.size()} <= ${this.numRows()}`);
        for (let i = 0; i < values.size(); ++i)
            this.set(row + i, col, values.get(i));
    }
    setSubRow(values: Vector, row: number, col: number): void {
        assert(col + values.size() <= this.numCols(), `Incorrect arguments: ${col} + ${values.size()} <= ${this.numCols()}`);
        for (let i = 0; i < values.size(); ++i)
            this.set(row, col + i, values.get(i));
    }
    subRow(row: number, col: number, numCols: number) {
        assert(col + numCols <= this.numRows(), `Incorrect arguments: ${col} + ${numCols} <= ${this.numCols()}`);
        let v = Vector.empty(numCols);
        for (let i = 0; i < numCols; ++i)
            v.set(i, this.get(row, col + i));
        return v;
    }
    subColumn(row: number, col: number, numRows: number) {
        assert(row + numRows <= this.numRows(), `Incorrect arguments: ${col} + ${numRows} <= ${this.numRows()}`);
        let v = Vector.empty(numRows);
        for (let i = 0; i < numRows; ++i)
            v.set(i, this.get(row + i, col));
        return v;
    }
    shrinkCols(numCols: number) {
        assert(numCols <= this._numCols, "Incorrect number of columns");
        if (numCols == this.numCols()) return;
        // copy values for cols which will be kept
        for (let row = 1; row < this.numRows(); ++row) {
            for (let col = 0; col < numCols; ++col)
                this.data[row * numCols + col] = this.data[row * this.numCols() + col];
        }
        this._numCols = numCols;
        this.data.splice(this._numRows * this._numCols);
    }
    shrinkRows(numRows: number) {
        assert(numRows <= this._numRows, "Incorrect number of rows");
        if (numRows == this._numRows) return;
        this.data.splice(numRows * this._numCols);
        this._numRows = numRows;
    }
    private toIndex(row: number, column: number): number {
        return row * this._numCols + column;
    }
    swapColumns(column1: number, column2: number) {
        assert(Math.max(column1, column2) < this._numCols && Math.min(column1, column2) >= 0, "Invalid indices");
        if (column1 == column2) return;
        for (let i = 0; i < this._numRows; ++i)
            swap(this.data, this.toIndex(i, column1), this.toIndex(i, column2));
    }
    swapRows(row1: number, row2: number) {
        assert(Math.max(row1, row2) < this._numRows && Math.min(row1, row2) >= 0, "Invalid indices");
        if (row1 == row2) return;
        for (let i = 0; i < this._numCols; ++i)
            swap(this.data, this.toIndex(row1, i), this.toIndex(row2, i));
    }
    constructor(data: number[], numRows: number, numCols: number) {
        super(data, numRows, numCols);
        assert(this.data.length == numRows * numCols, "Wrong size of data array");
    }
    static generate(numRows: number, numCols: number, f: (r: number, c: number) => number): Matrix {
        let result = Matrix.empty(numRows, numCols);
        for (let j = 0; j < numRows; ++j) {
            for (let i = 0; i < numCols; ++i)
                result.set(j, i, f(j, i));
        }
        return result;
    }
    numCols(): number {
        return this._numCols;
    }
    numRows(): number {
        return this._numRows;
    }
    static random(numRows: number, numCols: number): Matrix {
        let data = [];
        for (let i = 0; i < numRows * numCols; i++)
            data.push(Math.random());
        return new Matrix(data, numRows, numCols);
    }
    static diag(data: number[]): Matrix {
        let m = Matrix.empty(data.length, data.length);
        for (let i = 0; i < data.length; ++i)
            m.set(i, i, data[i]);
        return m;
    }
    static empty(numRows: number, numCols: number): Matrix {
        let data: number[];
        (data = []).length = numRows * numCols;
        data.fill(0);
        return new Matrix(data, numRows, numCols);
    }
    static fromDiagonal(diagonal: Vector): Matrix {
        let mat = Matrix.empty(diagonal.size(), diagonal.size());
        for (let i = 0; i < diagonal.size(); ++i)
            mat.set(i, i, diagonal.get(i));
        return mat;
    }
    copy(): Matrix {
        return new Matrix(this.data.slice(), this._numRows, this._numCols);
    }
    clone(): Matrix {
        return new Matrix(this.data.slice(), this._numRows, this._numCols);
    }
    width(): number {
        return this._numCols;
    }
    height(): number {
        return this._numRows;
    }
    static identity(size: number): Matrix {
        let data: number[];
        (data = []).length = size * size;
        data.fill(0);
        for (let i = 0; i < size; i++) {
            data[i + i * size] = 1;
        }
        return new Matrix(data, size, size);
    }
    static kroneckerProduct(a: Matrix, b: Matrix): Matrix {
        let result = Matrix.empty(a.numRows() * b.numRows(), a.numCols() * b.numCols());
        for (let aRow = 0, resRow = 0; aRow < a.numRows(); ++aRow) {
            for (let bRow = 0; bRow < b.numRows(); ++bRow, ++resRow) {
                for (let aCol = 0, resCol = 0; aCol < a.numCols(); ++aCol) {
                    let aVal = a.get(aRow, aCol);
                    if (aVal == 0.0) continue;
                    for (let bCol = 0; bCol < b.numCols(); ++bCol, resCol) {
                        result.set(resRow, resCol, aVal * b.get(bRow, bCol));
                    }
                }
            }
        }
        return result;
    }
    static mul(a: Matrix, b: Matrix): Matrix {
        assert(a._numCols == b._numRows, "Matrix dimensions aren't compatible");
        let result = Matrix.empty(a.numRows(), b.numCols());
        //for each cell in the result
        for (let row = 0; row < a.numRows(); row++) {
            for (let col = 0; col < b.numCols(); col++) {
                let value = 0;
                for (let i = 0; i < a.numCols(); i++) {
                    value += a.get(row, i) * b.get(i, col);
                }
                result.set(row, col, value);
            }
        }
        return result;
    }
    static postMulVec(a: Matrix, b: Vector): Vector {
        assert(a._numCols == b.data.length, "Width of matrix isn't compatible with vector's length");
        let result = Vector.empty(a._numRows);
        for (let j = 0; j < a._numRows; j++) {
            let v = 0;
            for (let i = 0; i < a._numCols; i++) {
                v += a.get(j, i) * b.get(i);
            }
            result.set(j, v);
        }
        return result;
    }
    static preMulVec(a: Vector, b: Matrix): Vector {
        assert(b._numRows == a.data.length, "Width of matrix isn't compatible with vector's length");
        let result = Vector.empty(b._numCols);
        for (let row = 0; row < b._numRows; ++row) {
            let multiplier = a.get(row);
            for (let col = 0; col < b._numCols; ++col)
                result.set(col, result.get(col) + multiplier * b.get(row, col));
        }
        return result;
    }
    get(row: number, column: number): number {
        // assert(row < this.numRows() && row >= 0, `Invalid row ${row} < ${this.numRows()}`);
        // assert(column < this.numCols() && column >= 0, `Invalid column ${column} < ${this.numCols()}`);
        return this.data[this.toIndex(row, column)];
    }
    set(row: number, column: number, value: number): void {
        // assert(row < this.numRows() && row >= 0, `Invalid row ${row} < ${this.numRows()}`);
        // assert(column < this.numCols() && column >= 0, `Invalid column ${column} < ${this.numCols()}`);
        this.data[this.toIndex(row, column)] = value;
    }
    transposeInPlace(): Matrix {
        if (!this.isSquare()) {
            let copy = this.data.slice();
            let temp = this._numRows;
            this._numRows = this._numCols;
            this._numCols = temp;
            for (let row = 0, oldIdx = 0; row < this._numCols; ++row) {
                for (let col = 0; col < this._numRows; ++col, ++oldIdx)
                    this.data[this.toIndex(col, row)] = copy[oldIdx];
            }
            /*
            let isInRightPlace = new Array().fill(false);
            isInRightPlace[0] = true;
            isInRightPlace[-1] = true;
            for (let row = 0; row < this._numRows; ++row)
            {
                for (let col = 0; col < this._numCols; ++col)
                {
                    let initialIdx = this.toIndex(row, col);
                    if (isInRightPlace(initialIdx)) continue;
                    let curRow = row;
                    let curCol = col;
                    let curIdx = initialIdx;
                    let targetIndex = 0;
                    let curValue = this.data[idx];
                    {
                        targetIndex = curRow + curCol * this._numRows;
                        isInRightPlace[targetIndex] = true;
                        const temp = this.data[targetIndex];
                        this.data[targetIndex] = curValue;
                        curValue = temp;
                        curCol = targetIndex % this._numCols;
                        curRow = Math.floor(targetIndex / this._numRows);
                    }
                    while(targetIndex != initialIdx)
                }
            }
            */
        }
        else {
            for (let row = 0; row < this._numRows; ++row) {
                for (let col = row + 1; col < this._numCols; ++col) {
                    swap(this.data, col + row * this._numCols, col * this._numRows + row);
                }
            }
        }
        return this;
    }
    transpose(): Matrix {
        let result = [];
        for (let col = 0; col < this._numCols; col++) {
            for (let row = 0; row < this._numRows; row++) {
                result.push(this.data[this.toIndex(row, col)]);
            }
        }
        return new Matrix(result, this._numCols, this._numRows);
    }
    // partial pivot gauss elimination
    static solve(A: Matrix, b: Vector): Vector {
        assert(A._numCols == b.data.length, "Width of matrix isn't compatible with vector's length");
        assert(A._numCols == A._numRows, "Non-square matrix");
        var rang = b.size();
        var x = Vector.empty(rang);
        let epsilon = 0.001
        var indexes = new Array(rang);
        for (var i = 0; i < rang; i++) {
            indexes[i] = i;
        }
        for (var l = 0; l < rang; l++) {
            var max = l;
            for (var i = l + 1; i < rang; i++) {
                if (Math.abs(A.get(indexes[i], l)) > Math.abs(A.get(indexes[max], l)))
                    max = i;
            }
            if (max != l) {
                var temp = indexes[l];
                indexes[l] = indexes[max];
                indexes[max] = temp;
            }
            if (Math.abs(A.get(indexes[l], l)) < epsilon) {
                for (var i = 0; i < rang; i++)
                    x.set(i, 0.0);
                return x;
            }
            for (var i = l + 1; i < rang; i++)
                A.set(indexes[l], i, A.get(indexes[l], i) / A.get(indexes[l], l));
            b.set(indexes[l], b.get(indexes[l]) / A.get(indexes[l], l));
            A.set(indexes[l], l, 1);

            for (var i = l + 1; i < rang; i++) {
                for (var k = l + 1; k < rang; k++)
                    A.set(indexes[i], k, A.get(indexes[i], k) - A.get(indexes[i], l) * A.get(indexes[l], k));
                b.set(indexes[i], b.get(indexes[i]) - A.get(indexes[i], l) * b.get(indexes[l]));
                A.set(indexes[i], l, 0);
            }
        }
        x.set(rang - 1, b.get(indexes[rang - 1]));
        for (var i = rang - 2; i > -1; i--) {
            var k = 0.;
            for (var j = i + 1; j < rang; j++) {
                k = k + A.get(indexes[i], j) * x.get(j);
            }
            x.set(i, b.get(indexes[i]) - k);
        }
        return x;
    }
    static fromTriplets(numRows: number, numCols: number, triplets: Triplet[]): Matrix {
        let result = Matrix.empty(numRows, numCols);
        for (const triplet of triplets)
            result.set(triplet.row, triplet.column, triplet.value);
        return result;
    }
    toTriplets(tolerance: number = SmallestTolerance): Triplet[] {
        let result: Triplet[] = [];
        for (let row = 0; row < this._numRows; ++row) {
            for (let column = 0; column < this._numCols; ++column) {
                let value = this.get(row, column);
                if (Math.abs(value) <= tolerance) continue;
                result.push({ row, column, value });
            }
        }
        return result;
    }
    determinant(): number {
        return new PartialPivLU(this).determinant();
    }
    inverse(tolerance: number = SmallTolerance): Matrix {
        // return new PartialPivLU(this).inverse();
        return PartialPivLU.solveMatrix(this, Matrix.identity(this.width()));
    }
    minor(row: number, col: number): Matrix {
        let matrix = Matrix.empty(this._numRows, this._numCols);
        let i = 0;
        for (let rowIdx = 0; rowIdx < this._numRows; ++rowIdx) {
            if (rowIdx == row) continue;
            i++;
            for (let column = 0; column < this._numCols; ++column) {
                if (column == col) continue;
                let j = 0;
                matrix.set(i, j, this.get(rowIdx, column));
                j++;
            }
        }
        return matrix;
    }
    // analytic solution
    inverseNaive(): Matrix {
        assert(this.isSquare(), "Non-square matrix");
        assert(this.width() < 7, "Factorial complexity");
        let d = this.determinantNaive();
        let result = Matrix.empty(this._numCols, this._numRows);
        let minors = this.clone();
        for (let column = 0; column < this._numCols; ++column) {
            for (let row = 0; row < this._numRows; ++row) {
                let minorValue = this.cofactor(column, row);
                minors.set(row, column, ((row + column) & 1) ? -minorValue : minorValue);
                minorValue = minorValue / d;
                result.set(row, column, (row + column) & 1 ? -minorValue : minorValue);
            }
        }
        return result;
    }
    determinantNaive() {
        assert(this.isSquare(), "Non-square matrix");
        assert(this.width() < 7, "Factorial complexity");
        if (this._numCols != this._numRows) return 0.0;

        const self = this;
        const step = (r: number, c: number, rowList: number[], colList: number[], depth: number): number => {
            const curValue = self.get(rowList[r], colList[c]);
            if (Math.abs(curValue) < SmallestTolerance) return 0.0;

            let result = 0.0;
            if (self._numCols - depth > 0) {
                swap(rowList, r, self._numCols - depth);
                swap(colList, c, self._numCols - depth);
                for (let i = 0; i < self._numCols - depth; ++i)
                    result += step(0, i, rowList, colList, depth + 1);
                swap(rowList, r, self._numCols - depth);
                swap(colList, c, self._numCols - depth);
            } else {
                return curValue;
            }
            const curNumCols = this._numCols - depth;
            if (Math.max(curNumCols - r - 1, 0) + Math.max(curNumCols - c - 1, 0) & 1)
                result = -result;

            result *= curValue * (((r + c) & 1) == 1 ? -1 : 1);

            return result;
        };
        let rowList = new Array(this._numCols);
        let colList = new Array(this._numCols);
        for (let i = 0; i < this._numCols; ++i)
            rowList[i] = colList[i] = i;
        let result = 0.0;
        for (let i = 0; i < this._numCols; ++i)
            result += step(0, i, rowList, colList, 1);
        return result;
    }
    trace(): number {
        let result = 0;
        for (let i = 0; i < Math.min(this.numRows(), this.numCols()); ++i)
            result += this.get(i, i);
        return result;
    }
    cofactor(r: number, c: number): number {
        assert(this.isSquare(), "Non-square matrix");
        if (this._numCols == 1) return 1.0;
        let subMatrix = Matrix.empty(this._numCols - 1, this._numRows - 1);
        for (let row = 0, i = 0; row < this._numRows; ++row) {
            if (row == r) continue;
            for (let col = 0, j = 0; col < this._numCols; ++col) {
                if (col == c) continue;
                subMatrix.set(i, j, this.get(row, col));
                ++j;
            }
            ++i;
        }
        return subMatrix.determinantNaive();
    }
    print(fractionDigits: number): string {
        if (!fractionDigits)
            fractionDigits = 4;
        let result = "";
        for (let j = 0; j < this._numRows; j++) {
            result += j > 0 ? "\n| " : "| ";
            for (let i = 0; i < this._numCols; i++)
                result += this.data[i + j * this._numCols].toFixed(fractionDigits) + " "
            result += "|";
        }
        return result;
    }
    toString(precision: number = 4): string {
        let result = "[";
        for (let j = 0; j < this._numRows; ++j) {
            if (j != 0)
                result += ",";
            result += "\n\t[";
            for (let i = 0; i < this._numCols; ++i) {
                if (i != 0) {
                    result += ", "
                }
                let value = this.get(j, i);
                if (Math.abs(value) < SmallestTolerance)
                    value = 0;
                result += value.toFixed(precision);
            }
            result += "]";
        }
        return result + "\n]";
    }
    static near(a: AbstractMatrix, b: AbstractMatrix, threshold: number = SmallTolerance): boolean {
        assert(a.numRows() == b.numRows() && a.numCols() == b.numCols(), "Sizes don't match");
        for (let i = 0; i < a.data.length; i++) {
            if (Math.abs(a.data[i] - b.data[i]) > threshold)
                return false;
        }
        return true;
    }
    // Frobenius norm
    static l2Distance(a: AbstractMatrix, b: AbstractMatrix): number {
        assert(a.numRows() == b.numRows() && a.numCols() == b.numCols(), "Sizes don't match");
        let result = 0.0;
        for (let i = 0; i < a.data.length; ++i) {
            const value = a.data[i] - b.data[i];
            result = value * value;
        }
        return result
    }
    // max norm
    static lInfDistance(a: AbstractMatrix, b: AbstractMatrix): number {
        assert(a.numRows() == b.numRows() && a.numCols() == b.numCols(), "Sizes don't match");
        let result = 0.0;
        for (let i = 0; i < a.data.length; ++i) {
            const value = a.data[i] - b.data[i];
            result = Math.max(result, Math.abs(value));
        }
        return result
    }
    getColumn(col: number): Vector {
        let values = Vector.empty(this.numRows());
        for (let i = 0; i < this.numRows(); ++i)
            values.set(i, this.get(i, col));
        return values;
    }
    getRow(row: number): Vector {
        let values = Vector.empty(this.numCols());
        for (let i = 0; i < this.numCols(); ++i)
            values.set(i, this.get(row, i));
        return values;
    }
    setColumn(col: number, values: Vector): void {
        assert(values.size() == this.numRows(), "Invalid size");
        for (let i = 0; i < values.size(); ++i)
            this.set(i, col, values.get(i));
    }
    setRow(row: number, values: Vector): void {
        assert(values.size() == this.numCols(), "Invalid size");
        for (let i = 0; i < values.size(); ++i)
            this.set(row, i, values.get(i));
    }
    static sub(a: Matrix, b: Matrix) {
        let result = a.clone();
        result.subSelf(b);
        return result;
    }
    static add(a: Matrix, b: Matrix) {
        let result = a.clone();
        result.addSelf(b);
        return result;
    }
    makeSquare(): Matrix {
        if (this.isSquare()) return this.clone();
        let size = Math.max(this.width(), this.height());
        let result = Matrix.empty(size, size);
        for (let j = 0; j < this.height(); ++j) {
            for (let i = 0; i < this.width(); ++i)
                result.set(j, i, this.get(j, i));
        }
        return result;
    }

    isOrthogonal(): boolean {
        let S = Matrix.mul(this, this.transpose());
        return S.isIdentity();
    }
    isDiagonallyDominant(strict: boolean = false): boolean {
        for (let row = 0; row < this.numRows(); ++row) {
            let sum = 0;
            for (let col = 0; col < this.numCols(); ++col)
                sum += Math.abs(this.get(row, col));
            if (sum > 2 * Math.abs(this.get(row, row))) return false;
            else if (strict && sum == 2 * Math.abs(this.get(row, row))) return false;
        }
        return true;
    }
    isSymmetric(tolerance: number = SmallTolerance): boolean {
        if (!this.isSquare()) return false;
        for (let i = 1; i < this._numCols; ++i) {
            for (let j = 0; j < i; ++j) {
                if (!near(this.get(i, j), this.get(j, i), tolerance)) return false;
            }
        }
        return true;
    }
    // test
    /** Checks that all of the elements apart from (i, i + offset) are zero
     */
    isDiagonal(tolerance: number = SmallTolerance, offset: number = 0): boolean {
        for (let row = 0; row < this._numRows; ++row) {
            for (let col = 0; col < this._numCols; ++col) {
                if (row == col + offset) continue;
                if (Math.abs(this.get(row, col)) > tolerance) return false;
            }
        }
        return true;
    }
    isTriangular(upper: boolean = false, tolerance: number = SmallTolerance) {
        for (let i = 1; i < Math.min(this._numCols, this._numRows); ++i) {
            for (let j = 0; j < i; ++j) {
                let value = upper ? this.get(i, j) : this.get(j, i);
                if (Math.abs(value) > tolerance) return false;
            }
        }
        return true;
    }
    isIdentity() {
        for (let row = 0; row < this._numRows; ++row) {
            for (let col = 0; col < this._numCols; ++col) {
                if (!near(this.get(row, col), row == col ? 1 : 0)) return false;
            }
        }
        return true;
    }
    isHessenberg(upper: boolean = true, tolerance: number = SmallTolerance) {
        if (!this.isSquare()) return false;
        if (upper) {
            for (let row = 2; row < this.numRows(); ++row) {
                for (let col = 0; col + 1 < row; ++col) {
                    let value = this.get(row, col);
                    if (Math.abs(value) > tolerance) return false;
                }
            }
        } else {
            for (let row = 0; row + 2 < this.numRows(); ++row) {
                for (let col = row + 2; col < this.numCols(); ++col) {
                    let value = this.get(row, col);
                    if (Math.abs(value) > tolerance) return false;
                }
            }
        }
        return true;
    }
    isTridiagonal(tolerance: number = SmallTolerance): boolean {
        if (!this.isSquare()) return false;
        for (let row = 0; row + 2 < this.numRows(); ++row) {
            for (let col = row + 2; col < this.numCols(); ++col) {
                let value = this.get(row, col);
                if (Math.abs(value) > tolerance) return false;
                value = this.get(col, row);
                if (Math.abs(value) > tolerance) return false;
            }
        }
        return true;
    }
}