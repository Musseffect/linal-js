import Matrix from "../../dense/denseMatrix";
import { assert, assertFail } from "../../utils";
import Vector from "../../dense/vector";
import { givens, applyGivensFromLeft, applyGivensFromRight, applyTransposeGivensFromRight } from "./givensRotation";
import { applyHouseholderFromLeft, applyHouseholderFromRight, calcHouseholderVectorCol } from "./hausholderReflection";

const SolverName = "QR";

export enum ZeroingMethod {
    Givens = 0,
    Housholder = 1
};

export enum OrthogonalDecompositionType {
    QR,
    RQ,
    LQ,
    QL
}

export class OrthogonalDecomposition {
    A: Matrix = null;
    q: Matrix;
    t: Matrix;
    _zeroingMethod: ZeroingMethod;
    _makeCompact: boolean;
    _type: OrthogonalDecompositionType;
    _determinantSign: number;

    constructor(A: Matrix | null, zeroingMethod: ZeroingMethod = ZeroingMethod.Housholder, makeCompact: boolean = false, type: OrthogonalDecompositionType) {
        this._makeCompact = makeCompact;
        this._zeroingMethod = zeroingMethod;
        this._type = type;
        this.factorize(A);
    }
    get Q(): Matrix {
        return this.q;
    }
    get T(): Matrix {
        return this.t;
    }
    private makeQR(A: Matrix) {
        let r: Matrix = A.clone();
        if (this.A.numCols() > this.A.numRows())
            r.transposeInPlace();
        let q: Matrix = Matrix.identity(r.numRows());
        if (this._zeroingMethod == ZeroingMethod.Housholder) {
            let numOps = 0;
            for (let col = 0; col < this.A.numCols(); ++col) {
                // numRows cannot be less than numCols, when numRows == numCols last column should be skipped
                if (this.A.numRows() == col + 1) continue;
                numOps++;
                let v = calcHouseholderVectorCol(r, col, col);
                applyHouseholderFromLeft(v, r, col);
                applyHouseholderFromRight(v, q, col);
            }
            this._determinantSign = (numOps & 1 ? -1.0 : 1.0);

        } else if (this._zeroingMethod == ZeroingMethod.Givens) {
            for (let j = 0; j != r.numCols(); ++j) {
                for (let i = r.numRows() - 1; i > j; --i) {
                    let a = r.get(j, j);
                    let b = r.get(i, j);
                    let givensCoeffs = givens(a, b);
                    applyGivensFromLeft(r, givensCoeffs, i, j);
                    applyTransposeGivensFromRight(q, givensCoeffs, i, j);
                }
            }
            this._determinantSign = 1.0;
        }
        if (this._makeCompact) {
            r.shrinkRows(r.numCols());
            q.shrinkCols(r.numCols());
        }
        this.q = q;
        this.t = r;
    }
    private makeRQ(A: Matrix) {
        let r: Matrix = A.clone();
        if (this.A.numRows() > this.A.numCols())
            r.transposeInPlace();
        let q: Matrix = Matrix.identity(r.numCols());
        throw new Error("Not implemented");
        if (this._zeroingMethod == ZeroingMethod.Housholder) {



        } else if (this._zeroingMethod == ZeroingMethod.Givens) {

        }
        if (this._makeCompact) {
            r.shrinkCols(r.numRows());
            q.shrinkRows(r.numRows());
        }
        this.q = q;
        this.t = r;
    }
    private makeQL(A: Matrix) {
        let q: Matrix;
        let l: Matrix;
        throw new Error("Not implemented");
        if (this._zeroingMethod == ZeroingMethod.Housholder) {



        } else if (this._zeroingMethod == ZeroingMethod.Givens) {

        }
        this.q = q;
        this.t = l;
    }
    private makeLQ(A: Matrix) {
        let q: Matrix;
        let l: Matrix;
        throw new Error("Not implemented");
        if (this._zeroingMethod == ZeroingMethod.Housholder) {



        } else if (this._zeroingMethod == ZeroingMethod.Givens) {

        }
        this.q = q;
        this.t = l;
    }
    factorize(A: Matrix) {
        this.A = A;
        if (this.A == null) return;
        switch (this._type) {
            case OrthogonalDecompositionType.LQ:
                this.makeLQ(A); break;
            case OrthogonalDecompositionType.QL:
                this.makeQL(A); break;
            case OrthogonalDecompositionType.QR:
                this.makeQR(A); break;
            case OrthogonalDecompositionType.RQ:
                this.makeRQ(A); break;
        }
    }
    set type(value: OrthogonalDecompositionType) {
        this._type = value;
    }
    get type() {
        return this._type;
    }
}

export class QR {
    A: Matrix = null;
    q: Matrix;
    r: Matrix;
    qDet: number = 0;
    _zeroingMethod: ZeroingMethod;
    _makeCompact: boolean;
    private factorizeGivens() {
        // make zeroes for each column from the bottom to rowIdx - 1
        for (let j = 0; j != this.r.numCols(); ++j) {
            // | a
            // | x
            // | b  
            // | 0  /|\
            // | 0 / | \
            for (let i = this.r.numRows() - 1; i > j; --i) {
                let a = this.r.get(j, j);
                let b = this.r.get(i, j);
                let givensCoeffs = givens(a, b);
                // apply to r
                applyGivensFromLeft(this.r, givensCoeffs, i, j);
                //this.r.set(j, j, givensCoeffs.r);
                //this.r.set(i, j, 0.0);
                // apply to q
                applyTransposeGivensFromRight(this.q, givensCoeffs, i, j);
            }
        }
        this.qDet = 1.0;
    }
    private factorizeHousholder() {
        let numOps = 0;
        for (let col = 0; col < this.r.numCols(); ++col) {
            // numRows cannot be less than numCols, when numRows == numCols last column should be skipped
            if (this.r.numRows() == col + 1) continue;
            numOps++;
            // calc housholder
            let v = calcHouseholderVectorCol(this.r, col, col);
            applyHouseholderFromLeft(v, this.r, col);

            /*this.r.set(col, col, v.get(0));
            for (let row = col + 1; row < this.A.numRows(); ++row)
                this.r.set(row, col, 0.0);*/

            // R = (Q3Q2Q1)^A = QT*A
            // QT = Q3(Q2(Q1*I))
            // Q = ((I*Q1)Q2)Q3)
            // QNT = QN because householder matrix is symmetric
            applyHouseholderFromRight(v, this.q, col);
        }
        this.qDet = (numOps & 1 ? -1.0 : 1.0);
    }
    // Calc A=QR if numRows >= numCols, otherwise AT=QR is computed
    public factorize(A: Matrix | null) {
        this.q = null;
        this.r = null;
        this.A = A;
        if (this.A == null) return;

        this.r = this.A.clone();
        if (this.A.numCols() > this.A.numRows())
            this.r.transposeInPlace();
        this.q = Matrix.identity(this.r.numRows());
        switch (this._zeroingMethod) {
            case ZeroingMethod.Givens:
                this.factorizeGivens();
                break;
            case ZeroingMethod.Housholder:
                this.factorizeHousholder();
                break;
            default:
                assertFail("Invalid value");
        }
        if (this.makeCompact) {
            this.r.shrinkRows(this.r.numCols());
            this.q.shrinkCols(this.r.numCols());
        }
    }
    constructor(A: Matrix | null, zeroingMethod: ZeroingMethod = ZeroingMethod.Housholder, makeCompact: boolean = false) {
        this._makeCompact = makeCompact;
        this._zeroingMethod = zeroingMethod;
        this.factorize(A);
    }
    set makeCompact(value: boolean) {
        this._makeCompact = value;
    }
    set zeroingMethod(value: ZeroingMethod) {
        this._zeroingMethod = value;
    }
    get Q(): Matrix {
        return this.q;
    }
    get R(): Matrix {
        return this.r;
    }
    private solveMatrix(rhs: Matrix): Matrix {
        throw new Error("Not implemented");
    }
    private solveVector(rhs: Vector): Vector {
        assert(this.A.numRows() == rhs.size(), "Incompatible sizes");
        if (this.A.numRows() >= this.A.numCols()) {
            // overdetermined system
            // Ax = b -> QRx = b -> Rx = QT b
            // x = R_1^-1(Q_1^T * b), solve by back substitution
            let b = Matrix.preMulVec(rhs, this.Q);
            let x = Vector.empty(this.A.numCols());
            // back substitution
            for (let row = this.r.numCols() - 1; row >= 0; --row) {
                let sum = b.get(row);
                for (let col = row + 1; col < this.r.numCols(); ++col)
                    sum -= this.R.get(row, col) * x.get(col);
                x.set(row, sum / this.R.get(row, row));
            }
            return x;
        }
        else if (this.A.numCols() > this.A.numRows()) {
            // underdetermined system
            // AT=QR, x = Q(RT^-1 b)
            // forward substitution RTx = b
            // !!! this is not least square norm solution
            let x = Vector.empty(this.A.numCols());
            for (let row = 0; row < this.A.numRows(); ++row) {
                let sum = rhs.get(row);
                for (let col = 1; col < row; ++col)
                    sum -= this.R.get(col, row) * x.get(col);
                x.set(row, sum / this.R.get(row, row));
            }
            x = Matrix.postMulVec(this.Q, x);
            return x;
        }
    }
    solve(rhs: Matrix | Vector): Matrix | Vector {
        if (rhs instanceof Matrix)
            return this.solveMatrix(rhs);
        return this.solveVector(rhs);
    }
    // returns inverse or pseudoinverse depending on the rank of the system
    inverse(): Matrix {
        let inverse: Matrix = Matrix.empty(this.A.numCols(), this.A.numRows());
        if (this.A.numRows() >= this.A.numCols()) {
            for (let row = this.A.numCols() - 1; row >= 0; --row) {
                for (let col = 0; col < this.A.numRows(); ++col) {
                    let sum = this.Q.get(col, row);
                    for (let idx = row + 1; idx < this.A.numCols(); ++idx)
                        sum -= this.R.get(row, idx) * inverse.get(idx, col);
                    inverse.set(row, col, sum / this.R.get(row, row));
                }
            }
            // x = R_1^-1 * Q_1^T 
            // solve R_1 x = Q_1^T
        } else {
            // RT A^-1 = Q
            /*
            for (let row = 0; row < this.A.numRows(); ++row) {
                for (let col = 0; col <= row; ++col) {
                    let rhs = (col == row ? 1 : 0);
                    for (let idx = col; idx < row; ++idx)
                        rhs -= inverse.get(idx, col) * this.R.get(idx, row);
                    inverse.set(row, col, rhs / this.R.get(row, row));
                }
            }
            inverse = Matrix.mul(this.Q, inverse);
            */
            // Alternative
            for (let row = this.A.numRows() - 1; row >= 0; --row) {
                for (let col = 0; col < this.A.numCols(); ++col) {
                    let sum = this.Q.get(col, row);
                    for (let idx = row + 1; idx < this.A.numRows(); ++idx)
                        sum -= this.R.get(row, idx) * inverse.get(col, idx);
                    inverse.set(col, row, sum / this.R.get(row, row));
                }
            }
        }
        return inverse;
    }
    determinant(): any {
        let result = this.qDet;
        for (let i = 0; i < Math.min(this.r.numCols(), this.r.numCols()); ++i)
            result *= this.r.get(i, i);
        return result;
    }
}

// todo (NI): column pivoting, full pivoting

/**
class PivotedQR
{
    p:PermutationMatrix = null;
    q:Matrix = null;
    r:Matrix = null;
    get Q() {
        return this.q;
    }
    get R() {
        return this.r;
    }
    get P() {
        return this.p;
    }
    constructor(A:Matrix) {
        this.factorize(A);
    }
    factorize(A:Matrix) {
        this.p = null;
        if (A == undefined) return;
        let r = A.clone();
    }
    determinant(): number {

    }
    solve(rhs:Vector):Vector {

    }
    inverse():Matrix {
        
    }
}
 */