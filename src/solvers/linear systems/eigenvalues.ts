import Matrix from "../../dense/denseMatrix";
import { SmallestTolerance, Tolerance, assert } from "../../utils";
import Vector from "../../dense/vector";
import { ConvergenseFailureException } from "./exceptions";
import { makeHessenbergInplace, makeTridiagonalInplace } from "./hessenbergMatrix";
import { applyHouseholderFromLeft, applyHouseholderFromRight } from "./hausholderReflection";
import { givens, givensCoeffs } from "./givensRotation";
import { complex } from "../../complex";
import { jacobiRotation } from "./jacobiRotation";

// todo (NI): Givens rotations, complex eigenvalues
/**
 * Compute real parts of eigenvalues with Francis double step QR algorithm for arbitary square matrix
 * @param A Input matrix
 * @param numIters number of QR iterations
 * @param tolerance tolerance for zeroed elements
 * @returns array of real eigenvalues
 */
export function calcEigenvalues(A: Matrix, numIters: number, tolerance: number): number[] {
    assert(A.isSquare(), "Expected square matrix");
    let eigenvalues: number[] = new Array(A.numCols());
    if (A.numCols() == 1) {
        eigenvalues[0] = A.get(0, 0);
        return eigenvalues;
    }
    else if (A.numCols() == 2) {
        let b = A.get(0, 0) + A.get(1, 1);
        let c = A.get(0, 0) * A.get(1, 1) - A.get(0, 1) * A.get(1, 0);
        let D = (b * b - 4.0 * c);
        eigenvalues[0] = b * 0.5;
        eigenvalues[1] = b * 0.5;
        if (D > 0) {
            D = Math.sqrt(D);
            eigenvalues[0] += D * 0.5;
            eigenvalues[1] -= D * 0.5;
        }
        return eigenvalues;
    }
    A = A.clone();

    let u = Vector.empty(A.numCols());
    if (!A.isHessenberg()) {
        // turn into hessenberg matrix
        for (let i = 0; i < u.size() - 2; i++) {
            let xNorm = 0.0;
            for (let k = 0, j = i + 1; j < u.size(); j++, k++) {
                u.set(k, A.get(j, i));
                xNorm += u.get(k) * u.get(k);
            }
            let ro = -Math.sign(A.get(i + 1, i));
            let uNorm = xNorm - A.get(i + 1, i) * A.get(i + 1, i);
            u.set(0, u.get(0) - ro * Math.sqrt(xNorm));
            uNorm += u.get(0) * u.get(0);
            uNorm = Math.sqrt(uNorm);
            u.scaleSelf(1.0 / uNorm);
            let u_a = Vector.empty(u.size() - i); //uk* Ak+1:n,k:n
            // premultiply by Q_i
            for (let j = i; j < u.size(); j++) {
                let value = 0.0;
                for (let k = i + 1; k < u.size(); k++)
                    value += u.get(k - i - 1) * A.get(k, j);
                u_a.set(j - i, value);
            }

            for (let j = i + 1; j < u.size(); j++) {
                for (let k = i; k < u.size(); k++)
                    A.set(j, k, A.get(j, k) - u.get(j - i - 1) * 2.0 * u_a.get(k - i));
            }
            // postmultiply by Q_i
            u_a = Vector.empty(u.size());
            for (let j = 0; j < u.size(); j++) {
                let value = 0.0;
                for (let k = i + 1; k < u.size(); k++)
                    value += u.get(k - i - 1) * A.get(j, k);
                u_a.set(j, value);
            }

            for (let j = 0; j < u.size(); j++) {
                for (let k = i + 1; k < u.size(); k++)
                    A.set(j, k, A.get(j, k) - 2.0 * u_a.get(j) * u.get(k - i - 1));
            }
        }
    }
    //console.log(`Hessenberg ${A.toString()}`);
    // A = makeHessenberg(A);

    for (let i = 0; i < u.size() - 2; i++) {
        for (let j = i + 2; j < u.size(); j++)
            A.set(j, i, 0.0);
    }
    // Francis double step QR
    let iter = 0;
    for (let p = u.size() - 1; p > 1;) {
        let q = p - 1;
        let s = A.get(q, q) + A.get(p, p);
        let t = A.get(q, q) * A.get(p, p) - A.get(p, q) * A.get(q, p);
        let x = A.get(0, 0) * A.get(0, 0) + A.get(0, 1) * A.get(1, 0) - s * A.get(0, 0) + t;
        let y = A.get(1, 0) * (A.get(0, 0) + A.get(1, 1) - s);
        let z = A.get(1, 0) * A.get(2, 1);
        for (let k = 0; k <= p - 2; k++) {
            let r = Math.max(0, k - 1);
            let p_v = new Vector([x, y, z]);
            let ro = -Math.sign(x);
            p_v.set(0, p_v.get(0) - ro * p_v.l2Norm());
            p_v.normalize();

            let p_t = new Array(u.size() - r);
            for (let j = r, m = 0; j < u.size(); j++, m++) {
                let temp = 0.0;
                for (let i = k, l = 0; l < 3; i++, l++)
                    temp += p_v.get(l) * A.get(i, j);
                p_t[m] = temp;
            }
            for (let j = k, l = 0; l < 3; j++, l++) {
                for (let i = r; i < u.size(); i++)
                    A.set(j, i, A.get(j, i) - 2.0 * p_v.get(l) * p_t[i - r]);
            }
            r = Math.min(k + 3, p);
            p_t = new Array(r + 1);
            for (let j = 0; j <= r; j++) {
                let value = 0.0;
                for (let i = k, l = 0; l < 3; i++, l++)
                    value += p_v.get(l) * A.get(j, i);
                p_t[j] = value;
            }

            for (let i = 0; i <= r; i++) {
                for (let j = k, l = 0; l < 3; j++, l++)
                    A.set(i, j, A.get(i, j) - 2.0 * p_v.get(l) * p_t[i]);
            }
            x = A.get(k + 1, k);
            y = A.get(k + 2, k);
            if (k < p - 2)
                z = A.get(k + 3, k);
        }

        let p_v = new Vector([x, y]);
        let ro = -Math.sign(x);
        p_v.set(0, p_v.get(0) - ro * p_v.l2Norm());
        p_v.normalize();

        let p_t = new Array(u.size() - p + 2);
        for (let j = p - 2, m = 0; j < u.size(); j++, m++) {
            let temp = 0.0;
            for (let i = q; i <= p; i++)
                temp += p_v.get(i - q) * A.get(i, j);
            p_t[m] = temp;
        }
        for (let i = q; i <= p; i++) {
            for (let j = p - 2, m = 0; j < u.size(); j++, m++)
                A.set(i, j, A.get(i, j) - 2.0 * p_v.get(i - q) * p_t[m]);
        }


        p_t = new Array(p + 1);
        for (let j = 0; j <= p; j++) {
            let value = 0.0;
            for (let i = p - 1, m = 0; i <= p; i++, m++)
                value += p_v.get(m) * A.get(j, i);
            p_t[j] = value;
        }

        for (let i = 0; i <= p; i++) {
            for (let j = p - 1, m = 0; j <= p; j++, m++)
                A.set(i, j, A.get(i, j) - 2.0 * p_v.get(m) * p_t[i]);
        }
        if (Math.abs(A.get(p, q)) < tolerance * (Math.abs(A.get(q, q)) + Math.abs(A.get(p, p)))) {
            A.set(p, q, 0);
            p = p - 1;
            q = p - 1;
        } else if (Math.abs(A.get(p - 1, q - 1)) < tolerance * (Math.abs(A.get(q - 1, q - 1)) + Math.abs(A.get(q, q)))) {
            A.set(p - 1, q - 1, 0);
            p = p - 2;
            q = p - 1;
        }
        iter++;
        if (iter > numIters)
            throw new ConvergenseFailureException("EignevaluesSolver");
    }
    for (let i = 0; i < A.numCols(); i++) {
        if (i > 0 && Math.abs(A.get(i, i - 1)) > tolerance * 10.0) //complex eigenvalues
        {
            let b = A.get(i - 1, i - 1) + A.get(i, i);
            let c = A.get(i - 1, i - 1) * A.get(i, i) - A.get(i - 1, i) * A.get(i, i - 1);
            let D = b * b - 4.0 * c;
            let x1 = b * 0.5;
            let x2 = b * 0.5;
            if (D > 0) {
                D = Math.sqrt(D);
                x1 += D * 0.5;
                x2 -= D * 0.5;
            }
            eigenvalues[i - 1] = x1;
            eigenvalues[i] = x2;
        } else {
            eigenvalues[i] = A.get(i, i);
        }
    }
    return eigenvalues;
}

/**
 * Q - orthogonal matrix, D - diagonal matrix of eigenvalues of A
 */
export class SymmetricEigendecomposition {
    private q: Matrix = null;
    private d: Vector = null;
    public tolerance: number = Tolerance;
    constructor(A: Matrix | null = null, numIters: number = 10) {
        if (A == null) return;
        this.factorize(A, numIters);
    }
    public factorize(A: Matrix, numIters: number = 10) {
        this.q = null;
        this.d = null;
        let T = A.clone();
        let Q = Matrix.identity(A.numCols());
        if (!T.isTridiagonal(SmallestTolerance)) {
            makeTridiagonalInplace(T, Q);
            Q.transposeInPlace();
        }
        // symmetric matrices have only real eigenvalues so single shift algorithm can be used

        let activeMatrixSize = A.numRows() - 1;
        for (let it = 0; it < numIters && activeMatrixSize > 0; ++it) {
            const p = activeMatrixSize;
            const q = p - 1;
            let ap = T.get(p, p);
            let bp = T.get(p, q);
            let d = (T.get(q, q) - ap) / 2;
            let s = 0;
            if (Math.abs(d) == 0)
                s = ap - Math.abs(bp);
            else
                s = ap - bp * bp / (d + Math.sign(d) * Math.sqrt(d * d + bp * bp));
            let x = T.get(0, 0) - s;
            let y = T.get(1, 0);
            for (let k = 0; k <= q; ++k) {
                let c: number = 0;
                let s: number = 0;
                if (activeMatrixSize > 1) {
                    let givensCoeffs = givens(x, y);
                    c = givensCoeffs.c;
                    s = givensCoeffs.s;
                } else {
                    // jacobi
                    let jacobi = jacobiRotation(T.get(0, 0), T.get(1, 1), y);
                    c = jacobi.c;
                    s = jacobi.s;
                }
                const w = c * x - s * y;
                const t = T.get(k, k) - T.get(k + 1, k + 1);
                const z = (2 * c * T.get(k + 1, k) + s * t) * s;
                T.set(k, k, T.get(k, k) - z);
                T.set(k + 1, k + 1, T.get(k + 1, k + 1) + z);
                T.set(k + 1, k, t * c * s + (c * c - s * s) * T.get(k + 1, k));
                x = T.get(k + 1, k);
                if (k > 0)
                    T.set(k, k - 1, w);
                if (k < q) {
                    y = -s * T.get(k + 2, k + 1);
                    T.set(k + 2, k + 1, c * T.get(k + 2, k + 1));
                }
                // apply givens [c, s][-s, c] from the right
                for (let i = 0; i < A.numRows(); ++i) {
                    let q1 = Q.get(i, k);
                    let q2 = Q.get(i, k + 1);
                    let q1New = q1 * c - q2 * s;
                    let q2New = q1 * s + q2 * c;
                    Q.set(i, k, q1New);
                    Q.set(i, k + 1, q2New);
                }
            }
            const aq = T.get(q, q);
            bp = T.get(p, q);
            if (Math.abs(bp) < this.tolerance * (Math.abs(ap) + Math.abs(aq)))
                --activeMatrixSize;
        }
        if (activeMatrixSize > 0) return;
        this.q = Q;
        this.d = T.diag();
    }
    public get D(): Vector {
        return this.d;
    }
    public get Q(): Matrix {
        return this.q;
    }
}

export class RealSchurDecomposition {
    private q: Matrix = null;
    private d: Matrix = null;
    tolerance: number = Tolerance;
    constructor(A: Matrix | null = null, numIters: number = 10) {
        if (A == null) return;
        this.factorize(A, numIters);
    }
    public factorize2(A: Matrix, numIters: number = 10) {
        assert(A.isSquare(), "A should be square");
        this.d = A.clone();
        this.q = Matrix.identity(A.numCols());
        if (A.numCols() < 3) return;
        if (!this.d.isHessenberg(true, SmallestTolerance)) {
            makeHessenbergInplace(this.d, this.q);
            this.q.transposeInPlace();
        }
        let activeMatrixSize = A.numRows();
        for (let it = 0; it < numIters && activeMatrixSize > 2; ++it) {
            const p = activeMatrixSize;
            const q = activeMatrixSize - 1;
            let s = this.d.get(q - 1, q - 1) + this.d.get(p - 1, p - 1);
            let t = this.d.get(q - 1, q - 1) * this.d.get(p - 1, p - 1) - this.d.get(p - 1, q - 1) * this.d.get(q - 1, p - 1);
            let x = this.d.get(0, 0) * this.d.get(0, 0) + this.d.get(0, 1) * this.d.get(1, 0) - s * this.d.get(0, 0) + t;
            let y = this.d.get(1, 0) * (this.d.get(0, 0) + this.d.get(1, 1) - s);
            let z = this.d.get(1, 0) * this.d.get(2, 1);
            for (let k = 0; k <= p - 3; k++) {
                let r = Math.max(1, k);
                let v = new Vector([x, y, z]);
                let ro = -Math.sign(x);
                v.set(0, v.get(0) - ro * v.l2Norm());
                let l = v.l2Norm();
                if (l >= SmallestTolerance * SmallestTolerance)
                    v.scaleSelf(1 / l);
                else if (l == 0) { v = new Vector([1, 0, 0]); } else {
                    if (Math.abs(v.get(0)) > Math.abs(v.get(1))) {
                        if (Math.abs(v.get(0)) > Math.abs(v.get(2))) {
                            v.set(0, 1);
                            v.set(1, v.get(1) / v.get(0));
                            v.set(2, v.get(2) / v.get(0));
                        } else {
                            v.set(2, 1);
                            v.set(1, v.get(1) / v.get(2));
                            v.set(0, v.get(0) / v.get(2));
                        }
                    } else if (Math.abs(v.get(1)) > Math.abs(v.get(2))) {
                        v.set(1, 1);
                        v.set(2, v.get(2) / v.get(1));
                        v.set(0, v.get(0) / v.get(1));
                    } else {
                        v.set(2, 1);
                        v.set(1, v.get(1) / v.get(2));
                        v.set(0, v.get(0) / v.get(2));
                    }
                    v.normalize();
                }
                //console.log(`x:${x}, y:${y}, z:${z} v:${v.toString()}`);

                // apply householder reflection from the left PT * H
                for (let col = r; col <= this.d.numCols(); ++col) {
                    let vDotX = 0.0;
                    for (let row = k + 1; row < k + 1 + v.size(); ++row)
                        vDotX += v.get(row - k - 1) * this.d.get(row - 1, col - 1);
                    vDotX *= 2;
                    for (let row = k + 1; row < k + 1 + v.size(); ++row)
                        this.d.set(row - 1, col - 1, this.d.get(row - 1, col - 1) - v.get(row - k - 1) * vDotX);
                }
                r = Math.min(k + 4, p);
                // apply householder from the right side H * P
                for (let row = 1; row <= r; ++row) {
                    let vDotX = 0.0;
                    for (let col = k + 1; col < k + 1 + v.size(); ++col)
                        vDotX += v.get(col - k - 1) * this.d.get(row - 1, col - 1);
                    vDotX *= 2;
                    for (let col = k + 1; col < k + 1 + v.size(); ++col)
                        this.d.set(row - 1, col - 1, this.d.get(row - 1, col - 1) - v.get(col - k - 1) * vDotX);
                }

                // update Q from the right
                for (let row = 1; row <= A.numRows(); ++row) {
                    let vDotX = 0.0;
                    for (let col = k + 1; col < k + 1 + v.size(); ++col)
                        vDotX += v.get(col - k - 1) * this.q.get(row - 1, col - 1);
                    vDotX *= 2;
                    for (let col = k + 1; col < k + 1 + v.size(); ++col)
                        this.q.set(row - 1, col - 1, this.q.get(row - 1, col - 1) - v.get(col - k - 1) * vDotX);
                }

                x = this.d.get(k + 2 - 1, k + 1 - 1);
                y = this.d.get(k + 3 - 1, k + 1 - 1);
                if (k < p - 3)
                    z = this.d.get(k + 4 - 1, k + 1 - 1);
            }
            // Givens rotations
            if (true) {
                let { c, s, r } = givens(x, y);
                //console.log(`x: ${x}, y: ${y} c: ${c} s:${s}`);

                for (let i = q - 1; i <= A.numCols(); ++i) {
                    let d1 = this.d.get(q - 1, i - 1);
                    let d2 = this.d.get(p - 1, i - 1);
                    let d1New = d1 * c - d2 * s;
                    let d2New = d1 * s + d2 * c;
                    this.d.set(q - 1, i - 1, d1New);
                    this.d.set(p - 1, i - 1, d2New);
                }
                for (let i = 1; i <= A.numRows(); ++i) {
                    let d1 = this.d.get(i - 1, q - 1);
                    let d2 = this.d.get(i - 1, p - 1);
                    let d1New = d1 * c - d2 * s;
                    let d2New = d1 * s + d2 * c;
                    this.d.set(i - 1, q - 1, d1New);
                    this.d.set(i - 1, p - 1, d2New);
                }
                for (let i = 1; i <= this.q.numRows(); ++i) {
                    let q1 = this.q.get(i - 1, q - 1);
                    let q2 = this.q.get(i - 1, p - 1);
                    let q1New = q1 * c - q2 * s;
                    let q2New = q1 * s + q2 * c;
                    this.q.set(i - 1, q - 1, q1New);
                    this.q.set(i - 1, p - 1, q2New);
                }
            }
            //console.log(`Iter ${it} ${activeMatrixSize}`);
            //console.log(`${Math.abs(this.d.get(p - 1, q - 1))} >= ${this.tolerance * (Math.abs(this.d.get(q - 1, q - 1)) + Math.abs(this.d.get(p - 1, p - 1)))}`);
            //console.log(`${Math.abs(this.d.get(q - 1, q - 1 - 1))} >= ${this.tolerance * (Math.abs(this.d.get(q - 1 - 1, q - 1 - 1)) + Math.abs(this.d.get(q - 1, q - 1)))}`);
            if (Math.abs(this.d.get(p - 1, q - 1)) < this.tolerance * (Math.abs(this.d.get(q - 1, q - 1)) + Math.abs(this.d.get(p - 1, p - 1)))) {
                this.d.set(p - 1, q - 1, 0);
                activeMatrixSize = q;
            } else if (Math.abs(this.d.get(q - 1, q - 1 - 1)) < this.tolerance * (Math.abs(this.d.get(q - 1 - 1, q - 1 - 1)) + Math.abs(this.d.get(q - 1, q - 1)))) {
                this.d.set(q - 1, q - 1 - 1, 0);
                activeMatrixSize = q - 1;
            }
        }
        //console.log(activeMatrixSize);
        //if (activeMatrixSize > 2) throw new Error("Not converged");
    }
    public factorize(A: Matrix, numIters: number = 10) {
        assert(A.isSquare(), "A should be square");
        this.d = A.clone();
        this.q = Matrix.identity(A.numCols());
        if (A.numCols() < 3) return;
        if (!this.d.isHessenberg(true, SmallestTolerance)) {
            makeHessenbergInplace(this.d, this.q);
            this.q.transposeInPlace();
        }
        //console.log(this.d.toString());
        let it = 0;
        let activeMatrixSize = A.numRows() - 1;
        for (it = 0; it < numIters && activeMatrixSize > 1; ++it) {
            const p = activeMatrixSize;
            const q = activeMatrixSize - 1;
            let s = 0;
            let t = 0;
            let hpp = this.d.get(p, p);
            let hqq = this.d.get(q, q);
            let hpqhqp = this.d.get(p, q) * this.d.get(q, p);
            if ((it % 10) == 0) {
                const COEFF1 = 0.75;
                const COEFF2 = -0.4375;
                const s = this.d.get(p, q) + this.d.get(q, q - 1);
                hpp = hpp + COEFF1 * s;
                hqq = hpp;
                hpqhqp = COEFF2 * s * s;
            } else {
                let disc = (hqq - hpp) / 2;
                disc = disc * disc + hpqhqp;
                if (disc > 0) {
                    disc = Math.sqrt(disc);
                    let average = (hqq + hpp) / 2;
                    if (Math.abs(hqq) - Math.abs(hpp) > 0) {
                        hqq = hqq * hpp - hpqhqp;
                        hpp = hqq / (Math.sign(average) * disc + average);
                    } else {
                        hpp = Math.sign(average) * disc + average;
                    }
                    hqq = hpp;
                    hpqhqp = 0;
                }
            }
            s = hqq + hpp;
            t = hqq * hpp - hpqhqp;

            let x = this.d.get(0, 0) * this.d.get(0, 0) + this.d.get(0, 1) * this.d.get(1, 0) - s * this.d.get(0, 0) + t;
            let y = this.d.get(1, 0) * (this.d.get(0, 0) + this.d.get(1, 1) - s);
            let z = this.d.get(1, 0) * this.d.get(2, 1);
            for (let k = 0; k <= p - 2; k++) {
                let r = Math.max(0, k - 1);
                let v = new Vector([x, y, z]);
                let ro = -Math.sign(x);
                v.set(0, v.get(0) - ro * v.l2Norm());
                let l = v.l2Norm();
                if (l >= SmallestTolerance)
                    v.scaleSelf(1 / l);
                else if (l == 0) {
                    v = new Vector([1, 0, 0]);
                } else {
                    if (Math.abs(v.get(0)) > Math.abs(v.get(1))) {
                        if (Math.abs(v.get(0)) > Math.abs(v.get(2))) {
                            v.set(1, v.get(1) / v.get(0));
                            v.set(2, v.get(2) / v.get(0));
                            v.set(0, 1);
                        } else {
                            v.set(1, v.get(1) / v.get(2));
                            v.set(0, v.get(0) / v.get(2));
                            v.set(2, 1);
                        }
                    } else if (Math.abs(v.get(1)) > Math.abs(v.get(2))) {
                        v.set(2, v.get(2) / v.get(1));
                        v.set(0, v.get(0) / v.get(1));
                        v.set(1, 1);
                    } else {
                        v.set(1, v.get(1) / v.get(2));
                        v.set(0, v.get(0) / v.get(2));
                        v.set(2, 1);
                    }
                    v.normalize();
                }

                // apply householder reflection from the left PT * H
                // todo (NI): change applyHouseholderFromRight to have starting index and replace this code block with function call
                // applyHouseholderFromLeft(v, d, k, r);
                for (let col = r; col < this.d.numCols(); ++col) {
                    let vDotX = 0.0;
                    for (let row = k; row < k + v.size(); ++row)
                        vDotX += v.get(row - k) * this.d.get(row, col);
                    vDotX *= 2;
                    for (let row = k; row < k + v.size(); ++row)
                        this.d.set(row, col, this.d.get(row, col) - v.get(row - k) * vDotX);
                }
                r = Math.min(k + 3, p);
                // apply householder from the right side H * P
                // applyHouseholderFromRight(v, d, k, r);
                for (let row = 0; row <= r; ++row) {
                    let vDotX = 0.0;
                    for (let col = k; col < k + v.size(); ++col)
                        vDotX += v.get(col - k) * this.d.get(row, col);
                    vDotX *= 2;
                    for (let col = k; col < k + v.size(); ++col)
                        this.d.set(row, col, this.d.get(row, col) - v.get(col - k) * vDotX);
                }

                // update Q from the right
                for (let row = 0; row < this.Q.numRows(); ++row) {
                    let vDotX = 0.0;
                    for (let col = k; col < k + v.size(); ++col)
                        vDotX += v.get(col - k) * this.q.get(row, col);
                    vDotX *= 2;
                    for (let col = k; col < k + v.size(); ++col)
                        this.q.set(row, col, this.q.get(row, col) - v.get(col - k) * vDotX);
                }

                x = this.d.get(k + 1, k);
                y = this.d.get(k + 2, k);
                if (k < p - 2)
                    z = this.d.get(k + 3, k);
            }
            // Givens rotations
            if (true) {
                let { c, s, r } = givens(x, y);

                for (let i = q - 1; i < A.numCols(); ++i) {
                    let d1 = this.d.get(q, i);
                    let d2 = this.d.get(p, i);
                    let d1New = d1 * c - d2 * s;
                    let d2New = d1 * s + d2 * c;
                    this.d.set(q, i, d1New);
                    this.d.set(p, i, d2New);
                }
                for (let i = 0; i < A.numRows(); ++i) {
                    let d1 = this.d.get(i, q);
                    let d2 = this.d.get(i, p);
                    let d1New = d1 * c - d2 * s;
                    let d2New = d1 * s + d2 * c;
                    this.d.set(i, q, d1New);
                    this.d.set(i, p, d2New);
                }
                for (let i = 0; i < A.numRows(); ++i) {
                    let q1 = this.q.get(i, q);
                    let q2 = this.q.get(i, p);
                    let q1New = q1 * c - q2 * s;
                    let q2New = q1 * s + q2 * c;
                    this.q.set(i, q, q1New);
                    this.q.set(i, p, q2New);
                }
            } else {
                let v = new Vector([x, y]);
                let ro = -Math.sign(x);
                v.set(0, v.get(0) - ro * v.l2Norm());
                v.normalize();
                // from the left
                for (let col = p - 2; col < this.d.numCols(); ++col) {
                    let vDotX = 0.0;
                    for (let row = q; row < q + v.size(); ++row)
                        vDotX += v.get(row - q) * this.d.get(row, col);
                    vDotX *= 2;
                    for (let row = q; row < q + v.size(); ++row)
                        this.d.set(row, col, this.d.get(row, col) - v.get(row - q) * vDotX);
                }
                // from the right
                for (let row = 0; row <= p; ++row) {
                    let vDotX = 0.0;
                    for (let col = q; col < q + v.size(); ++col)
                        vDotX += v.get(col - q) * this.d.get(row, col);
                    vDotX *= 2;
                    for (let col = q; col < q + v.size(); ++col)
                        this.d.set(row, col, this.d.get(row, col) - v.get(col - q) * vDotX);
                }
                // update Q from the right
                for (let row = 0; row < this.Q.numRows(); ++row) {
                    let vDotX = 0.0;
                    for (let col = q; col < q + v.size(); ++col)
                        vDotX += v.get(col - q) * this.q.get(row, col);
                    vDotX *= 2;
                    for (let col = q; col < q + v.size(); ++col)
                        this.q.set(row, col, this.q.get(row, col) - v.get(col - q) * vDotX);
                }
            }
            //console.log(this.d.toString());
            if (Math.abs(this.d.get(p, q)) < this.tolerance * (Math.abs(this.d.get(q, q)) + Math.abs(this.d.get(p, p)))) {
                this.d.set(p, q, 0);
                activeMatrixSize = q;
            } else if (Math.abs(this.d.get(q, q - 1)) < this.tolerance * (Math.abs(this.d.get(q - 1, q - 1)) + Math.abs(this.d.get(q, q)))) {
                this.d.set(q, q - 1, 0);
                activeMatrixSize = q - 1;
            }
        }
        // console.log(`${activeMatrixSize <= 1 ? "Converged" : "Didn't converge"} Iter ${it}, matrix size ${activeMatrixSize}`);
    }
    public get D(): Matrix {
        return this.d;
    }
    public get Q(): Matrix {
        return this.q;
    }
    public realEigenvalues(): number[] {
        let eigenvalues: number[] = new Array(this.d.numCols()).fill(0);
        for (let i = 0; i < this.d.numCols(); i++) {
            if (i > 0 && Math.abs(this.d.get(i, i - 1)) > this.tolerance * 10.0) //complex eigenvalues
            {
                let b = this.d.get(i - 1, i - 1) + this.d.get(i, i);
                let c = this.d.get(i - 1, i - 1) * this.d.get(i, i) - this.d.get(i - 1, i) * this.d.get(i, i - 1);
                let D = b * b - 4.0 * c;
                let x1 = b * 0.5;
                let x2 = b * 0.5;
                if (D > 0) {
                    D = Math.sqrt(D);
                    x1 += D * 0.5;
                    x2 -= D * 0.5;
                }
                eigenvalues[i - 1] = x1;
                eigenvalues[i] = x2;
            } else {
                eigenvalues[i] = this.d.get(i, i);
            }
        }
        return eigenvalues;
    }
    public eigenvalues(): complex[] {
        let eigenvalues: complex[] = new Array(this.d.numCols()).fill(null);
        for (let i = 0; i < this.d.numCols(); i++) {
            if (i > 0 && Math.abs(this.d.get(i, i - 1)) > this.tolerance * 10.0) //complex eigenvalues
            {
                let b = this.d.get(i - 1, i - 1) + this.d.get(i, i);
                let c = this.d.get(i - 1, i - 1) * this.d.get(i, i) - this.d.get(i - 1, i) * this.d.get(i, i - 1);
                let D = b * b - 4.0 * c;
                let x1 = b * 0.5;
                let x2 = x1;
                let y1 = 0;
                let y2 = 0;
                if (D > 0) {
                    D = Math.sqrt(D);
                    x1 += D * 0.5;
                    x2 -= D * 0.5;
                } else {
                    y1 = Math.sqrt(Math.abs(D)) / 2;
                    y2 = -y1;
                }
                eigenvalues[i - 1] = new complex(x1, y1);
                eigenvalues[i] = new complex(x2, y2);
            } else {
                eigenvalues[i] = complex.real(this.d.get(i, i));
            }
        }
        return eigenvalues;
    }
}

// todo (NI): Jacobi method for eigenvalues of symmetric matrix, also for singular values too