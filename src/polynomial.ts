import { complex } from "./complex";
import Matrix from "./dense/denseMatrix";
import { SmallTolerance, SmallestTolerance, assert, assertFail } from "./utils";

/** 
 * Polynomial represented as an array of coefficients from 0 to N: c_0 + c_1 * x + c_2 * x^2 + ... + c_N * x^N
 * */
export class Polynomial {
    coeffs: number[];
    constructor(coeffs: number[]) {
        assert(coeffs.length >= 1, "Invalid size");
        this.coeffs = coeffs;
    }
    clone(): Polynomial {
        return new Polynomial(this.coeffs.slice());
    }
    isEmpty(): boolean {
        for (const value of this.coeffs) {
            if (Math.abs(value) > 0)
                return false;
        }
        return true;
    }
    static mul(a: Polynomial, b: Polynomial): Polynomial {
        let resultCoeffs = new Array(a.degree() + b.degree() + 1).fill(0);
        for (let i = 0; i < a.numCoeffs(); ++i) {
            for (let j = 0; j < b.numCoeffs(); ++j)
                resultCoeffs[i + j] += a.coeffs[i] * b.coeffs[j];
        }
        return new Polynomial(resultCoeffs);
    }
    static add(a: Polynomial, b: Polynomial): Polynomial {
        let result = new Array(Math.max(a.degree(), b.degree()) + 1).fill(0);
        for (let i = 0; i < a.coeffs.length; ++i)
            result[i] = a.coeffs[i];
        for (let i = 0; i < b.coeffs.length; ++i)
            result[i] += b.coeffs[i];
        return new Polynomial(result);
    }
    static sub(a: Polynomial, b: Polynomial): Polynomial {
        let result = new Array(Math.max(a.degree(), b.degree()) + 1).fill(0);
        for (let i = 0; i < a.coeffs.length; ++i)
            result[i] = a.coeffs[i];
        for (let i = 0; i < b.coeffs.length; ++i)
            result[i] -= b.coeffs[i];
        return new Polynomial(result);
    }
    static scale(a: Polynomial, s: number): Polynomial {
        let result = a.clone();
        for (let i = 0; i < result.coeffs.length; ++i)
            result.coeffs[i] *= s;
        return result;
    }
    shiftLeft(power: number) {
        for (let i = 0; i < power; ++i)
            this.coeffs.unshift(0);
    }
    /**
     * Computes quotient Q and remainder R such that a = bq + r
     * @param a first polynomial
     * @param b second polynomial
     * @returns quotient and remainder
     */
    static div(a: Polynomial, b: Polynomial): { Q: Polynomial, R: Polynomial } {
        let Q = new Polynomial([0]);
        let R = a.clone();
        if (a.degree() < b.degree()) {
            return { Q, R };
        }
        while (R.degree() >= b.degree() && !R.isEmpty()) {
            let t = R.coeffs[R.coeffs.length - 1] / b.coeffs[b.coeffs.length - 1];
            Q.coeffs.unshift(t);
            for (let i = 1; i <= b.coeffs.length; ++i)
                R.coeffs[R.coeffs.length - i] -= b.coeffs[b.coeffs.length - i] * t;
            R.shrink();
        }
        if (Q.coeffs.length > 1)
            Q.coeffs.pop();
        return { Q, R };
    }
    numCoeffs(): number {
        return this.coeffs.length;
    }
    degree(): number {
        return this.coeffs.length - 1;
    }
    eval(x: number): number {
        let value = 0.0;
        for (let i = this.numCoeffs() - 1; i > 0; --i) {
            value += this.coeffs[i];
            value *= x;
        }
        return value + this.coeffs[0];
    }
    derivative(): Polynomial {
        assert(this.coeffs.length > 0, "Invalid size");
        if (this.coeffs.length == 1) new Polynomial([0]);
        let result: number[] = [];
        for (let i = 1; i < this.numCoeffs(); ++i) {
            result.push(this.coeffs[i] * i);
        }
        return new Polynomial(result);
    }
    integral(): Polynomial {
        let result: number[] = [0];
        for (let i = 0; i < this.numCoeffs(); ++i)
            result.push(this.coeffs[i] / (i + 1));
        return new Polynomial(result);
    }
    definiteIntegral(x0: number, x1: number): number {
        let ip = this.integral();
        return ip.eval(x1) - ip.eval(x0);
    }
    shrink(tolerance: number = 0): void {
        while (this.coeffs.length > 1 && Math.abs(this.coeffs[this.coeffs.length - 1]) <= tolerance)
            this.coeffs.pop();
    }
    companionMatrix(): Matrix {
        let matrix = Matrix.empty(this.degree(), this.degree());
        let lastCoeff = this.coeffs[this.coeffs.length - 1];
        for (let i = 0; i < this.degree(); ++i) {
            if (i + 1 != this.degree())
                matrix.set(i + 1, i, 1);
            matrix.set(i, this.degree() - 1, -this.coeffs[i] / lastCoeff)
        }
        return matrix;
    }
    toString(): string {
        let result = "P(x) = ";
        result += this.coeffs[0];
        if (this.coeffs.length > 1)
            result += ` + ${this.coeffs[1]}x`
        for (let i = 2; i < this.coeffs.length; ++i)
            result += ` + ${this.coeffs[i]}x^${i}`;
        return result;
    }
}

export class PolynomialSolver {
    protected numIters: number = 10;
    protected tol: number = SmallestTolerance;
    protected delta: number = 1.0;
    constructor(numIters: number = 10, tol: number = SmallestTolerance, delta: number = 1) {
        this.numIters = numIters;
        this.tol = tol;
        this.delta = delta;
    }
    // bisection + newton
    // todo (NI): try false position and fancy false position method from matlab paper
    protected findRoot(f: Polynomial, df: Polynomial, xMin: number, xMax: number, fMin: number, fMax: number): number {
        assert(Math.sign(fMin) != Math.sign(fMax), "Incorrect interval");
        // handle case with both infinity bounds, it should only arise when polynomial is odd
        if (xMin == Number.NEGATIVE_INFINITY && xMax == Number.POSITIVE_INFINITY) {
            let x0 = 0.0;
            let value = f.eval(x0);
            if (Math.sign(value) == Math.sign(fMin)) {
                xMin = x0;
                fMin = value;
            } else {
                xMax = x0;
                fMax = value;
            }
        }
        for (let i = 0; i < this.numIters; ++i) {
            // handle infinity on one of the sides
            let xCenter: number;
            if (xMin == Number.NEGATIVE_INFINITY)
                xCenter = xMax - this.delta;
            else if (xMax == Number.POSITIVE_INFINITY)
                xCenter = xMin + this.delta;
            else
                xCenter = (xMax + xMin) * 0.5;

            let value = f.eval(xCenter);
            let derivative = df.eval(xCenter);
            if (Math.abs(derivative) > SmallestTolerance) {
                let xNext = xCenter - value / derivative;
                if (xNext > xMin && xNext < xMax) {
                    xCenter = xNext;
                    value = f.eval(xNext);
                    if (Math.abs(xMin - xNext) <= this.tol) {
                        if (Math.sign(fMin) == Math.sign(value))
                            xCenter += this.tol;
                        else xCenter -= this.tol;
                    }
                }
            }
            if (Math.min(Math.abs(xCenter - xMax), Math.abs(xCenter - xMin)) < this.tol)
                return xCenter;
            if (Math.sign(value) == Math.sign(fMin)) {
                xMin = xCenter;
                fMin = value;
            }
            else { xMax = xCenter; fMax = value; }
        }
        return (xMax + xMin) * 0.5;
    }
    private infSign(x: number, f: Polynomial) {
        for (let i = f.numCoeffs() - 1; i >= 0; --i) {
            if (f.coeffs[i] == 0) continue;
            let sign = Math.sign(f.coeffs[i]);
            if (i & 1)
                return Math.sign(x) * sign;
            return sign;
        }
        assertFail("Empty polynomial");
    }
    // ax + b
    static solveLinear(a: number, b: number, xStart: number = Number.NEGATIVE_INFINITY, xEnd: number = Number.POSITIVE_INFINITY): number[] {
        let result: number[] = [];
        if (Math.abs(a) == 0) return result;
        result.push(-b / a);
        if (result[0] > xEnd || result[0] < xStart) return [];
        return result;
    }
    // ax^2 + bx + c
    static solveQuadratic(a: number, b: number, c: number, xStart: number = Number.NEGATIVE_INFINITY, xEnd: number = Number.POSITIVE_INFINITY): number[] {
        let d = b * b - 4 * a * c;
        if (d < 0.0)
            return [];
        d = (b + Math.sign(b) * Math.sqrt(d)) / 2;
        let x0 = -c / d;
        let x1 = -d / a;
        let result: number[] = [];
        if (x0 >= xStart && x0 <= xEnd)
            result.push(x0);
        if (x1 >= xStart && x1 <= xEnd)
            result.push(x1);
        return result
    }
    solveCubic(a: number, b: number, c: number, d: number, deflation: boolean = true, xStart: number = Number.NEGATIVE_INFINITY, xEnd: number = Number.POSITIVE_INFINITY): number[] {
        let f = new Polynomial([d, c, b, a]);
        let df = new Polynomial([c, 2 * b, 3 * a]);
        let criticalPoints = PolynomialSolver.solveQuadratic(df.coeffs[2], df.coeffs[1], df.coeffs[0], xStart, xEnd);
        criticalPoints.unshift(xStart);
        criticalPoints.push(xEnd);
        let roots: number[] = [];
        let i = 0;
        for (; i < criticalPoints.length - 1; ++i) {
            let xMin = criticalPoints[i];
            let xMax = criticalPoints[i + 1];
            let fMin = Number.isFinite(xMin) ? f.eval(xMin) : this.infSign(xMin, f);
            let fMax = Number.isFinite(xMax) ? f.eval(xMax) : this.infSign(xMax, f);
            let root: number = null;
            if (fMax == 0) {
                roots.push(xMax);
                continue;
            }
            if (Math.sign(fMin) != Math.sign(fMax)) {
                root = this.findRoot(f, df, xMin, xMax, fMin, fMax);
                deflation = false;
            }
            else {
                if (Math.abs(fMax) < this.tol) {
                    root = xMax;
                    //newRoots.push(this.refineRoot());
                }
                if (Math.abs(fMin) < this.tol) {
                    root = xMin;
                    //newRoots.push(this.refineRoot());
                }
            }
            if (root != null) {
                roots.push(root);
                if (!deflation) continue;
                // deflation
                let aq = a;
                let bq = b + aq * root;
                let cq = c + bq * root;
                let otherRoots = PolynomialSolver.solveQuadratic(aq, bq, cq, xMax, xEnd);
                roots = roots.concat(otherRoots);
                break;
            }
        }
        if (deflation && roots.length > 1) {
            for (let j = i + 1; j < criticalPoints.length; ++j) {
                let xMin = criticalPoints[i];
                let xMax = criticalPoints[i + 1];
                let fMin = Number.isFinite(xMin) ? f.eval(xMin) : this.infSign(xMin, f);
                let fMax = Number.isFinite(xMax) ? f.eval(xMax) : this.infSign(xMax, f);
                if (Math.sign(fMin) != Math.sign(fMax)) {
                    let rootIndex = roots.findIndex((value) => {
                        return value >= xMin && value <= xMax;
                    });
                    if (rootIndex >= 0) {
                        let root = roots[rootIndex];
                        let fRoot = f.eval(root);
                        if (Math.sign(fMin) == Math.sign(fRoot)) {
                            xMin = root;
                            fMin = fRoot;
                        }
                        else {
                            xMax = root;
                            fMax = fRoot;
                        }
                        // refine root
                        root = this.findRoot(f, df, xMin, xMax, fMin, fMax);
                        roots[rootIndex] = root;
                    }
                }
            }
        }
        return roots;
    }
    static eval(x: number, a: number[]) {
        let value = 0.0;
        for (let l = a.length - 1; l > 0; --l) {
            value += a[l];
            value *= x;
        }
        return value + a[0];
    }
    /**
     * Find roots inside region
     * @param polynomial polynomial
     * @param xStart start of interval, default value: NEGATIVE_INFINITY
     * @param xEnd end of interval, default value: POSITIVE_INFINITY
     * @returns sorted array of real roots
     */
    public solveInRegion(polynomial: Polynomial, solveCubic: boolean = false, xStart: number = Number.NEGATIVE_INFINITY, xEnd: number = Number.POSITIVE_INFINITY): number[] {
        assert(xStart < xEnd, "Invalid interval");
        polynomial.shrink();
        if (polynomial.degree() < 1) return [];
        if (polynomial.degree() == 1) return PolynomialSolver.solveLinear(polynomial.coeffs[1], polynomial.coeffs[0], xStart, xEnd);
        let stack: Polynomial[] = [polynomial];

        let criticalPoints: number[] = [];
        if (solveCubic) {
            if (polynomial.degree() == 2) return PolynomialSolver.solveQuadratic(polynomial.coeffs[2], polynomial.coeffs[1], polynomial.coeffs[0], xStart, xEnd);
            for (let i = polynomial.degree(); i > 3; --i)
                stack.push(stack[stack.length - 1].derivative());

            const cubicPolynomial = stack[stack.length - 1];
            assert(cubicPolynomial.numCoeffs() == 4, "Cubic polynomial expected");
            criticalPoints = this.solveCubic(cubicPolynomial.coeffs[3], cubicPolynomial.coeffs[2], cubicPolynomial.coeffs[1], cubicPolynomial.coeffs[0], true, xStart, xEnd);
        } else {
            for (let i = polynomial.degree(); i > 2; --i)
                stack.push(stack[stack.length - 1].derivative());

            const quadraticPolynomial = stack[stack.length - 1];
            assert(quadraticPolynomial.numCoeffs() == 3, "Quadratic polynomial expected");
            criticalPoints = PolynomialSolver.solveQuadratic(quadraticPolynomial.coeffs[2], quadraticPolynomial.coeffs[1], quadraticPolynomial.coeffs[0], xStart, xEnd);
        }

        criticalPoints.unshift(xStart);
        criticalPoints.push(xEnd);
        while (stack.length != 1) {
            //console.log(`roots of P[${stack[stack.length - 1].degree()}]: ${criticalPoints}`);
            let df = stack.pop();
            let f = stack[stack.length - 1];
            let newRoots: number[] = [];
            newRoots.push(xStart);
            for (let i = 0; i < criticalPoints.length - 1; ++i) {
                let xMin = criticalPoints[i];
                let xMax = criticalPoints[i + 1];
                let fMin = Number.isFinite(xMin) ? f.eval(xMin) : this.infSign(xMin, f);
                let fMax = Number.isFinite(xMax) ? f.eval(xMax) : this.infSign(xMax, f);
                if (Math.sign(fMin) != Math.sign(fMax)) {
                    let newRoot = this.findRoot(f, df, xMin, xMax, fMin, fMax);
                    newRoots.push(newRoot);
                } else {
                    // todo (NI): case with multiple roots - one side of the interval will be close to the root and the comparison above may fail
                    if (Math.abs(fMax) < this.tol) {
                        newRoots.push(xMax);
                        //newRoots.push(this.refineRoot());
                    }
                    if (Math.abs(fMin) < this.tol) {
                        newRoots.push(xMin);
                        //newRoots.push(this.refineRoot());
                    }
                }
            }
            newRoots.push(xEnd);
            criticalPoints = newRoots;
        }
        assert(criticalPoints.length >= 0, "Invalid number of critical points");
        criticalPoints.pop();
        criticalPoints.shift();
        return criticalPoints;
    }
}
/**
 * Calculate coefficients of polynomial from list of complex roots by unwrapping \prod (z - z_i).
 * It is expected that resulting coefficients will be real.
 */
export function generatePolynomialWithComplexRoots(roots: complex[]) {
    assert(roots.length != 0, "Zero roots");
    assert(roots.length <= 63, "Too many roots");
    let coeffs: complex[] = [];
    for (let i = 0; i < roots.length + 1; ++i)
        coeffs.push(complex.empty());
    let numProducts = 1 << roots.length;
    for (let i = 0; i < numProducts; ++i) {
        let coeffID = 0;
        let value = new complex(1.0, 0.0);
        for (let j = 0, k = i; j < roots.length; ++j, k = k >> 1) {
            coeffID += k & 1;
            value = (k & 1 ? value.negateSelf() as complex : complex.mul(value, roots[j]));
        }
        coeffs[coeffID].addSelf(value);
    }
    let realCoeffs: number[] = [];
    for (const coeff of coeffs) {
        assert(Math.abs(coeff.y) < SmallTolerance, "Complex coefficient");
        realCoeffs.push(coeff.x);
    }
    return new Polynomial(realCoeffs);
}

/** 
*    Calculate coefficients of polynomial from list of roots by unwrapping \prod (x - x_i)
*/
export function generatePolynomial(roots: number[]): Polynomial {
    assert(roots.length != 0, "Zero roots");
    assert(roots.length <= 63, "Too many roots");
    let coeffs = new Array(roots.length + 1).fill(0);
    let numProducts = 1 << roots.length;
    for (let i = 0; i < numProducts; ++i) {
        let coeffID = 0;
        let value = 1.0;
        for (let j = 0, k = i; j < roots.length; ++j, k = k >> 1) {
            coeffID += k & 1;
            value *= k & 1 ? -1 : roots[j];
        }
        coeffs[coeffID] += value;
    }
    return new Polynomial(coeffs);
}

// todo (NI): RPoly and complex polynomials