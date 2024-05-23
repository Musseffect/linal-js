import Matrix from "../../dense/denseMatrix";
import { assert, clamp, SmallestTolerance } from "../../utils";
import Vector from "../../dense/vector";

interface CubicSpline {
    a: number;
    b: number;
    c: number;
    d: number;
}

// Regular splines
class CubicSplines {
    splines: CubicSpline[];
    min: number;
    max: number;
    constructor(splines: CubicSpline[], min: number, max: number) {
        this.min = min;
        this.max = max;
        this.splines = splines;
        assert(splines.length > 0, "Empty splines array");
    }
    calc(param: number): number {
        let normalizedParam = this.max - this.min > SmallestTolerance ? (param - this.min) / (this.max - this.min) : 0.0;
        normalizedParam *= this.splines.length;
        let iParam = clamp(Math.floor(normalizedParam), 0, this.splines.length - 1);
        let fParam = (normalizedParam - iParam);
        let spline = this.splines[iParam];
        let f = spline.d + fParam * (spline.c + fParam * (spline.b + fParam * spline.a));
        return f;
    }
}

class PiecewiseCubicSpline {
    static run(x: number[], y: number[], intervals: number, isPeriodic: boolean): { splines: CubicSplines, error: number } {
        const variables = 2;
        let p = [
            (x: number) => {
                return x * x * (2.0 * x - 3) + 1;
            },
            (x: number) => {
                return x * (1 + x * (x - 2));
            }];
        let q = [
            (x: number) => {
                return x * x * (3 - 2 * x);
            },
            (x: number) => {
                return x * x * (x - 1);
            }];
        let A: Matrix = Matrix.empty((intervals + 1) * variables, (intervals + 1) * variables);
        let B: Vector = Vector.empty((intervals + 1) * variables);
        let bins: number[][] = [];
        for (let i = 0; i < intervals; i++)
            bins.push([]);
        let max = x[0];
        let min = x[0];
        for (let i = 0; i < x.length; i++) {
            max = Math.max(max, x[i]);
            min = Math.min(min, x[i])
        }
        let dx = (max - min) / intervals;
        for (let i = 0; i < x.length; i++) {
            let bin = Math.min(Math.floor((x[i] - min) / dx), intervals - 1);
            bins[bin].push(i);
        }
        let a: number[] = [];
        for (let i = 0; i < intervals + 1; i++) {
            let aCur = min + dx * i;
            a.push(aCur);
            let varId = i * variables;
            if (i - 1 >= 0) {
                let aPrev = aCur - dx;
                let bin = bins[i - 1];
                for (let j = 0; j < bin.length; j++) {
                    let index = bin[j];
                    let x_j = x[index];
                    let y_j = y[index];
                    let t_j = (x_j - aPrev) / (aCur - aPrev);
                    for (let k = 0; k < variables; k++) {
                        let dfdk = q[k](t_j);
                        B.set(B.get(varId + k) + y_j * dfdk, varId + k);
                        for (let l = 0; l < variables; l++) {
                            A.set(A.get(varId + k, varId - variables + l) + p[l](t_j) * dfdk, varId + k, varId - variables + l);
                            A.set(A.get(varId + k, varId + l) + q[l](t_j) * dfdk, varId + k, varId + l);
                        }
                    }
                }
            }
            if (i + 1 <= intervals) {
                let aNext = aCur + dx;
                let bin = bins[i];
                for (let j = 0; j < bin.length; j++) {
                    let index = bin[j];
                    let x_j = x[index];
                    let y_j = y[index];
                    let t_j = (x_j - aCur) / (aNext - aCur);
                    for (let k = 0; k < variables; k++) {
                        let dfdk = p[k](t_j);
                        B.set(B.get(varId + k) + y_j * dfdk, varId + k);
                        for (let l = 0; l < variables; l++) {
                            A.set(A.get(varId + k, varId + l) + p[l](t_j) * dfdk, varId + k, varId + l);
                            A.set(A.get(varId + k, varId + variables + l) + q[l](t_j) * dfdk, varId + k, varId + variables + l);
                        }
                    }
                }
            }
        }
        if (isPeriodic) {
            for (let k = 0; k < variables; k++) {
                let nextToLast = intervals * variables;
                let thirdToLast = nextToLast - variables;
                B.set(B.get(k) + B.get(nextToLast + k), k);
                B.set(0, nextToLast + k);
                for (let l = 0; l < variables; l++) {
                    A.set(A.get(k, l) + A.get(nextToLast + k, nextToLast + l), k, l);
                    A.set(A.get(nextToLast + k, thirdToLast + l) + A.get(k, thirdToLast + l), k, thirdToLast + l);
                    A.set(A.get(thirdToLast + k, l) + A.get(thirdToLast + k, nextToLast + l), thirdToLast + k, l);

                    A.set(0, k, nextToLast + l);
                    for (let i = 0; i <= intervals; i++) {
                        A.set(0, nextToLast + k, i * variables + l);
                        A.set(0, i * variables + k, nextToLast + l);
                    }
                }
                A.set(1, nextToLast + k, k);
                A.set(-1, nextToLast + k, nextToLast + k);
            }
        }
        let b = Matrix.solve(A, B);
        let splines = [];
        for (let i = 0; i < intervals; i++) {
            let fCur = b.get(i * 2);
            let dfCur = b.get(i * 2 + 1);
            let fNext = b.get(i * 2 + 2);
            let dfNext = b.get(i * 2 + 3);
            splines.push({
                a: dfNext + dfCur + 2 * (-fNext + fCur),
                b: 3 * (fNext - fCur) - dfNext - 2 * dfCur,
                c: dfCur,
                d: fCur
            });
        }
        let error = 0;
        for (let i = 0; i < intervals; i++) {
            let aCur = min + dx * i;
            let aNext = aCur + dx;
            let bin = bins[i];
            let spline = splines[i];
            for (let j = 0; j < bin.length; j++) {
                let index = bin[j];
                let _x = x[index];
                let _y = y[index];
                let t = (_x - aCur) / (aNext - aCur);
                error += Math.pow(spline.d + t * (spline.c + t * (spline.b + spline.a * t)) - _y, 2);
            }
        }
        return { splines: new CubicSplines(splines, min, max), error: error };
    }
}

export { CubicSpline, CubicSplines, PiecewiseCubicSpline };