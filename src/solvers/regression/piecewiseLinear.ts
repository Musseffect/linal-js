import Matrix from "../../dense/denseMatrix";
import Vector from "../../dense/vector";
// find peacewise linear least squares approximation with C0 knots
export class PiecewiseLinear {
    static run(x: number[], y: number[], numIntervals: number, isPeriodic: boolean) {
        let A: Matrix = Matrix.empty(numIntervals + 1, numIntervals + 1);
        let B: Vector = Vector.empty(numIntervals + 1);
        let bins: number[][] = [];
        for (let i = 0; i < numIntervals; i++)
            bins.push([]);
        let max = x[0];
        let min = x[0];
        for (let i = 0; i < x.length; i++) {
            max = Math.max(max, x[i]);
            min = Math.min(min, x[i])
        }
        let dx = (max - min) / numIntervals;
        for (let i = 0; i < x.length; i++) {
            let bin = Math.min(Math.floor((x[i] - min) / dx), numIntervals - 1);
            bins[bin].push(i);
        }
        let a: number[] = [];
        for (let i = 0; i < numIntervals + 1; i++) {
            let ai = min + dx * i;
            a.push(ai);
            if (i - 1 >= 0) {
                let aPrev = ai - dx;
                let bin = bins[i - 1];
                for (let j = 0; j < bin.length; j++) {
                    let index = bin[j];
                    let _x = x[index];
                    let _y = y[index];
                    B.set(B.get(i) + _y * (_x - aPrev) / dx, i);
                    A.set(A.get(i, i - 1) + (ai - _x) * (_x - aPrev) / (dx * dx), i, i - 1);
                    A.set(A.get(i, i) + (_x - aPrev) * (_x - aPrev) / (dx * dx), i, i);
                }
            }
            if (i + 1 <= numIntervals) {
                let aNext = ai + dx;
                let bin = bins[i];
                for (let j = 0; j < bin.length; j++) {
                    let index = bin[j];
                    let _x = x[index];
                    let _y = y[index];
                    B.set(B.get(i) + _y * (aNext - _x) / dx, i);
                    A.set(A.get(i, i) + (aNext - _x) * (aNext - _x) / (dx * dx), i, i);
                    A.set(A.get(i, i + 1) + (_x - ai) * (aNext - _x) / (dx * dx), i, i + 1);
                }
            }
        }
        if (isPeriodic) {
            B.set(B.get(0) + B.get(numIntervals), 0);
            A.set(A.get(0, 0) + A.get(numIntervals, numIntervals), 0, 0);
            A.set(A.get(numIntervals, numIntervals - 1) + A.get(0, numIntervals - 1), 0, numIntervals - 1);
            A.set(A.get(numIntervals - 1, 0) + A.get(numIntervals - 1, numIntervals), numIntervals - 1, 0);

            A.set(1, numIntervals, 0);
            A.set(-1, numIntervals, numIntervals);
            A.set(0, 0, numIntervals);
            for (let i = 1; i < numIntervals; i++) {
                A.set(0, numIntervals, i);
                A.set(0, i, numIntervals);
            }
            B.set(0, numIntervals);
        }
        let b = Matrix.solve(A, B);
        let error = 0;
        for (let i = 0; i < numIntervals; i++) {
            let ai = min + dx * i;
            let aNext = ai + dx;
            let bin = bins[i];
            for (let j = 0; j < bin.length; j++) {
                let index = bin[j];
                let _x = x[index];
                let _y = y[index];
                error += Math.pow((b.get(i + 1) * (_x - ai) + b.get(i) * (aNext - _x)) / dx - _y, 2);
            }
        }
        return { a: a, b: b.data, error: error };
    }
}