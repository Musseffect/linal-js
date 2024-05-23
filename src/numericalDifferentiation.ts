import Matrix from "./dense/denseMatrix";
import Vector from "./dense/vector";

type ScalarFunc = (x: Vector) => number;

export function forwardDifference(f: ScalarFunc, p: Vector, step: number): Vector {
    let dimensions = p.size();
    let value = f(p);
    p = p.clone();
    let result = Vector.empty(dimensions);
    for (let i = 0; i < dimensions; ++i) {
        let x = p.get(i);
        p.set(i, x + step);
        result.set(i, f(p) - value);
        p.set(i, x);
    }
    return result.scaleSelf(1.0 / step);
}

export function backwardDifference(f: ScalarFunc, p: Vector, step: number): Vector {
    let dimensions = p.size();
    let value = f(p);
    p = p.clone();
    let result = Vector.empty(dimensions);
    for (let i = 0; i < dimensions; ++i) {
        let x = p.get(i);
        p.set(i, x - step);
        result.set(i, value - f(p));
        p.set(i, x);
    }
    return result.scaleSelf(1.0 / step);
}

export function centralDifference(f: ScalarFunc, p: Vector, step: number): Vector {
    let dimensions = p.size();
    p = p.clone();
    let result = Vector.empty(dimensions);
    for (let i = 0; i < dimensions; ++i) {
        let x = p.get(i);
        p.set(i, x + step);
        let delta = f(p);
        p.set(i, x - step);
        delta -= f(p);
        p.set(i, x);
        result.set(i, delta);
    }
    return result.scaleSelf(0.5 / step);
}

export function secondOrderDifference(f: ScalarFunc, p: Vector, step: number): Matrix {
    let size = p.size();
    let f00 = f(p);
    p = p.clone();
    let result = Matrix.empty(size, size);
    for (let i = 0; i < size; ++i) {
        for (let j = i; j < size; ++j) {
            let value = 0.0;
            if (i == j) {
                let x = p.get(i);
                p.set(i, x + step);
                const f10 = f(p);
                p.set(i, x - step);
                const f_10 = f(p);
                p.set(i, x);
                value = (f10 + f_10 - 2 * f00) / (step * step);
                result.set(i, i, value);
            } else {
                let x = p.get(i);
                let y = p.get(j);
                p.set(i, x + step);
                p.set(j, y + step);
                const f11 = f(p);
                p.set(i, x - step);
                p.set(j, y + step);
                const f_11 = f(p);
                p.set(i, x + step);
                p.set(j, y - step);
                const f1_1 = f(p);
                p.set(i, x - step);
                p.set(j, y - step);
                const f_1_1 = f(p);
                p.set(i, x);
                p.set(j, y);
                value = (f11 - f1_1 - f_11 + f_1_1) / (4 * step * step);
                /*
                                p.set(i, x + step);
                                const f10 = f(p);
                                p.set(i, x - step);
                                const f_10 = f(p);
                                p.set(i, x);
                                p.set(j, y + step);
                                const f01 = f(p);
                                p.set(j, y - step);
                                const f0_1 = f(p);
                                p.set(i, x + step);
                                p.set(j, y + step);
                                const f11 = f(p);
                                p.set(i, x - step);
                                p.set(j, y - step);
                                const f_1_1 = f(p);
                                p.set(i, x);
                                p.set(j, y);
                                value = (f11 - f10 - f01 + 2 * f00 - f_10 - f0_1 + f_1_1) / (2 * step * step);*/
                result.set(i, j, value);
                result.set(j, i, value);
            }
        }
    }
    return result;
}