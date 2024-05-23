import { performance } from "perf_hooks";

export const Tolerance = 1e-4;
export const SmallTolerance = 1e-6;
export const SmallestTolerance = 1e-8;

export function assert(condition: boolean, message: string): void {
    if (!condition)
        throw new Error(message);
}

export function assertFail(message: string): void {
    throw new Error(message);
}

export function radians(value: number): number {
    return value / 180 * Math.PI;
}

export function degrees(value: number): number {
    return value / Math.PI * 180;
}

export function clamp(value: number, min: number, max: number): number {
    return Math.max(Math.min(value, max), min);
}

export function near(a: number, b: number, absTolerance: number = SmallTolerance, relTolerance: number = 0): boolean {
    if (relTolerance == 0)
        return Math.abs(b - a) <= absTolerance;
    return Math.abs(b - a) <= Math.max(absTolerance, relTolerance * Math.max(Math.abs(a), Math.abs(b)));
}

export function lerp(a: number, b: number, t: number): number {
    return b * t + a * (1 - t);
}

export function determinant2x2(m11: number, m12: number, m21: number, m22: number): number {
    return m11 * m22 - m12 * m21;
}

export function determinant3x3(m11: number, m12: number, m13: number,
    m21: number, m22: number, m23: number,
    m31: number, m32: number, m33: number) {
    return m11 * (m22 * m33 - m23 * m32)
        - m12 * (m21 * m33 - m31 * m23)
        + m13 * (m21 * m32 - m31 * m22);
}
export function determinant4x4(m11: number, m12: number, m13: number, m14: number,
    m21: number, m22: number, m23: number, m24: number,
    m31: number, m32: number, m33: number, m34: number,
    m41: number, m42: number, m43: number, m44: number): number {
    return m11 * determinant3x3(m22, m23, m24, m32, m33, m34, m42, m43, m44)
        - m12 * determinant3x3(m21, m23, m24, m31, m33, m34, m41, m43, m44)
        + m13 * determinant3x3(m21, m22, m24, m31, m32, m34, m41, m42, m44)
        - m14 * determinant3x3(m21, m22, m23, m31, m32, m33, m41, m42, m43);
}

export function swap<T>(array: Array<T>, firstIdx: number, secondIdx: number): void {
    let temp = array[firstIdx];
    array[firstIdx] = array[secondIdx];
    array[secondIdx] = temp;
}

export function sign(x: number) {
    return x > 0 ? 1 : -1;
}

export function binomial(n: number, k: number) {
    assert(Number.isInteger(n) && Number.isInteger(k), "Non-integer arguments");
    assert(n >= k && k >= 0, "Incorrect arguments");
    if (k > n - k)
        k = n - k;
    let result = 1;
    for (let i = 1; i <= k; ++i, --n)
        result = Math.floor(result / i) * n + (result % i) * Math.floor(n / i);
    return result;
}

export class StopWatch {
    private timestamp: number = 0;
    constructor() {
        this.timestamp = performance.now();
    }
    public elapsed(): number {
        return performance.now() - this.timestamp;
    }
    public reset(): void {
        this.timestamp = performance.now();
    }
    public resetElapsed(): number {
        let oldTimestamp = this.timestamp;
        this.timestamp = performance.now();
        return this.timestamp - oldTimestamp;
    }
}

export function randomArray(size: number, min: number, max: number): number[] {
    let values = [];
    for (let i = 0; i < size; ++i)
        values.push(Math.random());
    return values;
}