import Matrix from "./denseMatrix";
import { assert, clamp, Tolerance, SmallestTolerance, swap } from "../utils";

function reduceVectorsDiff(a: Vector, b: Vector, f: (prev: number, cur: number) => number) {
    assert(a.size() == b.size(), "Sizes don't match");
    let result = 0;
    for (let i = 0; i < a.size(); ++i)
        result = f(result, Math.abs(b.data[i] - a.data[i]));
    return result;
}

export default class Vector {
    resize(newSize: number, defaultValue: number): void {
        if (newSize > this.data.length) {
            for (let i = this.data.length; i < newSize; ++i)
                this.data.push(defaultValue);
        } else {
            this.data.splice(newSize, this.data.length - newSize);
        }
    }
    swap(firstIdx: number, secondIdx: number) {
        if (firstIdx != secondIdx)
            swap(this.data, firstIdx, secondIdx);
    }
    data: number[];
    constructor(data: number[]) {
        this.data = data;
    }
    toMatrix(rowMajor: boolean = false): Matrix {
        if (rowMajor)
            return new Matrix(this.data, 1, this.data.length);
        return new Matrix(this.data, this.data.length, 1);
    }
    static outer(a: Vector, b: Vector): Matrix {
        let result: Matrix = Matrix.empty(a.size(), b.size());
        for (let j = 0; j < a.size(); j++) {
            for (let i = 0; i < b.size(); i++)
                result.set(j, i, a.get(j) * b.get(i));
        }
        return result;
    }
    static near(a: Vector, b: Vector, threshold?: number): boolean {
        assert(a.size() == b.size(), "Vectors should have equal sizes");
        if (!threshold)
            threshold = Tolerance;

        for (let i = 0; i < a.size(); i++) {
            if (Math.abs(a.get(i) - b.get(i)) > threshold)
                return false;
        }
        return true;
    }
    static lerp(a: Vector, b: Vector, t: number): Vector {
        return b.sub(a).scaleSelf(t).addSelf(a);
    }
    copy(): Vector {
        return new Vector(this.data.slice());
    }
    clone(): Vector {
        return new Vector(this.data.slice());
    }
    static negate(v: Vector): Vector {
        let result = [];
        for (let el of v.data)
            result.push(-el);
        return new Vector(result);
    }
    static empty(length: number): Vector {
        let data: number[];
        (data = []).length = length;
        data.fill(0);
        return new Vector(data);
    }
    static generate(size: number, f: (i: number) => number) {
        let result = Vector.empty(size);
        for (let i = 0; i < size; ++i)
            result.set(i, f(i));
        return result;
    }
    static dot(a: Vector, b: Vector) {
        let result = 0;
        for (let i = 0; i < a.size(); i++)
            result += a.data[i] * b.data[i];
        return result;
    }
    static add(a: Vector, b: Vector) {
        let result = [];
        for (let i = 0; i < a.data.length; i++)
            result.push(a.data[i] + b.data[i]);
        return new Vector(result);
    }
    static sub(a: Vector, b: Vector) {
        let result = [];
        for (let i = 0; i < a.data.length; i++)
            result.push(a.data[i] - b.data[i]);
        return new Vector(result);
    }
    static mul(a: Vector, b: Vector) {
        let result = [];
        for (let i = 0; i < a.data.length; i++)
            result.push(a.data[i] * b.data[i]);
        return new Vector(result);
    }
    static div(a: Vector, b: Vector) {
        let result = [];
        for (let i = 0; i < a.data.length; i++)
            result.push(a.data[i] / b.data[i]);
        return new Vector(result);
    }
    static scale(a: Vector, s: number) {
        let result = [];
        for (let i = 0; i < a.data.length; i++)
            result.push(a.data[i] * s);
        return new Vector(result);
    }
    addSelf(b: Vector) {
        for (let i = 0; i < this.data.length; i++)
            this.data[i] += b.data[i];
        return this;
    }
    subSelf(b: Vector) {
        for (let i = 0; i < this.data.length; i++)
            this.data[i] -= b.data[i];
        return this;
    }
    scaleSelf(s: number) {
        for (let i = 0; i < this.data.length; i++)
            this.data[i] *= s;
        return this;
    }
    get(i: number): number {
        //assert(i >= 0 && i < this.data.length, `Incorrect index ${i} < ${this.data.length}`);
        return this.data[i];
    }
    set(i: number, value: number): void {
        //assert(i >= 0 && i < this.data.length, `Incorrect index ${i} < ${this.data.length}`);
        this.data[i] = value;
    }
    size(): number {
        return this.data.length;
    }
    getSubVector(offset: number, length: number) {
        let resultData = new Array(length);
        for (let i = 0; i < length; i++)
            resultData[i] = this.data[offset + i];
        return new Vector(resultData);
    }
    addSubVector(b: Vector, offset: number): Vector {
        assert(this.size() == b.size() + offset, "Vectors should have matching sizes");
        for (let i = 0; i < b.size(); i++)
            this.data[i + offset] += b.get(i);
        return this;
    }
    subSubVector(b: Vector, offset: number): Vector {
        assert(this.size() == b.size() + offset, "Vectors should have matching sizes");
        for (let i = 0; i < b.size(); i++)
            this.data[i + offset] -= b.get(i);
        return this;
    }
    add(b: Vector, dest?: Vector): Vector {
        if (!dest)
            dest = this;
        assert(this.size() == b.size(), "Vectors should be of equal size");
        for (let i = 0; i < this.data.length; i++)
            dest.data[i] = this.data[i] + b.data[i];
        return dest;
    }
    sub(b: Vector, dest?: Vector): Vector {
        if (!dest)
            dest = this;
        assert(this.size() == b.size(), "Vectors should be of equal size");
        for (let i = 0; i < this.data.length; i++)
            dest.data[i] = this.data[i] - b.data[i];
        return dest;
    }
    mul(b: Vector, dest?: Vector): Vector {
        if (!dest)
            dest = this;
        assert(this.size() == b.size(), "Vectors should be of equal size");
        for (let i = 0; i < this.data.length; i++)
            dest.data[i] = this.data[i] * b.data[i];
        return dest;
    }
    scale(b: number, dest?: Vector): Vector {
        if (!dest)
            dest = this;
        for (let i = 0; i < this.data.length; i++)
            dest.data[i] = this.data[i] * b;
        return dest;
    }
    l1Norm(): number {
        return this.data.reduce((prev: number, cur: number) => { return prev + Math.abs(cur); }, 0);
    }
    squaredLength(): number {
        return this.data.reduce((prev: number, cur: number) => { return prev + cur * cur }, 0);
    }
    l2Norm(): number {
        return Math.sqrt(this.squaredLength());
    }
    lInfNorm(): number {
        return this.data.reduce((prev: number, cur: number) => { return Math.max(prev, Math.abs(cur)); }, 0);
    }
    lpNorm(p: number): number {
        return Math.pow(this.data.reduce((prev: number, cur: number) => {
            return prev + Math.pow(Math.abs(cur), p);
        }, 0), 1 / p);
    }
    /** L1 norm (sum of abs coordinates of vector from a to b) between two vectors */
    static l1Distance(a: Vector, b: Vector): number {
        return reduceVectorsDiff(a, b, (prev: number, cur: number) => { return prev + cur; });
    }
    /** Euclidian norm (sum of squared coordinates of vector from a to b) between two vectors */
    static l2Distance(a: Vector, b: Vector): number {
        return Math.sqrt(reduceVectorsDiff(a, b, (prev: number, cur: number) => { return prev + cur * cur; }));
    }
    /** LInf norm (max of absolute coordinates of vector from a to b) between two vectors */
    static lInfDistance(a: Vector, b: Vector): number {
        return reduceVectorsDiff(a, b, (prev: number, cur: number) => { return Math.max(prev, cur); });
    }
    /** LP norm between two vectors */
    static lpDistance(a: Vector, b: Vector, p: number): number {
        return Math.pow(reduceVectorsDiff(a, b, (prev: number, cur: number) => { return prev + Math.pow(cur, p); }), 1 / p);
    }
    clamp(min: Vector, max: Vector): void {
        for (let i = 0; i < this.data.length; i++)
            this.data[i] = clamp(this.data[i], min.data[i], max.data[i]);
    }
    clampScalar(min: number, max: number): void {
        for (let i = 0; i < this.data.length; i++)
            this.data[i] = clamp(this.data[i], min, max);
    }
    normalize(tolerance: number = SmallestTolerance) {
        let l = this.l2Norm();
        if (l >= tolerance) {
            for (let i = 0; i < this.data.length; ++i)
                this.data[i] /= l;
        }
        return this;
    }
    print(fractionDigits: number): string {
        if (!fractionDigits)
            fractionDigits = 4;
        let result = "[ ";
        this.data.forEach((item) => result += item.toFixed(fractionDigits) + " ");
        return result + "]";
    }
    toString(): string {
        let result = "[";
        this.data.forEach((item, i) => { result += `${i != 0 ? ", " : ""}${item}` });
        return result + "]";
    }
}