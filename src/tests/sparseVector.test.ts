import { SparseVector, SparseVectorTwinIterator } from "../sparse/sparseVector";
import { SmallTolerance, Tolerance } from "../utils";
import Vector from "../dense/vector";
import { SparseMatrixCSR } from "../sparse/sparseMatrix";
import Matrix from "../dense/denseMatrix";
import { makeTriplet } from "../sparse/triplet";

describe('Vector operations', () => {
    test('Construction', () => {
        let dense: Vector = new Vector([0, -1, 2, -1e-8, 0.0, 3, -5, .0]);
        let v: SparseVector = SparseVector.fromVector(dense.data, SmallTolerance);
        expect(Vector.near(v.toDense(), dense, SmallTolerance)).toBeTruthy();
        expect(v.isIndexPresent(0)).toBeFalsy();
        expect(v.isIndexPresent(1)).toBeTruthy();
        expect(v.isIndexPresent(2)).toBeTruthy();
        expect(v.isIndexPresent(3)).toBeFalsy();
        expect(v.isIndexPresent(4)).toBeFalsy();
        expect(v.isIndexPresent(5)).toBeTruthy();
        expect(v.isIndexPresent(6)).toBeTruthy();
        expect(v.isIndexPresent(7)).toBeFalsy();
        for (let i = 0; i < dense.size(); ++i)
            expect(v.get(i)).toBeCloseTo(dense.get(i));
        expect(v.isValid()).toBeTruthy();

        let v2: SparseVector = SparseVector.empty(8);
        v2.set(2, 2);
        v2.set(1, -1);
        v2.set(5, 3);
        v2.set(6, -5);
        expect(SparseVector.near(v, v2, Tolerance)).toBeTruthy();
        for (let i = 0; i + 1 < v2.elements.length; ++i)
            expect(v2.elements[i].index).toBeLessThan(v2.elements[i + 1].index);
        expect(v.isValid()).toBeTruthy();

        v.set(1, 0);
        expect(v.isIndexPresent(1)).toBeFalsy();
        expect(v.get(1)).toBe(0);
        v.set(1, -1);
        expect(v.isIndexPresent(1)).toBeTruthy();
        expect(v.get(1)).toBe(-1);
        v.set(2, 5);
        expect(v.isIndexPresent(2)).toBeTruthy();
        expect(v.get(2)).toBe(5);
        v.set(3, -2);
        expect(v.isIndexPresent(3)).toBeTruthy();
        expect(v.get(3)).toBe(-2);
        expect(v.isValid()).toBeTruthy();
        expect(Vector.lInfDistance(v.toDense(), new Vector([0, -1, 5, -2, 0, 3, -5, 0]))).toBeCloseTo(0);
    });
    const v1Values = [{ value: -1, index: 1 }, { value: 2, index: 2 }, { value: 0.001, index: 4 }, { value: 3, index: 5 }, { value: -5, index: 6 }];
    let v1 = new SparseVector(8, v1Values);
    let d1 = v1.toDense();
    const v2Values = [{ value: 1, index: 0 }, { value: 10.5, index: 1 }, { value: 3, index: 2 }, { value: -2, index: 4 }, { value: 1, index: 7 }];
    let v2 = new SparseVector(8, v2Values);
    let d2 = v2.toDense();
    test('Scalar product', () => {
        expect(SparseVector.dot(v1, v2)).toBeCloseTo(Vector.dot(d1, d2));
    });
    test('Pointwise addition', () => {
        expect(Vector.near(SparseVector.add(v1, v2).toDense(), Vector.add(d1, d2))).toBeTruthy();
    });
    test('Pointwise subtraction', () => {
        expect(Vector.near(SparseVector.sub(v1, v2).toDense(), Vector.sub(d1, d2))).toBeTruthy();
    });
    test('Pointwise product', () => {
        expect(Vector.near(SparseVector.mul(v1, v2).toDense(), Vector.mul(d1, d2))).toBeTruthy();
    });
    test('Norms', () => {
        expect(v1.l2Norm()).toBeCloseTo(d1.l2Norm());
        expect(v1.l1Norm()).toBeCloseTo(d1.l1Norm());
        expect(v1.lInfNorm()).toBeCloseTo(d1.lInfNorm());
        expect(v1.squaredLength()).toBeCloseTo(d1.squaredLength());

        expect(SparseVector.lInfDistance(v1, v1)).toBeCloseTo(0);
        expect(SparseVector.l2Distance(v1, v1)).toBeCloseTo(0);
        expect(SparseVector.l1Distance(v1, v1)).toBeCloseTo(0);
        expect(SparseVector.lpDistance(v1, v1, 3)).toBeCloseTo(0);

        expect(SparseVector.lInfDistance(v2, v1)).toBeCloseTo(Vector.lInfDistance(d1, d2));
        expect(SparseVector.l2Distance(v2, v1)).toBeCloseTo(Vector.l2Distance(d1, d2));
        expect(SparseVector.l1Distance(v2, v1)).toBeCloseTo(Vector.l1Distance(d1, d2));
        expect(SparseVector.lpDistance(v2, v1, 3)).toBeCloseTo(Vector.lpDistance(d1, d2, 3));

        expect(SparseVector.lInfDistance(v1, v2)).toBeCloseTo(Vector.lInfDistance(d1, d2));
        expect(SparseVector.l2Distance(v1, v2)).toBeCloseTo(Vector.l2Distance(d1, d2));
        expect(SparseVector.l1Distance(v1, v2)).toBeCloseTo(Vector.l1Distance(d1, d2));
        expect(SparseVector.lpDistance(v1, v2, 3)).toBeCloseTo(Vector.lpDistance(d1, d2, 3));
    });
    test('Conversion', () => {
        expect(Vector.l1Distance(v1.toDense(), d1)).toBeCloseTo(0);
    });
    test('Outer product', () => {
        let expected = SparseMatrixCSR.fromTriplets(8, 8, [
            makeTriplet(1, 0, -1),
            makeTriplet(1, 1, -10.5),
            makeTriplet(1, 2, -3),
            makeTriplet(1, 4, 2),
            makeTriplet(1, 7, -1),

            makeTriplet(2, 0, 2),
            makeTriplet(2, 1, 21),
            makeTriplet(2, 2, 6),
            makeTriplet(2, 4, -4),
            makeTriplet(2, 7, 2),

            makeTriplet(4, 0, 0.001),
            makeTriplet(4, 1, 0.0105),
            makeTriplet(4, 2, 0.003),
            makeTriplet(4, 4, -0.002),
            makeTriplet(4, 7, 0.001),

            makeTriplet(5, 0, 3),
            makeTriplet(5, 1, 31.5),
            makeTriplet(5, 2, 9),
            makeTriplet(5, 4, -6),
            makeTriplet(5, 7, 3),

            makeTriplet(6, 0, -5),
            makeTriplet(6, 1, -52.5),
            makeTriplet(6, 2, -15),
            makeTriplet(6, 4, 10),
            makeTriplet(6, 7, -5)
        ]);
        expect(SparseMatrixCSR.lInfDistance(expected, SparseVector.outerProduct(v1, v2))).toBeCloseTo(0);
        expect(Matrix.lInfDistance(expected.toDense(), SparseVector.outerProductDense(v1, v2))).toBeCloseTo(0);
    });
    test('Iterators', () => {
        let twinIt = new SparseVectorTwinIterator(v1, v2);
        const expected = [{ value1: 0, value2: 1, index: 0 }, { value1: -1, value2: 10.5, index: 1 }, { value1: 2, value2: 3, index: 2 }, { value1: 0.001, value2: -2, index: 4 }, { value1: 3, value2: 0, index: 5 }, { value1: -5, value2: 0, index: 6 }, { value1: 0, value2: 1, index: 7 }];
        let values: { value1: number, value2: number, index: number }[] = [];
        while (!twinIt.isDone())
            values.push(twinIt.advance());
        expect(values.length).toBe(expected.length);
        for (let i = 0; i < expected.length; ++i) {
            expect(values[i].value1).toBe(expected[i].value1);
            expect(values[i].value2).toBe(expected[i].value2);
            expect(values[i].index).toBe(expected[i].index);
        }
    });
});