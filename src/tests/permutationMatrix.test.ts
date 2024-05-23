import Matrix from "../dense/denseMatrix";
import { PermutationType, PermutationMatrix } from "../permutationMatrix";
import { assert, near } from "../utils";
import Vector from "../dense/vector";
import { SparseMatrixCSR } from "../sparse/sparseMatrix";

test('Random permutation', () => {
    for (let i = 0; i < 10; ++i) {
        let size = Math.round(Math.random() * 8 + 2);
        let permutation = PermutationMatrix.random(size, PermutationType.Row);
        expect(permutation.isValid()).toBeTruthy();
    }
});

describe("Permutation matrix", () => {
    const rowPermutations = new PermutationMatrix([1, 6, 8, 2, 5, 4, 9, 3, 0, 7], PermutationType.Row);
    const colPermutations = new PermutationMatrix([1, 6, 8, 2, 5, 4, 9, 3, 0, 7], PermutationType.Col);
    test("Permutation matrix properties", () => {
        expect(rowPermutations.isValid()).toBeTruthy();
        expect(colPermutations.isValid()).toBeTruthy();
        expect(rowPermutations.determinant()).toBeCloseTo(rowPermutations.toMatrix().determinant());
        expect(colPermutations.determinant()).toBeCloseTo(colPermutations.toMatrix().determinant());
    });
    assert(rowPermutations.isValid(), "Invalid permutation");
    assert(colPermutations.isValid(), "Invalid permutation");
    test("Dense matrix", () => {

        const rowPermuted = Matrix.mul(rowPermutations.toMatrix(), Matrix.identity(10));
        const colPermuted = Matrix.mul(Matrix.identity(10), colPermutations.toMatrix());
        const fullPermuted = Matrix.mul(rowPermuted, colPermutations.toMatrix());
        for (let i = 0; i < 10; ++i) {
            const idx1 = rowPermutations.permuteIndex(i, i);
            expect(rowPermuted.get(idx1.row, idx1.column)).toBeCloseTo(1.0);
            expect(rowPermutations.value(idx1.row)).toEqual(i);
            const idx2 = colPermutations.permuteIndex(i, i);
            expect(colPermuted.get(idx2.row, idx2.column)).toBeCloseTo(1.0);
            expect(colPermutations.value(idx2.column)).toEqual(i);
            const idx3 = colPermutations.permuteIndex(i, i);
            const idx4 = rowPermutations.permuteIndex(idx3.row, idx3.column);
            expect(idx4.column).toEqual(idx3.column);
            expect(idx3.row).toEqual(i);
            expect(fullPermuted.get(idx4.row, idx4.column)).toBeCloseTo(1.0);
            expect(colPermutations.value(idx4.column)).toEqual(i);
            expect(rowPermutations.value(idx3.column)).toEqual(i);
        }
        expect(Matrix.near(rowPermuted, rowPermutations.toMatrix())).toBeTruthy();
        expect(Matrix.near(colPermuted, colPermutations.toMatrix())).toBeTruthy();
        expect(Matrix.near(fullPermuted, Matrix.mul(rowPermutations.toMatrix(), colPermutations.toMatrix()))).toBeTruthy();
        expect(Matrix.near(Matrix.mul(colPermutations.inverse().toMatrix(), colPermutations.toMatrix()), Matrix.identity(10))).toBeTruthy();
        expect(Matrix.near(Matrix.mul(rowPermutations.inverse().toMatrix(), rowPermutations.toMatrix()), Matrix.identity(10))).toBeTruthy();

        expect(Matrix.near(rowPermutations.toMatrix().inverse(), rowPermutations.inverse().toMatrix()));
        expect(Math.abs(rowPermutations.toMatrix().determinant())).toBeCloseTo(1.0);
        expect(Matrix.near(colPermutations.toMatrix().inverse(), colPermutations.inverse().toMatrix()));
        expect(Math.abs(colPermutations.toMatrix().determinant())).toBeCloseTo(1.0);

        const generatedMat = Matrix.generate(10, 10, (r: number, c: number) => r * 10 + c);
        const rowPermutedMat = rowPermutations.permuteMatrix(generatedMat);
        const colPermutedMat = colPermutations.permuteMatrix(generatedMat);
        const fullPermutedMat = colPermutations.permuteMatrix(rowPermutedMat);

        expect(Matrix.lInfDistance(rowPermutedMat, Matrix.mul(rowPermutations.toMatrix(), generatedMat))).toBeCloseTo(0);
        expect(Matrix.lInfDistance(colPermutedMat, Matrix.mul(generatedMat, colPermutations.toMatrix()))).toBeCloseTo(0);
        expect(Matrix.lInfDistance(fullPermutedMat, Matrix.mul(Matrix.mul(rowPermutations.toMatrix(), generatedMat), colPermutations.toMatrix()))).toBeCloseTo(0);
    });
    test("Dense vector", () => {
        const generatedVec = Vector.generate(10, (i: number) => i);
        const rowPermutedVec = rowPermutations.permuteVector(generatedVec);
        const colPermutedVec = colPermutations.permuteVector(generatedVec);
        const fullPermutedVec = colPermutations.permuteVector(rowPermutedVec);

        expect(Vector.lInfDistance(rowPermutedVec, Matrix.postMulVec(rowPermutations.toMatrix(), generatedVec))).toBeCloseTo(0);
        expect(Vector.lInfDistance(colPermutedVec, Matrix.preMulVec(generatedVec, colPermutations.toMatrix()))).toBeCloseTo(0);
        expect(Vector.lInfDistance(fullPermutedVec, Matrix.preMulVec(Matrix.postMulVec(rowPermutations.toMatrix(), generatedVec), colPermutations.toMatrix()))).toBeCloseTo(0);
    });
    test("Sparse matrix", () => {

        const initialMatrix = SparseMatrixCSR.fromDense(new Matrix([
            1, 0, 0, 0, 5, 0, 0, 0, 8, 1,
            0, 0, 0, 2, 0, 3, 0, 4, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            6, 0, 5, 0, 0, 0, 7, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 3,
            0, 0, 4, 0, 0, 0, 0, 0, 0, 0,
            0, 12, 0, 0, -3, 0, 0, 5, 0, 0,
            13, 0, 0, 0, 0, 0, 0, 33, 0, 0
        ], 10, 10), 0);
        const expectedRowPermuted = SparseMatrixCSR.fromDense(new Matrix([
            0, 0, 0, 2, 0, 3, 0, 4, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 3,
            0, 12, 0, 0, -3, 0, 0, 5, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            6, 0, 5, 0, 0, 0, 7, 0, 0, 0,
            13, 0, 0, 0, 0, 0, 0, 33, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 5, 0, 0, 0, 8, 1,
            0, 0, 4, 0, 0, 0, 0, 0, 0, 0
        ], 10, 10), 0);
        const expectedColPermuted = SparseMatrixCSR.fromDense(new Matrix([
            0, 0, 8, 0, 0, 5, 1, 0, 1, 0,
            0, 0, 0, 0, 3, 0, 0, 2, 0, 4,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 7, 0, 5, 0, 0, 0, 0, 6, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 3, 0, 0, 0,
            0, 0, 0, 4, 0, 0, 0, 0, 0, 0,
            12, 0, 0, 0, 0, -3, 0, 0, 0, 5,
            0, 0, 0, 0, 0, 0, 0, 0, 13, 33
        ], 10, 10), 0);
        const rowPermuted = rowPermutations.permuteSparseMatrix(initialMatrix);
        expect(rowPermuted.isValid()).toBeTruthy();
        expect(SparseMatrixCSR.lInfDistance(rowPermuted, expectedRowPermuted)).toBeCloseTo(0);
        const colPermuted = colPermutations.permuteSparseMatrix(initialMatrix);
        expect(colPermuted.isValid()).toBeTruthy();
        expect(SparseMatrixCSR.lInfDistance(colPermuted, expectedColPermuted)).toBeCloseTo(0);
    });
});