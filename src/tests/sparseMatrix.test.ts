import Matrix from "../dense/denseMatrix";
import { PermutationType, PermutationMatrix } from "../permutationMatrix";
import { SparseMatrixCSR, SparseMatrixRowIterator, SparseMatrixTwinRowIterator } from "../sparse/sparseMatrix";
import { SparseVector } from "../sparse/sparseVector";
import Triplet, { makeTriplet } from "../sparse/triplet";
import { SmallTolerance, Tolerance, assert } from "../utils";
import Vector from "../dense/vector";
import { SparseMatrixTriplets, SparseMatrixTripletsTwinIterator, TwinIteratorValue } from "../sparse/sparseMatrixTriplets";

/*
test.only("General sparse matrix", () => {
    expect(Matrix.lInfDistance(singularSparseTrivialMatrix.toDense(), singularDenseTrivialMatrix)).toBeCloseTo(0.0);
    expect(Matrix.lInfDistance(singularSparseNonTrivialMatrix.toDense(), singularDenseNonTrivialMatrix)).toBeCloseTo(0.0);
    expect(Matrix.lInfDistance(nonSingularSparseMatrix.toDense(), nonSingularDenseMatrix)).toBeCloseTo(0.0);

    //expect(SparseMatrixCSR.near(SparseMatrixCSR.identity(10).inverse(), SparseMatrixCSR.identity(10))).toBeTruthy();
    let permutationMatrix = new PermutationMatrix([1, 6, 8, 2, 5, 4, 9, 3, 0, 7], PermutationType.Row);
    assert(permutationMatrix.isValid(), "Invalid permutation");
    expect(SparseMatrixCSR.near(permutationMatrix.toSparseMatrix().inverse(), permutationMatrix.inverse().toSparseMatrix()));
    expect(Math.abs(permutationMatrix.toSparseMatrix().determinant())).toBeCloseTo(1.0);
    expect(nonSingularSparseMatrix.determinant()).not.toBeCloseTo(0.0);
    expect(SparseMatrixCSR.near(SparseMatrixCSR.mul(nonSingularSparseMatrix.inverse(), nonSingularSparseMatrix), SparseMatrixCSR.identity(nonSingularDenseMatrix.width())));
    // test trivial singular matrix
    expect(singularSparseTrivialMatrix.determinant()).toBeCloseTo(0.0);
    expect(singularSparseNonTrivialMatrix.determinant()).toBeCloseTo(0.0);
});
*/
describe('Sparse Matrix operations', () => {
    let triplets = [
        { column: 0, row: 0, value: 1 },
        { column: 2, row: 1, value: 1 },
        { column: 8, row: 1, value: 4 },
        { column: 3, row: 9, value: 8 },
        { column: 1, row: 1, value: 8 },
        { column: 1, row: 7, value: 7 },
        { column: 9, row: 7, value: 3 },
        { column: 7, row: 7, value: 10 },
        { column: 4, row: 4, value: 2 },
        { column: 6, row: 4, value: -3 },
        { column: 1, row: 4, value: -1 },
        { column: 5, row: 3, value: 1 },
        { column: 3, row: 2, value: 7 }];
    let denseMat = Matrix.fromTriplets(10, 10, triplets);
    let sparseCSRMat = SparseMatrixCSR.fromTriplets(10, 10, triplets, 0);
    let sparseTripletsMat = SparseMatrixTriplets.fromTriplets(10, 10, triplets);
    test('Construction', () => {
        expect(sparseCSRMat.isValid()).toBeTruthy();
        expect(sparseTripletsMat.isValid()).toBeTruthy();
        for (let { column, row, value } of triplets) {
            expect(sparseCSRMat.get(row, column)).toBeCloseTo(value);
            expect(sparseTripletsMat.get(row, column)).toBeCloseTo(value);
        }
        for (let row = 0; row < 10; ++row) {
            for (let col = 0; col < 10; ++col) {
                const element = denseMat.get(row, col);
                expect(sparseCSRMat.get(row, col)).toBeCloseTo(element);
                expect(sparseTripletsMat.get(row, col)).toBeCloseTo(element);
            }
        }

        const smallMatTriplets = [makeTriplet(0, 1, 1), makeTriplet(1, 2, -1), makeTriplet(2, 2, 2)];
        const smallCSRMat = SparseMatrixCSR.fromTriplets(3, 3, smallMatTriplets);
        const smallTripletsMat = SparseMatrixTriplets.fromTriplets(3, 3, smallMatTriplets);
        const ExpectedInitial = new Matrix([
            0, 1, 0,
            0, 0, -1,
            0, 0, 2
        ], 3, 3);
        expect(Matrix.lInfDistance(smallCSRMat.toDense(), ExpectedInitial)).toBeCloseTo(0);
        expect(Matrix.lInfDistance(smallTripletsMat.toDense(), ExpectedInitial)).toBeCloseTo(0);

        let insertions = [makeTriplet(1, 2, -5), makeTriplet(1, 1, 10), makeTriplet(2, 1, 0.3), makeTriplet(0, 2, 3)];
        const ExpectedModified = new Matrix([
            0, 1, 3,
            0, 10, -5,
            0, 0.3, 2
        ], 3, 3);
        for (let { row, column, value } of insertions) {
            smallCSRMat.set(row, column, value);
            smallTripletsMat.set(row, column, value);
        }
        expect(Matrix.lInfDistance(smallCSRMat.toDense(), ExpectedModified)).toBeCloseTo(0);
        expect(Matrix.lInfDistance(smallTripletsMat.toDense(), ExpectedModified)).toBeCloseTo(0);
    });
    test('Conversions', () => {
        let denseToTripletsMat = SparseMatrixTriplets.fromDense(denseMat, 0);
        let denseToCSRmat = SparseMatrixCSR.fromDense(denseMat, 0);
        let denseToTriplets = denseMat.toTriplets(0);
        expect(denseToTriplets.length).toBe(triplets.length);
        let sortedTriplets = triplets.slice();
        sortedTriplets.sort((a: Triplet, b: Triplet) => {
            let aIdx = a.column + a.row * denseMat.numCols();
            let bIdx = b.column + b.row * denseMat.numCols();
            return aIdx - bIdx;
        });
        for (let i = 0; i < triplets.length; ++i) {
            expect(sortedTriplets[i].column).toBe(denseToTriplets[i].column);
            expect(sortedTriplets[i].row).toBe(denseToTriplets[i].row);
            expect(sortedTriplets[i].value).toBe(denseToTriplets[i].value);
        }
        expect(SparseMatrixCSR.lInfDistance(denseToCSRmat, sparseCSRMat)).toBeCloseTo(0);
        expect(SparseMatrixTriplets.lInfDistance(denseToTripletsMat, sparseTripletsMat)).toBeCloseTo(0);

        expect(Matrix.lInfDistance(denseMat, sparseCSRMat.toDense())).toBeLessThan(SmallTolerance);
        expect(Matrix.lInfDistance(denseMat, sparseTripletsMat.toDense())).toBeLessThan(SmallTolerance);
    });
    test('Transpose', () => {
        let transposeCSR = sparseCSRMat.transpose();
        expect(transposeCSR.numCols()).toBe(sparseCSRMat.numRows());
        expect(transposeCSR.numRows()).toBe(sparseCSRMat.numCols());
        let transposeTriplets = sparseTripletsMat.transpose();
        expect(transposeTriplets.numCols()).toBe(sparseTripletsMat.numRows());
        expect(transposeTriplets.numRows()).toBe(sparseTripletsMat.numCols());
        for (let i = 0; i < transposeCSR.numCols(); ++i) {
            for (let j = 0; j < transposeCSR.numRows(); ++j)
                expect(transposeCSR.get(j, i)).toBe(sparseCSRMat.get(i, j));
        }
        for (let i = 0; i < transposeCSR.numCols(); ++i) {
            for (let j = 0; j < transposeCSR.numRows(); ++j)
                expect(transposeTriplets.get(j, i)).toBe(sparseTripletsMat.get(i, j));
        }
        expect(SparseMatrixCSR.near(transposeCSR, SparseMatrixCSR.fromDense(denseMat.transpose(), Tolerance), Tolerance)).toBeTruthy();
        expect(SparseMatrixTriplets.near(transposeTriplets, SparseMatrixTriplets.fromDense(denseMat.transpose(), Tolerance), Tolerance)).toBeTruthy();
        expect(transposeCSR.isValid()).toBeTruthy();
        expect(transposeTriplets.isValid()).toBeTruthy();
    });
    let vec = new Vector([0, 1, 3, 0, 2, 4, 0, -1, -2, 0]);
    let sparseVec = SparseVector.fromVector(vec.data);
    test('Vector multiplication', () => {
        let postMulRes = Matrix.postMulVec(denseMat, vec);
        let preMulRes = Matrix.preMulVec(vec, denseMat);
        expect(Vector.l1Distance(SparseMatrixCSR.postMul(sparseCSRMat, vec), postMulRes)).toBeCloseTo(0);
        expect(Vector.l1Distance(SparseMatrixCSR.preMul(vec, sparseCSRMat), preMulRes)).toBeCloseTo(0);

        expect(Vector.l1Distance(SparseMatrixTriplets.postMul(sparseTripletsMat, vec), postMulRes)).toBeCloseTo(0);
        expect(Vector.l1Distance(SparseMatrixTriplets.preMul(vec, sparseTripletsMat), preMulRes)).toBeCloseTo(0);

        expect(SparseVector.near(SparseMatrixCSR.postMulSparse(sparseCSRMat, sparseVec), SparseVector.fromVector(postMulRes.data, Tolerance), Tolerance)).toBeTruthy();
        expect(SparseVector.near(SparseMatrixCSR.preMulSparse(sparseVec, sparseCSRMat), SparseVector.fromVector(preMulRes.data, Tolerance), Tolerance)).toBeTruthy();

        expect(SparseVector.near(SparseMatrixTriplets.postMulSparse(sparseTripletsMat, sparseVec), SparseVector.fromVector(postMulRes.data, Tolerance), Tolerance)).toBeTruthy();
        expect(SparseVector.near(SparseMatrixTriplets.preMulSparse(sparseVec, sparseTripletsMat), SparseVector.fromVector(preMulRes.data, Tolerance), Tolerance)).toBeTruthy();
    });
    test('Norms', () => {
        let copyCSRMat = sparseCSRMat.clone();
        expect(SparseMatrixCSR.lInfDistance(sparseCSRMat, sparseCSRMat)).toBeCloseTo(0);
        expect(SparseMatrixCSR.lInfDistance(copyCSRMat, sparseCSRMat)).toBeCloseTo(0);
        expect(SparseMatrixCSR.near(sparseCSRMat, copyCSRMat)).toBeTruthy();
        expect(SparseMatrixCSR.near(sparseCSRMat, sparseCSRMat)).toBeTruthy();
        copyCSRMat.set(5, 5, copyCSRMat.get(5, 5) + 10);
        expect(SparseMatrixCSR.lInfDistance(copyCSRMat, sparseCSRMat)).toBeCloseTo(10);
        expect(SparseMatrixCSR.lInfDistance(sparseCSRMat, copyCSRMat)).toBeCloseTo(10);
        expect(SparseMatrixCSR.near(sparseCSRMat, copyCSRMat)).toBeFalsy();
    });
    test('Multiplication', () => {
        const otherTriplets: Triplet[] = [
            { column: 0, row: 0, value: 1 },
            { column: 0, row: 1, value: 1 },
            { column: 2, row: 1, value: -1 },
            { column: 1, row: 2, value: 2 },
            { column: 0, row: 4, value: -3 },
            { column: 1, row: 4, value: 2 },
            { column: 1, row: 6, value: -4 },
            { column: 2, row: 6, value: 1 },
            { column: 1, row: 7, value: 5 },
            { column: 0, row: 8, value: -6 },
            { column: 1, row: 8, value: 6 },
            { column: 2, row: 8, value: 6 },
            { column: 2, row: 9, value: -3 },
        ];
        const expectedTriplets: Triplet[] = [
            { column: 0, row: 0, value: 1 },
            { column: 0, row: 1, value: -16 },
            { column: 1, row: 1, value: 26 },
            { column: 2, row: 1, value: 16 },
            { column: 0, row: 4, value: -7 },
            { column: 1, row: 4, value: 16 },
            { column: 2, row: 4, value: -2 },
            { column: 0, row: 7, value: 7 },
            { column: 1, row: 7, value: 50 },
            { column: 2, row: 7, value: -16 }
        ];
        const expectedCSRMat = SparseMatrixCSR.fromTriplets(10, 3, expectedTriplets);
        const expectedDenseMat = Matrix.fromTriplets(10, 3, expectedTriplets);
        const expectedTripletMat = SparseMatrixTriplets.fromTriplets(10, 3, expectedTriplets);
        let otherCSRMat = SparseMatrixCSR.fromTriplets(10, 3, otherTriplets, 0);
        let otherTripletsMat = SparseMatrixTriplets.fromTriplets(10, 3, otherTriplets);
        let denseResult = Matrix.mul(sparseCSRMat.toDense(), otherCSRMat.toDense());
        let sparseCSRResult = SparseMatrixCSR.mul(sparseCSRMat, otherCSRMat);
        let sparseTripletResult = SparseMatrixTriplets.mul(sparseTripletsMat, otherTripletsMat);
        expect(Matrix.lInfDistance(denseResult, expectedDenseMat)).toBeLessThan(SmallTolerance);
        expect(SparseMatrixCSR.lInfDistance(sparseCSRResult, expectedCSRMat)).toBeLessThan(SmallTolerance);
        expect(SparseMatrixTriplets.lInfDistance(sparseTripletResult, expectedTripletMat)).toBeLessThan(SmallTolerance);

        expect(Matrix.lInfDistance(denseResult, sparseCSRResult.toDense())).toBeLessThan(SmallTolerance);
        expect(Matrix.lInfDistance(denseResult, sparseTripletResult.toDense())).toBeLessThan(SmallTolerance);
    });
    test('Iterators', () => {
        let rowIterator: SparseMatrixRowIterator = new SparseMatrixRowIterator(sparseCSRMat, 7);
        let rowValues: { value: number, colIdx: number }[] = [];
        while (!rowIterator.isDone()) {
            rowValues.push(rowIterator.advance());
        }
        const expectedRowValues = [{ value: 7, colIdx: 1 }, { value: 10, colIdx: 7 }, { value: 3, colIdx: 9 }];
        expect(expectedRowValues.length == rowValues.length);
        for (let i = 0; i < expectedRowValues.length; ++i) {
            expect(expectedRowValues[i].value).toBe(rowValues[i].value);
            expect(expectedRowValues[i].colIdx).toBe(rowValues[i].colIdx);
        }
        let otherMat = SparseMatrixCSR.fromTriplets(5, 10, [
            makeTriplet(4, 3, 1),
            makeTriplet(4, 4, -3),
            makeTriplet(4, 1, 0.102),
            makeTriplet(4, 0, -100),
            makeTriplet(4, 8, 10.35),
            makeTriplet(4, 5, 7.0131)], 0);
        expect(otherMat.isValid()).toBeTruthy();
        let twinIterator: SparseMatrixTwinRowIterator = new SparseMatrixTwinRowIterator(sparseCSRMat, otherMat, 4);
        let reversedTwinIterator: SparseMatrixTwinRowIterator = new SparseMatrixTwinRowIterator(otherMat, sparseCSRMat, 4);
        const expectedTwinIteratorValues = [{ value1: 0, value2: -100, colIdx: 0 },
        { value1: -1, value2: 0.102, colIdx: 1 }, { value1: 0, value2: 1, colIdx: 3 }, { value1: 2, value2: -3, colIdx: 4 }, { value1: 0, value2: 7.0131, colIdx: 5 }, { value1: -3, value2: 0, colIdx: 6 }, { value1: 0, value2: 10.35, colIdx: 8 }];
        let twinItRowValues: { value1: number, value2: number, colIdx: number }[] = [];
        let reversedTwinItRowValues: { value1: number, value2: number, colIdx: number }[] = []
        while (!twinIterator.isDone())
            twinItRowValues.push(twinIterator.advance());
        while (!reversedTwinIterator.isDone())
            reversedTwinItRowValues.push(reversedTwinIterator.advance());
        expect(expectedTwinIteratorValues.length == twinItRowValues.length);
        expect(expectedTwinIteratorValues.length == reversedTwinItRowValues.length);
        for (let i = 0; i < expectedTwinIteratorValues.length; ++i) {
            expect(expectedTwinIteratorValues[i].value1).toBe(twinItRowValues[i].value1);
            expect(expectedTwinIteratorValues[i].value2).toBe(twinItRowValues[i].value2);
            expect(expectedTwinIteratorValues[i].colIdx).toBe(twinItRowValues[i].colIdx);
            expect(expectedTwinIteratorValues[i].value1).toBe(reversedTwinItRowValues[i].value2);
            expect(expectedTwinIteratorValues[i].value2).toBe(reversedTwinItRowValues[i].value1);
            expect(expectedTwinIteratorValues[i].colIdx).toBe(reversedTwinItRowValues[i].colIdx);
        }

        let tripletMat1 = SparseMatrixTriplets.fromTriplets(5, 4, [makeTriplet(0, 0, 1), makeTriplet(0, 2, -2), makeTriplet(1, 0, 13), makeTriplet(1, 1, 14), makeTriplet(2, 1, -2), makeTriplet(2, 3, -3), makeTriplet(4, 3, 71)]);
        let tripletMat2 = SparseMatrixTriplets.fromTriplets(5, 4, [makeTriplet(0, 1, 3), makeTriplet(0, 3, 2), makeTriplet(1, 1, -3), makeTriplet(2, 0, 4), makeTriplet(2, 1, 5), makeTriplet(2, 3, 4), makeTriplet(3, 2, 6)]);
        let tripletsTwinIt = new SparseMatrixTripletsTwinIterator(tripletMat1, tripletMat2);
        const expectedTripletsTwinItValues: TwinIteratorValue[] = [
            { row: 0, column: 0, value1: 1, value2: 0 },
            { row: 0, column: 1, value1: 0, value2: 3 },
            { row: 0, column: 2, value1: -2, value2: 0 },
            { row: 0, column: 3, value1: 0, value2: 2 },
            { row: 1, column: 0, value1: 13, value2: 0 },
            { row: 1, column: 1, value1: 14, value2: -3 },
            { row: 2, column: 0, value1: 0, value2: 4 },
            { row: 2, column: 1, value1: -2, value2: 5 },
            { row: 2, column: 3, value1: -3, value2: 4 },
            { row: 3, column: 2, value1: 0, value2: 6 },
            { row: 4, column: 3, value1: 71, value2: 0 }
        ];

        let actualTripletsTwinItValues: TwinIteratorValue[] = [];
        while (!tripletsTwinIt.isDone())
            actualTripletsTwinItValues.push(tripletsTwinIt.advance());
        tripletsTwinIt = new SparseMatrixTripletsTwinIterator(tripletMat2, tripletMat1);
        let reversedTripletsTwinItValues: TwinIteratorValue[] = [];
        while (!tripletsTwinIt.isDone())
            reversedTripletsTwinItValues.push(tripletsTwinIt.advance());

        expect(expectedTripletsTwinItValues.length == actualTripletsTwinItValues.length);
        expect(expectedTripletsTwinItValues.length == reversedTripletsTwinItValues.length);
        for (let i = 0; i < expectedTwinIteratorValues.length; ++i) {
            expect(expectedTripletsTwinItValues[i].value1).toBe(actualTripletsTwinItValues[i].value1);
            expect(expectedTripletsTwinItValues[i].value2).toBe(actualTripletsTwinItValues[i].value2);
            expect(expectedTripletsTwinItValues[i].column).toBe(actualTripletsTwinItValues[i].column);
            expect(expectedTripletsTwinItValues[i].row).toBe(actualTripletsTwinItValues[i].row);
            expect(expectedTripletsTwinItValues[i].value1).toBe(reversedTripletsTwinItValues[i].value2);
            expect(expectedTripletsTwinItValues[i].value2).toBe(reversedTripletsTwinItValues[i].value1);
            expect(expectedTripletsTwinItValues[i].column).toBe(reversedTripletsTwinItValues[i].column);
            expect(expectedTripletsTwinItValues[i].row).toBe(reversedTripletsTwinItValues[i].row);
        }
    });
    let firstMat = SparseMatrixCSR.fromTriplets(3, 3, [
        makeTriplet(0, 0, 1),
        makeTriplet(0, 2, 3),
        makeTriplet(1, 0, 2),
        makeTriplet(2, 1, 2)
    ]);
    let secondMat = SparseMatrixCSR.fromTriplets(3, 3, [
        makeTriplet(0, 0, -7),
        makeTriplet(0, 1, -7),
        makeTriplet(1, 0, 6),
        makeTriplet(2, 1, -6),
        makeTriplet(2, 2, 2)
    ]);
    test('Entrywise product', () => {
        let expected = SparseMatrixCSR.fromTriplets(3, 3, [
            makeTriplet(0, 0, -7),
            makeTriplet(1, 0, 12),
            makeTriplet(2, 1, -12)
        ]);
        expect(SparseMatrixCSR.lInfDistance(SparseMatrixCSR.entrywiseProduct(firstMat, secondMat), expected)).toBeCloseTo(0);
    });
    test('Kronecker product', () => {
        let expected = SparseMatrixCSR.fromTriplets(9, 9, [
            makeTriplet(0, 0, -7),
            makeTriplet(0, 1, -7),
            makeTriplet(1, 0, 6),
            makeTriplet(2, 1, -6),
            makeTriplet(2, 2, 2),

            makeTriplet(0, 6, -3 * 7),
            makeTriplet(0, 7, -3 * 7),
            makeTriplet(1, 6, 3 * 6),
            makeTriplet(2, 7, -3 * 6),
            makeTriplet(2, 8, 3 * 2),

            makeTriplet(3, 0, -2 * 7),
            makeTriplet(3, 1, -2 * 7),
            makeTriplet(4, 0, 2 * 6),
            makeTriplet(5, 1, -2 * 6),
            makeTriplet(5, 2, 2 * 2),

            makeTriplet(6, 3, -2 * 7),
            makeTriplet(6, 4, -2 * 7),
            makeTriplet(7, 3, 2 * 6),
            makeTriplet(8, 4, -2 * 6),
            makeTriplet(8, 5, 2 * 2)
        ]);
        expect(SparseMatrixCSR.lInfDistance(SparseMatrixCSR.kroneckerProduct(firstMat, secondMat), expected)).toBeCloseTo(0);
    });
    test('Distances', () => {
        let mat1 = SparseMatrixCSR.fromTriplets(4, 4, [
            makeTriplet(0, 0, 3),
            makeTriplet(2, 1, 4),
            makeTriplet(3, 2, 5),
            makeTriplet(3, 3, -1)
        ]);
        let mat2 = SparseMatrixCSR.fromTriplets(4, 4, [
            makeTriplet(0, 0, 4),
            makeTriplet(1, 1, -5),
            makeTriplet(1, 2, 1),
            makeTriplet(3, 2, 3)
        ]);
        expect(SparseMatrixCSR.lInfDistance(mat1, mat2)).toBeCloseTo(5);
        expect(SparseMatrixCSR.lInfDistance(mat2, mat1)).toBeCloseTo(5);

        expect(SparseMatrixCSR.near(mat1, mat2, Tolerance)).toBeFalsy();
        expect(SparseMatrixCSR.near(mat2, mat1, Tolerance)).toBeFalsy();

        expect(SparseMatrixCSR.near(mat1, mat1, Tolerance)).toBeTruthy();
        expect(SparseMatrixCSR.lInfDistance(mat1, mat1)).toBeCloseTo(0);
    });
    const TestPrintMethods = false;
    if (TestPrintMethods) {
        console.log(sparseTripletsMat.toString());
        console.log(sparseCSRMat.toString());
        console.log(sparseTripletsMat.printAsMatrix());
        console.log(sparseCSRMat.printAsMatrix());
    }
});