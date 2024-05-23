import Matrix from "../dense/denseMatrix";
import mat2 from "../dense/mat2";
import mat3 from "../dense/mat3";
import mat4 from "../dense/mat4";
import Triplet from "../sparse/triplet";
import { Tolerance, SmallTolerance, assert } from "../utils";
import vec2 from "../dense/vec2";
import vec3 from "../dense/vec3";
import vec4 from "../dense/vec4";
import Vector from "../dense/vector";

describe('Dense tests', () => {
    const singularTrivialMatrixTriplets: Triplet[] = [{ row: 1, column: 1, value: 1 },
    { row: 2, column: 2, value: 1 }, { row: 0, column: 0, value: 2 }, { row: 3, column: 3, value: 1 }, { row: 5, column: 5, value: 1 }];
    const singularDenseTrivialMatrix = Matrix.fromTriplets(6, 6, singularTrivialMatrixTriplets);

    const singularNonTrivialMatrixTriplets: Triplet[] = [
        { row: 0, column: 0, value: 1 },
        { row: 0, column: 2, value: 3 },
        { row: 0, column: 4, value: 1 },
        { row: 1, column: 0, value: 2 },
        { row: 1, column: 1, value: 2 },
        { row: 1, column: 5, value: 2 },
        { row: 2, column: 0, value: 2 },
        { row: 2, column: 1, value: 2 },
        { row: 2, column: 2, value: 3 },
        { row: 2, column: 3, value: 1 },
        { row: 2, column: 5, value: 3 },
        { row: 3, column: 4, value: 4 },
        { row: 4, column: 0, value: 3 },
        { row: 4, column: 1, value: -3 },
        { row: 4, column: 2, value: -1 },
        { row: 4, column: 4, value: 2 },
        { row: 5, column: 1, value: 2 },
        { row: 5, column: 3, value: 2 },
        { row: 5, column: 5, value: 4 }
    ];
    const singularDenseNonTrivialMatrix = Matrix.fromTriplets(6, 6, singularNonTrivialMatrixTriplets);

    const nonSingularMatrixTriplets: Triplet[] = [{ row: 0, column: 0, value: 1 }, { row: 0, column: 2, value: 3 }, { row: 0, column: 4, value: 1 },
    { row: 1, column: 0, value: 2 }, { row: 1, column: 1, value: 2 }, { row: 1, column: 5, value: 2 }, { row: 2, column: 1, value: 2 }, { row: 2, column: 3, value: 1 }
        , { row: 2, column: 5, value: 3 }, { row: 3, column: 4, value: 4 }, { row: 4, column: 0, value: 3 }, { row: 4, column: 1, value: -3 }, { row: 4, column: 2, value: -1 }
        , { row: 4, column: 4, value: 2 }, { row: 5, column: 1, value: 2 }, { row: 5, column: 3, value: 2 }, { row: 5, column: 5, value: 4 }];
    const nonSingularDenseMatrix = Matrix.fromTriplets(6, 6, nonSingularMatrixTriplets);
    const nonSingularMatrixDeterminant = 144.0;

    describe('Matrix operations', () => {
        test('Determinant', () => {
            expect(mat2.identity().determinant()).toBeCloseTo(1);
            expect(mat3.identity().determinant()).toBeCloseTo(1);
            expect(mat4.identity().determinant()).toBeCloseTo(1);
            expect(mat2.empty().determinant()).toBeCloseTo(0);
            expect(mat3.empty().determinant()).toBeCloseTo(0);
            expect(mat4.empty().determinant()).toBeCloseTo(0);
            interface TestData {
                matrix: Matrix;
                expectedValue: number;
            };
            let testData: TestData[] = [
                { matrix: Matrix.empty(4, 4), expectedValue: 0 },
                { matrix: Matrix.identity(4), expectedValue: 1 },
                { matrix: new Matrix([1, 2, 3, 4], 2, 2), expectedValue: -2 },
                { matrix: new Matrix([3, 2, 4, 1, 2, 3, 1, 5, 2], 3, 3), expectedValue: -19 },
                {
                    matrix: new Matrix([4, 2, 0, 2, 0,
                        2, 2, 0, 0, 0,
                        3, 2, 3, 1, 0,
                        0, 0, 0, 0, 4,
                        0, -3, -1, 0, 2], 5, 5), expectedValue: 144
                },
                {
                    matrix: new Matrix([0, 2, -1, 0,
                        2, 0, 0, 0, 3, 0, 3, 1, 0, 4, 0, 0], 4, 4), expectedValue: 8
                },
                { matrix: singularDenseTrivialMatrix, expectedValue: 0 },
                { matrix: singularDenseNonTrivialMatrix, expectedValue: 0 },
                { matrix: nonSingularDenseMatrix, expectedValue: nonSingularMatrixDeterminant },
            ];

            for (const { matrix, expectedValue } of testData) {
                expect(matrix.determinantNaive()).toBeCloseTo(expectedValue);
                expect(matrix.determinant()).toBeCloseTo(expectedValue);
            }
        });
        test('Inverse', () => {
            let testMatrix = new mat4(
                1, 3, 2, 4,
                6, 8, 3, -2,
                -5, 3, 2, 1,
                3, 4, 5, 2);
            let expectedMatrix = new mat4(
                0.036036, 0.027027, -0.144144, 0.027027,
                0.122265, 0.127413, 0.082368, -0.158301,
                -0.223938, -0.096525, 0.038610, 0.332046,
                0.261261, -0.054054, -0.045045, -0.054054);
            expect(Matrix.lInfDistance(testMatrix.inverse(), expectedMatrix)).toBeCloseTo(0);
            expect(Matrix.near(mat4.identity(), mat4.mul(testMatrix, testMatrix.inverse()), Tolerance)).toBeTruthy();
        });
        test('Queries', () => {
            let testMatrix = new Matrix(
                [1, 3, 2, 4,
                    6, 8, 3, -2,
                    -5, 3, -2, 1,
                    3, 4, 5, 2], 4, 4);
            expect(testMatrix.trace()).toBeCloseTo(9);
        });
        test('Transpose', () => {
            let testMatrix = new mat4(
                1, 3, 2, 4,
                6, 8, 3, -2,
                -5, 3, 2, 1,
                3, 4, 5, 2);
            expect(Matrix.lInfDistance(testMatrix.transpose().transpose(), testMatrix)).toBeCloseTo(0);
            let squareMatrix = new Matrix([
                1, 3, 2, 4,
                6, 8, 3, -2,
                -5, 3, 2, 1,
                3, 4, 5, 2], 4, 4);
            let squareTranspose = new Matrix([
                1, 6, -5, 3,
                3, 8, 3, 4,
                2, 3, 2, 5,
                4, -2, 1, 2
            ], 4, 4);
            expect(Matrix.lInfDistance(squareMatrix.transpose(), squareTranspose)).toBeCloseTo(0);
            expect(Matrix.lInfDistance(squareMatrix.clone().transposeInPlace(), squareTranspose)).toBeCloseTo(0);
            let rectMatrix = new Matrix([
                1, 3, 2, 4,
                6, 8, 3, -2], 2, 4);
            let rectTranspose = new Matrix([
                1, 6,
                3, 8,
                2, 3,
                4, -2
            ], 4, 2);
            expect(Matrix.lInfDistance(rectMatrix.transpose(), rectTranspose)).toBeCloseTo(0);
            expect(Matrix.lInfDistance(rectMatrix.clone().transposeInPlace(), rectTranspose)).toBeCloseTo(0);
        });
        test('Addition/Subtraction', () => {
            let a = new Matrix([1, 2, 3, 4], 2, 2);
            let b = new Matrix([2, 3, 4, 5], 2, 2);
            expect(Matrix.near(Matrix.add(a, b), new Matrix([3, 5, 7, 9], 2, 2), SmallTolerance)).toBeTruthy();
            expect(Matrix.near(Matrix.sub(a, b), new Matrix([-1, -1, -1, -1], 2, 2), SmallTolerance)).toBeTruthy();
        })
        test('Multiplication', () => {
            // multiply rectangular matrices
            let a = new Matrix([
                1, 3, 1,
                2, 2, 4,
                -1, 1, -1,
                3, 5, 0], 4, 3);
            let b = new Matrix([
                2, 3,
                -2, 2,
                1, -3], 3, 2);
            expect(Matrix.lInfDistance(Matrix.mul(a, b), new Matrix([-3, 6, 4, -2, -5, 2, -4, 19], 4, 2))).toBeCloseTo(0);

            let m3a = new mat3(1, 3, 2, 4, 6, 8, 3, -2, -5);
            let m3b = new mat3(4, -2, -1, -31, 21, -4, 51, -13, 10);
            let m3c = new mat3(13, 35, 7, 238, 14, 52, -181, 17, -45);
            expect(Matrix.near(m3a, mat3.mul(m3a, mat3.identity()), Tolerance)).toBeTruthy();
            expect(Matrix.near(m3c, mat3.mul(m3a, m3b), Tolerance)).toBeTruthy();
            expect(Matrix.near(mat3.identity(), mat3.mul(m3a, m3a.inverse()), Tolerance)).toBeTruthy();
        })
    });

    test('Vector operations', () => {
        let a3D = new vec3(1., 3., 2.);
        let b3D = new vec3(2., -1., -4.);
        expect(a3D.l1norm()).toBeCloseTo(6, Tolerance);
        expect(a3D.l2norm()).toBeCloseTo(Math.sqrt(vec3.dot(a3D, a3D)), Tolerance);
        expect(a3D.lInfnorm()).toBeCloseTo(3, Tolerance);
        expect(vec3.dot(a3D, vec3.cross(a3D, b3D))).toBeCloseTo(0.0);
        expect(vec3.near(vec3.add(a3D, b3D), new vec3(3, 2, -2), Tolerance)).toBeTruthy();
        expect(vec3.near(vec3.mul(a3D, b3D), new vec3(2, -3, -8), Tolerance)).toBeTruthy();
        expect(vec3.near(vec3.sub(a3D, b3D), new vec3(-1, 4, 6), Tolerance)).toBeTruthy();
        expect(vec3.near(vec3.div(a3D, b3D), new vec3(0.5, -3, -0.5), Tolerance)).toBeTruthy();
    });

    test('Matrix-vector operations', () => {
        let p2D = new vec2(2., -3.);
        let p3D = new vec3(1.6, 0.5, -0.2);
        let p4D = new vec4(-1., 0.3, 2., -4.);
        let m2D = new mat2(
            2., 0.3,
            7., -13.
        );
        let m3D = new mat3(
            0.3, 1.3, -60.3,
            -0.9, 0.51, 3.1,
            10.23, 1.2, 3.4
        );
        let m4D = new mat4(
            0.3, 1.3, -60.3, 0.12,
            -0.9, 0.51, 3.1, 14,
            10.23, 1.2, 3.4, 1.7,
            0, -6.75, 1, -0.8
        );
        let m2DInv = m2D.inverse();
        let m3DInv = m3D.inverse();
        let m4DInv = m4D.inverse();
        expect(vec2.near(p2D, m2DInv.postMulVec(m2D.postMulVec(p2D))));
        expect(vec2.near(p2D, m2DInv.preMulVec(m2D.preMulVec(p2D))));
        expect(vec3.near(p3D, m3DInv.postMulVec(m3D.postMulVec(p3D))));
        expect(vec3.near(p3D, m3DInv.preMulVec(m3D.preMulVec(p3D))));
        expect(vec4.near(p4D, m4DInv.postMulVec(m4D.postMulVec(p4D))));
        expect(vec4.near(p4D, m4DInv.preMulVec(m4D.preMulVec(p4D))));

        /*m4D = new mat4(
            0, 1, 0, 1,
            1, 0, 0, 0,
            0, 0, 0, 1,
            0, 0, 1, 0
        );*/
        let mat = m4D.toMatrix();
        let rhs = p4D.toVector();
        expect(Vector.near(Matrix.postMulVec(mat, rhs), m4D.postMulVec(p4D).toVector(), SmallTolerance)).toBeTruthy();
        expect(Vector.near(Matrix.preMulVec(rhs, mat), m4D.preMulVec(p4D).toVector(), SmallTolerance)).toBeTruthy();

        expect(new Matrix([1, 2, 3, 4, 5, 6, 7, 8, 9], 3, 3).determinantNaive()).toBeCloseTo(0.0);
        expect(Vector.near(m4D.inverse().postMulVec(p4D).toVector(), Matrix.solve(mat.clone(), rhs), SmallTolerance)).toBeTruthy();
        expect(Matrix.near(mat.transpose(), m4D.transpose().toMatrix(), SmallTolerance)).toBeTruthy();
    });
    test("General dense matrix", () => {
        // test nonsingular matrices
        expect(Matrix.near(Matrix.identity(4).inverseNaive(), Matrix.identity(4))).toBeTruthy();
        //expect(nonSingularDenseMatrix.determinant()).not.toBeCloseTo(0.0);
        //expect(Matrix.near(Matrix.mul(nonSingularDenseMatrix.inverse(), nonSingularDenseMatrix), Matrix.identity(nonSingularDenseMatrix.width())));
        // test trivial singular matrix
        //expect(singularDenseTrivialMatrix.determinant()).toBeCloseTo(0.0);
        //expect(singularDenseNonTrivialMatrix.determinant()).toBeCloseTo(0.0);

        // test non-trivial singular matrix
    });
    test("Row/column manipulations", () => {
        let mat = new Matrix([
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
            10, 11, 12,
            13, 14, 15
        ], 5, 3);
        expect(Vector.lInfDistance(mat.diag(0), new Vector([1, 5, 9]))).toBeCloseTo(0);
        expect(Vector.lInfDistance(mat.diag(-1), new Vector([2, 6]))).toBeCloseTo(0);
        expect(Vector.lInfDistance(mat.diag(-2), new Vector([3]))).toBeCloseTo(0);
        expect(Vector.lInfDistance(mat.diag(2), new Vector([7, 11, 15]))).toBeCloseTo(0);
        expect(Vector.lInfDistance(mat.diag(3), new Vector([10, 14]))).toBeCloseTo(0);
        expect(Vector.lInfDistance(mat.diag(4), new Vector([13]))).toBeCloseTo(0);
        let row1 = mat.getRow(1);
        expect(Vector.lInfDistance(row1, new Vector([4, 5, 6]))).toBeCloseTo(0);
        let col2 = mat.getColumn(2);
        expect(Vector.lInfDistance(col2, new Vector([3, 6, 9, 12, 15]))).toBeCloseTo(0);
        let subCol2 = mat.subColumn(1, 2, 3);
        expect(Vector.lInfDistance(subCol2, new Vector([6, 9, 12]))).toBeCloseTo(0);
        let subRow1 = mat.subRow(1, 1, 2);
        expect(Vector.lInfDistance(subRow1, new Vector([5, 6]))).toBeCloseTo(0);
        mat.setSubRow(new Vector([3, 4]), 3, 1);
        expect(Matrix.lInfDistance(mat, new Matrix([
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
            10, 3, 4,
            13, 14, 15
        ], 5, 3))).toBeCloseTo(0);
        mat.setSubColumn(new Vector([1, 2]), 2, 2);
        expect(Matrix.lInfDistance(mat, new Matrix([
            1, 2, 3,
            4, 5, 6,
            7, 8, 1,
            10, 3, 2,
            13, 14, 15
        ], 5, 3))).toBeCloseTo(0);
        mat.setRow(3, new Vector([6, 5, 4]));
        expect(Matrix.lInfDistance(mat, new Matrix([
            1, 2, 3,
            4, 5, 6,
            7, 8, 1,
            6, 5, 4,
            13, 14, 15
        ], 5, 3))).toBeCloseTo(0);
        mat.setColumn(1, new Vector([1, 1, 1, 1, 1]));
        expect(Matrix.lInfDistance(mat, new Matrix([
            1, 1, 3,
            4, 1, 6,
            7, 1, 1,
            6, 1, 4,
            13, 1, 15
        ], 5, 3))).toBeCloseTo(0);
        mat.swapColumns(0, 2);
        expect(Matrix.lInfDistance(mat, new Matrix([
            3, 1, 1,
            6, 1, 4,
            1, 1, 7,
            4, 1, 6,
            15, 1, 13
        ], 5, 3))).toBeCloseTo(0);
        mat.swapRows(2, 4);
        expect(Matrix.lInfDistance(mat, new Matrix([
            3, 1, 1,
            6, 1, 4,
            15, 1, 13,
            4, 1, 6,
            1, 1, 7
        ], 5, 3))).toBeCloseTo(0);
        mat.shrinkCols(2);
        expect(Matrix.lInfDistance(mat, new Matrix([
            3, 1,
            6, 1,
            15, 1,
            4, 1,
            1, 1
        ], 5, 2))).toBeCloseTo(0);
        mat.shrinkRows(2);
        expect(Matrix.lInfDistance(mat, new Matrix([
            3, 1,
            6, 1
        ], 2, 2))).toBeCloseTo(0);
    });
});

describe('Classification tests', () => {
    const upperHessenberg = new Matrix([
        1, 4, 2, 3,
        3, 4, 1, 7,
        0, 2, 3, 4,
        0, 0, 1, 3], 4, 4);
    const lowerHessenberg = upperHessenberg.transpose();
    const upperTriangular = new Matrix([
        1, -1, 2, -2,
        0, 2, -4, 3,
        0, 0, 2, -5,
        0, 0, 0, 5], 4, 4);
    const lowerTriangular = upperTriangular.transpose();
    const diagonal = new Matrix([
        1, 0, 0, 0,
        0, 2, 0, 0,
        0, 0, -1, 0,
        0, 0, 0, -13
    ], 4, 4);
    const symmetric = new Matrix([
        1, 2, 3, 4,
        2, 0, 2, 1,
        3, 2, 4, 0,
        4, 1, 0, -2
    ], 4, 4);
    const tridiagonal = new Matrix([
        1, 2, 0, 0,
        4, 3, 5, 0,
        0, -2, 4, 10,
        0, 0, 2, 3
    ], 4, 4);
    const identity = Matrix.identity(4);
    expect(upperHessenberg.isHessenberg(true)).toBeTruthy();
    expect(lowerHessenberg.isHessenberg(true)).toBeFalsy();
    expect(tridiagonal.isHessenberg(true)).toBeTruthy();

    expect(lowerHessenberg.isHessenberg(false)).toBeTruthy();
    expect(upperHessenberg.isHessenberg(false)).toBeFalsy();
    expect(tridiagonal.isHessenberg(false)).toBeTruthy();

    expect(upperTriangular.isTriangular(true)).toBeTruthy();
    expect(lowerTriangular.isTriangular(true)).toBeFalsy();

    expect(lowerTriangular.isTriangular(false)).toBeTruthy();
    expect(upperTriangular.isTriangular(false)).toBeFalsy();

    expect(diagonal.isDiagonal()).toBeTruthy();
    expect(lowerTriangular.isDiagonal()).toBeFalsy();

    expect(identity.isIdentity()).toBeTruthy();
    expect(diagonal.isIdentity()).toBeFalsy();

    expect(symmetric.isSymmetric()).toBeTruthy()
    expect(lowerTriangular.isSymmetric()).toBeFalsy();

    expect(tridiagonal.isTridiagonal()).toBeTruthy();
    expect(lowerTriangular.isTridiagonal()).toBeFalsy();
});