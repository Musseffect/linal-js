import Matrix from "../dense/denseMatrix";
import { DiagonalMatrixView, TriMatrixType, TriMatrixView } from "../dense/matrixView";



test("TriMatrixView", () => {
    let matrix = new Matrix([
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    ], 4, 4);
    const testMatrix = (view: TriMatrixView, expectedMatrix: Matrix) => {
        for (let row = 0; row < expectedMatrix.numRows(); ++row) {
            for (let col = 0; col < expectedMatrix.numCols(); ++col) {
                expect(view.get(row, col)).toBeCloseTo(expectedMatrix.get(row, col));
            }
        }
        expect(Matrix.lInfDistance(view.toMatrix(), expectedMatrix)).toBeCloseTo(0);
    };
    let lowerUnitDiag = new TriMatrixView(matrix, TriMatrixType.lower, 1);
    testMatrix(lowerUnitDiag, new Matrix([
        1, 0, 0, 0,
        5, 1, 0, 0,
        9, 10, 1, 0,
        13, 14, 15, 1
    ], 4, 4));

    let lowerExistingDiag = new TriMatrixView(matrix, TriMatrixType.lower);
    testMatrix(lowerExistingDiag, new Matrix([
        1, 0, 0, 0,
        5, 6, 0, 0,
        9, 10, 11, 0,
        13, 14, 15, 16
    ], 4, 4));
    let upperUnitDiag = new TriMatrixView(matrix, TriMatrixType.upper, 1);
    testMatrix(upperUnitDiag, new Matrix([
        1, 2, 3, 4,
        0, 1, 7, 8,
        0, 0, 1, 12,
        0, 0, 0, 1
    ], 4, 4));
    let upperExistingDiag = new TriMatrixView(matrix, TriMatrixType.upper);
    testMatrix(upperExistingDiag, new Matrix([
        1, 2, 3, 4,
        0, 6, 7, 8,
        0, 0, 11, 12,
        0, 0, 0, 16
    ], 4, 4));
});

test("DiagonalView", () => {
    let sqrMatrix = new Matrix([
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    ], 4, 4);
    const testMatrix = (view: DiagonalMatrixView, expectedMatrix: Matrix) => {
        for (let row = 0; row < expectedMatrix.numRows(); ++row) {
            for (let col = 0; col < expectedMatrix.numCols(); ++col) {
                expect(view.get(row, col)).toBeCloseTo(expectedMatrix.get(row, col));
            }
        }
        expect(Matrix.lInfDistance(view.toMatrix(), expectedMatrix)).toBeCloseTo(0);
    };
    testMatrix(new DiagonalMatrixView(sqrMatrix), new Matrix([
        1, 0, 0, 0,
        0, 6, 0, 0,
        0, 0, 11, 0,
        0, 0, 0, 16
    ], 4, 4));
    let rectMatrix1 = new Matrix([
        1, 2, 3, 4,
        5, -1, 2, 3
    ], 2, 4);
    testMatrix(new DiagonalMatrixView(rectMatrix1), new Matrix([
        1, 0, 0, 0,
        0, -1, 0, 0
    ], 2, 4));
    let rectMatrix2 = new Matrix([
        1, 2,
        3, 4,
        5, 1,
        2, 3
    ], 4, 2);
    testMatrix(new DiagonalMatrixView(rectMatrix2), new Matrix([
        1, 0,
        0, 4,
        0, 0,
        0, 0
    ], 4, 2));
});