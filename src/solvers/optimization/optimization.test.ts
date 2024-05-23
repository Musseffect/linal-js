

import { SmallTolerance, Tolerance, assert } from "../../utils";
import Vector from "../../dense/vector";
import { TruncatedLLT } from "./newton";
import {
    OptimizationTestFunction,
    AckleyFunc,
    Beale,
    RastriginFunc,
    RosenbrockFunc,
    SphereFunc,
    GoldsteinPriceFunc,
    BoothFunc,
    BukinFunc6,
    MatyasFunc,
    LeviFunc13,
    HimmelblauFunc,
    ThreeHumpCamelFunc,
    EasomFunc,
    CrossInTrayFunc,
    EggholderFunc,
    HolderTableFunc,
    McCormickFunc,
    SchafferFunc2,
    SchafferFunc4,
    StyblinskiTangFunc
} from "./testFunctions";
import Matrix from "../../dense/denseMatrix";
import { SymmetricEigendecomposition } from "../linear systems/eigenvalues";
import LLT from "../linear systems/llt";
import { PermutationType } from "../../permutationMatrix";

interface TestFunction1D {
    name: String;
    f: (x: number) => number;
    min: number,
    max: number,
    solutions: { x: number, y: number }[]
}

const TestFunctions1D: TestFunction1D[] = [
    {
        name: "Convex unimodal",
        f: (x: number) => {
            return 5 + Math.pow(x, 2);
        },
        min: -5,
        max: 5,
        solutions: [{
            x: 0,
            y: 5
        }]
    },
    {
        name: "Non-convex unimodal",
        f: (x: number) => {
            return -(x + Math.sin(x)) * Math.exp(-Math.pow(x, 2));
        },
        min: -10,
        max: 10,
        solutions: [{
            x: 0.67957866002,
            y: -0.824239398476
        }]
    },
    {
        name: "Multimodal 1",
        f: (x: number) => {
            return Math.sin(x) + Math.sin(10 / 3 * x);
        },
        min: -2.7,
        max: 7.5,
        solutions: [{ x: 5.14573529026, y: -1.89959934915 }]
    },
    {
        name: "Multimodal 2",
        f: (x: number) => {
            return -(1.4 - 3 * x) * Math.sin(18 * x);
        },
        min: 0,
        max: 1.2,
        solutions: [{ x: 0.966085803827, y: -1.48907253869 }]
    },
    {
        name: "Multimodal 3",
        f: (x: number) => {
            return -x * Math.sin(x);
        },
        min: 0,
        max: 10,
        solutions: [{ x: 7.97866571241, y: -7.91672737159 }]
    }
];

const TestFunctionsND: OptimizationTestFunction[] = [
    RastriginFunc,
    AckleyFunc,
    SphereFunc,
    RosenbrockFunc,
    Beale,
    GoldsteinPriceFunc,
    BoothFunc,
    BukinFunc6,
    MatyasFunc,
    LeviFunc13,
    HimmelblauFunc,
    ThreeHumpCamelFunc,
    EasomFunc,
    CrossInTrayFunc,
    EggholderFunc,
    HolderTableFunc,
    McCormickFunc,
    SchafferFunc2,
    SchafferFunc4,
    StyblinskiTangFunc
];

describe('Positive definite projection (regularization)', () => {
    const LogProjection = false;
    let posDefMatrix = new Matrix([
        10, -1, 2, 0,
        -1, 11, -1, 3,
        2, -1, 10, -1,
        0, 3, -1, 8
    ], 4, 4);
    let negDefMatrix = new Matrix([
        1, 2, 3,
        2, 5, 4,
        3, 4, 9
    ], 3, 3);
    let llt = new LLT();
    llt.factorize(posDefMatrix);
    expect(Matrix.lInfDistance(Matrix.mul(llt.L.toMatrix(), llt.LT.toMatrix()), posDefMatrix)).toBeCloseTo(0);
    llt.factorize(negDefMatrix);
    expect(llt.llt).toBeNull();

    test("Truncated LLT", () => {
        let solver = new TruncatedLLT();
        solver.factorize(posDefMatrix);
        expect(Matrix.lInfDistance(solver.PLLTPT, posDefMatrix)).toBeCloseTo(0.0);
        expect(Vector.lInfDistance(solver.solve(new Vector([6, 25, -11, 15])), new Vector([1, 2, -1, 1]))).toBeCloseTo(0.0);

        solver.factorize(negDefMatrix);
        let projection = solver.PLLTPT;
        if (LogProjection)
            console.log(projection.toString());
        expect(Matrix.lInfDistance(projection, negDefMatrix)).not.toBeCloseTo(0);
        let eigenvalues = (new SymmetricEigendecomposition(projection)).D;
        expect(eigenvalues).not.toBeNull();
        expect(eigenvalues.size()).toBe(3);
        for (let i = 0; i < eigenvalues.size(); ++i)
            expect(eigenvalues.get(i)).toBeGreaterThanOrEqual(0);

        llt.factorize(projection);
        expect(llt.llt != null).not.toBeNull();
        if (llt.llt != null)
            expect(Matrix.lInfDistance(Matrix.mul(llt.L.toMatrix(), llt.LT.toMatrix()), projection)).toBeCloseTo(0);
    });

    test("Eigendecomposition", () => {
        let eigendecomposition = new SymmetricEigendecomposition();
        eigendecomposition.factorize(negDefMatrix, 20);

        let clampedDiag = eigendecomposition.D.clone();
        for (let i = 0; i < clampedDiag.size(); ++i) {
            if (clampedDiag.get(i) < Tolerance) {
                clampedDiag.set(i, Tolerance);
            }
        }
        let projection = Matrix.mul(Matrix.mul(eigendecomposition.Q, Matrix.fromDiagonal(clampedDiag)), eigendecomposition.Q.transpose());
        if (LogProjection)
            console.log(projection.toString());
        llt.factorize(projection);
        expect(llt.llt).not.toBeNull();
        if (llt.llt != null)
            expect(Matrix.lInfDistance(Matrix.mul(llt.L.toMatrix(), llt.LT.toMatrix()), projection)).toBeCloseTo(0);
    });
    test("Tikhonov", () => {
        let lambda = SmallTolerance;
        let solver = new LLT();
        for (let i = 0; i < 20; ++i) {
            let A = Matrix.add(negDefMatrix, Matrix.fromDiagonal(Vector.generate(negDefMatrix.numRows(), () => lambda)));
            solver.factorize(A);
            if (solver.llt !== null) break;
            lambda *= 2;
        }
        expect(solver.llt).not.toBeNull();
        if (solver.llt != null) {
            if (LogProjection)
                console.log((solver.A as Matrix).toString());
            expect(Matrix.lInfDistance(Matrix.mul(solver.L.toMatrix(), solver.LT.toMatrix()), solver.A as Matrix)).toBeCloseTo(0);
        }
    });
});

describe('Validate optimization test cases', () => {
    test.each(TestFunctionsND)('Test function ND $name', (testFunction: OptimizationTestFunction) => {
        if (typeof testFunction.min.p == "number") {
            expect(testFunction.numDimensions).toBe(-1);
            for (let numDimensions = 2; numDimensions <= 4; ++numDimensions) {
                let p = new Vector(new Array(numDimensions).fill(testFunction.min.p));
                let value = testFunction.f(p);
                expect(value).toBeCloseTo(testFunction.min.value);
            }
            expect(typeof testFunction.searchDomain.min == "number").toBeTruthy()
            expect(typeof testFunction.searchDomain.max == "number").toBeTruthy()
            expect(testFunction.min.p).toBeGreaterThanOrEqual(testFunction.searchDomain.min as number);
            expect(testFunction.min.p).toBeLessThanOrEqual(testFunction.searchDomain.max as number);
        } else {
            expect(testFunction.searchDomain.min instanceof Vector).toBeTruthy();
            expect(testFunction.searchDomain.max instanceof Vector).toBeTruthy();
            const min = testFunction.searchDomain.min as Vector;
            const max = testFunction.searchDomain.max as Vector;
            expect(min.size() == testFunction.numDimensions);
            expect(max.size() == testFunction.numDimensions);
            if (testFunction.min.p instanceof Vector) {
                expect(testFunction.min.p.size()).toBe(testFunction.numDimensions);
                expect(testFunction.f(testFunction.min.p)).toBeCloseTo(testFunction.min.value);
                for (let dimension = 0; dimension < testFunction.numDimensions; ++dimension) {
                    expect(testFunction.min.p.get(dimension)).toBeLessThanOrEqual(max.get(dimension));
                    expect(testFunction.min.p.get(dimension)).toBeGreaterThan(min.get(dimension));
                }
            } else if (Array.isArray(testFunction.min.p)) {
                for (let p of testFunction.min.p) {
                    expect(p.size()).toBe(testFunction.numDimensions);
                    expect(testFunction.f(p)).toBeCloseTo(testFunction.min.value);
                    for (let dimension = 0; dimension < testFunction.numDimensions; ++dimension) {
                        expect(p.get(dimension)).toBeLessThanOrEqual(max.get(dimension));
                        expect(p.get(dimension)).toBeGreaterThan(min.get(dimension));
                    }
                }
            }
        }
    });
});