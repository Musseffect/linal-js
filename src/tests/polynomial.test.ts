
import { complex } from "../complex";
import Matrix from "../dense/denseMatrix";
import { Polynomial, PolynomialSolver, generatePolynomial, generatePolynomialWithComplexRoots } from "../polynomial";
import { calcEigenvalues } from "../solvers/linear systems/eigenvalues";
import { SmallTolerance, SmallestTolerance, Tolerance, assert, near } from "../utils";


interface TestData {
    roots: number[],
    polynomial: Polynomial;

};

let polynomialRootsTests: TestData[] = [];
(function () {
    let rootsList = [
        [1, 2],
        [-1, -1],
        [0, 0, 1],
        [100, 1, 23],
        [10, 10, 3],
        [1, 4, 5, 0],
        [4, -10, 2, 23],
        [-3.125, 23, 2.1, 3, -2.2],
        [7, 12, -14, 24, 0, 7]
    ];
    for (const roots of rootsList)
        polynomialRootsTests.push({ roots, polynomial: generatePolynomial(roots) });
})();

function logPolynomial(polynomial: Polynomial) {
    console.log(`Polynomial: ${polynomial.toString()}`);
    let der = polynomial;
    for (let i = 0; i < polynomial.degree() - 1; ++i) {
        der = der.derivative();
        console.log(`Der ${i}: ${der.toString()}`);
    }
}

function compareRoots(actual: number[], expected: number[], tolerance: number = Tolerance) {
    actual.sort((a, b) => { return a - b; });
    expected.sort((a, b) => { return a - b; });
    if (actual.length != expected.length)
        console.log(`Expected: ${expected.toString()}, actual: ${actual.toString()}`);
    expect(actual.length).toBe(expected.length);
    for (let i = 0; i < actual.length; ++i)
        expect(Math.abs(actual[i] - expected[i])).toBeLessThanOrEqual(tolerance);
}

test("Polynomial operations", () => {
    expect(Polynomial.add(new Polynomial([1, 2, 3, -1]), new Polynomial([2, 4, 3, 1, 1])).coeffs).toEqual(expect.arrayContaining([3, 6, 6, 0, 1]));

    expect(Polynomial.sub(new Polynomial([1, 2, 3, -1, 0]), new Polynomial([2, 4, 3, 1, 2])).coeffs).toEqual(expect.arrayContaining([-1, -2, 0, 0, -2]));

    expect(Polynomial.mul(new Polynomial([0.2, 2, 3]), new Polynomial([-2, -1, 0.1, 3])).coeffs).toEqual(expect.arrayContaining([-0.4, -4.2, -7.98, -2.2, 6.3, 9]));
    const testDiv = (a: Polynomial, b: Polynomial, expectedQ: number[], expectedR: number[]) => {
        let div = Polynomial.div(a, b);
        expect(Polynomial.add(Polynomial.mul(b, div.Q), div.R).coeffs).toEqual(expect.arrayContaining(a.coeffs));
        expect(div.Q.coeffs).toEqual(expect.arrayContaining(expectedQ));
        expect(div.R.coeffs).toEqual(expect.arrayContaining(expectedR));
    };
    testDiv(new Polynomial([-4, 0, -2, 1]), new Polynomial([-3, 1]), [3, 1, 1], [5]);
    testDiv(new Polynomial([1, 2]), new Polynomial([2]), [1, 0.5], [0]);
    testDiv(new Polynomial([1]), new Polynomial([2]), [0.5], [0]);
    testDiv(new Polynomial([1]), new Polynomial([2, 3]), [0], [1]);

    // derivative
    expect(new Polynomial([4, 3, 2, 1]).derivative().coeffs).toEqual(expect.arrayContaining([3, 4, 3]));

    // integration
    expect(new Polynomial([3, 4, 3]).integral().coeffs).toEqual(expect.arrayContaining([3, 2, 1]));
    expect(new Polynomial([0.2, 2, 3]).definiteIntegral(0, 1)).toBeCloseTo(2.2);
});

describe(`Polynomial generation`, () => {
    test(`Real polynomial with real roots`, () => {
        // test generator of polynomials
        expect(generatePolynomial([1, 2]).coeffs).toEqual(expect.arrayContaining([2, -3, 1]));
        expect(generatePolynomial([1, 3, 5]).coeffs).toEqual(expect.arrayContaining([15, -23, 9, -1]));
        expect(generatePolynomial([1, 4, 5, 0]).coeffs).toEqual(expect.arrayContaining([0, -20, 29, -10, 1]));
    });

    test(`Real polynomial with complex roots`, () => {
        // test generator of polynomials with complex roots
        expect(() => generatePolynomialWithComplexRoots([new complex(2, 1), new complex(2, -1)])).not.toThrow();
        expect(generatePolynomialWithComplexRoots([new complex(2, 1), new complex(2, -1)]).coeffs).toEqual(expect.arrayContaining([5, -4, 1]));
        expect(generatePolynomialWithComplexRoots([new complex(0, 3), new complex(0, -3), new complex(5, 0)]).coeffs).toEqual(expect.arrayContaining([45, -9, 5, -1]));
        expect(() => generatePolynomialWithComplexRoots([new complex(2, 1), new complex(2, 1)])).toThrow();
    });
    // todo (NI): complex coefficient polynomials
})

describe('Polynomial root solvers', () => {
    describe("Special solvers", () => {
        test("Quadratic", () => {

            const testRoots = (polynomial: Polynomial, expectedRoots: number[]) => {
                compareRoots(PolynomialSolver.solveQuadratic(polynomial.coeffs[2], polynomial.coeffs[1], polynomial.coeffs[0]), expectedRoots);
            };
            testRoots(new Polynomial([-2, -1, 1]), [-1, 2]);
            testRoots(new Polynomial([1, -2, 1]), [1, 1]);
            testRoots(new Polynomial([1, 1, 1]), []);
        });
        test("Cubic", () => {
            const testRoots = (polynomial: Polynomial, expectedRoots: number[]) => {
                let solver = new PolynomialSolver(50, SmallestTolerance, 0.5);
                compareRoots(solver.solveCubic(polynomial.coeffs[3], polynomial.coeffs[2], polynomial.coeffs[1], polynomial.coeffs[0]), expectedRoots);
            };
            // three distinct roots
            testRoots(new Polynomial([6, 1, -4, 1]), [-1, 2, 3]);
            // three roots: two roots are the same
            testRoots(new Polynomial([-2, 5, -4, 1]), [1, 1, 2]);
            // one root
            testRoots(new Polynomial([-1, 3, -3, 1]), [1, 1, 1]);
        });
        test("Linear", () => {
            compareRoots(PolynomialSolver.solveLinear(3, -6), [2])
        });
    });
    describe.each(polynomialRootsTests)(`With roots %#`, (testData: TestData) => {
        let sortedRoots = testData.roots.slice().sort((a, b) => a - b);
        test("PolynomialSolver", () => {
            let solver = new PolynomialSolver(50, SmallestTolerance, 0.5);
            let solution = solver.solveInRegion(testData.polynomial, false);
            compareRoots(solution, sortedRoots, SmallTolerance);
        });
        test("PolynomialSolverCubic", () => {
            let solver = new PolynomialSolver(50, SmallestTolerance, 0.5);
            let solution = solver.solveInRegion(testData.polynomial, true);
            compareRoots(solution, sortedRoots, SmallTolerance);
        });
        test("EigenvalueSolver", () => {
            // generate matrix and solve eigenvalues
            let solution = calcEigenvalues(generatePolynomial(testData.roots).companionMatrix(), 20, SmallestTolerance);
            expect(solution.length).toEqual(sortedRoots.length);
            solution.sort((a, b) => { return a - b; });
            for (let i = 0; i < sortedRoots.length; ++i)
                expect(Math.abs(sortedRoots[i] - solution[i])).toBeLessThanOrEqual(Tolerance);
        })
    });
    const rootlessPolynomials: Polynomial[] = [
        new Polynomial([1]),
        new Polynomial([1, 0]),
        new Polynomial([-0.5, 1.5, -3.8]),
        new Polynomial([0.8, 0.4, -3.7, -1.5, 6.5]),
        new Polynomial([-1, 0, 4, 0, -4, 0, -1]),
        new Polynomial([0.25, 0, -1.5, 0, 2, 0, 1, 0, 10])
    ];
    test.each(rootlessPolynomials)('Without roots %#', (polynomial: Polynomial) => {
        let solver = new PolynomialSolver(50, SmallestTolerance, 0.5);
        let solution = solver.solveInRegion(polynomial);
        expect(solution.length).toBe(0);
    })
    // todo (NI): RPoly
});