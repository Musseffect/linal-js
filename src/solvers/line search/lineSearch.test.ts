import Matrix from "../../dense/denseMatrix";
import Vector from "../../dense/vector";
import { LineSearchProblem } from "./lineSearch";
import { ArmijoBacktracking } from "./armijo";
import WolfeLineSearch from "./wolfe";
import GoldsteinLineSearch from "./goldstein";

class SecondOrderCurve implements LineSearchProblem {
    private a: number;
    private b: number;
    private c: number;
    constructor(a: number, b: number, c: number) {
        this.a = a;
        this.b = b;
        this.c = c;
    }
    public f(x: Vector): number {
        return this.c + x.get(0) * (this.b + this.a * x.get(0));
    }
    public grad(x: Vector): Vector {
        return new Vector([this.b + 2 * this.a * x.get(0)]);
    }
    public hessian(x: Vector): Matrix {
        return new Matrix([2 * this.a], 1, 1);
    }
};


describe('Line search tests', () => {
    interface TestData {
        function: LineSearchProblem;
        x: Vector;
        dir: Vector;
        initialStep: number;
    }
    const convexCurve = new SecondOrderCurve(1, 0, 0);
    let multipleLocalMinimums: LineSearchProblem =
    {
        f: (x: Vector): number => {
            return x.get(0) * x.get(0) + 0.3 * Math.cos(Math.PI * 2 * x.get(0) + 1);
        },
        grad: (x: Vector): Vector => {
            return new Vector([2.0 * x.get(0) + 0.3 * 2 * Math.PI * Math.sin(Math.PI * 2 * x.get(0) + 1)]);
        },
        hessian: (x: Vector) => {
            return new Matrix([2.0 - 0.3 * 4 * Math.PI * Math.PI * Math.cos(Math.PI * 2 * x.get(0) + 1)], 1, 1);
        }
    };
    const problems: TestData[] = [
        { function: convexCurve, x: new Vector([-1]), dir: new Vector([1]), initialStep: 3 },
        { function: multipleLocalMinimums, x: new Vector([-1]), dir: new Vector([1]), initialStep: 3 }
    ];
    test.each(problems)('Armijo', (testData: TestData) => {
        let solver = new ArmijoBacktracking(testData.function);
        solver.tau = 0.75;
        let step = solver.step(testData.x, testData.dir, testData.initialStep);
        let nextArg = Vector.add(testData.x, Vector.scale(testData.dir, step));
        let curValue = testData.function.f(testData.x);
        let curGrad = testData.function.grad(testData.x);
        let nextValue = testData.function.f(nextArg);
        expect(nextValue).toBeLessThanOrEqual(curValue + solver.gradCoeff * step * Vector.dot(testData.dir, curGrad));
    });
    test.each(problems)('Wolfe', (testData: TestData) => {
        let solver = new WolfeLineSearch(testData.function);
        solver.tau = 0.75;
        let step = solver.step(testData.x, testData.dir, testData.initialStep);
        let nextArg = Vector.add(testData.x, Vector.scale(testData.dir, step));
        let curValue = testData.function.f(testData.x);
        let curGrad = testData.function.grad(testData.x);
        let nextValue = testData.function.f(nextArg);
        let nextGrad = testData.function.grad(nextArg);
        const dotDirGrad = Vector.dot(testData.dir, curGrad);
        expect(nextValue).toBeLessThanOrEqual(curValue + solver.gradCoeff * step * dotDirGrad);
        expect(-Vector.dot(testData.dir, nextGrad)).toBeLessThanOrEqual(-solver.curvatureCoeff * dotDirGrad);
    });
    test.each(problems)('Goldstein', (testData: TestData) => {
        let solver = new GoldsteinLineSearch(testData.function);
        solver.tau = 0.75;
        let step = solver.step(testData.x, testData.dir, testData.initialStep);
        let nextArg = Vector.add(testData.x, Vector.scale(testData.dir, step));
        let curValue = testData.function.f(testData.x);
        let curGrad = testData.function.grad(testData.x);
        let nextValue = testData.function.f(nextArg);
        const dotDirGrad = Vector.dot(testData.dir, curGrad);
        expect(curValue + (1 - solver.gradCoeff) * step * dotDirGrad).toBeLessThanOrEqual(nextValue);
        expect(nextValue).toBeLessThanOrEqual(curValue + solver.gradCoeff * step * dotDirGrad);
    });
});