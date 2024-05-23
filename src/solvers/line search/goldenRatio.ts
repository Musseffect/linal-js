import Matrix from "../../dense/denseMatrix";
import { SmallTolerance } from "../../utils";
import Vector from "../../dense/vector";
import { LineSearch, LineSearchProblem } from "./lineSearch";

/**
 * 1-dimensional continuous optimization
 */
export abstract class Problem1D {
    min: number;
    max: number;
    initial: number;
    getBounds(): { min: number, max: number } {
        return { min: this.min, max: this.max };
    }
    getInitial(): number {
        return this.initial;
    }
    abstract f(x: number): number;
    abstract dfdx(x: number): number;
    abstract dfdxdx(x: number): number;
}

export interface Solver1D {
    solve(problem: Problem1D): number;
}

class SearchAlongDirectionProblem extends Problem1D {
    x0: Vector;
    direction: Vector;
    problem: LineSearchProblem;
    maxStep: number;
    constructor(problem: LineSearchProblem, x0: Vector, direction: Vector, maxStep: number) {
        super();
        this.problem = problem;
        this.x0 = x0;
        this.direction = direction;
        this.maxStep = maxStep;
    }
    private arg(x: number): Vector {
        return Vector.scale(this.direction, x).addSelf(this.x0);
    }
    override f(x: number): number {
        return this.problem.f(this.arg(x));
    }
    dfdx(x: number): number {
        // f(x0 + x * dir)
        // df/dx = df/darg * dir
        return Vector.dot(this.problem.grad(this.arg(x)), this.direction);
    }
    // f(x0 + x * dir)
    // d^2f/dx^2 = (d^2f/darg * dir) * dir
    dfdxdx(x: number): number {
        return Vector.dot(Matrix.postMulVec(this.problem.hessian(this.arg(x)), this.direction), this.direction);
    }
}

export class GoldenSearch1D implements Solver1D {
    threshold: number;
    iterations: number;
    constructor(threshold: number, iterations: number) {
        this.threshold = threshold;
        this.iterations = iterations;
    }
    solve(problem: Problem1D): number {
        let { min, max } = problem.getBounds();
        let a = min, b = max;
        let t = (1.0 + Math.sqrt(5.0)) / 2.0;

        let x2: number = a + (b - a) / t;
        let x1: number = a + b - x2;
        let fx1: number = problem.f(x2);
        let fx2: number = problem.f(x1);
        let i = 0;
        while (i < this.iterations) {
            if (fx1 < fx2) {
                b = x2;
                x2 = x1;
                fx2 = fx1;
                x1 = a + b - x2;
                fx1 = problem.f(x1);
            }
            else {
                a = x1;
                x1 = x2;
                fx1 = fx2;
                x2 = a + (b - a) / t;
                fx2 = problem.f(x2);
            }
            if (Math.abs(a - b) < this.threshold) {
                break;
            }
            i++;
        }
        let x = (a + b) * 0.5;
        return x;
    }
}

export class GoldenLineSearch extends LineSearch {
    numIters: number = 100;
    tolerance: number = SmallTolerance;
    stMaxIters(numIters: number) {
        this.numIters = numIters;
    }
    setTolerance(tolerance: number) {
        this.tolerance = tolerance;
    }
    public override step(x: Vector, direction: Vector, initialStep: number = 1.0): number {
        let problem = new SearchAlongDirectionProblem(this.problem, x, direction, initialStep);
        let search = new GoldenSearch1D(this.tolerance, this.numIters);
        return search.solve(problem);
    }
}