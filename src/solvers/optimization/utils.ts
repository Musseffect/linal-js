import Matrix from "../../dense/denseMatrix";
import { assertFail } from "../../utils";
import Vector from "../../dense/vector";
import { ArmijoBacktracking } from "../line search/armijo";
import GoldsteinLineSearch from "../line search/goldstein";
import { LineSearch, LineSearchProblem } from "../line search/lineSearch";
import WolfeLineSearch from "../line search/wolfe";
import { OptimizationProblem } from "./optimizationProblem";


export enum LineSearchAlgorithm {
    Wolf = 0,
    Armijo = 1,
    Goldstein = 2
};

class OptimizationLineSearchProblemAdapter implements LineSearchProblem {
    private problem: OptimizationProblem;
    constructor(problem: OptimizationProblem) {
        this.problem = problem;
    }
    public f(x: Vector): number {
        return this.problem.f(x);
    }
    public grad(x: Vector): Vector {
        return this.problem.dfdx(x);
    }
    public hessian(x: Vector): Matrix {
        return this.problem.dfdxdy(x);
    }

}

export function initializeLineSearch(algorithm: LineSearchAlgorithm, problem: OptimizationProblem): LineSearch {
    let adapter = new OptimizationLineSearchProblemAdapter(problem);
    switch (algorithm) {
        case LineSearchAlgorithm.Wolf:
            return new WolfeLineSearch(adapter);
        case LineSearchAlgorithm.Armijo:
            return new ArmijoBacktracking(adapter);
        case LineSearchAlgorithm.Goldstein:
            return new GoldsteinLineSearch(adapter);
        default:
            assertFail("Unexpected variant");
    }
}

export function createLineSearch(algorithm: LineSearchAlgorithm, problem: LineSearchProblem): LineSearch {
    switch (algorithm) {
        case LineSearchAlgorithm.Wolf:
            return new WolfeLineSearch(problem);
        case LineSearchAlgorithm.Armijo:
            return new ArmijoBacktracking(problem);
        case LineSearchAlgorithm.Goldstein:
            return new GoldsteinLineSearch(problem);
        default:
            assertFail("Unexpected variant");
    }
}