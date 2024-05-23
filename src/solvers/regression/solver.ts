import { LeastSquaresFunction, LeastSquaresProblem } from "./problem";

export abstract class LeastSquaresSolver {
    abstract solve(problem:LeastSquaresFunction, samples:LeastSquaresProblem, initialParams?: number[]): number[] 
}