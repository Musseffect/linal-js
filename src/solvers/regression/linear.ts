import Matrix from "../../dense/denseMatrix";
import { assert } from "../../utils";
import Vector from "../../dense/vector";
import PartialPivLU from "../linear systems/partialPivLU";


export default class LinearLeastSquaresSolver {
    run(x: Vector[], y: number[]): number[] {
        assert(x.length == y.length, "Incorrect sizes");
        assert(x.length > 0, "Zero size");
        let numSamples = x.length;
        let numVariables = x[0].size();
        let numParams = numVariables + 1;
        let A = Matrix.empty(numParams, numParams);
        let b = Vector.empty(numParams);
        for (let i = 0; i < numVariables; ++i) {
            let bValue = 0.0;
            let lastCellValue = 0.0;
            for (let k = 0; k < numSamples; ++k) {
                let xVec = x[k];
                bValue += xVec.get(i) * y[k];
                lastCellValue += xVec.get(i);
            }
            b.set(i, bValue);
            A.set(i, numParams - 1, lastCellValue);
            A.set(numParams - 1, i, lastCellValue);
            for (let j = i; j < numVariables; ++i) {
                let cellValue = 0.0;
                for (let k = 0; k < numSamples; ++k) {
                    let xVec = x[k];
                    cellValue = xVec.get(i) * xVec.get(j);
                }
                A.set(i, j, cellValue);
                if (i != j)
                    A.set(j, i, cellValue);
            }
        }
        A.set(numParams - 1, numParams - 1, numSamples);
        let bValue = 0.0;
        for (let k = 0; k < numSamples; ++k) {
            bValue += y[k];
        }
        b.set(numParams - 1, bValue);
        let params = PartialPivLU.solve(A, b);
        return params.data;
    }
}