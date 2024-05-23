import { SparseMatrixCSR, SparseMatrixRowIterator } from "../../../sparse/sparseMatrix";
import { assert, SmallTolerance } from "../../../utils";
import Vector from "../../../dense/vector";
import { ConvergenseFailureException } from "../exceptions";
import IncompleteLL from "./incompleteLL";

export enum CGPreconditioner {
    None,
    Diagonal,
    IncompleteLL
};

interface Preconditioner {
    calc(r: Vector): Vector;
}

class IdentityPreconditioner implements Preconditioner {
    // will return reference to residual vector
    calc(r: Vector): Vector {
        return r;
    }
}

class DiagonalPreconditioner implements Preconditioner {
    diagonal: Vector;
    constructor(A: SparseMatrixCSR) {
        this.diagonal = Vector.empty(A.height());
        for (let row = 0; row < A.height(); ++row) {
            let it = new SparseMatrixRowIterator(A, row);
            while (!it.isDone()) {
                let { value, colIdx } = it.advance();
                if (colIdx > row) break;
                if (colIdx == row)
                    this.diagonal.set(row, value);
            }
        }
    };
    calc(r: Vector): Vector {
        let result = r.clone();
        for (let i = 0; i < r.size(); ++i)
            result.set(i, result.get(i) / this.diagonal.get(i));
        return result;
    }
}

class IncompleteLLPReconditioner implements Preconditioner {
    factorization: IncompleteLL;
    constructor(A: SparseMatrixCSR) {
        this.factorization = new IncompleteLL(A);
    }
    calc(r: Vector): Vector {
        return this.factorization.solve(r);
    }
}

export class ConjugateGradients {
    A: SparseMatrixCSR;
    preconditioner: Preconditioner
    constructor(A: SparseMatrixCSR, preconditioner: CGPreconditioner) {
        assert(A.isSquare(), "Non-square matrix");
        switch (preconditioner) {
            case CGPreconditioner.None:
                this.preconditioner = new IdentityPreconditioner();
                break;
            case CGPreconditioner.Diagonal:
                this.preconditioner = new DiagonalPreconditioner(A);
                break;
            case CGPreconditioner.IncompleteLL:
                this.preconditioner = new IncompleteLLPReconditioner(A);
                break;
            default:
                throw Error("Unkown preconditioner");
        }
        this.A = A;
    }
    compute(b: Vector, maxIterations: number = 20, tolerance: number = SmallTolerance): Vector {
        assert(this.A.width() == b.data.length, "Width of matrix isn't compatible with vector's length");
        let x = Vector.empty(b.size());
        let r = Vector.sub(b, SparseMatrixCSR.postMul(this.A, x))
        if (r.lInfNorm() < tolerance) return x;
        let z = this.preconditioner.calc(r);
        let p = z.clone();
        let iter = 0;
        while (iter < maxIterations) {
            let ap = SparseMatrixCSR.postMul(this.A, p);
            let rDotZ = Vector.dot(r, z);
            let alpha = rDotZ / Vector.dot(p, ap);
            x.addSelf(Vector.scale(p, alpha));
            r.subSelf(Vector.scale(ap, alpha));
            if (r.lInfNorm() < tolerance) return x;
            z = this.preconditioner.calc(r);
            let rDotZNew = Vector.dot(r, z);
            let beta = rDotZNew / rDotZ;
            p = p.scaleSelf(beta).addSelf(z);
            rDotZ = rDotZNew;
            ++iter;
        }
        throw new ConvergenseFailureException("Conjugate gradients");
    }
    static solve(A: SparseMatrixCSR, b: Vector, maxIterations: number = 20, tolerance: number = SmallTolerance): Vector {
        let solver = new ConjugateGradients(A, CGPreconditioner.Diagonal);
        return solver.compute(b, maxIterations, tolerance);
    }
}