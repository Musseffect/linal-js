import Matrix from "../../dense/denseMatrix";
import { forwardDifference, secondOrderDifference } from "../../numericalDifferentiation";
import { assert, SmallTolerance } from "../../utils";
import Vector from "../../dense/vector";

// https://en.wikipedia.org/wiki/Test_functions_for_optimization
export interface OptimizationTestFunction {
    name: String;
    numDimensions: number;
    f(x: Vector): number;
    min: { p: number | Vector | Vector[], value: number };
    searchDomain: { min: number | Vector, max: number | Vector };
}

// global minimum f(0,..., 0) = 0
export const RastriginFunc: OptimizationTestFunction = {
    name: "Rastrigin",
    f: function (p: Vector): number {
        let result = 10 * p.size();
        for (let i = 0; i < p.size(); ++i) {
            let x = p.get(i);
            result += (x * x - 10 * Math.cos(2.0 * Math.PI * x));
        }
        return result;
    },
    numDimensions: -1,
    min: {
        p: 0,
        value: 0
    },
    searchDomain: {
        min: -5.12,
        max: 5.12
    }
};

function ackley2DFunc(x: number, y: number) {
    return -20 * Math.exp(-0.2 * Math.sqrt(0.5 * (x * x + y * y))) - Math.exp(0.5 * (Math.cos(2 * Math.PI * x) + Math.cos(2 * Math.PI * y))) + Math.E + 20;
}

export const AckleyFunc: OptimizationTestFunction = {
    name: "Ackley",
    f: function (p: Vector): number {
        assert(p.size() == 2, "Invalid size");
        return ackley2DFunc(p.get(0), p.get(1));
    },
    numDimensions: 2,
    min: {
        p: new Vector([0, 0]),
        value: 0
    },
    searchDomain: {
        min: new Vector([-5, -5]),
        max: new Vector([5, 5])
    }
};

// global and one local minimum f(0,..., 0) = 0
export const SphereFunc: OptimizationTestFunction = {
    name: "Sphere",
    f: function (p: Vector) {
        return p.squaredLength();
    },
    numDimensions: -1,
    min: {
        p: 0,
        value: 0
    },
    searchDomain: {
        min: Number.NEGATIVE_INFINITY,
        max: Number.POSITIVE_INFINITY
    }
};

// global minimum f(1,..., 1) = 0
export const RosenbrockFunc: OptimizationTestFunction = {
    name: "Rosenbrock",
    f: function (p: Vector) {
        assert(p.size() > 1, "Invalid size");
        let result = 0.0;
        for (let i = 1; i < p.size() - 1; ++i) {
            let x = p.get(i);
            let xNext = p.get(i + 1);
            result += 100 * Math.pow(xNext - x * x, 2) + Math.pow(1 - x, 2);
        }
        return result;
    },
    numDimensions: -1,
    min: {
        p: 1,
        value: 0
    },
    searchDomain: {
        min: Number.NEGATIVE_INFINITY,
        max: Number.POSITIVE_INFINITY
    }
};

function beale2DFunc(x: number, y: number) {
    return Math.pow(1.5 - x + x * y, 2) + Math.pow(2.25 - x + x * y * y, 2) + Math.pow(2.625 - x + x * y * y * y, 2)
}
export const Beale: OptimizationTestFunction = {
    name: "Beale",
    f: function (p: Vector) {
        assert(p.size() == 2, "Invalid size");
        return beale2DFunc(p.get(0), p.get(1));
    },
    numDimensions: 2,
    min: {
        p: new Vector([3, 0.5]),
        value: 0
    },
    searchDomain: {
        min: new Vector([-4.5, -4.5]),
        max: new Vector([4.5, 4.5])
    }
};

function GoldsteinPrice2DFunc(x: number, y: number) {
    return (1 + Math.pow(x + y + 1, 2) * (19 + x * (3 * x - 14 + 6 * y) + y * (3 * y - 14))) *
        (30 + Math.pow(2 * x - 3 * y, 2) * (18 + x * (12 * x - 32 - 36 * y) + y * (48 + 27 * y)));
}

export const GoldsteinPriceFunc: OptimizationTestFunction = {
    name: "Goldstein-Price",
    f: function (p: Vector) {
        assert(p.size() == 2, "Invalid size");
        return GoldsteinPrice2DFunc(p.get(0), p.get(1));
    },
    numDimensions: 2,
    min: {
        p: new Vector([0, -1]),
        value: 3
    },
    searchDomain: {
        min: new Vector([-2, -2]),
        max: new Vector([2, 2])
    }
};

function BoothFunc2D(x: number, y: number) {
    return Math.pow(x + 2 * y - 7, 2) + Math.pow(2 * x + y - 5, 2)
}

export const BoothFunc: OptimizationTestFunction = {
    name: "Booth",
    f: function (p: Vector) {
        assert(p.size() == 2, "Invalid size");
        return BoothFunc2D(p.get(0), p.get(1));
    },
    numDimensions: 2,
    min: {
        p: new Vector([1, 3]),
        value: 0
    },
    searchDomain: {
        min: new Vector([-10, -10]),
        max: new Vector([10, 10])
    }
};

function BukinFunc6_2D(x: number, y: number) {
    return 100 * Math.sqrt(Math.abs(y - 0.01 * x * x)) + 0.01 * Math.abs(x + 10);
}

export const BukinFunc6: OptimizationTestFunction = {
    name: "Bukin 6",
    f: function (p: Vector) {
        assert(p.size() == 2, "Invalid size");
        return BukinFunc6_2D(p.get(0), p.get(1));
    },
    numDimensions: 2,
    min: {
        p: new Vector([-10, 1]),
        value: 0
    },
    searchDomain: {
        min: new Vector([-15, -3]),
        max: new Vector([5, 3])
    }
};

function MatyasFunc2D(x: number, y: number) {
    return 0.26 * (x * x + y * y) - 0.48 * x * y;
}

export const MatyasFunc: OptimizationTestFunction = {
    name: "Matyas",
    f: function (p: Vector) {
        assert(p.size() == 2, "Invalid size");
        return MatyasFunc2D(p.get(0), p.get(1));
    },
    numDimensions: 2,
    min: {
        p: new Vector([0, 0]),
        value: 0
    },
    searchDomain: {
        min: new Vector([-10, -10]),
        max: new Vector([10, 10])
    }
};

function Levi13Func2D(x: number, y: number) {
    return Math.pow(Math.sin(3 * Math.PI * x), 2) + Math.pow(x - 1, 2) * (1 + Math.pow(Math.sin(3 * Math.PI * y), 2)) +
        Math.pow(y - 1, 2) * (1 + Math.pow(Math.sin(2 * Math.PI * y), 2));
}

export const LeviFunc13: OptimizationTestFunction = {
    name: "Levi 13",
    f: function (p: Vector) {
        assert(p.size() == 2, "Invalid size");
        return Levi13Func2D(p.get(0), p.get(1));
    },
    numDimensions: 2,
    min: {
        p: new Vector([1, 1]),
        value: 0
    },
    searchDomain: {
        min: new Vector([-10, -10]),
        max: new Vector([10, 10])
    }
};

function Himmelblau2DFunc(x: number, y: number) {
    return Math.pow(x * x + y - 11, 2) + Math.pow(x + y * y - 7, 2);
}

export const HimmelblauFunc: OptimizationTestFunction = {
    name: "Himmelblau",
    f: function (p: Vector) {
        assert(p.size() == 2, "Invalid size");
        return Himmelblau2DFunc(p.get(0), p.get(1));
    },
    numDimensions: 2,
    min: {
        p: [
            new Vector([3, 2]),
            new Vector([-2.805118, 3.131312]),
            new Vector([-3.779310, -3.283186]),
            new Vector([3.584428, -1.848126])
        ],
        value: 0
    },
    searchDomain: {
        min: new Vector([-5, -5]),
        max: new Vector([5, 5])
    }
};

function ThreeHumpCamel2DFunc(x: number, y: number) {
    return x * (y + x * (2 + x * x * (x * x / 6 - 1.05)));
}

export const ThreeHumpCamelFunc: OptimizationTestFunction = {
    name: "Three Hump Camel",
    f: function (p: Vector) {
        assert(p.size() == 2, "Invalid size");
        return ThreeHumpCamel2DFunc(p.get(0), p.get(1));
    },
    numDimensions: 2,
    min: {
        p: new Vector([0, 0]),
        value: 0
    },
    searchDomain: {
        min: new Vector([-5, -5]),
        max: new Vector([5, 5])
    }
};

function Easom2DFunc(x: number, y: number) {
    return -Math.cos(x) * Math.cos(y) * Math.exp(-Math.pow(x - Math.PI, 2) - Math.pow(y - Math.PI, 2));
}

export const EasomFunc: OptimizationTestFunction = {
    name: "Easom",
    f: function (p: Vector) {
        assert(p.size() == 2, "Invalid size");
        return Easom2DFunc(p.get(0), p.get(1));
    },
    numDimensions: 2,
    min: {
        p: new Vector([Math.PI, Math.PI]),
        value: -1
    },
    searchDomain: {
        min: new Vector([-100, -100]),
        max: new Vector([100, 100])
    }
};

function CrossInTray2DFunc(x: number, y: number) {
    return -0.0001 * Math.pow(Math.abs(Math.sin(x) * Math.sin(y) * Math.exp(Math.abs(100 - Math.sqrt(x * x + y * y) / Math.PI))) + 1, 0.1);
}

export const CrossInTrayFunc: OptimizationTestFunction = {
    name: "Cross in tray",
    f: function (p: Vector) {
        assert(p.size() == 2, "Invalid size");
        return CrossInTray2DFunc(p.get(0), p.get(1));
    },
    numDimensions: 2,
    min: {
        p: [
            new Vector([1.34941, -1.34941]),
            new Vector([1.34941, 1.34941]),
            new Vector([-1.34941, 1.34941]),
            new Vector([-1.34941, -1.34941])
        ],
        value: -2.06261
    },
    searchDomain: {
        min: new Vector([-10, -10]),
        max: new Vector([10, 10])
    }
};

function Eggholder2DFunc(x: number, y: number) {
    return - (y + 47) * Math.sin(Math.sqrt(Math.abs(x / 2 + (y + 47)))) - x * Math.sin(Math.sqrt(Math.abs(x - (y + 47))));
}

export const EggholderFunc: OptimizationTestFunction = {
    name: "Eggholder",
    f: function (p: Vector) {
        assert(p.size() == 2, "Invalid size");
        return Eggholder2DFunc(p.get(0), p.get(1));
    },
    numDimensions: 2,
    min: {
        p: new Vector([512, 404.2319]),
        value: -959.6407
    },
    searchDomain: {
        min: new Vector([-512, -512]),
        max: new Vector([512, 512])
    }
};

function HölderTable2DFunc(x: number, y: number) {
    return - Math.abs(Math.sin(x) * Math.cos(y) * Math.exp(Math.abs(1 - Math.sqrt(x * x + y * y) / Math.PI)));
}

export const HolderTableFunc: OptimizationTestFunction = {
    name: "Hölder Table",
    f: function (p: Vector) {
        assert(p.size() == 2, "Invalid size");
        return HölderTable2DFunc(p.get(0), p.get(1));
    },
    numDimensions: 2,
    min: {
        p: [
            new Vector([8.05502, 9.66459]),
            new Vector([-8.05502, 9.66459]),
            new Vector([8.05502, -9.66459]),
            new Vector([-8.05502, -9.66459])
        ],
        value: -19.2085
    },
    searchDomain: {
        min: new Vector([-10, -10]),
        max: new Vector([10, 10])
    }
};

function McCormick2DFunc(x: number, y: number) {
    return Math.sin(x + y) + Math.pow(x - y, 2) - 1.5 * x + 2.5 * y + 1;
}

export const McCormickFunc: OptimizationTestFunction = {
    name: "McCormick",
    f: function (p: Vector) {
        assert(p.size() == 2, "Invalid size");
        return McCormick2DFunc(p.get(0), p.get(1));
    },
    numDimensions: 2,
    min: {
        p: new Vector([-0.54719, -1.54719]),
        value: -1.9133
    },
    searchDomain: {
        min: new Vector([-1.5, -3]),
        max: new Vector([4, 4])
    }
};

function Schaffer2DFunc2(x: number, y: number) {
    return 0.5 + (Math.pow(Math.sin(x * x - y * y), 2) - 0.5) / Math.pow(1 + 0.001 * (x * x + y * y), 2);
}

export const SchafferFunc2: OptimizationTestFunction = {
    name: "Schaffer2",
    f: function (p: Vector) {
        assert(p.size() == 2, "Invalid size");
        return Schaffer2DFunc2(p.get(0), p.get(1));
    },
    numDimensions: 2,
    min: {
        p: new Vector([0, 0]),
        value: 0
    },
    searchDomain: {
        min: new Vector([-100, -100]),
        max: new Vector([100, 100])
    }
};

function Schaffer2DFunc4(x: number, y: number) {
    return 0.5 + (Math.pow(Math.cos(Math.sin(Math.abs(x * x - y * y))), 2) - 0.5) / Math.pow(1 + 0.001 * (x * x + y * y), 2);
}

export const SchafferFunc4: OptimizationTestFunction = {
    name: "Schaffer4",
    f: function (p: Vector) {
        assert(p.size() == 2, "Invalid size");
        return Schaffer2DFunc4(p.get(0), p.get(1));
    },
    numDimensions: 2,
    min: {
        p: [
            new Vector([0, 1.25313]),
            new Vector([0, -1.25313]),
            new Vector([1.25313, 0]),
            new Vector([-1.25313, 0])
        ],
        value: 0.292579
    },
    searchDomain: {
        min: new Vector([-100, -100]),
        max: new Vector([100, 100])
    }
};

export const StyblinskiTangFunc: OptimizationTestFunction = {
    name: "Styblinski-Tang",
    f: function (p: Vector) {
        let result = 0;
        for (let i = 0; i < p.size(); ++i) {
            let value = p.get(i);
            result += value * (5 + value * (value * value - 16));
        }
        return result / (2 * p.size());
    },
    numDimensions: -1,
    min: {
        p: -2.903534,
        value: -39.166165
    },
    searchDomain: {
        min: -5,
        max: 5
    }
};

// constrained functions

export abstract class ConstrainedProblem {
    derivativeDelta: number = SmallTolerance;
    public abstract numConstraints(): number;
    public abstract numVariables(): number;
    public abstract constraints(p: Vector): Vector;
    public abstract f(p: Vector): number;
    public dfdx(p: Vector): Vector {
        return forwardDifference((x: Vector) => { return this.f(x); }, p, this.derivativeDelta);
    }
    public dfdxdx(p: Vector): Matrix {
        return secondOrderDifference((x: Vector) => { return this.f(x); }, p, this.derivativeDelta);
    }
}