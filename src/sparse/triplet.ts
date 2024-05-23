

export default interface Triplet {
    row: number,
    column: number,
    value: number
};

export function makeTriplet(row: number, column: number, value: number): Triplet {
    return { row, column, value };
}