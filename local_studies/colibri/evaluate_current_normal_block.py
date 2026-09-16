"""Construct the current captured physical operator without changing witnesses."""

import numpy as np

from local_studies.colibri.prototype_connected_normal_sweeps import main as evaluate


def main():
    snapshot = "/tmp/high_mass_current_245_1_before.npz"
    raw = np.load(snapshot)
    nb = len(raw["inverse_mass"])
    points = np.flatnonzero(raw["derived"][3] <= 0)
    J = np.zeros((len(points) * 3, nb * 6))
    W = np.zeros((nb * 6, nb * 6))
    for b in range(nb):
        W[6 * b : 6 * b + 3, 6 * b : 6 * b + 3] = np.eye(3) * raw["inverse_mass"][b]
        W[6 * b + 3 : 6 * b + 6, 6 * b + 3 : 6 * b + 6] = raw["inverse_inertia"][b]
    h = raw["headers"].view(np.int32)
    for c in range(int(raw["column_count"][0])):
        a, b = h[1:3, c]
        for index, p in enumerate(points):
            if not h[5, c] <= p < h[5, c] + h[6, c]:
                continue
            n = raw["lambdas"][:3, p].astype(float)
            t = raw["lambdas"][3:6, p].astype(float)
            axes = np.array([n, t, np.cross(n, t)])
            J[3 * index : 3 * index + 3, 6 * a : 6 * a + 6] = -np.concatenate(
                (axes, np.cross(raw["derived"][9:12, p], axes)), axis=1
            )
            J[3 * index : 3 * index + 3, 6 * b : 6 * b + 6] = np.concatenate(
                (axes, np.cross(raw["derived"][12:15, p], axes)), axis=1
            )
    A = J @ W @ J.T
    initial = raw["impulses"][:, points].T.astype(float).ravel()
    u = np.concatenate((raw["velocity"], raw["angular_velocity"]), axis=1).astype(float).ravel()
    output = "/tmp/high_mass_current_245_1_operator.npz"
    np.savez(
        output,
        A=A,
        J=J,
        inverse_mass=W,
        initial=initial,
        rhs=J @ u - A @ initial,
        selected_points=points,
        normal_regularization=np.zeros(len(points)),
    )
    evaluate(output, snapshot, "/tmp/high_mass_current_normal_sweeps")


if __name__ == "__main__":
    main()
