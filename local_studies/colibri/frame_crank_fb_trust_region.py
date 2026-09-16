"""Equivalent normal NCP merit with bounded dogleg globalization, original mu."""

import json
from pathlib import Path

import numpy as np

from local_studies.colibri.coulomb_semismooth import natural_map_evaluator


def fb_evaluator(A, rhs, mu):
    natural, scale, operator, friction = natural_map_evaluator(A, rhs, np.zeros(len(mu)), mu)

    def evaluate(value):
        residual, jac = natural(value)
        velocity = operator @ value + rhs
        for n in range(0, len(value), 3):
            a = scale[n] * value[n]
            b = velocity[n]
            norm = np.hypot(a, b)
            residual[n] = norm - a - b
            if norm > 0:
                da = a / norm - 1.0
                db = b / norm - 1.0
            else:
                da = db = 1.0 / np.sqrt(2.0) - 1.0
            jac[n] = db * operator[n]
            jac[n, n] += da * scale[n]
        return residual, jac

    return evaluate, natural, scale


def solve(A, rhs, seed, mu, max_steps=80):
    evaluate, natural, scale = fb_evaluator(A, rhs, mu)
    value = seed.copy()
    radius = max(float(np.linalg.norm(scale * value)), 1e-3)
    history = []
    for step in range(max_steps):
        residual, jac = evaluate(value)
        physical = float(np.max(np.abs(natural(value)[0])))
        if physical < 1e-8:
            history.append({"step": step, "physical_residual": physical, "accepted": True})
            break
        J = jac / scale[None, :]
        newton, _, rank, singular = np.linalg.lstsq(J, -residual, rcond=None)
        gradient = J.T @ residual
        normg = float(gradient @ gradient)
        image = J @ gradient
        cauchy = -gradient * (normg / max(float(image @ image), 1e-300))
        if np.linalg.norm(newton) <= radius:
            direction = newton
            kind = "newton"
        elif np.linalg.norm(cauchy) >= radius:
            direction = -radius * gradient / max(np.linalg.norm(gradient), 1e-300)
            kind = "cauchy_boundary"
        else:
            difference = newton - cauchy
            aa = float(difference @ difference)
            bb = 2 * float(cauchy @ difference)
            cc = float(cauchy @ cauchy) - radius**2
            tau = (-bb + np.sqrt(max(bb * bb - 4 * aa * cc, 0))) / (2 * aa)
            direction = cauchy + tau * difference
            kind = "dogleg_boundary"
        prediction = 0.5 * float(residual @ residual) - 0.5 * float(
            (residual + J @ direction) @ (residual + J @ direction)
        )
        trial = value + direction / scale
        next_residual, _ = evaluate(trial)
        actual = 0.5 * float(residual @ residual) - 0.5 * float(next_residual @ next_residual)
        ratio = actual / prediction if prediction > 0 else -np.inf
        take = ratio > 0.1
        history.append(
            {
                "step": step,
                "physical_residual": physical,
                "fb_merit": 0.5 * float(residual @ residual),
                "radius": radius,
                "ratio": float(ratio),
                "step_kind": kind,
                "accepted_step": take,
                "rank": int(rank),
                "smallest_retained": float(singular[rank - 1]),
                "linear_defect": float(np.max(np.abs(J @ newton + residual))),
            }
        )
        if ratio < 0.25:
            radius *= 0.25
        elif ratio > 0.75 and kind != "newton":
            radius *= 2
        if take:
            value = trial
        if radius < np.finfo(float).eps * max(1.0, np.linalg.norm(scale * value)):
            break
    return value, history, float(np.max(np.abs(natural(value)[0])))


def main():
    x = np.load("/tmp/colibri_physical_gs_combined_frame_crank_4.npz")
    value, history, error = solve(x["A"], x["rhs"], x["solution"], x["friction"])
    report = {
        "original_friction": True,
        "method": "FB_normal_plus_original_disk_dogleg",
        "maximum_steps": 80,
        "physical_residual": error,
        "accepted": error < 1e-8,
        "history": history,
    }
    Path("/tmp/colibri_frame_crank_fb_trust_region.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed("/tmp/colibri_frame_crank_fb_trust_region.npz", solution=value)
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))
    print("last", history[-5:])


if __name__ == "__main__":
    main()
