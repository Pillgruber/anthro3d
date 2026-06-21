from pathlib import Path
import numpy as np
import math

BASE = Path("~/anthro3d").expanduser()
STATE = BASE / "aruco_state"

PAIRS = [
    "ELP1_to_ELP2",
    "OV9281_to_ELP2",
    "OV9281_to_ELP1",
]

CONTAINERS = [
    "active",
    "candidate",
    "backup",
]

def load_npz_pair(pair_name):
    for container in CONTAINERS:
        path = STATE / f"{pair_name}_{container}.npz"
        if not path.exists():
            continue

        try:
            data = np.load(path, allow_pickle=False)
            ok = bool(data["ok"][0])
            if not ok:
                continue

            R = np.array(data["R"], dtype=float)
            T = np.array(data["T"], dtype=float).reshape(3)
            source = str(data["source"][0]) if "source" in data.files else container

            return {
                "R": R,
                "T": T,
                "source": f"{container}:{source}",
                "path": str(path),
            }
        except Exception:
            continue

    return None

def load_legacy_pair(pair_name):
    files = {
        "ELP1_to_ELP2": (
            "R_rel_elp1_to_elp2.npy",
            "T_rel_elp1_to_elp2.npy",
        ),
        "OV9281_to_ELP2": (
            "R_rel_ov9281_to_elp2.npy",
            "T_rel_ov9281_to_elp2.npy",
        ),
    }

    if pair_name not in files:
        return None

    r_file, t_file = files[pair_name]
    r_path = BASE / r_file
    t_path = BASE / t_file

    if not r_path.exists() or not t_path.exists():
        return None

    return {
        "R": np.load(r_path),
        "T": np.load(t_path).reshape(3),
        "source": "legacy_npy",
        "path": f"{r_path.name}, {t_path.name}",
    }

def load_pair(pair_name):
    item = load_npz_pair(pair_name)
    if item is not None:
        return item

    return load_legacy_pair(pair_name)

def compose(first, second):
    R1 = first["R"]
    T1 = first["T"].reshape(3)
    R2 = second["R"]
    T2 = second["T"].reshape(3)

    R = R1 @ R2
    T = R1 @ T2 + T1

    return {"R": R, "T": T}

def inv(rel):
    R = rel["R"]
    T = rel["T"].reshape(3)

    Ri = R.T
    Ti = -Ri @ T

    return {"R": Ri, "T": Ti}

def rot_diff_deg(a, b):
    R_delta = a["R"] @ b["R"].T
    trace_val = (np.trace(R_delta) - 1.0) / 2.0
    trace_val = float(np.clip(trace_val, -1.0, 1.0))
    return math.degrees(math.acos(trace_val))

def trans_diff_cm(a, b):
    return float(np.linalg.norm(a["T"].reshape(3) - b["T"].reshape(3)) * 100.0)

def print_rel(name, rel):
    print()
    print(name)
    if rel is None:
        print("  fehlt")
        return

    print(f"  Quelle: {rel['source']}")
    print(f"  Datei: {rel['path']}")
    print(f"  T cm: {rel['T'] * 100.0}")
    print(f"  Distanz cm: {np.linalg.norm(rel['T']) * 100.0:.2f}")

def print_compare(name, direct, via):
    t = trans_diff_cm(direct, via)
    r = rot_diff_deg(direct, via)

    print()
    print(name)
    print(f"  Translation Differenz cm: {t:.2f}")
    print(f"  Rotation Differenz Grad: {r:.2f}")

    return t, r

elp1_to_elp2 = load_pair("ELP1_to_ELP2")
ov_to_elp2 = load_pair("OV9281_to_ELP2")
ov_to_elp1 = load_pair("OV9281_to_ELP1")

print("Dreiecksschluss Richtungstest")
print("============================")

print_rel("ELP1_to_ELP2", elp1_to_elp2)
print_rel("OV9281_to_ELP2 direkt", ov_to_elp2)
print_rel("OV9281_to_ELP1", ov_to_elp1)

if elp1_to_elp2 is None or ov_to_elp2 is None or ov_to_elp1 is None:
    print()
    print("Ergebnis:")
    print("  Dreieckstest nicht moeglich, weil mindestens eine Beziehung fehlt.")
    print("  Erst ArUcoMonitor im laufenden Kamera-Loop sammeln lassen oder calibrate_positions.py erneut ausfuehren.")
    raise SystemExit(0)

standard_via = compose(elp1_to_elp2, ov_to_elp1)

reverse_via = compose(ov_to_elp1, elp1_to_elp2)

inverse_standard_via = inv(standard_via)
inverse_direct = inv(ov_to_elp2)

t_standard, r_standard = print_compare(
    "Variante A, erwartete Richtung: OV_to_ELP2 direkt gegen ELP1_to_ELP2 mal OV_to_ELP1",
    ov_to_elp2,
    standard_via,
)

t_reverse, r_reverse = print_compare(
    "Variante B, umgekehrte Reihenfolge: OV_to_ELP2 direkt gegen OV_to_ELP1 mal ELP1_to_ELP2",
    ov_to_elp2,
    reverse_via,
)

t_inverse, r_inverse = print_compare(
    "Variante C, inverse Kontrolle: inverse OV_to_ELP2 gegen inverse der erwarteten Kette",
    inverse_direct,
    inverse_standard_via,
)

score_standard = t_standard + r_standard * 2.0
score_reverse = t_reverse + r_reverse * 2.0

print()
print("Bewertung")
print("=========")
print(f"  Score erwartete Richtung: {score_standard:.2f}")
print(f"  Score umgekehrte Reihenfolge: {score_reverse:.2f}")

if score_standard < score_reverse:
    print("  Ergebnis: Erwartete Verkettung ist plausibler.")
    print("  Formel: OV_to_ELP2 = ELP1_to_ELP2 mal OV_to_ELP1")
else:
    print("  Ergebnis: Umgekehrte Reihenfolge ist plausibler.")
    print("  Achtung: Dreiecksschluss im Monitor muss dann angepasst werden.")

print()
print("Hinweis:")
print("  Gute Werte waeren grob unter 8 bis 12 cm und unter 5 bis 8 Grad.")
print("  Wenn beide Varianten schlecht sind, ist nicht zwingend die Richtung falsch.")
print("  Dann sind eher eine Messbeziehung, Markerpose oder rechte Linsen-Umrechnung instabil.")
