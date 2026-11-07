"""Rebuild everything from the raw CSV:  python run_pipeline.py
Options:  --list  |  --from STEP  |  --only STEP"""
import argparse
import subprocess
import sys
import time

SCOPE_K = {"national": 3, "india_states": 3, "india_districts": 3}
TEMPORAL_K = {"national": 4, "india_states": 3, "india_districts": 3}


def mod(name, *args):
    return ["-m", name, *args]


STEPS = [
    ("load", mod("src.load_data")),
    ("profile", mod("src.profile_data")),
    ("preprocess", mod("src.preprocess")),
    ("features", mod("src.features")),
    ("outliers", mod("src.outliers")),
    ("cluster", mod("src.cluster")),
]
for scope, k in SCOPE_K.items():
    STEPS += [
        (f"evaluate_{scope}", mod("src.evaluate", "--scope", scope, "--k", str(k))),
        (f"hierarchical_{scope}", mod("src.hierarchical", "--scope", scope, "--k", str(k))),
        (f"interpret_{scope}", mod("src.interpret", "--scope", scope, "--k", str(k))),
        (f"temporal_{scope}", mod("src.temporal", "--scope", scope, "--k", str(TEMPORAL_K[scope]))),
    ]
STEPS.append(("tests", mod("pytest", "-q")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="list step names")
    ap.add_argument("--from", dest="start", help="resume from this step")
    ap.add_argument("--only", help="run a single step")
    args = ap.parse_args()

    names = [n for n, _ in STEPS]
    if args.list:
        print("\n".join(names))
        return
    for chosen in (args.start, args.only):
        if chosen and chosen not in names:
            sys.exit(f"Unknown step '{chosen}'. Use --list to see valid names.")

    todo = STEPS
    if args.only:
        todo = [s for s in STEPS if s[0] == args.only]
    elif args.start:
        todo = STEPS[names.index(args.start):]

    t_all = time.time()
    for name, cmd in todo:
        print(f"\n>>> {name}")
        t0 = time.time()
        result = subprocess.run([sys.executable, *cmd])
        if result.returncode != 0:
            sys.exit(f"FAILED at step '{name}'. Fix it, then resume with: "
                     f"python run_pipeline.py --from {name}")
        print(f"<<< {name} done in {time.time() - t0:.0f}s")
    print(f"\nPipeline finished in {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()