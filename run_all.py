"""Single entry point:  python run_all.py [e0 e1 ...]   (default: all)."""
import sys, time
from experiments import ALL

if __name__ == "__main__":
    names = sys.argv[1:] or list(ALL)
    for n in names:
        t = time.time()
        ALL[n]()
        print(f"{n} done in {time.time() - t:.0f}s", flush=True)
