import anyio
from repartee.memory import build_host
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--addr",
                    default="tcp://127.0.0.1:55855",
                    help="listen address (default tcp://127.0.0.1:55855)")
    ns = ap.parse_args()
    host = build_host()
    anyio.run(host.serve, ns.addr)

if __name__ == "__main__":
    main()
