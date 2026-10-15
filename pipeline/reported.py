"""
reported.py — shared by the reported_values.py scripts in each stage.

A Report compares a value recomputed from results/ with the value printed in
the paper. A value matches when it rounds to the printed digits (or meets the
stated bound for "about", ranges and inequalities). Nothing here changes the
analysis; it only reads deposited outputs.
"""
import math


def _decimals(x):
    s = repr(x)
    if "e" in s or "E" in s:
        return None
    return len(s.split(".")[1]) if "." in s else 0


class Report:
    def __init__(self, title):
        self.title = title
        self.n = 0
        self.bad = []
        self._server = []
        print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")

    def section(self, name):
        print(f"\n-- {name}")

    def check(self, label, value, reported, digits=None, kind="round", ref=None, lo=None, hi=None, note=""):
        self.n += 1
        if isinstance(reported, bool) or kind == "bool":
            ok = bool(value) == bool(reported)
            shown = str(bool(value))
        elif kind == "gt":
            ok = value > ref
            shown = f"{value:.4g}"
        elif kind == "range":
            ok = lo <= value <= hi
            shown = f"{value:.4g}"
        elif kind == "sci":
            # same mantissa to two significant figures and same exponent
            ok = f"{value:.1e}" == f"{float(reported):.1e}"
            shown = f"{value:.2e}"
        else:
            d = _decimals(reported) if digits is None else digits
            ok = round(float(value), d) == round(float(reported), d) if d is not None else math.isclose(value, reported, rel_tol=0.05)
            shown = f"{value:.{max(d, 0) + 1}f}" if d is not None and d >= 0 else f"{value:.4g}"
        mark = "OK " if ok else "MISMATCH"
        extra = f"   ({note})" if note else ""
        print(f"  {mark:8s} {label:<70s} here {shown:>12s}   paper {reported!s:>10s}{extra}")
        if not ok:
            self.bad.append(label)

    def server_only(self, items):
        self._server.extend(items)

    def finish(self):
        if self._server:
            print("\n-- Values that need data not deposited here (printed by the script named):")
            for what, where in self._server:
                print(f"  {what}\n      -> {where}")
        print(f"\n{self.title}: {self.n - len(self.bad)} of {self.n} values reproduced from results/")
        for b in self.bad:
            print(f"  MISMATCH: {b}")
        return 1 if self.bad else 0
