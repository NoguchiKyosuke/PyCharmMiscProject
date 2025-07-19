ImportError: Numba needs NumPy 2.2 or less. Got NumPy 2.3.

solve: pip install numpy==2.2

---

TypeError: time_stretch() takes 1 positional argument but 2 were given

solve: time_stretch(x, rate) -> time_stretch(x, rate = 1.0)