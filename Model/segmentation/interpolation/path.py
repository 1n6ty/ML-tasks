import numpy as np
import scipy.integrate

from collections.abc import Callable

class vPath:
    """Class to store skeletons of base spine-elements.
    
        This class contains methods to build up a skeleton 
        (natural middle curve) for basic spine-elements 
        (like vertebrae), computes it's length and provides
        public method to parametrize it.

        Attributes
        ----------
            length (np.float32)
                Path-length of `f`
            start_t (np.float32)
                t = 0 in absolute parametrization (from start of S1)
            f (Callable[np.float32, np.ndarray[np.float32]])
                Parametrized function of a path, `t` is inside `[0, 1]`
            df (Callable[np.float32, np.ndarray[np.float32]])
                Derivative vector-function of `f`, `t` is inside `[0, 1]`
            fn (Callable[np.float32, np.ndarray[np.float32]])
                Normal vector for df (counter-clockwise), `t` is inside `[0, 1]`
    """
    def __init__(self, 
                 start_p: np.ndarray[np.int32], 
                 end_p: np.ndarray[np.int32], 
                 start_normal: np.ndarray[np.int32], 
                 end_normal: np.ndarray[np.int32],
                 start_t: np.float32 = 0,
                 n: np.int32 = 4
                ) -> None:
        """Constructor for vPath.

            Makes a parametrization (natural curve, cubic spline by default) for skeleton path: `f(t) -> (x, y)` ,
            where `t` is parameter between `[start, end]` (default is `[0, 1]`);
            Also makes derivative vector-function for it; Makes function that returns normal for derivative vector
            (counter-clockwise 90deg); Computes length of final path.
        
            Args
            ----
                start_p (np.ndarray[np.int32])
                    Start point of curve
                end_p (np.ndarray[np.int32])
                    End point of curve
                start_normal (np.ndarray[np.int32])
                    Normal vector to curve in `start_p`
                end_normal (np.ndarray[np.int32])
                    Normal vector to curve in `end_p`
                start_t (np.float32 | None)
                    Start value for parameter t
                n (np.int32)
                    Dim of the curve (default is cubic spline -> 4). Should be greater then 3
        """
        self.start_t: np.float32 = start_t
        A: np.ndarray[np.float32] = np.array(
            [
                [0 ** i for i in range(n)] + [0 for i in range(n)],
                [0 for i in range(n)] + [0 ** i for i in range(n)],
                [1 ** i for i in range(n)] + [0 for i in range(n)],
                [0 for i in range(n)] + [1 ** i for i in range(n)],
                [0] + [start_normal[0]] + [0 for i in range(n - 2)] + [0] + [start_normal[1]] + [0 for i in range(n - 2)],
                [0] + [end_normal[0] * i for i in range(1, n)] + [0] + [end_normal[1] * i for i in range(1, n)],
                [0, 0] + [2] + [0 for i in range(n - 3)] + [0 for i in range(n)],
                [0 for i in range(n)] + [0, 0] + [2] + [0 for i in range(n - 3)],
                [0, 0] + [2] + [i * (i - 1) for i in range(3, n)] + [0 for i in range(n)],
                [0 for i in range(n)] + [0, 0] + [2] + [i * (i - 1) for i in range(3, n)],
            ],
            dtype=np.float32
        )
        b: np.ndarray[np.float32] = np.array(
            [
                [start_p[0]],
                [start_p[1]],
                [end_p[0]],
                [end_p[1]],
                [0],
                [0],
                [0],
                [0],
                [0],
                [0],
            ],
            dtype=np.float32
        )
        try:
            c: np.ndarray[np.float32] = np.linalg.solve(A.T @ A, A.T @ b)

            path_length: np.float32 = scipy.integrate.quad(
                vPath._dpath_length,
                0,
                1,
                args=(c, n)
            )[0]
            self.length: np.float32 = path_length

            def f(t: np.float32, start: np.float32 = 0, end: np.float32 = 1) -> np.ndarray[np.float32]:
                """Parametrized path function.
                
                    Args
                    ----
                        t (np.float32)
                            Relative parameter, which starts from `start` and ends at `end`
                        start (np.float32)
                            Parametristic start, default is 0
                        end (np.float32)
                            Parametristic end, default is 1

                    Returns
                    -------
                        point (np.ndarray[np.float32])
                            Point on path that corresponds to `t` 
                """
                tn: np.float32 = (t - start) / (end - start)
                return np.squeeze(
                    np.array(
                        [
                            [tn ** i for i in range(n)] + [0 for i in range(n)],
                            [0 for i in range(n)] + [tn ** i for i in range(n)]
                        ],
                        dtype=np.float32
                    ) @ c
                )
            self.f = f

            def df(t: np.float32, start: np.float32 = 0, end: np.float32 = 1) -> np.ndarray[np.float32]:
                """Parametrized derivative of path function.
                
                    Args
                    ----
                        t (np.float32)
                            Relative parameter, which starts from `0` and ends at `1`
                        start (np.float32)
                            Parametristic start, default is 0
                        end (np.float32)
                            Parametristic end, default is 1

                    Returns
                    -------
                        vector (np.ndarray[np.float32])
                            Normalized Derivative vector of function `f` in point that corresponds to `t`
                """
                tn: np.float32 = (t - start) / (end - start)
                new_p: np.ndarray[np.float32] = np.squeeze(
                    np.array(
                        [
                            [0] + [i * (tn ** (i - 1)) for i in range(1, n)] + [0 for i in range(n)],
                            [0 for i in range(n)] + [0] + [i * (tn ** (i - 1)) for i in range(1, n)],
                        ],
                        dtype=np.float32
                    ) @ c
                )
                new_p_norm: np.float32 = np.linalg.norm(new_p)
                return new_p / (1e6 if new_p_norm == 0 else new_p_norm)
            self.df = df
        except Exception as e:
            path_length: np.float32 = np.linalg.norm(start_p - end_p)
            self.length: np.float32 = path_length

            def f(t: np.float32, start: np.float32 = 0, end: np.float32 = 1) -> np.ndarray[np.int32]:
                """Parametrized path function.
                
                    Args
                    ----
                        t (np.float32)
                            Relative parameter, which starts from `start` and ends at `end`
                        start (np.float32)
                            Parametristic start, default is 0
                        end (np.float32)
                            Parametristic end, default is 1
                    
                    Returns
                    -------
                        point (np.ndarray[np.float32])
                            Point on path that corresponds to `t` 
                """
                tn: np.float32 = (t - start) / (end - start)
                return (1 - t) * start_p + tn * end_p
            self.f = f

            def df(t: np.float32, start: np.float32 = 0, end: np.float32 = 1) -> np.ndarray[np.float32]:
                """Parametrized derivative of path function.
                
                    Args
                    ----
                        t (np.float32)
                            Relative parameter, which starts from `start` and ends at `end`
                        start (np.float32)
                            Parametristic start, default is 0
                        end (np.float32)
                            Parametristic end, default is 1

                    Returns
                    -------
                        vector (np.ndarray[np.float32])
                            Derivative vector of function `f` in point that corresponds to `t`
                """
                return (end_p - start_p) / (1e6 if path_length == 0 else path_length)
            self.df = df

        def fn(t: np.float32, start: np.float32 = 0, end: np.float32 = 1) -> np.ndarray[np.float32]:
            """Parametrized normal vector for derivative of path function.

                Normal vector is computed counter-clockwise relative to the derivative vector.
            
                Args
                ----
                    t (np.float32)
                        Relative parameter, which starts from `start` and ends at `end`
                    start (np.float32)
                            Parametristic start, default is 0
                    end (np.float32)
                        Parametristic end, default is 1

                Returns
                -------
                    vector (np.ndarray[np.float32])
                        Normal vector for derivative vector of function `f` in point that corresponds to `t`
            """
            d: np.ndarray[np.float32] = self.df(t, start, end)
            return np.array([d[1], -d[0]], dtype=np.float32)
        self.fn = fn

    @staticmethod
    def _dpath_length(t: np.float32, c: np.ndarray[np.float32], n: np.int32) -> np.float32:
        A: np.ndarray = np.array(
            [
                [0] + [i * (t ** (i - 1)) for i in range(1, n)] + [0 for i in range(n)],
                [0 for i in range(n)] + [0] + [i * (t ** (i - 1)) for i in range(1, n)],
            ], 
            dtype=np.float32
        )
        p: np.ndarray = np.squeeze(A @ c)
        return np.sqrt(p[0] ** 2 + p[1] ** 2)