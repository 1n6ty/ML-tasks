import numpy as np

import scipy.interpolate, scipy.integrate

from typing import Callable

class lPath:
    """Class to store skeletons of base spine-elements - line-parametrization.
    
        This class contains methods to build up a skeleton 
        (middle curve) from simple lines for basic spine-elements 
        (like vertebrae), computes it's length and provides
        public method to parametrize it.

        Attributes
        ----------
            length (np.float32)
                Path-length of `f`
            start_t (np.float32)
                t = 0 in absolute parametrization (from bottom of S1)
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
                 start_t: np.float32 = 0,
                ) -> None:
        """Constructor for lPath.

            Makes a line interpolation for skeleton path: `f(t) -> (x, y)`,
            where `t` is parameter between `[start, end]` (default is `[0, 1]`);
            Also makes derivative vector-function for it; Makes function that returns normal for derivative vector
            (counter-clockwise 90deg); Computes length of final path.
        
            Args
            ----
                start_p (np.ndarray[np.int32])
                    Start point of curve
                end_p (np.ndarray[np.int32])
                    End point of curve
                start_t (np.float32 | None)
                    Start value for parameter t
        """
        self.start_t: np.float32 = start_t
        
        self.length: np.float32 = np.linalg.norm(start_p - end_p)
        
        self.f = self._get_f(start_p, end_p)
        self.df = self._get_df(start_p, end_p)
        self.fn = self._get_fn()
    
    def _get_f(self, start_p: np.ndarray[np.int32], end_p: np.ndarray[np.int32]) -> Callable[[np.float32, np.float32, np.float32], np.ndarray[np.float32]]:
        """Method to obtain f.

            Args
            ----
                start_p (np.ndarray[np.int32])
                    Point of path, that would be under `t = start`
                end_p (np.ndarray[np.int32])
                    Point of path, that would be under `t = end`
            
            Returns
            -------
                f ((float32, float32, float32) -> ndarray[float32])
                    Path function
        """
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
        return f

    def _get_df(self, start_p: np.ndarray[np.int32], end_p: np.ndarray[np.int32]) -> Callable[[np.float32, np.float32, np.float32], np.ndarray[np.float32]]:
        """Method to obtain derivative function of f.

            Args
            ----
                start_p (np.ndarray[np.int32])
                    Point of path, that would be under `t = start`
                end_p (np.ndarray[np.int32])
                    Point of path, that would be under `t = end`
            
            Returns
            -------
                df ((float32, float32, float32) -> ndarray[float32])
                    Derivative function of `f`
        """
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
                        df over dt vector in point that corresponds to `t`
            """
            return (end_p - start_p) / (start - end)
        return df
    
    def _get_fn(self) -> Callable[[np.float32, np.float32, np.float32], np.ndarray[np.float32]]:
        """Method to obtain normal of f.

            Returns
            -------
                fn ((float32, float32, float32) -> ndarray[float32])
                    Normal function of `f`
        """
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
        return fn
