"""
Polar Mesh and Derivative Operators
====================================

Implements the polar (r, θ) grid for the slipper bearing land region
and discrete derivative operators for the Reynolds equation.

The mesh covers the annular region R_in ≤ r ≤ R_out, 0 ≤ θ < 2π.

Key Features:
    - Cell-centred finite differences
    - Second-order accurate derivative operators
    - Periodic boundary handling in θ
    - Sparse matrix representation for efficiency

Coordinate System:
    - r: radial coordinate (cell centres at r[i])
    - θ: azimuthal coordinate (cell centres at theta[j])
    - Indexing: field[i, j] corresponds to (r[i], theta[j])

References:
    - LeVeque, R.J. (2007). Finite Difference Methods for ODEs and PDEs.
    - Dhlamini, N.C. (2025). PhD Thesis, Chapter 4.

Author: Nigel C. Dhlamini
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional
import numpy as np
from scipy.sparse import diags, lil_matrix, csr_matrix, eye
from scipy.sparse.linalg import spsolve

from .geometry import SlipperGeometry


@dataclass
class PolarMesh:
    """
    Polar mesh for the slipper bearing land region.
    
    The mesh is uniform in both r and θ directions with cell-centred nodes.
    
    Parameters
    ----------
    geometry : SlipperGeometry
        Slipper geometry defining the domain
    Nr : int
        Number of grid points in radial direction
    Ntheta : int
        Number of grid points in azimuthal direction
    
    Attributes
    ----------
    r : ndarray
        Radial node positions, shape (Nr,) [m]
    theta : ndarray
        Azimuthal node positions, shape (Ntheta,) [rad]
    R, THETA : ndarray
        Meshgrid arrays, shape (Nr, Ntheta)
    dr : float
        Radial grid spacing [m]
    dtheta : float
        Azimuthal grid spacing [rad]
    N : int
        Total number of nodes = Nr × Ntheta
    
    Examples
    --------
    >>> from src.geometry import SlipperGeometry
    >>> geom = SlipperGeometry(R_in=0.008, R_out=0.020, h_0=5e-6)
    >>> mesh = PolarMesh(geom, Nr=40, Ntheta=60)
    >>> mesh.dr
    0.0003  # (R_out - R_in) / Nr
    """
    geometry: SlipperGeometry
    Nr: int
    Ntheta: int
    
    # Grid arrays (computed in __post_init__)
    r: np.ndarray = field(init=False, repr=False)
    theta: np.ndarray = field(init=False, repr=False)
    R: np.ndarray = field(init=False, repr=False)
    THETA: np.ndarray = field(init=False, repr=False)
    dr: float = field(init=False)
    dtheta: float = field(init=False)
    N: int = field(init=False)
    
    # Cached operators
    _D_r: Optional[csr_matrix] = field(init=False, default=None, repr=False)
    _D_theta: Optional[csr_matrix] = field(init=False, default=None, repr=False)
    _D_rr: Optional[csr_matrix] = field(init=False, default=None, repr=False)
    _D_thetatheta: Optional[csr_matrix] = field(init=False, default=None, repr=False)
    
    def __post_init__(self):
        """Build the mesh and pre-compute operators."""
        if self.Nr < 5:
            raise ValueError(f"Nr must be at least 5, got {self.Nr}")
        if self.Ntheta < 6:
            raise ValueError(f"Ntheta must be at least 6, got {self.Ntheta}")
        
        R_in = self.geometry.R_in
        R_out = self.geometry.R_out
        
        # Cell-centred nodes
        self.dr = (R_out - R_in) / self.Nr
        self.dtheta = 2 * np.pi / self.Ntheta
        
        # Node positions (cell centres)
        self.r = R_in + (np.arange(self.Nr) + 0.5) * self.dr
        self.theta = (np.arange(self.Ntheta) + 0.5) * self.dtheta
        
        # Meshgrid (r varies along axis 0, theta along axis 1)
        self.R, self.THETA = np.meshgrid(self.r, self.theta, indexing='ij')
        
        self.N = self.Nr * self.Ntheta
    
    def _index(self, i: int, j: int) -> int:
        """
        Convert 2D index (i, j) to 1D index for sparse matrices.
        
        Uses row-major ordering: k = i * Ntheta + j
        
        Parameters
        ----------
        i : int
            Radial index, 0 ≤ i < Nr
        j : int
            Azimuthal index (with periodic wrap)
        
        Returns
        -------
        k : int
            1D index, 0 ≤ k < N
        """
        j_wrapped = j % self.Ntheta  # Periodic in θ
        return i * self.Ntheta + j_wrapped
    
    def to_2d(self, field_1d: np.ndarray) -> np.ndarray:
        """
        Reshape 1D field vector to 2D grid array.
        
        Parameters
        ----------
        field_1d : ndarray
            1D field of length N
        
        Returns
        -------
        field_2d : ndarray
            2D field of shape (Nr, Ntheta)
        """
        return field_1d.reshape((self.Nr, self.Ntheta))
    
    def to_1d(self, field_2d: np.ndarray) -> np.ndarray:
        """
        Flatten 2D grid array to 1D field vector.
        
        Parameters
        ----------
        field_2d : ndarray
            2D field of shape (Nr, Ntheta)
        
        Returns
        -------
        field_1d : ndarray
            1D field of length N
        """
        return field_2d.ravel()
    
    # =========================================================================
    # Derivative Operators
    # =========================================================================
    
    def build_D_r(self) -> csr_matrix:
        """
        Build first derivative operator in r direction: ∂/∂r.
        
        Uses second-order central differences in the interior.
        At boundaries:
            - r = R_in: forward difference (Neumann BC proxy)
            - r = R_out: backward difference (Neumann BC proxy)
        
        Returns
        -------
        D_r : csr_matrix
            Sparse matrix of shape (N, N). For field f, D_r @ f gives ∂f/∂r.
        """
        if self._D_r is not None:
            return self._D_r
        
        D = lil_matrix((self.N, self.N))
        dr = self.dr
        
        for i in range(self.Nr):
            for j in range(self.Ntheta):
                k = self._index(i, j)
                
                if i == 0:
                    # Forward difference: (-3f_0 + 4f_1 - f_2) / (2Δr)
                    D[k, self._index(i, j)] = -3.0 / (2.0 * dr)
                    D[k, self._index(i+1, j)] = 4.0 / (2.0 * dr)
                    D[k, self._index(i+2, j)] = -1.0 / (2.0 * dr)
                elif i == self.Nr - 1:
                    # Backward difference: (3f_N - 4f_{N-1} + f_{N-2}) / (2Δr)
                    D[k, self._index(i, j)] = 3.0 / (2.0 * dr)
                    D[k, self._index(i-1, j)] = -4.0 / (2.0 * dr)
                    D[k, self._index(i-2, j)] = 1.0 / (2.0 * dr)
                else:
                    # Central difference: (f_{i+1} - f_{i-1}) / (2Δr)
                    D[k, self._index(i+1, j)] = 1.0 / (2.0 * dr)
                    D[k, self._index(i-1, j)] = -1.0 / (2.0 * dr)
        
        self._D_r = D.tocsr()
        return self._D_r
    
    def build_D_theta(self) -> csr_matrix:
        """
        Build first derivative operator in θ direction: ∂/∂θ.
        
        Uses second-order central differences with periodic wrapping.
        
        Returns
        -------
        D_theta : csr_matrix
            Sparse matrix of shape (N, N). For field f, D_theta @ f gives ∂f/∂θ.
        """
        if self._D_theta is not None:
            return self._D_theta
        
        D = lil_matrix((self.N, self.N))
        dtheta = self.dtheta
        
        for i in range(self.Nr):
            for j in range(self.Ntheta):
                k = self._index(i, j)
                # Central difference with periodic wrapping
                jp1 = (j + 1) % self.Ntheta
                jm1 = (j - 1) % self.Ntheta
                D[k, self._index(i, jp1)] = 1.0 / (2.0 * dtheta)
                D[k, self._index(i, jm1)] = -1.0 / (2.0 * dtheta)
        
        self._D_theta = D.tocsr()
        return self._D_theta
    
    def build_D_rr(self) -> csr_matrix:
        """
        Build second derivative operator in r direction: ∂²/∂r².
        
        Uses second-order central differences.
        
        Returns
        -------
        D_rr : csr_matrix
            Sparse matrix of shape (N, N).
        """
        if self._D_rr is not None:
            return self._D_rr
        
        D = lil_matrix((self.N, self.N))
        dr2 = self.dr**2
        
        for i in range(self.Nr):
            for j in range(self.Ntheta):
                k = self._index(i, j)
                
                if i == 0:
                    # One-sided: (2f_0 - 5f_1 + 4f_2 - f_3) / Δr²
                    D[k, self._index(i, j)] = 2.0 / dr2
                    D[k, self._index(i+1, j)] = -5.0 / dr2
                    D[k, self._index(i+2, j)] = 4.0 / dr2
                    D[k, self._index(i+3, j)] = -1.0 / dr2
                elif i == self.Nr - 1:
                    D[k, self._index(i, j)] = 2.0 / dr2
                    D[k, self._index(i-1, j)] = -5.0 / dr2
                    D[k, self._index(i-2, j)] = 4.0 / dr2
                    D[k, self._index(i-3, j)] = -1.0 / dr2
                else:
                    # Central: (f_{i+1} - 2f_i + f_{i-1}) / Δr²
                    D[k, self._index(i+1, j)] = 1.0 / dr2
                    D[k, self._index(i, j)] = -2.0 / dr2
                    D[k, self._index(i-1, j)] = 1.0 / dr2
        
        self._D_rr = D.tocsr()
        return self._D_rr
    
    def build_D_thetatheta(self) -> csr_matrix:
        """
        Build second derivative operator in θ direction: ∂²/∂θ².
        
        Uses second-order central differences with periodic wrapping.
        
        Returns
        -------
        D_thetatheta : csr_matrix
            Sparse matrix of shape (N, N).
        """
        if self._D_thetatheta is not None:
            return self._D_thetatheta
        
        D = lil_matrix((self.N, self.N))
        dtheta2 = self.dtheta**2
        
        for i in range(self.Nr):
            for j in range(self.Ntheta):
                k = self._index(i, j)
                jp1 = (j + 1) % self.Ntheta
                jm1 = (j - 1) % self.Ntheta
                
                D[k, self._index(i, jp1)] = 1.0 / dtheta2
                D[k, self._index(i, j)] = -2.0 / dtheta2
                D[k, self._index(i, jm1)] = 1.0 / dtheta2
        
        self._D_thetatheta = D.tocsr()
        return self._D_thetatheta
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def compute_gap_height(self, h_0_override: Optional[float] = None) -> np.ndarray:
        """
        Compute gap height on the mesh.
        
        Parameters
        ----------
        h_0_override : float, optional
            Override central clearance for transient analysis
        
        Returns
        -------
        H : ndarray
            Gap height array, shape (Nr, Ntheta) [m]
        """
        return self.geometry.gap_height(self.R, self.THETA, h_0_override)
    
    def compute_field_gradient(self, field_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradient of a scalar field using the derivative operators.
        
        Parameters
        ----------
        field_2d : ndarray
            Scalar field, shape (Nr, Ntheta)
        
        Returns
        -------
        df_dr : ndarray
            Radial derivative, shape (Nr, Ntheta)
        df_dtheta : ndarray
            Azimuthal derivative, shape (Nr, Ntheta)
        """
        field_1d = self.to_1d(field_2d)
        
        D_r = self.build_D_r()
        D_theta = self.build_D_theta()
        
        df_dr_1d = D_r @ field_1d
        df_dtheta_1d = D_theta @ field_1d
        
        return self.to_2d(df_dr_1d), self.to_2d(df_dtheta_1d)
    
    def integrate_field(self, field_2d: np.ndarray) -> float:
        """
        Integrate a scalar field over the domain using trapezoidal rule.
        
        ∫∫ f(r,θ) r dr dθ
        
        Parameters
        ----------
        field_2d : ndarray
            Scalar field, shape (Nr, Ntheta)
        
        Returns
        -------
        integral : float
            Domain integral
        """
        # Trapezoidal integration
        # ∫∫ f r dr dθ ≈ Σᵢⱼ f[i,j] * r[i] * dr * dθ
        integrand = field_2d * self.R
        return np.sum(integrand) * self.dr * self.dtheta
    
    def apply_dirichlet_bc(self, matrix: csr_matrix, rhs: np.ndarray,
                           bc_inner: float, bc_outer: float) -> Tuple[csr_matrix, np.ndarray]:
        """
        Apply Dirichlet boundary conditions at inner and outer radii.
        
        Parameters
        ----------
        matrix : csr_matrix
            System matrix
        rhs : ndarray
            Right-hand side vector
        bc_inner : float
            Value at r = R_in
        bc_outer : float
            Value at r = R_out
        
        Returns
        -------
        matrix_bc : csr_matrix
            Modified matrix with BCs
        rhs_bc : ndarray
            Modified RHS with BCs
        """
        matrix_bc = matrix.tolil()
        rhs_bc = rhs.copy()
        
        # Inner boundary (i = 0)
        for j in range(self.Ntheta):
            k = self._index(0, j)
            matrix_bc[k, :] = 0
            matrix_bc[k, k] = 1.0
            rhs_bc[k] = bc_inner
        
        # Outer boundary (i = Nr - 1)
        for j in range(self.Ntheta):
            k = self._index(self.Nr - 1, j)
            matrix_bc[k, :] = 0
            matrix_bc[k, k] = 1.0
            rhs_bc[k] = bc_outer
        
        return matrix_bc.tocsr(), rhs_bc
    
    def summary(self) -> str:
        """Return formatted summary of mesh."""
        lines = [
            "Polar Mesh",
            "=" * 40,
            f"Radial nodes Nr          : {self.Nr}",
            f"Azimuthal nodes Nθ       : {self.Ntheta}",
            f"Total nodes N            : {self.N}",
            "-" * 40,
            f"Radial spacing Δr        : {self.dr*1e3:.4f} mm",
            f"Azimuthal spacing Δθ     : {np.degrees(self.dtheta):.2f}°",
            f"Radial range             : [{self.r[0]*1e3:.3f}, {self.r[-1]*1e3:.3f}] mm",
        ]
        return "\n".join(lines)


# =============================================================================
# Verification Functions
# =============================================================================

def verify_derivative_accuracy(mesh: PolarMesh, 
                               test_function: str = 'polynomial') -> dict:
    """
    Verify derivative operator accuracy against known functions.
    
    Parameters
    ----------
    mesh : PolarMesh
        Mesh to test
    test_function : str
        Type of test function: 'polynomial', 'trig', 'mixed'
    
    Returns
    -------
    dict
        Verification results with errors for each derivative
    """
    r, theta = mesh.R, mesh.THETA
    
    if test_function == 'polynomial':
        # f(r, θ) = r² + r·cos(θ)
        f = r**2 + r * np.cos(theta)
        df_dr_exact = 2*r + np.cos(theta)
        df_dtheta_exact = -r * np.sin(theta)
    elif test_function == 'trig':
        # f(r, θ) = sin(2θ) · (r - R_in)
        R_in = mesh.geometry.R_in
        f = np.sin(2*theta) * (r - R_in)
        df_dr_exact = np.sin(2*theta)
        df_dtheta_exact = 2 * np.cos(2*theta) * (r - R_in)
    else:  # mixed
        # f(r, θ) = r·sin(θ) + cos(2θ)
        f = r * np.sin(theta) + np.cos(2*theta)
        df_dr_exact = np.sin(theta)
        df_dtheta_exact = r * np.cos(theta) - 2*np.sin(2*theta)
    
    # Compute numerical derivatives
    df_dr_num, df_dtheta_num = mesh.compute_field_gradient(f)
    
    # Compute errors (exclude boundaries for radial)
    error_r = np.abs(df_dr_num[1:-1, :] - df_dr_exact[1:-1, :])
    error_theta = np.abs(df_dtheta_num - df_dtheta_exact)
    
    return {
        'test_function': test_function,
        'max_error_r': np.max(error_r),
        'max_error_theta': np.max(error_theta),
        'mean_error_r': np.mean(error_r),
        'mean_error_theta': np.mean(error_theta),
        'expected_order': 2,
        'passed': np.max(error_r) < 0.01 and np.max(error_theta) < 0.01
    }


def convergence_study(geometry: SlipperGeometry,
                      Nr_list: list = None) -> dict:
    """
    Perform grid convergence study for derivative operators.
    
    Parameters
    ----------
    geometry : SlipperGeometry
        Geometry to use
    Nr_list : list, optional
        List of Nr values to test. Default: [10, 20, 40, 80]
    
    Returns
    -------
    dict
        Convergence data with errors vs grid spacing
    """
    if Nr_list is None:
        Nr_list = [10, 20, 40, 80]
    
    results = {'Nr': [], 'dr': [], 'error_r': [], 'error_theta': []}
    
    for Nr in Nr_list:
        Ntheta = int(2 * Nr)  # Maintain aspect ratio
        mesh = PolarMesh(geometry, Nr, Ntheta)
        
        r, theta = mesh.R, mesh.THETA
        
        # Use non-polynomial test function: f = sin(πr_scaled) * cos(2θ)
        L = geometry.R_out - geometry.R_in
        r_scaled = (r - geometry.R_in) / L  # 0 to 1
        
        f = np.sin(np.pi * r_scaled) * np.cos(2 * theta)
        df_dr_exact = (np.pi / L) * np.cos(np.pi * r_scaled) * np.cos(2 * theta)
        df_dtheta_exact = -2 * np.sin(np.pi * r_scaled) * np.sin(2 * theta)
        
        df_dr, df_dtheta = mesh.compute_field_gradient(f)
        
        # Error in interior (exclude boundary rows)
        error_r = np.max(np.abs(df_dr[2:-2, :] - df_dr_exact[2:-2, :]))
        error_theta = np.max(np.abs(df_dtheta[2:-2, :] - df_dtheta_exact[2:-2, :]))
        
        results['Nr'].append(Nr)
        results['dr'].append(mesh.dr)
        results['error_r'].append(error_r)
        results['error_theta'].append(error_theta)
    
    # Convert to arrays
    results['dr'] = np.array(results['dr'])
    results['error_r'] = np.array(results['error_r'])
    results['error_theta'] = np.array(results['error_theta'])
    
    if len(Nr_list) >= 2:
        log_dr = np.log(results['dr'])
        log_err_r = np.log(np.maximum(results['error_r'], 1e-15))
        log_err_theta = np.log(np.maximum(results['error_theta'], 1e-15))
        
        # Linear regression for order
        order_r = np.polyfit(log_dr, log_err_r, 1)[0]
        order_theta = np.polyfit(log_dr, log_err_theta, 1)[0]
        
        results['order_r'] = order_r
        results['order_theta'] = order_theta
    
    return results
