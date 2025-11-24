import numpy as np

import mdtraj as md

from joblib import Parallel, delayed, cpu_count


class Wrappers:
    """
    Collection of wrappers around mdtraj functions for trajectory analysis.
    This way I can easily tweak parameters or add logging or pre/post-processing
    of the inputs in one place. Each method is implemented as a static method
    for easy access without instantiating the class.
    """

    @staticmethod
    def rmsd(
        traj1: md.Trajectory,
        traj2: md.Trajectory,
        frame: int = 0,
        atom_indices: np.ndarray | None = None,
        ref_atom_indices: np.ndarray | None = None,
        parallel: bool = True,
        precentered: bool = True,
        superpose: bool = True,
    ) -> np.ndarray:
        """Simple wrapper"""
        return md.rmsd(
            traj1,
            traj2,
            frame,
            atom_indices,
            ref_atom_indices,
            parallel,
            precentered,
            superpose,
        )

    @staticmethod
    def rmsf(
        traj1: md.Trajectory,
        traj2: md.Trajectory | None = None,
        frame: int = 0,
        atom_indices: np.ndarray | None = None,
        ref_atom_indices: np.ndarray | None = None,
        parallel: bool = True,
        precentered: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Simple wrapper"""
        rmsf_vals = md.rmsf(
            traj1,
            traj2,
            frame,
            atom_indices,
            ref_atom_indices,
            parallel,
            precentered,
        )
        return rmsf_vals

    @staticmethod
    def compute_rg(
        traj: md.Trajectory,
        masses: np.ndarray | None = None,
    ) -> np.ndarray:
        """Simple wrapper"""
        return md.compute_rg(
            traj,
            masses,
        )

    @staticmethod
    def compute_dssp(
        traj: md.Trajectory,
        simplified: bool = False,
        n_jobs: int = -1,
    ) -> np.ndarray:
        """
        Compute DSSP, optionally in parallel across frames.

        Parameters
        ----------
        - traj : md.Trajectory
            A trajectory
        - simplified : bool, default=True
            Use the simplified 3-category assignment scheme.
        - n_jobs : int, default=1
            Number of CPU cores to use.
            -  1: Serial execution (no parallelization).
            - -1: Use all available cores.

        Returns
        -------
        - dssp_array : np.ndarray
            Array of DSSP assignments for each frame and residue.
        """
        if n_jobs == 1 or traj.n_frames < n_jobs:
            return md.compute_dssp(
                traj,
                simplified,
            )

        if n_jobs == -1:
            n_jobs = cpu_count()

        def _worker(traj_chunk: md.Trajectory) -> np.ndarray:
            writable_chunk = md.Trajectory(
                xyz=traj_chunk.xyz.copy(),
                topology=traj_chunk.topology
            )
            return md.compute_dssp(writable_chunk, simplified)

        frame_indices = np.array_split(range(traj.n_frames), n_jobs)

        traj_chunks = [
            traj[indices]
            for indices in frame_indices if len(indices) > 0
        ]

        results = Parallel(n_jobs=n_jobs)(
            delayed(_worker)(chunk)
            for chunk in traj_chunks
        )

        return np.concatenate(results, axis=0)

    @staticmethod
    def compute_gyration_tensor(
        traj: md.Trajectory,
    ) -> np.ndarray:
        """Simple wrapper"""
        return md.compute_gyration_tensor(
            traj
        )

    @staticmethod
    def principal_moments(
        traj: md.Trajectory,
    ) -> np.ndarray:
        """Simple wrapper"""
        return md.principal_moments(
            traj
        )

    @staticmethod
    def asphericity(
        traj: md.Trajectory,
    ) -> np.ndarray:
        """Simple wrapper"""
        return md.asphericity(
            traj
        )

    @staticmethod
    def acylindricity(
        traj: md.Trajectory,
    ) -> np.ndarray:
        """Simple wrapper"""
        return md.acylindricity(
            traj
        )

    @staticmethod
    def relative_shape_anisotropy(
        traj: md.Trajectory,
    ) -> np.ndarray:
        """Simple wrapper"""
        return md.relative_shape_antisotropy(
            traj
        )

    @staticmethod
    def compute_phi(
        traj: md.Trajectory,
        periodic: bool = True,
        opt: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simple wrapper"""
        return md.compute_phi(
            traj,
            periodic,
            opt,
        )

    @staticmethod
    def compute_psi(
        traj: md.Trajectory,
        periodic: bool = True,
        opt: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simple wrapper"""
        return md.compute_psi(
            traj,
            periodic,
            opt,
        )

    @staticmethod
    def compute_omega(
        traj: md.Trajectory,
        periodic: bool = True,
        opt: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simple wrapper"""
        return md.compute_omega(
            traj,
            periodic,
            opt,
        )

    @staticmethod
    def compute_chi1(
        traj: md.Trajectory,
        periodic: bool = True,
        opt: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simple wrapper"""
        return md.compute_chi1(
            traj,
            periodic,
            opt,
        )
