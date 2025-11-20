import numpy as np

import mdtraj as md

from scipy.spatial.distance import pdist, squareform

from pyPTV.mdtraj_wrappers import Wrappers as mdw


class FeatureExtractors:
    """
    Collection of feature extraction methods for molecular dynamics trajectories.
    Each method is implemented as a static method for easy access without
    instantiating the class.
    """

    @staticmethod
    def compute_rmsd_1D(
        traj: md.Trajectory,
        refs: dict[str, md.Trajectory] | None = None,
        selection: str = "protein",
        rmsd_kwargs: dict | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Compute RMSD 1D features against reference structures.

        Parameters
        ----------
        - traj : md.Trajectory
            Input trajectory.
        - refs : dict[str, md.Trajectory] | None
            Dictionary of reference structures with labels as keys. If None,
            the first frame of traj is used as the only reference.
            Default is None.
        - selection : str
            Atom selection string for RMSD calculation.
            Default is "protein".
        - rmsd_kwargs : dict | None
            Additional keyword arguments to pass to the RMSD calculation.
            Default is None.

        Returns
        -------
        - rmsds : dict[str, np.ndarray]
            Dictionary of RMSD 1D feature arrays with labels as keys.
        """
        # If no reference structures are provided, use the first frame as
        # the reference.
        if not refs:
            refs = {
                "first frame": traj[0],
            }

        # Apply atom selection
        atom_indices = traj.topology.select(selection)
        n_atoms_selected = atom_indices.shape[0]
        if n_atoms_selected == 0:
            raise ValueError(
                f"Warning: Selection '{selection}' yielded 0 atoms."
                " Check selection string."
            )

        # Compute RMSD features
        rmsds = {
            key: mdw.rmsd(
                traj,
                ref,
                atom_indices=atom_indices,
                **rmsd_kwargs if rmsd_kwargs else None,
            )
            for key, ref in refs.items()
        }

        return rmsds

    @staticmethod
    def compute_rmsd_2D(
        traj: md.Trajectory,
        selection: str = "protein",
        distance_type: str = "euclidean",
    ) -> np.ndarray:
        """
        Compute pairwise dRMSD 2D feature matrix for the trajectory.

        dRMSD is used instead of cRMSD for performance reasons, as it avoids
        the need for structural alignments and the results are analogous.
        The dRMSD is defined as:
        dRMSD(X, Y) = sqrt( (1/k) * sum_{i<j} (d_ij(X) - d_ij(Y))^2 )
        where k = N(N-1)/2 is the number of pairwise distances for N atoms.

        Parameters
        ----------
        - traj : md.Trajectory
            Input trajectory.
        - selection : str
            Atom selection string for distance calculation.
            Default is "protein".
        - distance_type : str
            Distance calculation type for pdist.
            Default is "euclidean".

        Returns
        -------
        - drmsd_matrix : np.ndarray
            Pairwise dRMSD 2D feature matrix of shape (n_frames, n_frames).
        """
        # Apply atom selection
        atom_indices = traj.topology.select(selection)
        n_atoms_selected = atom_indices.shape[0]
        if n_atoms_selected < 2:
            raise ValueError(
                f"Selection '{selection}' resulted in {n_atoms_selected} atoms."
                " Need at least 2 atoms for pairwise distances."
            )

        # k = number of unique pairwise distances
        k = n_atoms_selected * (n_atoms_selected - 1) / 2

        # Get coordinates for selected atoms across all frames
        coords = traj.xyz[:, atom_indices, :]

        # Calculate all condensed distance matrices (one per frame)
        # all_pdists will be a list of 1D arrays
        all_pdists = [
            pdist(frame_coords, distance_type)
            for frame_coords in coords
        ]

        # Convert list of 1D arrays to a single 2D (n_frames, k_distances)
        # matrix
        all_pdists_matrix = np.array(all_pdists)

        # Calculate pairwise distances between the distance vectors
        # We want the RMSD, which is the Euclidean distance divided by sqrt(k)
        # pdist calculates E(Vi, Vj) = sqrt( sum( (Vil - Vjl)^2 ) )
        drmsd_condensed = pdist(all_pdists_matrix, "euclidean")

        # Scale by 1 / sqrt(k) to get the final dRMSD
        drmsd_condensed_scaled = drmsd_condensed / np.sqrt(k)

        # Convert to square matrix
        return squareform(drmsd_condensed_scaled)

    @staticmethod
    def compute_rmsf(
        traj: md.Trajectory,
        selection: str = "protein",
        rmsf_kwargs: dict | None = None,
    ) -> np.ndarray:
        """
        Compute RMSF block-averaged features for the trajectory.

        Parameters
        ----------
        - traj : md.Trajectory
            Input trajectory.
        - selection : str
            Atom selection string for RMSF calculation.
            Default is "protein".
        - rmsf_kwargs : dict | None
            Additional keyword arguments to pass to the RMSF calculation.
            Default is None.

        Returns
        -------
        - rmsfs : np.ndarray
            RMSF feature array of shape (n_frames).
        """
        # Apply atom selection
        atom_indices = traj.topology.select(selection)
        n_atoms_selected = atom_indices.shape[0]
        if n_atoms_selected == 0:
            raise ValueError(
                f"Warning: Selection '{selection}' yielded 0 atoms."
                " Check selection string."
            )

        rmsf, rmsf_error = mdw.rmsf(
            traj,
            atom_indices=atom_indices,
            **rmsf_kwargs if rmsf_kwargs else None,
        )
        return rmsf, rmsf_error

    @staticmethod
    def compute_rg(
        traj: md.Trajectory,
        masses: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute radius of gyration (Rg) features for the trajectory.

        Parameters
        ----------
        - traj : md.Trajectory
            Input trajectory.
        - masses : np.ndarray | None
            Atomic masses for Rg calculation. If None, all atoms are treated
            equally. Default is None.

        Returns
        -------
        - rgs : np.ndarray
            Rg feature array of shape (n_frames).
        """
        return mdw.compute_rg(
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
        Compute DSSP secondary structure features for the trajectory.

        Parameters
        ----------
        - traj : md.Trajectory
            Input trajectory.
        - simplified : bool
            Whether to use simplified DSSP classification.
            Default is False.
        - n_jobs : int
            Number of parallel jobs for DSSP calculation.
            Default is -1 (use all available cores).

        Returns
        -------
        - dssp_counts : np.ndarray
            Array of shape (n_frames, n_ss_types) with counts of each
            secondary structure type per frame.
        """
        if simplified:
            ss_map = {
                "H": "Helix",
                "E": "Strand",
                "C": "Coil",
            }
        else:
            ss_map = {
                "H": "Alpha helix",
                "B": "Residue in isolated beta-bridge",
                "E": "Extended strand, participates in beta ladder",
                "G": "3-helix (3/10 helix)",
                "I": "5 helix (pi helix)",
                "T": "Hydrogen bonded turn",
                "S": "Bend",
                " ": "Loops and irregular elements",
            }

        # Compute DSSP assignments
        dssp = mdw.compute_dssp(
            traj,
            simplified=simplified,
            n_jobs=n_jobs,
        )

        # Define the columns (SS types) and array dimensions
        ss_types = list(ss_map.keys())
        n_frames = dssp.shape[0]
        n_ss_types = len(ss_types)

        dssp_counts = np.zeros(
            (n_frames, n_ss_types),
            dtype=int
        )

        # Populate the array (vectorized)
        for i, ss_type in enumerate(ss_types):
            # Sum occurrences of this ss_type for each frame (axis=1)
            counts_per_frame = np.sum(dssp == ss_type, axis=1)
            dssp_counts[:, i] = counts_per_frame
    
        return dssp_counts

    @staticmethod
    def compute_shape_metrics(
        traj: md.Trajectory,
        selection: str = "protein",
    ) -> dict[str, np.ndarray]:
        """
        Compute shape metrics for the trajectory.

        Parameters
        ----------
        - traj : md.Trajectory
            Input trajectory.
        - selection : str
            Atom selection string for shape metric calculation.
            Default is "protein".

        Returns
        -------
        - shape_metrics : dict[str, np.ndarray]
            Dictionary of shape metric arrays with labels as keys.
        """

        # Apply atom selection
        atom_indices = traj.topology.select(selection)
        n_atoms_selected = atom_indices.shape[0]
        if n_atoms_selected == 0:
            raise ValueError(
                f"Warning: Selection '{selection}' yielded 0 atoms."
                " Check selection string."
            )
        traj_slice = traj.atom_slice(atom_indices)

        # Gyration tensor
        gyration_tensor = mdw.compute_gyration_tensor(traj_slice)

        # Principal moments
        principal_moments = mdw.principal_moments(traj_slice)

        # Asphericity
        asphericity = mdw.asphericity(traj_slice)

        # Acylindricity
        acylindricity = mdw.acylindricity(traj_slice)

        # Relative shape anisotropy
        rel_shape_anisotropy = mdw.relative_shape_anisotropy(traj_slice)

        return {
            "gyration tensor": gyration_tensor,
            "principal moments": principal_moments,
            "asphericity": asphericity,
            "acylindricity": acylindricity,
            "relative shape anisotropy": rel_shape_anisotropy,
        }

    @staticmethod
    def compute_dihedrals(
        traj: md.Trajectory,
        angle_types: list[str] = ["phi", "psi", "chi1", "omega"],
    ):
        """
        Compute backbone and sidechain dihedral angles for the trajectory.

        Parameters
        ----------
        - traj : md.Trajectory
            Input trajectory.
        - angle_types : list[str]
            List of dihedral angle types to compute.
            Supported types: "phi", "psi", "chi1", "omega".
            Default is ["phi", "psi", "chi1", "omega"].

        Returns
        -------
        - angles_per_frame : dict[str, np.ndarray]
            Dictionary of dihedral angle arrays with labels as keys.
        - raw_per_residue_data : list[tuple]
            List of tuples containing (angle_label, angle_data, residue_indices)
            for each computed dihedral type.
        - full_dihedral_matrix : np.ndarray
            Concatenated dihedral angle data matrix of shape
            (n_frames, n_total_dihedrals).
        - transformed_matrix : np.ndarray
            Transformed dihedral angle data matrix of shape
            (n_frames, 2 * n_total_dihedrals) with (sin, cos) pairs.
        - nans_replaced_with_zero : bool
            Whether any NaN values were found and replaced with zeros.
        """

        # Clean up names
        dihedrals = {
            t.strip().lower() for t in angle_types
        }

        angles_per_frame = {}
        raw_per_residue_data = []
        all_dihedral_data = []

        # Compute dihedrals
        if "phi" in dihedrals:
            idx, data = mdw.compute_phi(traj)
            if data.shape[1] > 0:
                angles_per_frame["φ"] = np.mean(data, axis=1)
                raw_per_residue_data.append(("φ", data, idx))
                all_dihedral_data.append(data)
        if "psi" in dihedrals:
            idx, data = mdw.compute_psi(traj)
            if data.shape[1] > 0:
                angles_per_frame["ψ"] = np.mean(data, axis=1)
                raw_per_residue_data.append(("ψ", data, idx))
                all_dihedral_data.append(data)
        if "chi1" in dihedrals:
            idx, data = mdw.compute_chi1(traj)
            if data.shape[1] > 0:
                angles_per_frame["χ1"] = np.mean(data, axis=1)
                raw_per_residue_data.append(("χ1", data, idx))
                all_dihedral_data.append(data)
        if "omega" in dihedrals:
            idx, data = mdw.compute_omega(traj)
            if data.shape[1] > 0:
                angles_per_frame["ω"] = np.mean(data, axis=1)
                raw_per_residue_data.append(("ω", data, idx))
                all_dihedral_data.append(data)

        # Concatenate all dihedral data into a single matrix
        full_dihedral_matrix = np.concatenate(
            all_dihedral_data,
            axis=1
        )

        # Handle any NaNs that may arise from angle calculation
        nans_replaced_with_zero = False
        if np.isnan(full_dihedral_matrix).any():
            nans_replaced_with_zero = True
            full_dihedral_matrix = np.nan_to_num(full_dihedral_matrix)

        # Transform circular angle data (in radians) into (sin, cos) pairs
        sin_data = np.sin(full_dihedral_matrix)
        cos_data = np.cos(full_dihedral_matrix)
        transformed_matrix = np.concatenate(
            [sin_data, cos_data],
            axis=1
        )

        return (
            angles_per_frame,
            raw_per_residue_data,
            full_dihedral_matrix,
            transformed_matrix,
            nans_replaced_with_zero,
        )