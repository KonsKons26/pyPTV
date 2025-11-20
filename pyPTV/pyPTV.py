from itertools import groupby
from random import choice, sample

import numpy as np

import pandas as pd

from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.signal import savgol_filter

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import umap

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.transforms import blended_transform_factory
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import seaborn as sns

import k3d

import mdtraj as md

from pyPTV.mdtraj_wrappers import Wrappers


XKCD_COLORS = [
    f"xkcd:{c}" for c in [
        "royal blue", "bright blue", "sky blue", "cyan", "baby blue",
        "bright red", "coral", "brick", "red orange", "salmon",
        "olive green", "neon green", "mint green", "aqua green",
        "golden yellow", "lemon", "sandy",
        "electric pink", "purpleish pink",
    ]
]


class PTV:
    """
    Class for visualizing molecular dynamics trajectories.
    """
    def __init__(
        self,
        name: str,
        traj: md.Trajectory,
        step: tuple[float, str] | None = None,
        save_calculations: bool = False,
    ) -> None:
        """
        Initializes the TrajectoryAnalyzer.

        Parameters
        ----------
        - name: str
            A descriptive name for the trajectory.
        - traj: md.Trajectory
            The md.Trajectory object to be analyzed.
        - step: tuple[float, str], optional
            How many time units per trajectory frame and what the unit is. If
            None, use frames instead of time.
            Default = None
        - save_calculations: bool, optional
            If True, saves the results of calculations as attributes of the
            class instance for later access.
            Default = False
        """
        self.name = name
        if not step:
            step = (1.0, "frames")
        self.step = step[0]
        self.time_units = step[1]
        self.save_calculations = save_calculations

        self.traj = traj
        self.n_frames = self.traj.n_frames
        self.time = np.arange(len(self.traj)) / self.step 

        # Initialize placeholders
        self.rmsds = None
        self.drmsd_matrix = None
        self.rmsf = None
        self.rog = None
        self.dssp = None
        self.gyration_tensor = None
        self.principal_moments = None
        self.asphericity = None
        self.acylindricity = None
        self.rel_shape_anisotropy = None
        self.dihedral_angles = None
        self.dihedral_dim_red = None
        self.dihedral_dim_red_smooth = None

    def plot_rmsd(
        self,
        pairwise: bool = False,
        params_1D: dict = {},
        params_2D: dict = {},
    ) -> None:
        """
        Plots either a 1D cRMSD of selected references against the trajectory,
        or a 2D pairwise dRMSD of the trajectory against itself.

        The 1D cRMSD is calculated using the rmsd function from mdtraj. In the
        2D case the dRMSD is used instead for performance reasons, as it avoids
        the need for structural alignments.
        The dRMSD is defined as:
        dRMSD(X, Y) = sqrt( (1/k) * sum_{i<j} (d_ij(X) - d_ij(Y))^2 )
        where k = N(N-1)/2 is the number of pairwise distances for N atoms.

        Parameters
        ----------
        - pairwise: bool, optional
            If True, enters the 2D dRMSD mode, otherwise enters the 1D cRMSD
            mode.
            Default = False
        - params_1D: dict, optional
            Parameters to be passed to the _plot_1D_RMSD internal method.
                - figsize: tuple[int, int], optional
                    Size of the figure to be plotted.
                    Default = (18, 8)
                - refs: dict[str, md.Trajectory] | None, optional
                    A dictionary of reference structures to calculate cRMSD
                    against. Keys are labels, values are md.Trajectory objects.
                    If None, defaults to the first frame of the trajectory.
                    Default = None
                - selection: str, optional
                    Atom selection string for calculating RMSD.
                    Default = "backbone"
                - conv_window: int, optional
                    Window size for the moving average smoothing.
                    Default = 50
        - params_2D: dict, optional
            Parameters to be passed to the _plot_2D_RMSD internal method.
                - figsize: tuple[int, int], optional
                    Size of the figure to be plotted.
                    Default = (8, 8)
                - cmap: str, optional
                    Colormap to be used for the heatmap.
                    Default = "plasma"
                - selection: str, optional
                    Atom selection string for calculating RMSD.
                    Default = "backbone"
        """
        if pairwise:
            self._plot_2D_RMSD(
                figsize=params_2D.get("figsize", (10, 10)),
                cmap=params_2D.get("cmap", "plasma"),
                selection=params_2D.get("selection", "backbone"),
            )
        else:
            self._plot_1D_RMSD(
                figsize=params_1D.get("figsize", (18, 8)),
                refs=params_1D.get("refs", None),
                selection=params_1D.get("selection", "backbone"),
                conv_window=params_1D.get("conv_window", 50),
            )

    def plot_rmsf(
        self,
        figsize: tuple[int, int] = (18, 8),
        selection: str = "protein",
        residue_label_rotation: int = 90,
        n_blocks: int = 1,
        error_metric: str = "sem",
    ) -> None:
        """
        Plots the root mean square fluctuation (RMSF) per atom.
        If n_blocks > 1, performs block averaging and plots error.

        Parameters
        ----------
        - figsize: tuple[int, int], optional
            Size of the figure to be plotted.
            Default = (18, 8)
        - selection: str, optional
            Atom selection string for calculating RMSF.
            Default = "protein"
        - residue_label_rotation: int, optional
            Rotation angle for residue labels on the x-axis.
            Default = 90
        - n_blocks: int, optional
            Number of blocks for error calculation. If 1, no
            error is calculated or plotted. Default = 5
        - error_metric: str, optional
            Metric for error bars: 'sem' (Standard Error of Mean)
            or 'std' (Standard Deviation). Default = "sem"
        """
        atom_indices = self.traj.topology.select(selection)
        if len(atom_indices) == 0:
            print(f"Warning: Selection '{selection}' yielded 0 atoms.")
            return

        # Calculate RMSF using the modified wrapper
        # The wrapper now handles block averaging internally.

        rmsf, rmsf_error = Wrappers.rmsf(
            self.traj,
            atom_indices=atom_indices,
            ref_atom_indices=atom_indices,
            n_blocks=n_blocks,
            error_metric=error_metric,
            precentered=False 
        )

        if self.save_calculations:
            self.rmsf = rmsf
            if rmsf_error is not None:
                self.rmsf_error = rmsf_error

        # Plot
        selected_atoms = [
            self.traj.topology.atom(i)
            for i in atom_indices
        ]
        residue_labels = [
            f"{atom.residue.name}{atom.residue.resSeq}"
            for atom in selected_atoms
        ]
        residue_groups = []
        current_index = 0
        for label, group in groupby(residue_labels):
            group_size = len(list(group))
            start_index = current_index
            end_index = current_index + group_size - 1
            residue_groups.append({
                "label": label,
                "start_edge": start_index - 0.5,
                "end_edge": end_index + 0.5,
                "center": (start_index + end_index) / 2.0
            })
            current_index += group_size

        sns.set_theme(style="whitegrid", context="talk")
        _, ax = plt.subplots(figsize=figsize)
        
        # --- Modified Plotting Commands (Line + Error) ---
        x_values = np.arange(len(rmsf))
        plot_color = choice(XKCD_COLORS)

        # Plot the mean RMSF line
        ax.plot(
            x_values,
            rmsf,
            color=plot_color,
            linewidth=2,
            label="Mean RMSF" if rmsf_error is not None else "RMSF"
        )
        
        # Plot the shaded error region if it was calculated
        if rmsf_error is not None:
            error_label = f"± 1 {error_metric.upper()}"
            ax.fill_between(
                x_values,
                rmsf - rmsf_error,
                rmsf + rmsf_error,
                color=plot_color,
                alpha=0.3,
                label=error_label
            )
            ax.legend()

        ax.set_ylabel("RMSF (nm)")
        title = f"RMSF per Atom ({selection})"
        if n_blocks > 1:
            title += f" ({n_blocks}-Block Averaged)"
        title += f"\n{self.name.upper() if self.name else ''}"
        ax.set_title(title)

        ax.set_xticks([])

        # Custom x-axis drawing
        if residue_groups:
            ax.set_xlim(
                residue_groups[0]["start_edge"],
                residue_groups[-1]["end_edge"]
            )
        
        y_line = -0.08
        y_tick_height = 0.02
        y_text = -0.12
        transform = blended_transform_factory(ax.transData, ax.transAxes)

        for group in residue_groups:
            start = group["start_edge"]
            end = group["end_edge"]
            center = group["center"]
            ax.plot(
                [start, start, end, end],
                [y_line + y_tick_height, y_line, y_line, y_line + y_tick_height],
                color='black',
                linewidth=1,
                transform=transform,
                clip_on=False,
            )
            ax.text(
                center,
                y_text,
                group["label"],
                ha="center",
                va="top",
                rotation=residue_label_rotation,
                transform=transform,
                clip_on=False,
                fontsize="small",
            )
            
        plt.tight_layout()
        plt.show()

    def plot_rog(
        self,
        figsize: tuple[int, int] = (18, 8),
        conv_window: int = 50,
    ) -> None:
        """
        Plots the radius of gyration (RoG) over time.

        Parameters
        ----------
        - figsize: tuple[int, int], optional
            Size of the figure to be plotted.
            Default = (18, 8)
        - conv_window: int, optional
            Window size for the moving average smoothing.
            Default = 50
        """
        rog = Wrappers.compute_rg(self.traj)

        if self.save_calculations:
            self.rog = rog

        # Plot
        color = choice(XKCD_COLORS)
        sns.set_theme(style="whitegrid", context="talk")
        plt.figure(figsize=figsize)
        sns.scatterplot(
            x=self.time,
            y=rog,
            s=35,
            color=color,
            alpha=0.2,
            label=None,
        )
        sns.lineplot(
            x=self.time[conv_window: -conv_window],
            y=np.convolve(
                rog,
                np.ones(conv_window)/conv_window,
                mode="same",
            )[conv_window: -conv_window],
            color=color,
            label="Radius of Gyration",
        )
        plt.xlabel(f"Time ({self.time_units})")
        plt.ylabel("Radius of Gyration (nm)")
        plt.title(
            "Radius of Gyration\n"
            f"{self.name.upper() if self.name else ''}"
        )
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_dssp(
        self,
        figsize: tuple[int, int] = (18, 8),
        conv_window: int = 50,
        simplified: bool = False,
    ) -> None:
        """
        Plot the DSSP assignment over time.

        Parameters
        ----------
        - figsize: tuple[int, int], optional
            Size of the figure to be plotted.
            Default = (18, 8)
        - conv_window: int, optional
            Window size for the moving average smoothing.
            Default = 50
        - simplified: bool, optional
            If True, use the simplified 3-state DSSP assignment (H, E, C).
            If False, use the full 8-state assignment.
            Default = False
        """
        if simplified:
            ss_map = {
                "H": "Helix",
                "E": "Strand",
                "C": "Coil",
            }
            colors = {
                "H": "xkcd:royal blue",
                "E": "xkcd:bright red",
                "C": "xkcd:grass green",
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
            colors = {
                "H": "xkcd:royal blue",
                "B": "xkcd:scarlet",
                "E": "xkcd:bright red",
                "G": "xkcd:blue violet",
                "I": "xkcd:electric blue",
                "T": "xkcd:grass green",
                "S": "xkcd:mint green",
                " ": "xkcd:puke green",
            }

        # Compute DSSP assignments
        dssp = Wrappers.compute_dssp(self.traj, simplified=simplified)

        # Define the columns (SS types) and array dimensions
        ss_types = list(ss_map.keys())
        n_frames = dssp.shape[0]
        n_ss_types = len(ss_types)

        dssp_counts = np.zeros((n_frames, n_ss_types), dtype=int)

        # Populate the array (vectorized)
        for i, ss_type in enumerate(ss_types):
            # Sum occurrences of this ss_type for each frame (axis=1)
            counts_per_frame = np.sum(dssp == ss_type, axis=1)
            dssp_counts[:, i] = counts_per_frame

        # Plot
        sns.set_theme(style="whitegrid", context="talk")
        plt.figure(figsize=figsize)

        for i, ss_type in enumerate(ss_types):
            y_data = dssp_counts[:, i]

            # Don't plot if this SS type never appears
            if np.all(y_data == 0):
                continue

            sns.scatterplot(
                x=self.time,
                y=y_data,
                alpha=0.2,
                s=35,
                label=None,
                color=colors[ss_type],
            )

            sns.lineplot(
                x=self.time[conv_window: -conv_window],
                y=np.convolve(
                    y_data,
                    np.ones(conv_window)/conv_window,
                    mode="same",
                )[conv_window: -conv_window],
                color=colors[ss_type],
                label=ss_map[ss_type],
            )

        plt.xlabel(f"Time ({self.time_units})")
        plt.ylabel("Number of Residues")
        plt.title(
            "DSSP Secondary Structure Assignment Over Time\n"
            f"{self.name.upper() if self.name else ''}"
        )
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.show()

    def plot_shape_metrics(
        self,
        figsize: tuple[int, int] = (18, 8),
        selection: str = "protein",
        conv_window: int = 50,
    ) -> None:
        """
        Plots the following shape metrics over time:
        - The trace of the gyration tensor
        - The three principal moments (the eigenvalues of the gyration tensor)
        - The asphericity
        - The acylindricity
        - The relative shape anisotropy

        These metrics should follow the same trend for a given trajectory, but
        they can have different magnitutes.

        Parameters
        ----------
        - figsize: tuple[int, int], optional
            Size of the figure to be plotted.
            Default = (18, 8)
        - selection: str, optional
            Atom selection string for calculating the metrics.
            Default = "protein"
        - conv_window: int, optional
            Window size for the moving average smoothing.
            Default = 50
        """
        atom_indices = self.traj.topology.select(selection)
        traj_slice = self.traj.atom_slice(atom_indices)

        # Gyration tensor
        gyration_tensor = Wrappers.compute_gyration_tensor(traj_slice)

        # Principal moments
        principal_moments = Wrappers.principal_moments(traj_slice)

        # Asphericity
        asphericity = Wrappers.asphericity(traj_slice)

        # Acylindricity
        acylindricity = Wrappers.acylindricity(traj_slice)

        # Relative shape anisotropy
        rel_shape_anisotropy = Wrappers.relative_shape_anisotropy(traj_slice)

        if self.save_calculations:
            self.gyration_tensor = gyration_tensor
            self.principal_moments = principal_moments
            self.asphericity = asphericity
            self.acylindricity = acylindricity
            self.rel_shape_anisotropy = rel_shape_anisotropy

        # Plot
        sns.set_theme(style="whitegrid", context="talk")
        plt.figure(figsize=figsize)

        for metric_name, data, color in [
            (
                "Gyration Tensor Trace",
                np.trace(gyration_tensor, axis1=1, axis2=2),
                "xkcd:pastel blue",
            ),
            (
                "Principal Moments",
                principal_moments,
                [
                    "xkcd:pinkish red",
                    "xkcd:deep pink",
                    "xkcd:blood red",
                ],
            ),
            (
                "Asphericity",
                asphericity,
                "xkcd:dull green",
            ),
            (
                "Acylindricity",
                acylindricity,
                "xkcd:sea blue",
            ),
            (
                "Relative Shape Anisotropy",
                rel_shape_anisotropy,
                "xkcd:dark violet",
            ),
        ]:
            if metric_name == "Principal Moments":
                for l in range(3):
                    sns.scatterplot(
                        x=self.time,
                        y=data[:, l],
                        alpha=0.2,
                        s=35,
                        label=None,
                        color=color[l],
                    )
                    sns.lineplot(
                        x=self.time[conv_window: -conv_window],
                        y=np.convolve(
                            data[:, l],
                            np.ones(conv_window)/conv_window,
                            mode="same",
                        )[conv_window: -conv_window],
                        label=f"{metric_name} (λ{l + 1})",
                        color=color[l],
                    )
            else:
                sns.scatterplot(
                    x=self.time,
                    y=data,
                    alpha=0.2,
                    s=35,
                    label=None,
                    color=color
                )
                sns.lineplot(
                    x=self.time[conv_window: -conv_window],
                    y=np.convolve(
                        data,
                        np.ones(conv_window)/conv_window,
                        mode="same"
                    )[conv_window: -conv_window],
                    label=metric_name,
                    color=color
                )

        plt.xlabel(f"Time ({self.time_units})")
        plt.ylabel("Metric Value")
        plt.title(
            "Shape Metrics Over Time\n"
            f"{self.name.upper() if self.name else ''}"
        )
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.show()

    def plot_dihedrals(
        self,
        figsize: tuple[int, int] = (18, 8),
        angle_types: list[str] = ["phi", "psi", "omega", "chi1"],
        conv_window: int = 50,
    ) -> None:
            """
            Computes and plots dihedral angles in two ways:
            1. Per-frame (residue-averaged) angles over time.
            2. Per-residue (full distribution) violin plots.

            Parameters
            ----------
            - figsize: tuple[int, int], optional
                Size of the figure to be plotted.
                Default = (18, 8)
            - angle_types: list[str], optional
                List of dihedral angle types to compute. Supported types: "phi",
                "psi", "omega", "chi1".
                Default = ["phi", "psi", "omega", "chi1"]
            - conv_window: int, optional
                Window size for the moving average smoothing.
                Default = 50
            """
            # --- 1. Data Computation ---
            processed_angle_types = {t.strip().lower() for t in angle_types}

            angles_per_frame = {}
            raw_per_residue_data = []

            if "phi" in processed_angle_types:
                indices, phi_data = Wrappers.compute_phi(self.traj)
                if phi_data.shape[1] > 0:
                    angles_per_frame["φ"] = np.mean(phi_data, axis=1)
                    raw_per_residue_data.append(("φ", phi_data, indices))
                else:
                    print("Info: No 'phi' angles found in trajectory.")
            if "psi" in processed_angle_types:
                indices, psi_data = Wrappers.compute_psi(self.traj)
                if psi_data.shape[1] > 0:
                    angles_per_frame["ψ"] = np.mean(psi_data, axis=1)
                    raw_per_residue_data.append(("ψ", psi_data, indices))
                else:
                    print("Info: No 'psi' angles found in trajectory.")
            if "omega" in processed_angle_types:
                indices, omega_data = Wrappers.compute_omega(self.traj)
                if omega_data.shape[1] > 0:
                    angles_per_frame["ω"] = np.mean(omega_data, axis=1)
                    raw_per_residue_data.append(("ω", omega_data, indices))
                else:
                    print("Info: No 'omega' angles found in trajectory.")
            if "chi1" in processed_angle_types:
                indices, chi1_data = Wrappers.compute_chi1(self.traj)
                if chi1_data.shape[1] > 0:
                    angles_per_frame["χ1"] = np.mean(chi1_data, axis=1)
                    raw_per_residue_data.append(("χ1", chi1_data, indices))
                else:
                    print("Info: No 'chi1' angles found in trajectory.")

            if self.save_calculations:
                self.dihedral_angles = {
                    "per_frame": angles_per_frame,
                    "raw_per_residue": raw_per_residue_data
                }

            # Check if any data was found before plotting
            if not angles_per_frame:
                print("No 'per frame' data to plot. Exiting.")
                return

            # One color scheme for all plots
            colors = dict(
                zip(
                    angles_per_frame.keys(),
                    sample(XKCD_COLORS, k=len(angles_per_frame))
                )
            )

            # Plot 1, per-frame angles
            sns.set_theme(style="whitegrid", context="talk")
            plt.figure(figsize=figsize)

            for angle_name, angle_data in angles_per_frame.items():
                current_color = colors[angle_name]
                sns.scatterplot(
                    x=self.time, y=angle_data, alpha=0.2, s=35,
                    label=None, color=current_color,
                )
                convolved_data = np.convolve(
                    angle_data, np.ones(conv_window) / conv_window,
                    mode="same",
                )[conv_window: -conv_window]
                sns.lineplot(
                    x=self.time[conv_window: -conv_window], y=convolved_data,
                    label=angle_name, color=current_color,
                )

            plt.xlabel(f"Time ({self.time_units})")
            plt.ylabel("Dihedral Angle (radians)")
            plt.title(
                "Residue-Averaged Dihedral Angles Over Time\n"
                f"{self.name.upper() if self.name else ""}"
            )
            plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
            plt.tight_layout()
            plt.show()

            # Plot 2, per-residue distributions
            if not raw_per_residue_data:
                print("No 'per residue' data to plot.")
                return 

            n_frames = self.traj.n_frames

            for angle_name, data_array, indices_array in raw_per_residue_data:

                current_residue_indices = []
                current_angle_values = []

                n_residues_for_this_angle = data_array.shape[1]
                residue_labels = [
                    self.traj.topology.residue(i[1]).resSeq
                    for i in indices_array
                ]

                for i in range(n_residues_for_this_angle):
                    residue_idx = residue_labels[i]
                    values_for_this_residue = data_array[:, i]

                    current_angle_values.extend(values_for_this_residue)
                    current_residue_indices.extend([residue_idx] * n_frames)

                plt.figure(figsize=figsize)
                ax = sns.violinplot(
                    x=current_residue_indices,
                    y=current_angle_values,
                    inner="quartile",
                    color=colors[angle_name]  # Use the shared color
                )

                plt.xlabel("Residue Index")
                plt.ylabel("Dihedral Angle (radians)")
                plt.title(
                    f"Per-Residue {angle_name} Angle Distributions\n"
                    f"{self.name.upper() if self.name else ''}"
                )

                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                plt.tight_layout()
                plt.show()

    def plot_dihedral_landscape(
        self,
        angle_types: list[str] = ["phi", "psi", "omega", "chi1"],
        dim_red_params: dict = {"method": "pca", "n_components": 4},
        cmap : str = "plasma",
        figsize: tuple[int, int] = (16, 16),
        show_explained_variance: bool = False,
    ) -> None:
        """
        Computes all specified dihedrals, transforms them to handle
        circularity (sin/cos), performs dimensionality reduction,
        and generates a pairplot of the resulting components.

        Parameters
        ----------
        - angle_types: list[str], optional
            List of dihedral angle types to compute. Supported types: "phi",
            "psi", "omega", "chi1".
            Default = ["phi", "psi", "omega", "chi1"]
        - dim_red_params: dict, optional
            Parameters for dimensionality reduction. Must include a "method"
            key specifying the method to use (e.g., "PCA", "t-SNE", "UMAP").
            Additional parameters for the method can also be included.
            Default = {"method": "pca", "n_components": 4}
        - cmap: str, optional
            Colormap to be used for the pairplot.
            Default = "plasma"
        - figsize: tuple[int, int], optional
            Size of the figure to be plotted.
            Default = (16, 16)
        """
        if "method" not in dim_red_params:
            raise ValueError(
                "dim_red_params must include a 'method' key."
            )

        method = dim_red_params["method"].upper()

        processed_angle_types = {
            t.strip().lower() for t in angle_types
        }
        all_dihedral_data = []

        if "phi" in processed_angle_types:
            _, data = Wrappers.compute_phi(self.traj)
            if data.shape[1] > 0:
                all_dihedral_data.append(data)
        if "psi" in processed_angle_types:
            _, data = Wrappers.compute_psi(self.traj)
            if data.shape[1] > 0:
                all_dihedral_data.append(data)
        if "omega" in processed_angle_types:
            _, data = Wrappers.compute_omega(self.traj)
            if data.shape[1] > 0:
                all_dihedral_data.append(data)
        if "chi1" in processed_angle_types:
            _, data = Wrappers.compute_chi1(self.traj)
            if data.shape[1] > 0:
                all_dihedral_data.append(data)

        full_dihedral_matrix = np.concatenate(
            all_dihedral_data,
            axis=1
        )

        # Handle any NaNs that may arise from angle calculation
        nans_replaced_with_zero = False
        if np.isnan(full_dihedral_matrix).any():
            # idk if i will use ths flag but just in case
            nans_replaced_with_zero = True
            full_dihedral_matrix = np.nan_to_num(full_dihedral_matrix)

        # Transform circular angle data (in radians) into (sin, cos) pairs
        # This is the correct way to feed angles to linear methods like PCA
        sin_data = np.sin(full_dihedral_matrix)
        cos_data = np.cos(full_dihedral_matrix)
        transformed_matrix = np.concatenate([sin_data, cos_data], axis=1)

        # Perform dimensionality reduction
        reduced_data = self._dim_red(
            transformed_matrix,
            show_explained_variance=show_explained_variance,
            **dim_red_params
        )

        if self.save_calculations:
            self.dihedral_dim_red = reduced_data

        self._plot_dihedral_pairplot(
            data=reduced_data,
            method=method,
            hue=self.time,
            cmap=cmap,
            figsize=figsize,
        )

    def plot_dihedral_path(
        self,
        dims: tuple[int] = (0, 1),
        angle_types: list[str] = ["phi", "psi", "omega", "chi1"],
        dim_red_params: dict = {"method": "pca", "n_components": 3},
        window_length: int = 101,
        poly_order: int = 4,
        cmap : str = "plasma",
        figsize: tuple[int, int] = (10, 10),
        height: int = 800,
    ) -> None:
        """
        Computes all specified dihedrals, transforms them to handle
        circularity (sin/cos), performs dimensionality reduction,
        smooths the resulting components, and plots the trajectory path
        in the reduced space.

        Parameters
        ----------
        - dims: tuple[int], optional
            Indices of the dimensions to plot. Must be either 2 or 3 dimensions.
            Default = (0, 1)
        - angle_types: list[str], optional
            List of dihedral angle types to compute. Supported types: "phi",
            "psi", "omega", "chi1".
            Default = ["phi", "psi", "omega", "chi1"]
        - dim_red_params: dict, optional
            Parameters for dimensionality reduction. Must include a "method"
            key specifying the method to use (e.g., "PCA", "t-SNE", "UMAP").
            Additional parameters for the method can also be included.
            Default = {"method": "pca", "n_components": 3}
        - window_length: int, optional
            Window length for the Savitzky-Golay filter smoothing. Must be odd.
            Default = 45
        - poly_order: int, optional
            Polynomial order for the Savitzky-Golay filter smoothing.
            Default = 4
        - cmap: str, optional
            Colormap to be used for the path plot.
            Default = "plasma"
        - figsize: tuple[int, int], optional
            Size of the figure to be plotted.
            Default = (10, 10)
        - height: int, optional
            Height of the 3D plot (if applicable).
            Default = 800
        """
        if len(dims) != 2 and len(dims) != 3:
            raise ValueError("dims must be either 2 or 3.")

        processed_angle_types = {
            t.strip().lower() for t in angle_types
        }
        all_dihedral_data = []

        if "phi" in processed_angle_types:
            _, data = Wrappers.compute_phi(self.traj)
            if data.shape[1] > 0:
                all_dihedral_data.append(data)
        if "psi" in processed_angle_types:
            _, data = Wrappers.compute_psi(self.traj)
            if data.shape[1] > 0:
                all_dihedral_data.append(data)
        if "omega" in processed_angle_types:
            _, data = Wrappers.compute_omega(self.traj)
            if data.shape[1] > 0:
                all_dihedral_data.append(data)
        if "chi1" in processed_angle_types:
            _, data = Wrappers.compute_chi1(self.traj)
            if data.shape[1] > 0:
                all_dihedral_data.append(data)

        full_dihedral_matrix = np.concatenate(
            all_dihedral_data,
            axis=1
        )

        # Transform circular angle data (in radians) into (sin, cos) pairs
        # This is the correct way to feed angles to linear methods like PCA
        sin_data = np.sin(full_dihedral_matrix)
        cos_data = np.cos(full_dihedral_matrix)
        transformed_matrix = np.concatenate([sin_data, cos_data], axis=1)

        # Perform dimensionality reduction
        reduced_data = self._dim_red(
            transformed_matrix,
            show_explained_variance=False,
            **dim_red_params
        )

        cols = reduced_data.columns
        selected_cols = [
            cols[i]
            for i in dims
        ]
        data = reduced_data[selected_cols].to_numpy()

        # perform smoothing with a Savitzky-Golay filter along each dimension
        for i in range(len(dims)):
            data[:, i] = savgol_filter(
                data[:, i],
                window_length=window_length,
                polyorder=poly_order,
            )
    
        if self.save_calculations:
            self.dihedral_dim_red_smooth = data

        # Plot the path
        if len(dims) == 2:
            self._plot_2D_path(
                data=data,
                axes_names=selected_cols,
                hue=self.time,
                cmap=cmap,
                figsize=figsize,
            )
        else:
            self._plot_3D_path(
                data=data,
                axes_names=selected_cols,
                hue=self.time,
                cmap=cmap,
                height=height,
            )

    def _plot_1D_RMSD(
        self,
        figsize: tuple[int, int] = (18, 8),
        refs: dict[str, md.Trajectory] | None = None,
        selection: str = "backbone",
        conv_window: int = 50,
    ) -> None:
        """Internal method to plot 1D cRMSD over time, called by 'plot_rmsd'"""
        # Default to calculating the cRMSD against the first frame
        if not refs:
            refs = {
                "first frame": self.traj[0],
            }

        atom_indices = self.traj.topology.select(selection)
        if atom_indices.size == 0:
            raise ValueError(
                f"Warning: Selection '{selection}' yielded 0 atoms."
                " Check selection string."
            )

        # Calculate the cRMSD for each structure against all frames
        rmsds = {
            key: Wrappers.rmsd(
                self.traj,
                ref,
                atom_indices=atom_indices,
            )
            for key, ref in refs.items()
        }

        if self.save_calculations:
            self.rmsds = rmsds

        # Plot
        sns.set_theme(style="whitegrid", context="talk")
        plt.figure(figsize=figsize)

        colors = dict(
            zip(
                rmsds.keys(),
                sample(XKCD_COLORS, k=len(rmsds))
            )
        )

        for key, rmsd in rmsds.items():
            sns.scatterplot(
                x=self.time,
                y=rmsd,
                s=35,
                alpha=0.2,
                label=None,
                color=colors[key],
            )
            sns.lineplot(
                x=self.time[conv_window: -conv_window],
                y=np.convolve(
                    rmsd,
                    np.ones(conv_window)/conv_window,
                    mode="same",
                )[conv_window: -conv_window],
                label=f"RMSD to {key}",
                color=colors[key],
        )

        plt.xlabel(f"Time ({self.time_units})")
        plt.ylabel("RMSD (nm)")
        plt.title(
            "RMSD\n"
            f"{self.name.upper() if self.name else ""}"
        )
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.show()

    def _plot_2D_RMSD(
        self,
        figsize: tuple[int, int] = (10, 10),
        cmap: str = "plasma",
        selection: str = "backbone",
        distance_type: str = "euclidean",
    ) -> None:
        """Internal method to plot 2D dRMSD heatmap, called by 'plot_rmsd'"""
        atom_indices = self.traj.topology.select(selection)
        n_atoms_selected = atom_indices.shape[0]
        
        if n_atoms_selected < 2:
            raise ValueError(
                f"Selection '{selection}' resulted in {n_atoms_selected} atoms."
                " Need at least 2 atoms for pairwise distances."
            )

        k = n_atoms_selected * (n_atoms_selected - 1) / 2

        # Get coordinates for selected atoms across all frames
        coords = self.traj.xyz[:, atom_indices, :]

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
        drmsd_matrix = squareform(drmsd_condensed_scaled)

        if self.save_calculations:
            self.drmsd_matrix = drmsd_matrix

        # Plot
        sns.set_theme(style="white", context="talk")
        plt.figure(figsize=figsize)
        ax = sns.heatmap(
            drmsd_matrix,
            cmap=cmap,
            cbar_kws={"label": "dRMSD (nm)"},
            square=True,
            xticklabels=False,
            yticklabels=False,
        )
        num_ticks = 10
        tick_indices = np.linspace(
            0,
            self.n_frames - 1,
            num=num_ticks,
            dtype=int,
        )
        tick_indices = np.clip(tick_indices, 0, self.n_frames - 1)
        tick_labels = [f"{self.time[i]:.1f}" for i in tick_indices]
        ax.set_xticks(tick_indices + 0.5)
        ax.set_yticks(tick_indices + 0.5)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.set_yticklabels(tick_labels, rotation=0)
        ax.set_xlabel(f"Time ({self.time_units})")
        ax.set_ylabel(f"Time ({self.time_units})")
        ax.set_title(
            f"Pairwise dRMSD ({selection})\n"
            f"{self.name.upper() if self.name else ""}"
        )
        plt.tight_layout()
        plt.show()

    def _plot_dihedral_pairplot(
        self,
        data: pd.DataFrame,
        method: str,
        hue: np.ndarray | pd.Series,
        cmap: str = "plasma",
        figsize: tuple[int, int] = (16, 16),
    ) -> None:
        """Internal method to plot dihedral landscape pairplot, called by
        'plot_dihedral_landscape'"""
        def _annotate_correlations(x, y, **kwargs):
            """Annotate the correlation coefficients on the pairplot."""
            pearson_coef, _ = pearsonr(x, y)
            spearman_coef, _ = spearmanr(x, y)
            kendall_coef, _ = kendalltau(x, y)

            text = "".join([
                f"Pearson: {pearson_coef:.2f}\n",
                f"Spearman: {spearman_coef:.2f}\n",
                f"Kendall: {kendall_coef:.2f}"
            ])

            plt.annotate(
                text,
                xy=(0.95, 0.95),
                xycoords="axes fraction",
                ha="right",
                va="top",
                fontsize="small",
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    fc="white",
                    ec="black",
                    alpha=0.65
                )
            )

        def _plot_mean_median(x, **kwargs):
            """Plot the mean and median of the data on the diagonal of the
            pairplot."""
            mean_val = np.mean(x)
            plt.axvline(
                mean_val,
                color="blue",
                linestyle="-",
                label=f"Mean: {mean_val:.2f}"
            )

            median_val = np.median(x)
            plt.axvline(
                median_val,
                color="red",
                linestyle="--",
                label=f"Median {median_val:.2f}"
            )

        sns.set_theme(style="white", context="talk")

        g = sns.pairplot(
            data,
            diag_kind="kde",
            plot_kws={
                "hue": hue,
                "palette": cmap,
            },
            diag_kws={
                "color": choice(XKCD_COLORS),
                "fill": True
            },
        )
        g.figure.set_size_inches(figsize)

        g.map_upper(_annotate_correlations)
        g.map_diag(_plot_mean_median)
        g.map_lower(sns.kdeplot, levels=4, color="black")

        # Create a normalized scalar mappable for the color bar
        norm = Normalize(vmin=hue.min(), vmax=hue.max())
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        g.figure.subplots_adjust(right=0.85)
        cbar_ax = g.figure.add_axes([0.88, 0.15, 0.02, 0.7])
        g.figure.colorbar(sm, cax=cbar_ax, label=f"Time ({self.time_units})")

        plt.suptitle(
            f"Dihedral Landscape Pairplot with {method.upper()}\n"
            f"{self.name.upper() if self.name else ""}",
            y=1.02,
            va="bottom",
        )
        plt.show()

    def _plot_2D_path(
        self,
        data: np.ndarray,
        axes_names: list[str],
        hue: np.ndarray | pd.Series,
        cmap: str = "plasma",
        figsize: tuple[int, int] = (10, 10),
    ):
        """Internal method to plot 2D dihedral path, called by
        'plot_dihedral_path'"""
        sns.set_theme(style="white", context="talk")
        fig, ax = plt.subplots(figsize=figsize)
        lines = colored_line(
            data[:, 0],
            data[:, 1],
            hue,
            ax,
            cmap=cmap,
            linewidth=2,
        )
        fig.colorbar(lines, ax=ax, label=f"Time ({self.time_units})")
        ax.set_xlabel(axes_names[0])
        ax.set_ylabel(axes_names[1])
        ax.set_title(
            "Dihedral Path in 2D Reduced Space\n"
            f"{self.name.upper() if self.name else ''}"
        )
        ax.set_xlim(np.min(data[:, 0])*1.01, np.max(data[:, 0])*1.01)
        ax.set_ylim(np.min(data[:, 1])*1.01, np.max(data[:, 1])*1.01)
        plt.tight_layout()
        plt.show()

    def _plot_3D_path(
        self,
        data: np.ndarray,
        axes_names: list[str],
        hue: np.ndarray | pd.Series,
        cmap: str = "Plasma",
        height: int = 800,
    ):
        """
        Internal method to plot 3D dihedral path using K3D.
        """
        # Ensure data is float32 for WebGL
        vertices = np.asarray(data, dtype=np.float32)
        attribute = np.asarray(hue, dtype=np.float32)

        # Resolve colormap
        cmap_obj = getattr(k3d.matplotlib_color_maps, cmap, k3d.matplotlib_color_maps.Plasma)

        # Calculate line width and cast to standard float
        data_range = np.max(vertices) - np.min(vertices)
        line_width = float(data_range / 500.0)  # <--- Fix: Explicit cast to float

        plot = k3d.plot(
            height=height,
            axes=axes_names,
            name="Dihedral Path"
        )

        line = k3d.line(
            vertices=vertices,
            attribute=attribute,
            color_map=cmap_obj,
            color_range=[np.min(attribute), np.max(attribute)],
            width=line_width,
            shader='mesh'
        )

        plot += line
        plot.display()

    @staticmethod
    def _dim_red(
        data: np.ndarray,
        show_explained_variance: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Internal method to perform dimensionality reduction on data using
        specified method and parameters, called by 'plot_dihedral_landscape'"""
        if "method" not in kwargs:
            raise ValueError(
                "Method must be specified in kwargs"
            )

        method = kwargs["method"].lower()

        if show_explained_variance and method != "pca":
            print(
                "Warning: 'show_explained_variance' is only applicable for"
                " PCA. Ignoring for other methods."
            )

        if method == "pca":
            if "n_components" in kwargs:
                n_components = kwargs["n_components"]
                pca = PCA(n_components=n_components)

            elif "percent" in kwargs:
                percent = kwargs["percent"]
                if not (0 < percent <= 1.0):
                    raise ValueError(
                        "percent must be in the range (0.0, 1.0]"
                    )
                pca = PCA(n_components=percent)

            else:
                raise ValueError(
                    "For PCA, either n_components"
                    " or percent must be specified"
                )

            reduced = pca.fit_transform(data)

            if show_explained_variance:
                variance = pca.explained_variance_ratio_
                cols = [
                    f"PC{i+1} {var:.2%}"
                    for i, var in enumerate(variance)
                ]
            else:
                cols = [
                    f"PC{i}"
                    for i in range(1, reduced.shape[1] + 1)
                ]

            return pd.DataFrame(
                reduced,
                columns=cols
            )

        elif method == "tsne":
            n_components = kwargs.get("n_components", 3)
            if n_components > 3:
                # t-SNE is generally only used for 2 or 3 components
                print(
                    f"Warning: t-SNE with n_components={n_components} > 3 is"
                    " computationally expensive and rarely used."
                )

            tsne = TSNE(n_components=n_components)

            reduced = tsne.fit_transform(data)

            return pd.DataFrame(
                reduced,
                columns=[
                    f"tSNE{i}"
                    for i in range(1, reduced.shape[1] + 1)
                ]
            )

        elif method == "umap":
            n_components = kwargs.get("n_components", 5)

            umap_model = umap.UMAP(n_components=n_components)

            reduced =  umap_model.fit_transform(data)

            return pd.DataFrame(
                reduced,
                columns=[
                    f"UMAP{i}"
                    for i in range(1, reduced.shape[1] + 1)
                ]
            )

        else:
            raise ValueError(
                "Method must be one of 'pca', 'tsne', or 'umap'"
            )


# Stolen from:
# https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html
def colored_line(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)


def colored_line_3d(x, y, z, c, ax, **lc_kwargs):
    """
    Plot a 3D line with a color specified along the line by a fourth value.
    
    Parameters
    ----------
    x, y, z : array-like
        The coordinates of the data points.
    c : array-like
        The color values, same size as x, y, and z.
    ax : Axes3D
        The 3D axis object.
    **lc_kwargs
        Arguments passed to Line3DCollection.
    """
    # Default capstyle
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Ensure arrays
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    # Compute midpoints for x, y, and z
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))
    z_midpts = np.hstack((z[0], 0.5 * (z[1:] + z[:-1]), z[-1]))

    # Determine start, middle, and end coordinates
    # Stack x, y, and z for start points
    coord_start = np.column_stack(
        (x_midpts[:-1], y_midpts[:-1], z_midpts[:-1])
    )[:, np.newaxis, :]
    
    # Stack x, y, and z for mid points
    coord_mid = np.column_stack(
        (x, y, z)
    )[:, np.newaxis, :]
    
    # Stack x, y, and z for end points
    coord_end = np.column_stack(
        (x_midpts[1:], y_midpts[1:], z_midpts[1:])
    )[:, np.newaxis, :]

    # Concatenate to form segments: (N, 3, 3)
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = Line3DCollection(segments, **default_kwargs)
    lc.set_array(c)

    return ax.add_collection(lc)