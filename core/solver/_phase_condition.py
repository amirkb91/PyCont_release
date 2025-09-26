import numpy as np


def add_phase_condition(self, J):
    """
    Augments the Jacobian with phase condition constraints.

    This function adds rows to the Jacobian to enforce phase conditions, which are
    used to constrain the solution and remove rank deficiency in the system
    when solving for periodic orbits.

    Args:
        J (np.ndarray): The original Jacobian matrix (n x n+1).

    Returns:
        np.ndarray: The augmented Jacobian matrix.
    """
    n_dof = J.shape[0]  # Number of system degrees of freedom
    n_cols = J.shape[1]

    # The number of velocity DOFs is assumed to be half of the total DOFs.
    n_vel = n_dof // 2

    constraint_indices = []
    idx_setting = self.prob.parameters["continuation"]["phase_condition_index"]

    if not idx_setting:
        raise ValueError("Phase condition index not specified in parameter file.")

    if idx_setting == "all":
        # Constrain all velocity degrees of freedom.
        # This assumes velocities are the second half of the state vector.
        constraint_indices = list(range(n_vel))
    else:
        # Parse comma-separated indices and colon-separated ranges (e.g., "0:2,4").
        parts = idx_setting.split(",")
        for part in parts:
            if ":" in part:
                start, end = part.split(":")
                constraint_indices.extend(range(int(start), int(end) + 1))
            else:
                constraint_indices.append(int(part))
        # Ensure indices are unique and sorted.
        constraint_indices = sorted(set(constraint_indices))

    num_constraints = len(constraint_indices)
    if num_constraints == 0:
        return J  # Return original Jacobian if no constraints are applied

    # Create the augmented Jacobian
    J_aug = np.zeros((n_dof + num_constraints, n_cols))
    J_aug[:n_dof, :] = J  # Copy original Jacobian

    # Add phase condition rows. Each row corresponds to fixing one velocity DOF.
    for i, v_idx in enumerate(constraint_indices):
        J_aug[n_dof + i, v_idx + n_vel] = 1.0

    return J_aug
