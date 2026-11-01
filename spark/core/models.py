import numpy as np

from spark.core.fitting.fit import (
    convolved_exp_vectorized,
    convolved_stretched_exp_vectorized,
    get_sequential_populations,
    damped_oscillation,
    get_coherent_artifact
)



class BaseModel:
    """
    Base class from which all kinetic models inherit.

    Parameters
    ----------
    num_components : int
        Number of kinetic components (e.g., exponential decays).
    use_artifact : bool, optional
        Whether to include the coherent artifact in the model. Defaults to False.
    artifact_mode : str, optional
        The type of artifact basis to use ('raman', 'xpm', or 'both'). Defaults to 'both'.
    """
    def __init__(self, num_components, use_artifact=False, artifact_mode='both'):
        self.num_components = num_components
        self.use_artifact = use_artifact
        self.artifact_mode = artifact_mode

    def get_concentration_matrix(self, x_nl, t):
        """
        Constructs the concentration matrix (C) for the kinetic model.
        Must be implemented by each specific model subclass.

        Parameters
        ----------
        x_nl : array_like
            Array of non-linear parameters (e.g., IRF width, time zero, taus).
        t : numpy.ndarray
            Time vector.

        Raises
        ------
        NotImplementedError
            If called directly on the base class.
        """
        raise NotImplementedError

class StretchedModel(BaseModel):
    """
    Stretched Exponential Model (Kohlrausch-Williams-Watts, KWW).
    Assumes excited states decay independently as exp(-(t/tau_i)^beta_i)
    convolved with a Gaussian IRF.
    """
    def get_concentration_matrix(self, x_nl, t):
        w, t0 = x_nl[0], x_nl[1]
        taus = x_nl[2 : 2 + self.num_components]
        betas = x_nl[2 + self.num_components : 2 + 2 * self.num_components]
        
        C = convolved_stretched_exp_vectorized(t, t0, taus, betas, w)
        
        if self.use_artifact:
            C = np.hstack([C, get_coherent_artifact(t, t0, w, mode=self.artifact_mode)])
        return C


class DASModel(BaseModel):
    """
    Parallel Model (Decay Associated Spectra). 
    Assumes all excited states decay independently to the ground state.
    """
    def get_concentration_matrix(self, x_nl, t):
        """
        Builds the DAS concentration matrix.

        Parameters
        ----------
        x_nl : array_like
            Array of non-linear parameters [w, t0, tau_1, ..., tau_N].
        t : numpy.ndarray
            Time vector.

        Returns
        -------
        numpy.ndarray
            The concentration matrix C.
        """
        w, t0 = x_nl[0], x_nl[1]
        taus = x_nl[2 : 2 + self.num_components]
        
        C = convolved_exp_vectorized(t, t0, taus, w)
        
        if self.use_artifact:
            C = np.hstack([C, get_coherent_artifact(t, t0, w, mode=self.artifact_mode)])
        return C


class SASModel(BaseModel):
    """
    Sequential Model (Species Associated Spectra). 
    Assumes a kinetic sequence: A -> B -> C...
    """
    def get_concentration_matrix(self, x_nl, t):
        """
        Builds the SAS concentration matrix using Bateman equations.

        Parameters
        ----------
        x_nl : array_like
            Array of non-linear parameters [w, t0, tau_1, ..., tau_N].
        t : numpy.ndarray
            Time vector.

        Returns
        -------
        numpy.ndarray
            The concentration matrix C.
        """
        w, t0 = x_nl[0], x_nl[1]
        taus = x_nl[2 : 2 + self.num_components]
        
        pops_list = get_sequential_populations(t, t0, w, taus)
        C = np.column_stack(pops_list)
        
        if self.use_artifact:
            C = np.hstack([C, get_coherent_artifact(t, t0, w, mode=self.artifact_mode)])
        return C


class OscillationModel(BaseModel):
    """
    Model consisting of exponential decays added to a damped oscillation.
    Useful for wavepacket dynamics or acoustic phonons.
    """
    def get_concentration_matrix(self, x_nl, t):
        """
        Builds the concentration matrix including the oscillation basis.

        Parameters
        ----------
        x_nl : array_like
            Array of non-linear parameters [w, t0, tau_1..N, alpha, omega, phi].
        t : numpy.ndarray
            Time vector.

        Returns
        -------
        numpy.ndarray
            The concentration matrix C.
        """
        w, t0 = x_nl[0], x_nl[1]
        taus = x_nl[2 : 2 + self.num_components]
        alpha, omega, phi = x_nl[2 + self.num_components : 2 + self.num_components + 3]
        
        basis_exp = convolved_exp_vectorized(t, t0, taus, w)
        basis_osc = damped_oscillation(t, t0, alpha, omega, phi, w).reshape(-1, 1)
        C = np.hstack([basis_exp, basis_osc])
        
        if self.use_artifact:
            C = np.hstack([C, get_coherent_artifact(t, t0, w, mode=self.artifact_mode)])
        return C

class KMatrixModel:
    """
    Custom Kinetic Model built from user-defined transitions and rate matrices.
    
    Dynamically constructs a K-matrix, diagonalizes it, and calculates the 
    population evolution over time based on eigenvectors and eigenvalues.

    Parameters
    ----------
    name : str, optional
        Name of the custom model. Defaults to "Custom Model".
    """
    def __init__(self, name="Custom Model"):
        self.name = name
        self.states = []        # List of excited state names
        self.transitions = []   # List of tuples: (source, target, param_type, label)
        self.param_labels = []  # Ordered names of the non-linear parameters
        
    def add_state(self, state_name):
        """
        Adds a physical state to the kinetic system (e.g., 'S1*', 'S1', '3CT').

        Parameters
        ----------
        state_name : str
            The identifier of the state.
        """
        if state_name not in self.states:
            self.states.append(state_name)
            
    def add_transition(self, source, target, param_type="tau", label=""):
        """
        Adds a population transfer (transition) between two states.

        Parameters
        ----------
        source : str
            The origin state of the transition.
        target : str
            The destination state. Use "S0" for decay to the ground state.
        param_type : str, optional
            The nature of the parameter ('tau' for lifetime, 'gamma' for yield). 
            Defaults to "tau".
        label : str, optional
            The shared label of the parameter to optimize.
        """
        self.add_state(source)
        if target != "S0": # S0 is the ground state
            self.add_state(target)
        self.transitions.append((source, target, param_type, label))
        
    def build_parameter_list(self):
        """
        Analyzes the defined transitions and extracts the unique list of 
        non-linear parameters to be optimized.

        Returns
        -------
        list of str
            The ordered list of unique parameter labels.
        """
        labels = []
        for src, tgt, p_type, label in self.transitions:
            if label and label not in labels:
                labels.append(label)
        self.param_labels = labels
        return self.param_labels

    def get_default_guesses_and_bounds(self):
        """
        Automatically generates staggered initial guesses and bounds 
        to assist the optimizer in ordering the lifetimes.

        Returns
        -------
        tuple
            A tuple (ini, low, upp) containing:
            - ini (numpy.ndarray): Initial guess values.
            - low (numpy.ndarray): Lower bounds.
            - upp (numpy.ndarray): Upper bounds.
        """
        self.build_parameter_list()
        ini, low, upp = [], [], []
        
        # Staggered initial values to help the optimizer sort the taus
        tau_defaults = [1.0, 15.0, 200.0, 1500.0, 5000.0]
        tau_idx = 0
        
        for label in self.param_labels:
            p_type = next(t[2] for t in self.transitions if t[3] == label)
            if p_type == "tau":
                val = tau_defaults[tau_idx] if tau_idx < len(tau_defaults) else 10.0
                ini.append(val)
                low.append(0.001)     
                upp.append(1e8)        
                tau_idx += 1
            elif p_type == "gamma":
                ini.append(0.5)        
                low.append(0.0)        
                upp.append(1.0)        
                
        return np.array(ini), np.array(low), np.array(upp)


    def get_concentration_matrix(self, x_nl_params, t, w, t0, use_art=False, artifact_mode='both'):
        """
        Constructs the K-matrix from the parameters, diagonalizes it, and returns 
        the population matrix over time.

        Parameters
        ----------
        x_nl_params : array_like
            Current values of the non-linear kinetic parameters.
        t : numpy.ndarray
            Time vector.
        w : float
            Width of the IRF.
        t0 : float
            Time zero.
        use_art : bool, optional
            Whether to include the coherent artifact. Defaults to False.
        artifact_mode : str, optional
            Mode for the coherent artifact. Defaults to 'both'.

        Returns
        -------
        numpy.ndarray
            The concentration matrix C resulting from the K-matrix diagonalization.
        """
        N = len(self.states)
        K = np.zeros((N, N))
        
        p_dict = dict(zip(self.param_labels, x_nl_params))
        
        tau_totals = {}
        for src, tgt, p_type, label in self.transitions:
            if p_type == "tau":
                tau_totals[src] = p_dict[label]

        for src, tgt, p_type, label in self.transitions:
            idx_src = self.states.index(src)
            val_param = p_dict[label]
            
            if p_type == "tau":
                k_val = 1.0 / max(val_param, 1e-12)
                K[idx_src, idx_src] -= k_val 
                if tgt != "S0":
                    idx_tgt = self.states.index(tgt)
                    K[idx_tgt, idx_src] += k_val 
                    
            elif p_type == "gamma":
                tau_total = tau_totals.get(src, 1.0)
                k_total = 1.0 / max(tau_total, 1e-12)
                k_via = val_param * k_total
                
                if tgt != "S0":
                    idx_tgt = self.states.index(tgt)
                    K[idx_tgt, idx_src] += k_via
                    
        # Deterministic perturbation to avoid degenerate eigenvalues
        diag_vals = np.diagonal(K).copy()
        eps_base = 1e-9
        perturbacion = eps_base * np.arange(N)
        np.fill_diagonal(K, diag_vals * (1 + perturbacion))
        
        eigenvalues, V = np.linalg.eig(K)   
        
        try:
            V_inv = np.linalg.inv(V)
        except np.linalg.LinAlgError:
            V_inv = np.linalg.pinv(V)

        targets = [tgt for src, tgt, p_type, label in self.transitions]
        roots = [s for s in self.states if s not in targets]
        
        P0 = np.zeros(N)
        if roots:
            # --- Populate ALL initial states ---
            for root_state in roots:
                idx_p0 = self.states.index(root_state)
                P0[idx_p0] = 1.0
        else:
            idx_p0 = 0
            P0[idx_p0] = 1.0
        
        c = V_inv @ P0
        
        eigenvalues = np.real(eigenvalues)
        taus_eff = np.zeros_like(eigenvalues)
        
        for i, ev in enumerate(eigenvalues):
            if ev > -1e-12:
                taus_eff[i] = 1e8
            else:
                taus_eff[i] = -1.0 / ev
                
        E_matrix = convolved_exp_vectorized(t, t0, taus_eff, w)
        
        pops = []
        for i in range(N):
            state_pop = np.zeros_like(t)
            for j in range(N):
                peso = np.real(V[i, j] * c[j])
                state_pop += peso * E_matrix[:, j]
            pops.append(state_pop)
            
        C = np.column_stack(pops)
        
        if use_art: 
            C = np.hstack([C, get_coherent_artifact(t, t0, w, mode=artifact_mode)])
            
        return C
