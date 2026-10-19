# ==========================================================
# sparse_PGD.py
# ==========================================================

import numpy as np
from numpy.linalg import pinv, norm
from scipy.io import loadmat
from scipy.sparse import eye
import pickle
import time
import pdb  # Debugger import
from scipy.sparse import eye


def _to_scalar(value):
    """Return a Python float from a NumPy scalar or 1x1 array-like."""
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(f"Expected a scalar-like value, got shape {arr.shape}")
    return float(arr.reshape(-1)[0])


class SparsePGD:
    """
    Sparse Proper Generalized Decomposition (sPGD)
    using Kriging interpolation
    """

    def __init__(self):
        self.MOR = None

   

    def MOR_construction_sparse(
        self,
        error_crit,
        x_train,
        p_train,
        str_MOR,
        solver,
        flag_reg,
        Ncp
    ):
        """
        Builds the sPGD model from training data (x_train, p_train)

        Parameters
        ----------
        error_crit : float
            Target error percentage
        x_train : ndarray
            Training data (Nsensors x Ndata)
        p_train : ndarray
            Parameter data (Nparam x Ndata)
        str_MOR : str
            Filename to save the MOR model (pickle)
        solver : str
            "Krig" or "Rad"
        flag_reg : int
            0 or 1, for L1 regularization
        Ncp : int
            Number of control points

        Returns
        -------
        MOR : dict
            Dictionary containing the sPGD model
        """

        # ----------------------------
        # Convert inputs to numpy
        # ----------------------------
        x_train = np.asarray(x_train)
        p_train = np.asarray(p_train)

        # ----------------------------
        # Validate shapes
        # ----------------------------
        assert x_train.ndim == 2, "x_train must be 2D (Nsensors x Ndata)"
        assert p_train.ndim == 2, "p_train must be 2D (Nparam x Ndata)"
        assert x_train.shape[1] == p_train.shape[1], "Mismatch: Ndata must match"

        # ----------------------------
        # Build data structure like MATLAB
        # ----------------------------
        data = {}
        data["res"] = x_train                # Nsensors x Ndata
        data["param"] = p_train.T            # Ndata x Nparam

        # ----------------------------
        # Parameter vectors
        # ----------------------------
        Nparam = p_train.shape[0]
        param_vec = []
        param_vec_lim = []
        np_vec = []
        dp_vec = []

        for i in range(Nparam):
            vec = p_train[i, :]
            param_vec.append(np.unique(np.sort(vec)))

            # Unique sorted values like MATLAB
            lim = np.unique(np.sort(vec))
            param_vec_lim.append(lim)

            # np and dp
            np_vec.append(1000)
            dp_vec.append(abs(lim[0] - lim[-1]) / (1000 - 1))

        

        # ----------------------------
        # Choose solver
        # ----------------------------

        if solver == "Krig":
            h = 1
            fP, d_fP, dd_fP, MOR, Error1 = self.sPGD_Krig(
                data, np_vec, param_vec, param_vec_lim, index_int=1,
                error_crit=error_crit, Ncp=Ncp, h=h
            )
        else:
            h = 7
            fP, d_fP, dd_fP, MOR, Error1 = self.sPGD_RBFs_approx_unstructured_opt_v2(
                data, np_vec, param_vec, param_vec_lim, index_int=1,
                error_crit=error_crit, Ncp=Ncp, h=h
            )

        # Final error
        Error1 = Error1[-1] if isinstance(Error1, (list, np.ndarray)) else Error1

        # ----------------------------
        # Save MOR structure
        # ----------------------------
        MOR["np_vec"] = np_vec
        MOR["dp_vec"] = dp_vec
        MOR["param_vec"] = param_vec
        MOR["param_vec_lim"] = param_vec_lim
        MOR["choice"] = 1
        MOR["Error"] = Error1
        MOR["solver"] = solver
        MOR["Ncp"] = Ncp
        MOR["fP"] = fP
        MOR["d_fP"] = d_fP
        MOR["dd_fP"] = dd_fP  



        # Save to pickle
        with open(str_MOR, "wb") as f:
            pickle.dump(MOR, f)

        return MOR





    # ------------------------------------------------------
    # Core sPGD solver (Kriging)
    # ------------------------------------------------------
    
    def sPGD_Krig(
        self,
        data,
        np_vec,
        param_vec,
        param_vec_lim,
        index_int,
        error_crit,
        Ncp,
        h
    ):
        t0 = time.time()
        
        Fres = data["res"].copy()
        param = data["param"]

        Ndata = Fres.shape[1]
        Nparams = len(param_vec)  # number of parameters (not including sensors)
        Nparam = Nparams + index_int  # total including physical dimension
        Nsensors = Fres.shape[0]

        # Parameter indexing - CORRECTED
        index_param = np.zeros((Ndata, Nparam), dtype=int)
        for i in range(Ndata):
            ind = 0
            for j in range(Nparam):
                if j < index_int:  # Physical dimension (MATLAB: j <= index_int)
                    index_param[i, j] = 0  # MATLAB sets this to 1, not 0!
                else:
                    # Parametric dimension
                    index_param[i, j] = np.where(
                        param_vec_lim[ind] == param[i, ind]
                    )[0][0]
                    ind += 1

        # Precomputation
        dim = np.zeros(Nparam, dtype=int)
        IMp_all = {}
        IMp = {}
        M_rhs_aux = {}
        termsp = {}
        d_termsp = {}
        dd_termsp = {}
        prod_aux = {}
        Mtot_aux = {}
        up = {}
        
        ind_aux = 0

        for i in range(Nparam):
            
            # Physical dimension
            if i < index_int:  # MATLAB: i <= index_int
                dim[i] = Nsensors
                IMp_all[i] = eye(dim[i], dim[i]).toarray()
                IMp[i] = eye(dim[i], dim[i]).toarray()                
                up[i] = np.empty((dim[i], 0))

            # Parametric dimensions
            else:

                ind_aux += 1
                dim[i] = Ncp
                
                x_sp = param_vec_lim[ind_aux - 1]
                x_cp = np.linspace(param_vec[ind_aux - 1][0], 
                                param_vec[ind_aux - 1][-1], Ncp)

                sig = h * abs(x_cp[1] - x_cp[0])
                Pi = np.pi
                
                # Correlation matrices
                Cor_cp_cp = np.exp(
                    -(x_cp[:, None] - x_cp[None, :])**2 / (2 * sig**2)
                ) / (sig * np.sqrt(2 * Pi))

                Cor_cp_sp = np.exp(
                    -(x_cp[:, None] - x_sp[None, :])**2 / (2 * sig**2)
                ) / (sig * np.sqrt(2 * Pi))

                # Lambda and Sf
                lam = np.linalg.solve(Cor_cp_cp, Cor_cp_sp)
                Sf = (1.0 - np.sum(lam, axis=0)) / Ncp + lam

                # IMp_all
                IMp_all[i] = Sf @ Sf.T

                # IMp as 3D array - one matrix per sampling point
                Nsp = len(x_sp)
                IMp_3d = np.zeros((Ncp, Ncp, Nsp))
                for k in range(Nsp):
                    v = Sf[:, k:k+1]
                    IMp_3d[:, :, k] = v @ v.T
                IMp[i] = IMp_3d


                # Extract slices based on index_param
                indi = index_param[:, i]  # (Ndata,)
                naux = Ncp
                prod_aux[i] = IMp[i][:, :, indi]  # (Ncp, Ncp, Ndata)

                # M_rhs_aux
                M_rhs_aux[i] = Sf[:, indi]  # (Ncp, Ndata)

                # Mtot_aux
                Sf_aux = Sf[:, indi]  # (Ncp, Ndata)
                Mtot_aux[i] = np.zeros((Ncp, Ncp, Ndata))
                for k in range(Ndata):
                    v = Sf_aux[:, k:k+1]  # (Ncp, 1)
                    Mtot_aux[i][:, :, k] = v @ v.T



                # Finer grid interpolation
                pi = np.linspace(x_sp[0], x_sp[-1], np_vec[ind_aux - 1])
                Cor_cp_sp_i = np.exp(
                    -(x_cp[:, None] - pi[None, :])**2 / (2 * sig**2)
                ) / (sig * np.sqrt(2 * Pi))

                lam_i = np.linalg.solve(Cor_cp_cp, Cor_cp_sp_i)
                dlam_i = np.linalg.solve(
                    Cor_cp_cp,
                    Cor_cp_sp_i * (x_cp[:, None] - pi[None, :]) / (sig**2)
                )
                ddlam_i = np.linalg.solve(
                    Cor_cp_cp,
                    Cor_cp_sp_i * (
                        ((x_cp[:, None] - pi[None, :]) / sig**2)**2 - 1.0 / sig**2
                    )
                )

                termsp[i] = (1.0 - np.sum(lam_i, axis=0)) / Ncp + lam_i
                d_termsp[i] = (-np.sum(dlam_i, axis=0)) / Ncp + dlam_i
                dd_termsp[i] = (-np.sum(ddlam_i, axis=0)) / Ncp + ddlam_i

                up[i] = np.empty((dim[i], 0))




        # PGD loop
        error = 100.0
        Niter_max = 100
        stagn_crit = 0.001

        Fres_i = Fres.copy()

        #Fres_vec = Fres.reshape(-1, 1, order='F')
        #norm_Fres = np.linalg.norm(Fres_vec)
        #norm_Fres = np.linalg.norm(Fres.ravel(order="F"))
        norm_Fres = norm(Fres.ravel())
        #norm_Fres = norm(Fres)

        upi = {}
        upl = {}
        m = 0

        while error > error_crit:
            # Initialize modes
            for i in range(Nparam):
                if m == 0:
                    #upi[i] = np.random.rand(dim[i], 1)
                    #upl[i] = np.random.rand(dim[i], 1)

                    upi[i] = np.ones((dim[i], 1))
                    upl[i] = np.ones((dim[i], 1))
                    
                    upi[i] /= norm(upi[i])
                    upl[i] /= norm(upl[i])
                else:
                    upi[i] = upi[i].copy()
                    upl[i] = upl[i].copy()

            Niter = 0
            stagn = 100.0
            
            while stagn > stagn_crit and Niter < Niter_max:
                Niter += 1

                for i in range(Nparam):


                    # %%%%%%%%%%% Matrix determination: %%%%%%%%%%%%%

                    prod_mup = {}
                    product = 1.0  # Start as scalar
                    
                    for j in range(Nparam):
                        if j < index_int:  # Physical dimension
                            #pdb.set_trace()
                            prod_mup[j] = _to_scalar(upi[j].T @ upi[j])
                            product = product * prod_mup[j]
                        else:
                            # Parametric dimension
                            # MATLAB: prod_mup{j} = reshape(sum(upi{j}'.*sum(upi{j}.*prod_aux{j},1),2),1,[])
                            
                            #pdb.set_trace()
                            if 1:
                                temp1 = upi[j][:, None] * prod_aux[j]  # (Ncp, Ncp, Ndata)
                                temp2 = np.sum(temp1, axis=0)  # (Ncp, Ndata)
                                temp3 = upi[j].T @ temp2  # (1, Ndata)
                                prod_mup[j] = temp3                                
                                product = product * prod_mup[j]
                            else:
                                u = upi[j].reshape(-1)   # or upi[j].ravel()
                                prod_mup[j] = np.einsum('i,ijk,j->k', u, prod_aux[j], u)
                                product = product * prod_mup[j]
                                #pdb.set_trace()

                                

                    # Division: a = product / prod_mup[i]
                    
                    if i < index_int:
                        
                        # Physical dimension: both are scalars
                        a = product / prod_mup[i]
                        a = a.reshape(1, 1, -1)
                        #a = np.array(a).reshape(1, 1, 1)

                    else:
                        # Parametric dimension: both are (1, Ndata)
                        epsilon = 0*1e-10
                        a = product / (prod_mup[i] + epsilon)
                        a = a.reshape(1, 1, -1)  # (1, 1, Ndata)

                    # Compute Mtot
                    if i < index_int:
                        a = np.sum(a)  # Scalar
                        Mtot = a
                    else:
                        Mtot_aux2 = Mtot_aux[i] * a  # (Ncp, Ncp, Ndata) * (1, 1, Ndata)
                        Mtot = np.sum(Mtot_aux2, axis=2)  # (Ncp, Ncp)

                    # %%%%%%%%%%% end Matrix determination: %%%%%%%%%%%%% (Correct)        

                    #pdb.set_trace()  # Debugger breakpoint
    

                    # %%%%%%%%%%% RHS determination: %%%%%%%%%%%%%

                    if 0:
                        vec = list(range(Nparam))
                        val = vec.pop(i)
                        vec = [val] + vec
                        vec = list(reversed(vec))

                        prod_rhs = 1.0
                        for jj in range(Nparam - 1):
                            idx = vec[jj]
                            if idx < 1:  # MATLAB: vec(j) == 1, checking if index_int
                                if index_int == 1:
                                    prod_rhs = prod_rhs * (upi[idx].T @ Fres_i)
                                else:
                                    prod_rhs = prod_rhs * (upi[idx].T @ M_rhs_aux[idx]) * Fres_i
                            else:
                                prod_rhs = prod_rhs * (upi[idx].T @ M_rhs_aux[idx])

                        # Final RHS assembly
                        if vec[-1] < 1:  # MATLAB: vec(end) == 1
                            if index_int == 1:
                                rhs_aux = prod_rhs * Fres_i
                            else:
                                rhs_aux = prod_rhs * M_rhs_aux[vec[-1]] * Fres_i
                        else:
                            rhs_aux = prod_rhs * M_rhs_aux[vec[-1]]

                        rhs = np.sum(rhs_aux, axis=1, keepdims=True)

                    else:

                        prod_rhs = 1.0

                        vec = list(range(Nparam))
                        val = vec.pop(i)
                        vec = [val] + vec
                        vec = vec[::-1]

                        for j in range(Nparam - 1):
                            vj = vec[j]

                            if vj == index_int - 1:
                                if index_int == 1:
                                    prod_rhs *= (upi[vj].T @ Fres_i)
                                else:
                                    prod_rhs *= (upi[vj].T @ M_rhs_aux[vj]) * Fres_i
                            else:
                                prod_rhs *= (upi[vj].T @ M_rhs_aux[vj])

                        v_last = vec[-1]

                        if v_last == index_int - 1:
                            rhs_aux = prod_rhs * Fres_i
                        else:
                            rhs_aux = prod_rhs * M_rhs_aux[v_last]

                        rhs = np.sum(rhs_aux, axis=1, keepdims=True)




                    #pdb.set_trace()

                    # %%%%%%%%%%% end RHS determination: %%%%%%%%%%%%%    


                    # %%%%%%%%%%% Resolution: %%%%%%%%%%%%%

                    
                    if i < index_int:
                        upi[i] = rhs / a
                        #pdb.set_trace()
                    else:
                        upi[i] = pinv(Mtot) @ rhs
                        #upi[i] = np.linalg.solve(Mtot, rhs)
                    # %%%%%%%%%%% end Resolution: %%%%%%%%%        


                # % Normalize the PGD functions:
                norm_aux = np.zeros(Nparam)
                for i in range(Nparam):
                    norm_aux[i] = norm(upi[i])
                    upi[i] = upi[i] / norm_aux[i]

                prodn = np.prod(norm_aux) ** (1 / Nparam)
                for i in range(Nparam):
                    upi[i] = upi[i] * prodn
                # % end Normalize the PGD functions.    


                # Stagnation
                term1 = 1.0
                term2 = 1.0
                term3 = 1.0
                for i in range(Nparam):
                    v1 = IMp_all[i] @ upi[i]
                    v2 = IMp_all[i] @ upl[i]
                    term1 = term1 * _to_scalar(v1.T @ v1)
                    term2 = term2 * _to_scalar(v1.T @ v2)
                    term3 = term3 * _to_scalar(v2.T @ v2)
                stagn = np.sqrt(term1 - 2 * term2 + term3) / np.sqrt(term3)
                # % end stagnation.

                # % Assignation of the solution to the last solution:    
                for i in range(Nparam):
                    upl[i] = upi[i].copy()
                # % end Assignation of the solution to the last solution.


            # end % while stagn > stagn_crit && Niter < Niter_max                

            #pdb.set_trace()

            # Build approximation
            Approx_aux = 1.0
            for j in range(Nparam):
                if j < index_int:  # MATLAB: j == index_int (note: uses ==, not <=)
                    vec = IMp_all[j] @ upi[j]
                else:
                    #print(j)
                    vec = upi[j].T @ M_rhs_aux[j]
                Approx_aux = Approx_aux * vec        


            # % Save the solution to the last one:
            for i in range(Nparam):
                up[i] = np.hstack((up[i], upi[i]))
            # % end Save the solution to the last one    

            #pdb.set_trace()

            Fres_i = Fres_i - Approx_aux

            #Fres_i_vec = Fres_i.reshape(-1, 1, order='F')            
            error = 100.0 * np.linalg.norm(Fres_i.ravel()) / norm_Fres

            #error = 100.0 * np.linalg.norm(Fres_i.ravel(order="F")) / norm_Fres
            #error = 100 * norm(Fres_i) / norm_Fres  # Cambiar a norm(Fres_i(:))  
            m += 1

            print(f"sPGD mode {m}: error = {error:.4f}%")

        # end % while error>error_crit    



        # Final interpolation
        fP, d_fP, dd_fP = {}, {}, {}
        for i in range(Nparam):
            if i < index_int:  # MATLAB: i <= index_int
                fP[i] = up[i].reshape(dim[i], m)
                d_fP[i] = np.zeros_like(fP[i])
                dd_fP[i] = np.zeros_like(fP[i])
            else:
                up_aux = up[i].reshape(dim[i], 1, m)
                temp = up_aux * termsp[i][:, :, None]
                fP[i] = np.sum(temp, axis=0)
                
                temp_d = up_aux * d_termsp[i][:, :, None]
                d_fP[i] = np.sum(temp_d, axis=0)
                
                temp_dd = up_aux * dd_termsp[i][:, :, None]
                dd_fP[i] = np.sum(temp_dd, axis=0)

        MOR = {
            "up": up,
            "fP": fP,
            "d_fP": d_fP,
            "dd_fP": dd_fP,
            "index_int": index_int,
            "error": error_crit,
            "M_rhs_aux": M_rhs_aux,
            "dim": dim
        }

        print(f"sPGD finished in {time.time() - t0:.2f} s")
        return fP, d_fP, dd_fP, MOR, error



# ------------------------------------------------------
# Batch evaluation of MOR
# ------------------------------------------------------

def eval_MOR_point(MOR, param_sel):
    """
    Faithful translation of MATLAB eval_MOR

    Parameters
    ----------
    MOR : dict
        sPGD model
    param_sel : ndarray, shape (Nparam,)
        One parameter point

    Returns
    -------
    prediction : ndarray, shape (Nsensors,)
    """

    param_sel = np.asarray(param_sel).flatten()

    # --------------------------------------------------
    # MATLAB:
    # Nparam  = numel(MOR.param_vec);
    # ns = size(MOR.fP{1},1);
    # --------------------------------------------------
    Nparam = len(MOR["param_vec"])
    ns = MOR["fP"][0].shape[0]

    # --------------------------------------------------
    # MATLAB:
    # param_vec_init = zeros(1,Nparam);
    # ip = {};
    # --------------------------------------------------
    param_vec_init = np.zeros(Nparam, dtype=float)
    ip = []

    # --------------------------------------------------
    # Index identification
    # --------------------------------------------------
    for i in range(Nparam):
        param_vec_init[i] = MOR["param_vec_lim"][i][0]

        # MATLAB:
        # ip{i} = floor((param_sel(1,i)-param_vec_init(1,i)) / MOR.dp_vec{i}) + 1;
        #
        # Python: zero-based indexing → NO +1
        idx = int(
            np.floor(
                (param_sel[i] - param_vec_init[i]) / MOR["dp_vec"][i]
            )
        )

        ip.append(idx)

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------
    if MOR["index_int"] == 0:

        # MATLAB:
        # aux = 1;
        aux = 1.0

        # MATLAB:
        # aux = aux .* MOR.fP{j}(ip{j},:)
        for j in range(Nparam):
            aux = aux * MOR["fP"][j][ip[j], :]

        # MATLAB:
        # prediction = sum(aux,2);
        prediction = np.sum(aux, axis=1)

    else:
        # MATLAB:
        # aux = 1;
        # aux = aux .* MOR.fP{1};
        aux = MOR["fP"][0].copy()

        # MATLAB:
        # aux = aux .* MOR.fP{j+1}(ip{j},:)
        for j in range(Nparam):
            aux = aux * MOR["fP"][j + 1][ip[j], :]

        # MATLAB:
        # prediction = sum(aux,2);
        prediction = np.sum(aux, axis=1)

    return prediction



def eval_MOR(MOR, param_batch):
    """
    param_batch : (Nparam, Nsamples)
    returns     : (Nsensors, Nsamples)
    """

    param_batch = np.asarray(param_batch)
    Nparam, Nsamples = param_batch.shape

    ns = MOR["fP"][0].shape[0]
    predictions = np.zeros((ns, Nsamples))

    for k in range(Nsamples):
        predictions[:, k] = eval_MOR_point(
            MOR, param_batch[:, k]
        )

    return predictions






import numpy as np
from numpy.linalg import norm, pinv


def Inv_LM_algo_new(MOR, data_to_fit, Imposed_var, opt_var):

    act_zone = 1

    # LM parameters
    iter_again_max = 30
    alfa = 0.9
    iter_max_z = 7
    Error_i_max = 0.7

    Update_Type = 1
    lambda_0_ref = 1e-2
    epsilon_4 = 1e-1

    lambda_UP_fac = 11
    lambda_DN_fac = 9

    data_to_fit = np.asarray(data_to_fit).reshape(-1)

    N_imp_param = len(Imposed_var)
    Nparam = len(MOR["param_vec"])
    param_vec = np.zeros(Nparam)
    inva_Param = np.zeros(Nparam)
    param_vec_init = np.zeros(Nparam)
    Nparam_opt = Nparam - N_imp_param

    dp = MOR["dp_vec"]
    np_vec = MOR["np_vec"]

    # --------------------------------------------------
    # Initial parameter guess
    # --------------------------------------------------
    ip = {}
    ip_try = {}

    for i in range(Nparam):
        if i < N_imp_param:
            param_vec[i] = Imposed_var[i]
        else:
            param_vec[i] = np.min(opt_var[i]) + (np.max(opt_var[i]) - np.min(opt_var[i])) * np.random.rand()

        param_vec_init[i] = MOR["param_vec"][i][0]

        ip[i] = int(np.floor((param_vec[i] - param_vec_init[i]) / dp[i]))
        ip_try[i] = ip[i]
        inva_Param[:] = param_vec

    # --------------------------------------------------
    # Optimization loop
    # --------------------------------------------------
    save_results = []
    again = True
    index_again = 0

    while again:
        index_again += 1

        ind_param = sorted(set(MOR["param_index"][0]))

        # Random restart
        ind = 0
        for i in range(Nparam):
            if i < N_imp_param:
                inva_Param[i] = Imposed_var[i]
            else:
                if ind < len(ind_param) and i == ind_param[ind]:
                    ind += 1
                    inva_Param[i] = np.min(opt_var[i]) + (np.max(opt_var[i]) - np.min(opt_var[i])) * np.random.rand()

            param_vec_init[i] = MOR["param_vec"][i][0]
            ip[i] = int(np.floor((inva_Param[i] - param_vec_init[i]) / dp[i]))
            ip_try[i] = ip[i]

        # Fix indices
        for p in range(Nparam):
            if ip[p] >= np_vec[p] or ip[p] < 0:
                ip[p] = int(0.2 * np_vec[p])
            inva_Param[p] = ip[p] * dp[p] + param_vec_init[p]
            ip_try[p] = ip[p]

        # --------------------------------------------------
        # LM iterations
        # --------------------------------------------------
        iter_z = 0
        Error_i = 100
        lambda_0 = lambda_0_ref
        Nsensors_z = len(data_to_fit)
        param_vec_last = inva_Param.copy()

        while Error_i > Error_i_max and iter_z < iter_max_z:
            iter_z += 1

            pq = list(range(N_imp_param, Nparam))
            Nparam_zone = len(pq)

            # ---------------- Jacobian ----------------
            J = np.zeros((Nsensors_z, Nparam_zone))

            for q, p in enumerate(pq):
                vec = list(range(Nparam))
                vec.remove(p)
                vec = [p] + vec
                vec = vec[::-1]

                aux = MOR["fP"][0].copy()
                for j in vec[:-1]:
                    aux *= MOR["fP"][j + 1][ip[j], :]

                aux = np.sum(aux * MOR["d_fP"][vec[-1] + 1][ip[vec[-1]], :], axis=1)
                J[:, q] = aux

            # ---------------- Residual ----------------
            tvide = tvide_estimation_zone(Nsensors_z, Nparam, ip, MOR)
            diff = data_to_fit - tvide

            W = np.diag(data_to_fit ** 2)
            JtWJ = J.T @ W @ J
            JtWdy = J.T @ W @ diff

            lambda_val = lambda_0
            h = pinv(JtWJ + lambda_val * np.diag(np.diag(JtWJ))) @ JtWdy
            d_param_try = h.copy()

            # Trial update
            for q, p in enumerate(pq):
                param_try = param_vec_last[p] + alfa * d_param_try[q]
                ip_try[p] = int(np.floor((param_try - param_vec_init[p]) / dp[p]))

                if ip_try[p] >= np_vec[p] or ip_try[p] < 0:
                    ip_try[p] = int((0.1 + 0.5 * np.random.rand()) * np_vec[p])
                    param_try = ip_try[p] * dp[p] + param_vec_init[p]
                    d_param_try[q] = (param_try - param_vec_last[p]) / alfa

            tvide_try = tvide_estimation_zone(Nsensors_z, Nparam, ip_try, MOR)
            diff_try = data_to_fit - tvide_try

            X2 = diff.T @ W @ diff
            X2_try = diff_try.T @ W @ diff_try

            rho = (X2 - X2_try) / (h.T @ (lambda_val * np.diag(np.diag(JtWJ)) @ h + JtWdy))

            if rho > epsilon_4:
                lambda_val = max(lambda_val / lambda_DN_fac, 1e-7)
                d_param = d_param_try
            else:
                lambda_val = min(lambda_val * lambda_UP_fac, 1e7)
                d_param = np.zeros_like(d_param_try)

            # Update parameters
            for q, p in enumerate(pq):
                param_try = param_vec_last[p] + alfa * d_param[q]
                ip[p] = int(np.floor((param_try - param_vec_init[p]) / dp[p]))

                if ip[p] >= np_vec[p] or ip[p] < 0:
                    ip[p] = int(0.2 * np_vec[p])

                inva_Param[p] = ip[p] * dp[p] + param_vec_init[p]
                ip_try[p] = ip[p]

            param_vec_last = inva_Param.copy()
            tvide = tvide_estimation_zone(Nsensors_z, Nparam, ip, MOR)
            Error_i = 100 * norm(tvide - data_to_fit) / norm(data_to_fit)

        # Save result
        Error = 100 * norm(tvide - data_to_fit) / norm(data_to_fit)
        save_results.append([Error] + list(inva_Param))

        if index_again >= iter_again_max:
            break

    save_results = np.array(save_results)
    idx = np.argmin(save_results[:, 0])

    Error_min = save_results[idx, 0]
    param_vec = save_results[idx, 1:]

    print("Optimization finished")
    return Error_min, param_vec



def tvide_estimation_zone(Nsensors_z, Nparam, ip, MOR):

    aux = MOR["fP"][0].copy()
    for j in range(Nparam):
        aux *= MOR["fP"][j + 1][ip[j], :]

    return np.sum(aux, axis=1)
