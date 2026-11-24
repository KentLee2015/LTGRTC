# ltrgtc_gpu.py
import numpy as np
import time
import scipy.io
import gc

# try to import cupy for GPU; fallback to numpy if not available
try:
    import cupy as cp
    gpu_available = True
except Exception:
    cp = None
    gpu_available = False


# choose backend: xp will be cp (GPU) if available else numpy
def get_xp(use_gpu=True):
    if use_gpu and gpu_available:
        return cp
    else:
        return np


# -------------------------
# Config: set to True to use GPU if available
# -------------------------
USE_GPU = True

# dtype choice to reduce memory
DTYPE = np.float32  # will be converted to xp.float32 automatically


# -------------------------
# helpers (use xp inside functions)
# -------------------------
def ten2mat0(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order='F')


def mat2ten0(mat, tensor_size, mode):
    index = list()
    index.append(mode)
    for i in range(tensor_size.shape[0]):
        if i != mode:
            index.append(i)
    return np.moveaxis(np.reshape(mat, tensor_size[index].tolist(), order='F'), 0, mode)


def compute_mape(var, var_hat, xp):
    # var and var_hat may be xp arrays (small arrays)
    eps = 1e-12
    denom = xp.where(var == 0, eps, var)
    return xp.sum(xp.abs(var - var_hat) / denom) / var.shape[0]


def compute_mae(var, var_hat, xp):
    """Mean Absolute Error (same unit as input data)"""
    return xp.mean(xp.abs(var - var_hat))


def compute_rmse(var, var_hat, xp):
    return xp.sqrt(xp.sum((var - var_hat) ** 2) / var.shape[0])


def print_result(it, tol, var_cpu, var_hat_cpu):
    """
    var_cpu and var_hat_cpu should be numpy arrays (small arrays).
    Print MAPE, MAE and RMSE computed on CPU numpy.
    """
    print('Iter: {}'.format(it))
    print('Tolerance: {:.6}'.format(tol))
    eps = 1e-12
    mape = np.sum(np.abs(var_cpu - var_hat_cpu) /
                  np.where(var_cpu == 0, eps, var_cpu)) / var_cpu.shape[0]
    mae = np.mean(np.abs(var_cpu - var_hat_cpu))                       # <-- 新增 MAE
    rmse = np.sqrt(np.sum((var_cpu - var_hat_cpu) ** 2) / var_cpu.shape[0])
    print('Imputation MAPE: {:.6}'.format(mape))
    print('Imputation MAE : {:.6}'.format(mae))                        # <-- 新增 MAE
    print('Imputation RMSE: {:.6}'.format(rmse))
    print()


# -------------------------
# Laplacian: local consistency + periodic + weekly
# -------------------------
def laplacian(T, tau=1, period=7, gamma=0.5, gamma2=0.5):
    """
    组合算子：时间局部一致性 + 近似周期性 + 周一致性

    ell = ell_time + gamma * ell_period + gamma2 * ell_day
    """
    if tau == 0:
        return np.zeros(T)

    # ---- (1) 时间差分项----
    ell_time = np.zeros(T)
    ell_time[0] = 2 * tau
    for k in range(tau):
        ell_time[k + 1] = -1
        ell_time[-k - 1] = -1

    # ---- (2) 周期差分项 ----
    ell_period = np.zeros(T)
    ell_period[0] = 2.0
    if period < T:
        ell_period[period] = -1.0
    if period < T - period:
        ell_period[-period] = -1.0

    # ---- (3) 周差分项 ----
    ell_day = np.zeros(T)
    ell_day[0] = 2.0
    ell_day[int(period / 7)] = -1.0
    ell_day[-int(period / 7)] = -1.0

    # ---- (4) 组合 ----
    ell = ell_time + gamma * ell_period + gamma2 * ell_day
    return ell


# -------------------------
# more robust SVT / TNN (uses xp.linalg.svd)
# -------------------------
def svt_tnn(mat, tau, theta, xp):
    """
    SVT + truncated-nuclear-norm style shrink using xp backend.
    mat: xp array
    Returns xp array
    """
    if mat.size == 0:
        return mat

    if mat.dtype != xp.float32 and mat.dtype != xp.float64:
        mat = mat.astype(xp.float32)

    U, s, Vt = xp.linalg.svd(mat, full_matrices=False)
    s_shrink = xp.maximum(s - tau, 0.0)
    theta = int(theta)
    if theta < len(s):
        s_shrink[:theta] = s[:theta]
    nonzero = s_shrink > 0
    if xp.sum(nonzero) == 0:
        return xp.zeros_like(mat)
    r = int(xp.sum(nonzero).item())
    U_r = U[:, :r]
    Vt_r = Vt[:r, :]
    S_r = xp.diag(s_shrink[:r])
    return U_r @ S_r @ Vt_r

# -------------------------
# ltgrtc
# -------------------------
def ltgrtc(dense_tensor_np, sparse_tensor_np, tau_t, alpha, rho0, lambda0,
         theta, peroid, gamma, gamma2,
         epsilon=1e-4, maxiter=100, K=1, use_gpu=USE_GPU,
         return_metrics=False):
    """

    Parameters
    ----------
    dense_tensor_np : np.ndarray
    sparse_tensor_np : np.ndarray
    time_lags : np.ndarray
    alpha : list
    rho0 : float
    lambda0 : float
    theta : int
    peroid : int
    gamma : float
    gamma2 : float
    epsilon : float
    maxiter : int
    K : int
    use_gpu : bool
    return_metrics : bool
        If True, return (tensor_hat_cpu, metrics_dict) where
        metrics_dict = {'MAPE':..., 'MAE':..., 'RMSE':...}

    Returns
    -------
    tensor_hat_cpu : np.ndarray
    metrics_dict : dict (only if return_metrics=True)
    """
    xp = get_xp(use_gpu)

    dim = np.array(sparse_tensor_np.shape, dtype=int)
    dim_time = int(np.prod(dim) / dim[0])

    dense_tensor = xp.asarray(dense_tensor_np, dtype=DTYPE)
    sparse_tensor = xp.asarray(sparse_tensor_np, dtype=DTYPE)

    sparse_mat = ten2mat(sparse_tensor, 0, xp)
    obs_mask = (sparse_mat != 0)
    pos_missing = xp.where(~obs_mask)

    pos_test_cpu = np.where((dense_tensor_np != 0) & (sparse_tensor_np == 0))
    dense_test_cpu = dense_tensor_np[pos_test_cpu]

    T = xp.zeros(dim, dtype=DTYPE)
    Z_tensor = sparse_tensor.copy().astype(DTYPE)
    Z = ten2mat(Z_tensor, 0, xp).astype(DTYPE)

    it = 0
    last_mat = ten2mat(Z_tensor, 0, xp).copy()
    snorm_val = float(xp.linalg.norm(sparse_mat, 'fro'))
    snorm = snorm_val if snorm_val != 0 else 1.0
    rho = rho0

    ell_t_cpu = laplacian(dim_time, tau_t, peroid, gamma, gamma2)
    ell_t = xp.asarray(ell_t_cpu, dtype=DTYPE)
    ell = xp.fft.fft(ell_t)
    ell2 = xp.abs(ell) ** 2

    tensor_hat = None
    tol = np.inf

    try:
        while True:
            # X-update
            tensor_hat = xp.zeros(dim, dtype=DTYPE)
            for p in range(len(dim)):
                M = ten2mat(Z_tensor - T / rho, p, xp)
                M_svt = svt_tnn(M, alpha[p] / rho, theta, xp)
                tensor_hat += alpha[p] * mat2ten(M_svt, dim, p, xp)

            # inner loop: Z-update and T-update
            for inner_k in range(K):
                rho = min(rho * 1.05, 1e5)               
                rhs_base = rho * ten2mat(tensor_hat, 0, xp) + ten2mat(T, 0, xp)
                mat_fft = xp.fft.fft(rhs_base, axis=1)
                denominator = rho + lambda0 * ell2
                mat_freq = mat_fft / denominator[xp.newaxis, :]
                Z_new_full = xp.fft.ifft(mat_freq, axis=1).real

                Z_new = Z_new_full.copy()
                Z_new[obs_mask] = sparse_mat[obs_mask]

                Z = Z_new
                Z_tensor = mat2ten(Z, dim, 0, xp)
            T = T + rho * (tensor_hat - Z_tensor)

            mat_hat = ten2mat(tensor_hat, 0, xp)
            tol = float(xp.linalg.norm((mat_hat - last_mat), 'fro') / snorm)
            last_mat = mat_hat.copy()
            it += 1

            if it % 200 == 0:
                pred_test = xp.asnumpy(tensor_hat[pos_test_cpu])
                print_result(it, tol, dense_test_cpu, pred_test)

            if (tol < epsilon) or (it >= maxiter):
                break

    finally:
        try:
            pred_test = xp.asnumpy(tensor_hat[pos_test_cpu])
            print_result(it, tol, dense_test_cpu, pred_test)
        except Exception:
            pass

    # ---- 返回 CPU tensor + 最终指标 ----
    tensor_hat_cpu = xp.asnumpy(tensor_hat)

    eps = 1e-12
    pred_test = tensor_hat_cpu[pos_test_cpu]
    mape_final = float(np.sum(np.abs(dense_test_cpu - pred_test) /
                              np.where(dense_test_cpu == 0, eps, dense_test_cpu))
                       / dense_test_cpu.shape[0])
    mae_final = float(np.mean(np.abs(dense_test_cpu - pred_test)))           # <-- 新增 MAE
    rmse_final = float(np.sqrt(np.mean((dense_test_cpu - pred_test) ** 2)))

    # clean up GPU memory
    try:
        del dense_tensor, sparse_tensor, sparse_mat, Z, Z_tensor, T, tensor_hat
        del mat_fft, Z_new_full, rhs_base, last_mat, mat_hat
    except Exception:
        pass
    gc.collect()
    if gpu_available and use_gpu:
        try:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass

    if return_metrics:
        return tensor_hat_cpu, {
            'MAPE': mape_final,
            'MAE': mae_final,
            'RMSE': rmse_final,
        }
    return tensor_hat_cpu

if __name__ == "__main__":    
    dense_mat = np.load('../input/datasets/kent2020/transdim/Portland-data-set/volume.npy')
    dim1, dim2 = dense_mat.shape
    dim = np.array([dim1, 96, 31])
    dense_tensor = mat2ten0(dense_mat, dim, 0)
    period = 96 * 7
    block = 96

##########################
    rmse =[]
    mae =[]
    for seed in [1000,700,300]:
        print('seed:',seed)
        np.random.seed(seed)
        mask, y_ture = create_missing_data_mask(
            dense_mat, missing_rate=0.3, missing_type='column_block',
            block_length=block, noise_type=None, noise_level=0
        )
        print('y_ture:', y_ture.shape)
        sparse_tensor = mat2ten0(y_ture, dim, 0)
    
        gamma2 = gamma = 0.2
        for theta in [1e-6]:
            for tau in [15]:
                
                alpha = np.ones(3) / 3          
                phi =1*theta
    
                for pi in [2]:
                    print('gamma=', gamma, 'theta=', theta, 'tau=', tau,
                          'pi=', pi, ' USE_GPU=', USE_GPU and gpu_available)
                    start = time.time()
                    tensor_hat, metrics = ltgrtc(
                        dense_tensor, sparse_tensor,
                        pi, alpha, theta, phi, tau,
                        period, gamma, gamma2,
                        epsilon=1e-4, maxiter=200, K=1,
                        use_gpu=USE_GPU, return_metrics=True
                    )
                    end = time.time()
    
                    print(f'\nTotal running time: {end - start:.2f} seconds')
                    rmse.append(metrics['RMSE'])
                    mae.append(metrics['MAE'])
                    print()
            
    # 打印汇总结果
    print(f'\n{"="*60}')
    print(f"  MAE  : {np.mean(mae):.4f} ± {np.std(mae):.4f}")
    print(f"  RMSE : {np.mean(rmse):.4f} ± {np.std(rmse):.4f}")
    print(f"{'='*60}")

    gc.collect()
    if gpu_available and USE_GPU:
        try:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass
