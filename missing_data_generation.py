
import numpy as np


def create_missing_data_mask(y_true, missing_rate=0.1, missing_type='random',
                             block_length=2, noise_type=None, noise_level=0.1):
    """
    创建缺失掩码并添加噪声

    missing_type:
        'random'       - 随机独立缺失
        'row_block'    - 每行连续时间块缺失
        'column_block' - 所有行在相同连续时间块上缺失

    block_length:
        块状缺失的长度（固定值）

    noise_type:
        None           - 无噪声
        'half_mixed'   - 50%的观测值添加混合噪声
        'full_mixed'   - 100%的观测值添加混合噪声
    """

    # 确保是 NumPy 数组（如果输入是 CuPy 数组）
    if 'cupy' in str(type(y_true)):
        y_true = y_true.get()

    y_true = y_true.astype(np.float64)
    N, T = y_true.shape
    mask = np.ones((N, T), dtype=bool)

    # 创建缺失模式
    if missing_type == 'random':

        # 随机缺失
        missing_mask = np.random.rand(N, T) < missing_rate
        mask[missing_mask] = False

    elif missing_type == 'row_block':

        # 原始 row_block 逻辑，保持不变
        total_elements = N * T
        target_missing = int(total_elements * missing_rate)

        if target_missing > 0:
            current_missing = 0
            max_attempts = 1000
            attempts = 0

            row_targets = [int(T * missing_rate) for _ in range(N)]

            diff = target_missing - sum(row_targets)

            for i in range(abs(diff)):
                row_targets[i % N] += 1 if diff > 0 else -1

            while current_missing < target_missing and attempts < max_attempts:

                i = np.random.randint(0, N)

                current_row_missing = T - np.sum(mask[i])

                if current_row_missing >= row_targets[i]:
                    continue

                start_t = np.random.randint(0, T - block_length + 1)
                end_t = start_t + block_length

                if np.all(mask[i, start_t:end_t]):

                    if current_row_missing + block_length <= row_targets[i]:

                        mask[i, start_t:end_t] = False
                        current_missing += block_length

                    else:

                        remaining_needed = row_targets[i] - current_row_missing

                        if remaining_needed > 0:

                            available_cols = np.where(mask[i, start_t:end_t])[0]

                            if len(available_cols) >= remaining_needed:

                                selected = np.random.choice(
                                    available_cols,
                                    remaining_needed,
                                    replace=False
                                )

                                mask[i, start_t + selected] = False
                                current_missing += remaining_needed

                attempts += 1

            # 如果没有达到目标，使用随机缺失补足
            if current_missing < target_missing:

                remaining_needed = target_missing - current_missing
                missing_positions = np.where(mask)

                if len(missing_positions[0]) > 0:

                    n_to_miss = min(
                        remaining_needed,
                        len(missing_positions[0])
                    )

                    indices = np.random.choice(
                        len(missing_positions[0]),
                        n_to_miss,
                        replace=False
                    )

                    for idx in indices:

                        i, j = missing_positions[0][idx], missing_positions[1][idx]
                        mask[i, j] = False

    elif missing_type == 'column_block':

        # 精确控制总体缺失数量
        total_elements = N * T
        target_missing = int(total_elements * missing_rate)

        if target_missing > 0:

            # 优先生成完整同步缺失列块
            full_columns = target_missing // N
            remaining_elements = target_missing % N

            available = np.ones(T, dtype=bool)
            selected_blocks = []

            remaining_columns = full_columns

            while remaining_columns > 0:

                current_length = min(block_length, remaining_columns)

                # 寻找不重叠的连续列块
                valid_starts = [
                    s for s in range(T - current_length + 1)
                    if np.all(available[s:s + current_length])
                ]

                if valid_starts:

                    start_col = np.random.choice(valid_starts)
                    end_col = start_col + current_length

                    mask[:, start_col:end_col] = False
                    available[start_col:end_col] = False

                    selected_blocks.append((start_col, end_col))
                    remaining_columns -= current_length

                else:

                    # 无法继续生成连续块时，随机选择剩余可用列
                    available_cols = np.where(available)[0]

                    selected_cols = np.random.choice(
                        available_cols,
                        remaining_columns,
                        replace=False
                    )

                    mask[:, selected_cols] = False
                    available[selected_cols] = False

                    remaining_columns = 0

            # 处理剩余不足一整列的缺失数量
            if remaining_elements > 0:

                available_cols = np.where(available)[0]

                partial_col = np.random.choice(available_cols)

                selected_rows = np.random.choice(
                    N,
                    remaining_elements,
                    replace=False
                )

                mask[selected_rows, partial_col] = False

    else:

        raise ValueError(
            "missing_type must be 'random', 'row_block' or 'column_block'."
        )

    # 验证实际缺失率
    actual_missing_rate = 1 - np.mean(mask)

    print(
        f"目标缺失率: {missing_rate:.4f}, "
        f"实际缺失率: {actual_missing_rate:.4f}"
    )

    # 创建带缺失的数据（缺失位置设为0）
    y_obs_with_missing = y_true.copy()
    y_obs_with_missing[~mask] = 0

    # 添加噪声到观测数据
    if noise_type is not None:

        if noise_type == 'half_mixed':

            y_obs_with_missing = add_half_mixed_noise(
                y_obs_with_missing,
                mask,
                noise_level
            )

        elif noise_type == 'full_mixed':

            y_obs_with_missing = add_full_mixed_noise(
                y_obs_with_missing,
                mask,
                noise_level
            )

        else:

            raise ValueError(
                "noise_type must be None, 'half_mixed' or 'full_mixed'."
            )

    return mask, y_obs_with_missing


def add_half_mixed_noise(y_true, mask, noise_level):
    """
    添加50%观测值的混合噪声：高斯噪声 + 异常值
    只对50%的观测值添加噪声
    """

    y_noisy = y_true.copy()
    obs_positions = np.where(mask)
    n_obs = len(obs_positions[0])

    if n_obs > 0:

        n_noisy = int(n_obs * 0.5)

        noisy_indices = np.random.choice(
            n_obs,
            n_noisy,
            replace=False
        )

        noisy_positions = (
            obs_positions[0][noisy_indices],
            obs_positions[1][noisy_indices]
        )

        if len(noisy_positions[0]) > 0:

            data_std = np.std(y_true[noisy_positions])

            gaussian_noise = np.random.normal(
                0,
                data_std * noise_level * 0.9,
                size=len(noisy_positions[0])
            )

            y_noisy[noisy_positions] += gaussian_noise

            n_noisy_points = len(noisy_positions[0])
            n_outliers = int(n_noisy_points * noise_level * 0.1)

            if n_outliers > 0:

                outlier_indices = np.random.choice(
                    n_noisy_points,
                    n_outliers,
                    replace=False
                )

                data_mean = np.mean(y_true[noisy_positions])
                data_std = np.std(y_true[noisy_positions])

                for idx in outlier_indices:

                    i = noisy_positions[0][idx]
                    j = noisy_positions[1][idx]

                    outlier_magnitude = np.random.uniform(2, 3) * data_std
                    sign = 1 if np.random.rand() > 0.5 else -1

                    y_noisy[i, j] = data_mean + sign * outlier_magnitude

    return y_noisy


def add_full_mixed_noise(y_true, mask, noise_level):
    """
    添加100%观测值的混合噪声：高斯噪声 + 异常值
    对所有观测值都添加噪声
    """

    y_noisy = y_true.copy()
    obs_positions = np.where(mask)
    n_obs = len(obs_positions[0])

    if n_obs > 0:

        data_std = np.std(y_true[obs_positions])

        gaussian_noise = np.random.normal(
            0,
            data_std * noise_level * 0.9,
            size=n_obs
        )

        y_noisy[obs_positions] += gaussian_noise

        n_outliers = int(n_obs * noise_level * 0.1)

        if n_outliers > 0:

            outlier_indices = np.random.choice(
                n_obs,
                n_outliers,
                replace=False
            )

            data_mean = np.mean(y_true[obs_positions])
            data_std = np.std(y_true[obs_positions])

            for idx in outlier_indices:

                i = obs_positions[0][idx]
                j = obs_positions[1][idx]

                outlier_magnitude = np.random.uniform(4, 5) * data_std
                sign = 1 if np.random.rand() > 0.5 else -1

                y_noisy[i, j] = data_mean + sign * outlier_magnitude

    return y_noisy
