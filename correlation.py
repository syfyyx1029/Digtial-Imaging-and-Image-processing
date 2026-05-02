import numpy as np
import cv2
from scipy.signal import fftconvolve


def _prepare_feature_map(img):
    """Use coarse gradient + contrast to match emoji body shapes, less sensitive to sharp text edges."""
    img = img.astype(np.float32)
    
    # Apply bilateral filter to smooth noise while preserving main structure
    bilateral = cv2.bilateralFilter(img, 5, 20, 20).astype(np.float32)
    
    # Compute coarse gradient (smooth transitions preferred over sharp edges)
    blurred = cv2.GaussianBlur(bilateral, (5, 5), 1.0)
    grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=5)
    gradient = cv2.magnitude(grad_x, grad_y)
    
    # Local contrast using morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph_close = cv2.morphologyEx(bilateral, cv2.MORPH_CLOSE, kernel)
    contrast = cv2.absdiff(bilateral, morph_close)
    
    # Combine: gradient-dominant to resist sharp text edges and emphasize body silhouette
    feature = gradient * 0.65 + contrast * 0.35
    
    max_value = float(np.max(feature))
    if max_value > 1e-6:
        feature = feature / max_value

    return feature

def get_block_sum(img_array, h, w):
    """
    【核心加速技巧 1】利用 Numpy 的累加和 (cumsum) 替代 for 循环的积分图运算。
    用于极速计算滑动窗口内的像素总和，彻底消灭滑动窗口循环！
    """
    # 1. 计算行方向的滑动和
    row_cumsum = np.cumsum(img_array, axis=1)
    row_cumsum = np.c_[np.zeros((img_array.shape[0], 1)), row_cumsum] # 补零方便相减
    row_sum = row_cumsum[:, w:] - row_cumsum[:, :-w]
    
    # 2. 在行滑动和的基础上，计算列方向的滑动和
    col_cumsum = np.cumsum(row_sum, axis=0)
    col_cumsum = np.r_[np.zeros((1, row_sum.shape[1])), col_cumsum]
    block_sum = col_cumsum[h:, :] - col_cumsum[:-h, :]
    
    return block_sum

def calculate_ncc(image, template):
    """
    纯矩阵运算版：标准归一化互相关 (Fast NCC)
    """
    image = _prepare_feature_map(image)
    template = _prepare_feature_map(template)

    th, tw = template.shape
    ih, iw = image.shape
    
    # 防止切图越界
    if ih < th or iw < tw:
        return (0, 0), -1.0

    # 统一转为 float32 防止计算溢出
    image = image.astype(np.float32)
    template = template.astype(np.float32)
    
    # 1. 模板处理：去均值，计算方差
    t_mean = float(np.mean(template))
    t_diff = template - t_mean
    t_var = float(np.sum(t_diff ** 2))
    
    # 纯色模板直接返回
    if t_var < 1e-4:
        return (0, 0), -1.0

    # =========================================================
    # 【核心加速技巧 2】计算 NCC 分子 (互相关) -> 利用 FFT 加速
    # 在频域中，互相关等价于将模板旋转 180 度后的卷积。
    # fftconvolve 的耗时是 O(N log N)，远小于纯循环的 O(N*M)
    # =========================================================
    t_diff_flipped = np.rot90(t_diff, 2)
    numerator = fftconvolve(image, t_diff_flipped, mode='valid')

    # =========================================================
    # 计算 NCC 分母 (图像局部方差)
    # 局部方差公式：Var = E[I^2] - (E[I])^2
    # 利用 FFT 加速卷积来计算局部均值和局部均方值
    # =========================================================
    ones = np.ones((th, tw), dtype=np.float32)
    ones_flipped = np.rot90(ones, 2)
    
    sum_I = fftconvolve(image, ones_flipped, mode='valid')
    sum_sqI = fftconvolve(image ** 2, ones_flipped, mode='valid')
    
    local_mean = sum_I / (th * tw)
    local_var = sum_sqI / (th * tw) - (local_mean ** 2)
    local_var = np.maximum(local_var, 0)
    denominator = np.sqrt(local_var * t_var)
    
    # =========================================================
    # 计算最终得分
    # =========================================================
    score_map = np.zeros_like(numerator) - 1.0
    valid_mask = denominator > 1e-4
    if np.any(valid_mask):
        score_map[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
    
    max_idx = np.argmax(score_map)
    best_y, best_x = np.unravel_index(max_idx, score_map.shape)
    best_score = score_map[best_y, best_x]
    
    return (int(best_x), int(best_y)), best_score

def hierarchical_search(image, template, scale=0.5):
    """
    基于 FFT 相关运算的分层搜索。
    缩放比例设为 0.5，既保证了极速，又防止模板缩放后过于模糊。
    """
    # 1. 粗匹配
    small_img = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    small_tpl = cv2.resize(template, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    coarse_pos, _ = calculate_ncc(small_img, small_tpl)
    
    approx_x = int(coarse_pos[0] / scale)
    approx_y = int(coarse_pos[1] / scale)
    
    # 2. 原图精确匹配区域提取
    margin = 40
    h, w = image.shape
    th, tw = template.shape
    
    x1 = max(0, approx_x - margin)
    y1 = max(0, approx_y - margin)
    x2 = min(w, approx_x + tw + margin)
    y2 = min(h, approx_y + th + margin)
    
    roi_image = image[y1:y2, x1:x2]
    
    # 精确匹配
    fine_pos, best_score = calculate_ncc(roi_image, template)
    
    final_x = x1 + fine_pos[0]
    final_y = y1 + fine_pos[1]
    
    return (final_x, final_y), best_score