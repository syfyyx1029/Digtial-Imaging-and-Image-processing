import cv2
import numpy as np
from pathlib import Path
import shutil
import tempfile
from correlation import calculate_ncc, hierarchical_search


def _clamp_roi(x1, y1, x2, y2, width, height):
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def _smooth_point(prev_point, new_point, alpha=0.35):
    if prev_point is None:
        return new_point
    prev_x, prev_y = prev_point
    new_x, new_y = new_point
    smooth_x = int(round(prev_x * (1.0 - alpha) + new_x * alpha))
    smooth_y = int(round(prev_y * (1.0 - alpha) + new_y * alpha))
    return smooth_x, smooth_y


def _rank_candidates(image, templates, roi_origin=(0, 0), predicted_center=None, preferred_tpl_idx=None, search_radius=120):
    candidates = []
    roi_x, roi_y = roi_origin

    for idx, tpl in enumerate(templates):
        pos, score = calculate_ncc(image, tpl)
        th, tw = tpl.shape
        center_x = roi_x + pos[0] + tw // 2
        center_y = roi_y + pos[1] + th // 2

        adjusted_score = float(score)
        if predicted_center is not None and search_radius > 0:
            dist = float(np.hypot(center_x - predicted_center[0], center_y - predicted_center[1]))
            adjusted_score -= min(0.30, dist / max(1.0, search_radius * 2.5))

        if preferred_tpl_idx is not None and idx == preferred_tpl_idx:
            adjusted_score += 0.015

        candidates.append({
            'idx': idx,
            'pos': pos,
            'score': float(score),
            'adjusted_score': adjusted_score,
            'shape': tpl.shape,
            'center': (center_x, center_y),
        })

    candidates.sort(key=lambda item: item['adjusted_score'], reverse=True)
    return candidates


def clamp_roi(center_x, center_y, template_shape, width, height, radius):
    th, tw = template_shape
    x1 = max(0, int(center_x - tw // 2 - radius))
    y1 = max(0, int(center_y - th // 2 - radius))
    x2 = min(width, int(center_x + tw // 2 + radius))
    y2 = min(height, int(center_y + th // 2 + radius))
    return x1, y1, x2, y2


def load_gray_image(image_path):
    image_data = np.fromfile(str(image_path), dtype=np.uint8)
    if image_data.size == 0:
        return None
    return cv2.imdecode(image_data, cv2.IMREAD_GRAYSCALE)


def trim_template(template, threshold=245, padding=12):
    """Crop the white border around a template with generous padding to retain full character shape."""
    if template is None or template.size == 0:
        return template

    foreground = template < threshold
    if not np.any(foreground):
        return template

    ys, xs = np.where(foreground)
    y1 = max(0, int(ys.min()) - padding)
    y2 = min(template.shape[0], int(ys.max()) + padding + 1)
    x1 = max(0, int(xs.min()) - padding)
    x2 = min(template.shape[1], int(xs.max()) + padding + 1)
    return template[y1:y2, x1:x2]


def get_template_geometry(template, threshold=245):
    """Return foreground bounding box and centroid relative to a template image."""
    height, width = template.shape[:2]
    foreground = template < threshold

    if not np.any(foreground):
        return {
            "bbox": (0, 0, width, height),
            "anchor": (width // 2, height // 2),
            "size": (width, height),
        }

    ys, xs = np.where(foreground)
    x1 = int(xs.min())
    x2 = int(xs.max()) + 1
    y1 = int(ys.min())
    y2 = int(ys.max()) + 1
    anchor_x = int(round((x1 + x2) / 2.0))
    anchor_y = int(round((y1 + y2) / 2.0))

    return {
        "bbox": (x1, y1, x2, y2),
        "anchor": (anchor_x, anchor_y),
        "size": (x2 - x1, y2 - y1),
    }


def select_best_template(roi_img, templates, template_geometries, last_tpl_idx, pred_center=None, roi_offset=(0, 0), search_radius=100):
    candidates = []
    x_offset, y_offset = roi_offset

    for idx, tpl in enumerate(templates):
        meta = template_geometries[idx]
        pos, score = calculate_ncc(roi_img, tpl)
        if score < -0.9:
            continue

        abs_x = x_offset + pos[0]
        abs_y = y_offset + pos[1]
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = meta["bbox"]
        anchor_x, anchor_y = meta["anchor"]
        cand_cx = abs_x + anchor_x
        cand_cy = abs_y + anchor_y

        # Spatial filtering: reject candidates too far from prediction
        if pred_center is not None:
            dist = np.hypot(cand_cx - pred_center[0], cand_cy - pred_center[1])
            max_allowed_dist = max(100, search_radius * 1.5)  # Relaxed: allow farther matches for fast motion
            if dist > max_allowed_dist:
                continue

        adjusted_score = float(score)
        
        if pred_center is not None:
            dist = np.hypot(cand_cx - pred_center[0], cand_cy - pred_center[1])
            # Strong but not excessive penalty for far candidates
            adjusted_score -= dist * 0.025

        if idx == last_tpl_idx:
            adjusted_score += 0.05  # Strong template stability bias
        else:
            adjusted_score -= 0.03  # Penalty for template switch

        candidates.append({
            "idx": idx,
            "pos": pos,
            "abs_x": abs_x,
            "abs_y": abs_y,
            "bbox": meta["bbox"],
            "size": meta["size"],
            "anchor": meta["anchor"],
            "center": (cand_cx, cand_cy),
            "raw_score": float(score),
            "adjusted_score": adjusted_score,
        })

    if not candidates:
        return None, []

    candidates.sort(key=lambda item: item["adjusted_score"], reverse=True)
    return candidates[0], candidates

def main():
    print("正在初始化模板...")
    base_dir = Path(__file__).resolve().parent
    temp_dir = Path(tempfile.gettempdir()) / "emoji_tracker_task1"
    temp_dir.mkdir(parents=True, exist_ok=True)
    # 1. 读取所有的模板，并转换为灰度图
    templates = []
    template_geometries = []
    template_paths = [
        base_dir / 'data' / 'small_template_0.png',
        base_dir / 'data' / 'small_template_1.png', 
        base_dir / 'data' / 'small_template_2.png', 
        base_dir / 'data' / 'small_template_3.png',
        base_dir / 'data' / 'small_template_4.png',
        base_dir / 'data' / 'small_template_5.png'
    ]
    for path in template_paths:
        tpl = load_gray_image(path)
        if tpl is None:
            print(f"找不到模板文件: {path}，请检查路径。")
            return
        tpl = trim_template(tpl)
        templates.append(tpl)
        template_geometries.append(get_template_geometry(tpl))

    # 2. 打开视频文件
    video_path = base_dir / 'data' / '动画表情视频.mp4'
    temp_input_video = temp_dir / 'input_video.mp4'
    shutil.copyfile(str(video_path), str(temp_input_video))
    cap = cv2.VideoCapture(str(temp_input_video))
    
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    # 获取视频属性用于保存输出文件
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # 3. 设置视频输出
    output_dir = base_dir / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / 'result_tracked.mp4'
    temp_output_path = temp_dir / 'result_tracked.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(temp_output_path), fourcc, fps, (width, height))

    # 初始化跟踪变量
    is_first_frame = True
    track_history = []  # 保存历史中心点用于绘制轨迹
    last_cx, last_cy = 0, 0
    last_valid_cx, last_valid_cy = 0, 0  # 记录最后一个成功跟踪的位置
    last_tpl_shape = (0, 0)  # 【修复点】新增变量：记录上一帧目标的大小
    last_tpl_bbox = (0, 0, 0, 0)
    last_tpl_anchor = (0, 0)
    last_tpl_idx = 0
    last_dx, last_dy = 0, 0
    smooth_cx, smooth_cy = 0.0, 0.0
    smooth_bw, smooth_bh = 0.0, 0.0
    smooth_box_cx, smooth_box_cy = 0.0, 0.0
    tracking_lost = False
    search_radius = 200  # Very expanded search radius
    min_search_radius = 150  # Increased
    max_search_radius = 350  # Much larger
    smooth_alpha = 0.10  # Maximum smoothing to minimize trajectory jitter
    accept_threshold = 0.02  # Very lenient: accept almost any local match
    first_frame_confirm_threshold = 0.30  # Relaxed: easier first frame confirmation
    reacquire_threshold = 0.25  # Relaxed: easier re-acquisition
    reacquire_gap = 0.10  # Relaxed: smaller gap requirement
    consecutive_loss_limit = 5  # Allow more consecutive loss frames before giving up
    consecutive_loss_count = 0  # Track how many consecutive frames have been lost
    frame_count = 0
    trajectory_max_points = 45
    trajectory_min_step = 3
    trajectory_jump_limit = 80

    print("开始处理视频，这可能需要一点时间，请耐心等待...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (3, 3), 0)
        
        # 这些是当前帧的临时最优记录
        best_global_score = -2.0
        best_global_pos = (0, 0)
        best_tpl_idx = 0
        best_tpl_shape = (0, 0)
        best_raw_score = -2.0
        display_x, display_y = 0, 0
        frame_visible = True

        # ================= 第一帧：全局金字塔搜索 =================
        if is_first_frame or tracking_lost:
            print("正在全图搜索第一帧，请稍候...")
            second_best_score = -2.0
            for i, tpl in enumerate(templates):
                # 首帧采用全图直接 NCC 获得最大精度
                pos, score = calculate_ncc(gray_frame, tpl)
                if score > best_global_score:
                    second_best_score = best_global_score
                    best_global_score = score
                    best_global_pos = pos
                    best_tpl_idx = i
                    best_tpl_shape = tpl.shape
                    best_raw_score = score
                elif score > second_best_score:
                    second_best_score = score
            
            # Double-check first frame: if top match is close to second match or below threshold, use hierarchical search
            if best_raw_score < first_frame_confirm_threshold or (second_best_score > best_raw_score - 0.12):
                print(f"第一帧初始匹配质量不足 (得分 {best_raw_score:.2f}，次优 {second_best_score:.2f})，进行层级精细搜索...")
                best_global_score = -2.0
                for i, tpl in enumerate(templates):
                    pos, score = hierarchical_search(gray_frame, tpl, scale=0.6)
                    if score > best_global_score:
                        best_global_score = score
                        best_global_pos = pos
                        best_tpl_idx = i
                        best_tpl_shape = tpl.shape
                        best_raw_score = score

            # First frame is only accepted when it is clearly better than alternatives.
            if best_raw_score < first_frame_confirm_threshold or (second_best_score > best_raw_score - 0.05):
                tracking_lost = True
                frame_visible = False
                is_first_frame = False
            else:
                best_x, best_y = best_global_pos
                first_meta = template_geometries[best_tpl_idx]
                last_tpl_bbox = first_meta["bbox"]
                last_tpl_anchor = first_meta["anchor"]
                last_cx = best_x + last_tpl_anchor[0]
                last_cy = best_y + last_tpl_anchor[1]
                smooth_cx, smooth_cy = float(last_cx), float(last_cy)
                display_x, display_y = best_x, best_y
                
                # 【修复点】第一帧结束后，记录目标的长宽，供下一帧使用
                last_tpl_shape = best_tpl_shape 
                last_tpl_idx = best_tpl_idx
                is_first_frame = False
                tracking_lost = False
                last_valid_cx, last_valid_cy = last_cx, last_cy
                consecutive_loss_count = 0

                bbox_x1, bbox_y1, bbox_x2, bbox_y2 = last_tpl_bbox
                smooth_bw = float((bbox_x2 - bbox_x1) + 24)
                smooth_bh = float((bbox_y2 - bbox_y1) + 24)
                smooth_box_cx = float(display_x + (bbox_x1 + bbox_x2) / 2.0)
                smooth_box_cy = float(display_y + (bbox_y1 + bbox_y2) / 2.0)
                box_x1 = max(0, int(smooth_box_cx - smooth_bw / 2))
                box_y1 = max(0, int(smooth_box_cy - smooth_bh / 2))
                box_x2 = min(width, int(smooth_box_cx + smooth_bw / 2))
                box_y2 = min(height, int(smooth_box_cy + smooth_bh / 2))
                cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), 2)
                center_x = (box_x1 + box_x2) // 2
                center_y = (box_y1 + box_y2) // 2
            

        # ================= 后续帧：局部 ROI 预测追踪 =================
        else:
            th, tw = last_tpl_shape

            pred_cx = int(last_cx + last_dx)
            pred_cy = int(last_cy + last_dy)
            motion_radius = int(max(min_search_radius, min(max_search_radius, max(abs(last_dx), abs(last_dy)) * 3 + search_radius // 2)))

            # 目标位置预测：使用最近位移估计来缩小搜索区域，减少误检跳变
            x1, y1, x2, y2 = clamp_roi(pred_cx, pred_cy, last_tpl_shape, width, height, motion_radius)
            
            roi_img = gray_frame[y1:y2, x1:x2]

            pred_center = (pred_cx, pred_cy)
            best_local, candidates = select_best_template(
                roi_img,
                templates,
                template_geometries,
                last_tpl_idx,
                pred_center=pred_center,
                roi_offset=(x1, y1),
                search_radius=motion_radius,
            )

            if best_local is None:
                best_global_score = -2.0
            else:
                best_global_score = best_local["adjusted_score"]
                best_raw_score = best_local["raw_score"]
                best_global_pos = best_local["pos"]
                best_tpl_idx = best_local["idx"]
                best_tpl_shape = templates[best_tpl_idx].shape
                last_tpl_bbox = best_local["bbox"]
                last_tpl_anchor = best_local["anchor"]

            if best_local is None or best_local["raw_score"] < accept_threshold:
                tracking_lost = True
                frame_visible = False
            else:
                candidate_cx, candidate_cy = best_local["center"]
                predicted_cx = pred_cx if last_cx != 0 or last_cy != 0 else candidate_cx
                predicted_cy = pred_cy if last_cx != 0 or last_cy != 0 else candidate_cy
                jump_limit = max(120, int(motion_radius * 1.2))  # Relaxed: allow large motion in handwaving
                jump_dist = float(np.hypot(candidate_cx - last_cx, candidate_cy - last_cy))

                if jump_dist > jump_limit:
                    tracking_lost = True
                    frame_visible = False
                else:
                    best_x = best_local["abs_x"]
                    best_y = best_local["abs_y"]

                    new_cx = best_x + last_tpl_anchor[0]
                    new_cy = best_y + last_tpl_anchor[1]

                    if last_cx == 0 and last_cy == 0:
                        smooth_cx, smooth_cy = float(new_cx), float(new_cy)
                    else:
                        smooth_cx = smooth_alpha * new_cx + (1 - smooth_alpha) * smooth_cx
                        smooth_cy = smooth_alpha * new_cy + (1 - smooth_alpha) * smooth_cy

                    last_dx = int(round(smooth_cx)) - last_cx
                    last_dy = int(round(smooth_cy)) - last_cy
                    last_cx, last_cy = int(round(smooth_cx)), int(round(smooth_cy))

                    # 【修复点】当前帧处理完，更新记录供下一帧使用
                    last_tpl_shape = best_tpl_shape
                    last_tpl_idx = best_tpl_idx
                    tracking_lost = False
                    frame_visible = True
                    display_x, display_y = best_x, best_y
                    last_valid_cx, last_valid_cy = last_cx, last_cy  # Update valid position
                    consecutive_loss_count = 0  # Reset loss counter on successful tracking

            if tracking_lost:
                consecutive_loss_count += 1
                # Only try to re-acquire if we haven't been lost for too many consecutive frames
                if consecutive_loss_count <= consecutive_loss_limit:
                    best_global_score = -2.0
                    second_best_score = -2.0
                    for i, tpl in enumerate(templates):
                        pos, score = hierarchical_search(gray_frame, tpl)
                        if score > best_global_score:
                            second_best_score = best_global_score
                            best_global_score = score
                            best_global_pos = pos
                            best_tpl_idx = i
                            best_tpl_shape = tpl.shape
                            best_raw_score = score
                        elif score > second_best_score:
                            second_best_score = score
                else:
                    best_raw_score = -2.0  # Force re-acquire to fail if lost too long

                if best_raw_score >= reacquire_threshold and (best_raw_score - second_best_score) >= reacquire_gap:
                    best_x, best_y = best_global_pos
                    best_meta = template_geometries[best_tpl_idx]
                    last_tpl_bbox = best_meta["bbox"]
                    last_tpl_anchor = best_meta["anchor"]
                    new_cx = best_x + last_tpl_anchor[0]
                    new_cy = best_y + last_tpl_anchor[1]
                    
                    # Sanity check: new position should not be too far from last valid position
                    reacquire_max_dist = 400  # Max distance allowed for re-acquire (more lenient)
                    if last_valid_cx > 0 and last_valid_cy > 0:
                        reacquire_dist = np.hypot(new_cx - last_valid_cx, new_cy - last_valid_cy)
                        if reacquire_dist > reacquire_max_dist:
                            # Reject this re-acquire, position is too far
                            pass
                        else:
                            last_cx = new_cx
                            last_cy = new_cy
                            smooth_cx, smooth_cy = float(last_cx), float(last_cy)
                            last_tpl_shape = best_tpl_shape
                            last_tpl_idx = best_tpl_idx
                            tracking_lost = False
                            frame_visible = True
                            display_x, display_y = best_x, best_y
                    else:
                        # No valid previous position, accept re-acquire
                        last_cx = new_cx
                        last_cy = new_cy
                        smooth_cx, smooth_cy = float(last_cx), float(last_cy)
                        last_tpl_shape = best_tpl_shape
                        last_tpl_idx = best_tpl_idx
                        tracking_lost = False
                        frame_visible = True
                        display_x, display_y = best_x, best_y


            # 绘制目标的边界框 (绿色) - 根据当前中心点提取连通域，适应目标缩放
            roi_rad = 300
            bx1 = max(0, int(last_cx) - roi_rad)
            by1 = max(0, int(last_cy) - roi_rad)
            bx2 = min(width, int(last_cx) + roi_rad)
            by2 = min(height, int(last_cy) + roi_rad)
            
            roi_gray = gray_frame[by1:by2, bx1:bx2]
            _, thresh = cv2.threshold(roi_gray, 240, 255, cv2.THRESH_BINARY_INV)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            thresh = cv2.dilate(thresh, kernel, iterations=2)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            best_rect = None
            center_pt = (last_cx - bx1, last_cy - by1)
            
            if contours:
                for cnt in contours:
                    if cv2.pointPolygonTest(cnt, center_pt, False) >= 0:
                        best_rect = cv2.boundingRect(cnt)
                        break
                        
                if best_rect is None:
                    for cnt in contours:
                        x, y, w, h = cv2.boundingRect(cnt)
                        if x <= center_pt[0] <= x + w and y <= center_pt[1] <= y + h:
                            best_rect = (x, y, w, h)
                            break
                            
                if best_rect is None:
                    min_dist = float('inf')
                    for cnt in contours:
                        x, y, w, h = cv2.boundingRect(cnt)
                        cx = x + w / 2.0
                        cy = y + h / 2.0
                        dist = (cx - center_pt[0])**2 + (cy - center_pt[1])**2
                        if dist < min_dist:
                            min_dist = dist
                            best_rect = (x, y, w, h)

            if best_rect is not None:
                x, y, w, h = best_rect
                target_w = w + 20
                target_h = h + 20
                target_box_cx = bx1 + x + w / 2.0
                target_box_cy = by1 + y + h / 2.0
            else:
                bbox_x1, bbox_y1, bbox_x2, bbox_y2 = last_tpl_bbox
                target_w = (bbox_x2 - bbox_x1) + 24
                target_h = (bbox_y2 - bbox_y1) + 24
                target_box_cx = display_x + (bbox_x1 + bbox_x2) / 2.0
                target_box_cy = display_y + (bbox_y1 + bbox_y2) / 2.0

            if smooth_bw == 0.0:
                smooth_bw, smooth_bh = float(target_w), float(target_h)
                smooth_box_cx, smooth_box_cy = float(target_box_cx), float(target_box_cy)
            else:
                box_alpha = 0.25
                smooth_bw = smooth_bw * (1 - box_alpha) + target_w * box_alpha
                smooth_bh = smooth_bh * (1 - box_alpha) + target_h * box_alpha
                smooth_box_cx = smooth_box_cx * (1 - box_alpha) + target_box_cx * box_alpha
                smooth_box_cy = smooth_box_cy * (1 - box_alpha) + target_box_cy * box_alpha

            box_x1 = max(0, int(smooth_box_cx - smooth_bw / 2))
            box_y1 = max(0, int(smooth_box_cy - smooth_bh / 2))
            box_x2 = min(width, int(smooth_box_cx + smooth_bw / 2))
            box_y2 = min(height, int(smooth_box_cy + smooth_bh / 2))
            cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), 2)
            
            # 使用边界框的中心作为实际显示的中心点，更贴合物体整体
            center_x = (box_x1 + box_x2) // 2
            center_y = (box_y1 + box_y2) // 2
            
            # 绘制中心像素点 (红色)
            
        # 4. 记录运动轨迹
        if frame_visible:
            current_point = (center_x, center_y)
            if track_history and track_history[-1] is not None:
                last_point = track_history[-1]
                step = float(np.hypot(current_point[0] - last_point[0], current_point[1] - last_point[1]))
                if step < trajectory_min_step:
                    track_history[-1] = current_point
                else:
                    track_history.append(current_point)
            else:
                track_history.append(current_point)
        else:
            if not track_history or track_history[-1] is not None:
                track_history.append(None)

        if len(track_history) > trajectory_max_points:
            track_history = track_history[-trajectory_max_points:]
        
        # 绘制运动轨迹 (蓝色) - 增加滑动窗口平滑
        if len(track_history) > 1:
            smoothed_path = []
            prev_point = None
            for point in track_history:
                if point is None:
                    smoothed_path.append(None)
                    prev_point = None
                    continue
                if prev_point is not None:
                    jump = float(np.hypot(point[0] - prev_point[0], point[1] - prev_point[1]))
                    if jump > trajectory_jump_limit:
                        smoothed_path.append(None)
                        prev_point = point
                        continue
                prev_point = _smooth_point(prev_point, point, alpha=0.45)
                smoothed_path.append(prev_point)

            for pt1, pt2 in zip(smoothed_path, smoothed_path[1:]):
                if pt1 is not None and pt2 is not None:
                    cv2.line(frame, pt1, pt2, (255, 0, 0), 2, cv2.LINE_AA)
                
        # 打印进度并在画面上输出中心点文字
        if frame_visible:
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1, cv2.LINE_AA)
            text_origin = (box_x1, max(20, box_y1 - 10))
            cv2.putText(frame, f"Pos: ({center_x}, {center_y})", text_origin, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 将绘制好的帧写入新视频
        out.write(frame)
        
        if frame_count % 10 == 0:
            print(f"已处理 {frame_count} 帧... (当前得分: {best_raw_score:.2f})")

    # 5. 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    if temp_output_path.exists():
        shutil.copyfile(str(temp_output_path), str(out_path))
    print(f"处理完成！跟踪结果已保存至: {out_path}")

if __name__ == "__main__":
    main()
