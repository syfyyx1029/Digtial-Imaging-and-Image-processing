import cv2
import numpy as np
from pathlib import Path

def evaluate_tracking():
    """Evaluate tracking performance metrics"""
    base_dir = Path(__file__).resolve().parent
    video_path = base_dir / 'output' / 'result_tracked.mp4'
    
    if not video_path.exists():
        print(f"视频文件不存在: {video_path}")
        return
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print("=" * 60)
    print("视频跟踪性能评估报告")
    print("=" * 60)
    print(f"视频分辨率: {width}x{height}")
    print(f"总帧数: {total_frames}")
    print(f"帧率: {fps} fps")
    print()
    
    # 提取绿框位置信息
    frames_with_box = 0
    frames_without_box = 0
    box_positions = []
    trajectory_distances = []
    
    frame_count = 0
    prev_center = None
    box_stability = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 检测绿框 (BGR: 0, 255, 0) - 调整范围以捕捉所有绿色像素
        green_mask = cv2.inRange(frame, (0, 150, 0), (100, 255, 150))
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 找最大的矩形（绿框）
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            if w > 10 and h > 10:  # 有效的框
                frames_with_box += 1
                center_x = x + w // 2
                center_y = y + h // 2
                box_positions.append((frame_count, center_x, center_y, w, h))
                
                # 计算轨迹稳定性
                if prev_center is not None:
                    dist = np.hypot(center_x - prev_center[0], center_y - prev_center[1])
                    trajectory_distances.append(dist)
                    box_stability.append({
                        'frame': frame_count,
                        'distance': dist
                    })
                
                prev_center = (center_x, center_y)
            else:
                frames_without_box += 1
        else:
            frames_without_box += 1
    
    cap.release()
    
    # 计算指标
    print("【跟踪覆盖率】")
    print(f"  有框的帧数: {frames_with_box}")
    print(f"  无框的帧数: {frames_without_box}")
    coverage_rate = frames_with_box / total_frames * 100
    print(f"  跟踪覆盖率: {coverage_rate:.1f}%")
    print(f"  漏检率: {100 - coverage_rate:.1f}%")
    print()
    
    print("【轨迹稳定性】")
    if trajectory_distances:
        avg_distance = np.mean(trajectory_distances)
        max_distance = np.max(trajectory_distances)
        std_distance = np.std(trajectory_distances)
        
        print(f"  相邻帧距离平均值: {avg_distance:.2f} pixels")
        print(f"  相邻帧距离最大值: {max_distance:.2f} pixels")
        print(f"  相邻帧距离标准差: {std_distance:.2f} pixels")
        
        # 计算突跳次数（相邻帧距离 > 50 pixels）
        jumps = sum(1 for d in trajectory_distances if d > 50)
        print(f"  轨迹突跳次数 (>50px): {jumps}")
        
        if jumps > 0:
            print(f"  轨迹稳定性评价: 一般（存在{jumps}次跳动）")
        elif std_distance < 5:
            print(f"  轨迹稳定性评价: 优秀（平滑无跳动）")
        else:
            print(f"  轨迹稳定性评价: 良好")
    print()
    
    print("【位置精度评估】")
    if box_positions:
        widths = [p[3] for p in box_positions]
        heights = [p[4] for p in box_positions]
        
        avg_width = np.mean(widths)
        avg_height = np.mean(heights)
        std_width = np.std(widths)
        std_height = np.std(heights)
        
        print(f"  平均框宽度: {avg_width:.1f} ± {std_width:.1f} pixels")
        print(f"  平均框高度: {avg_height:.1f} ± {std_height:.1f} pixels")
        
        # 框大小变化系数（越小越稳定）
        if avg_width > 0:
            width_cv = std_width / avg_width
            height_cv = std_height / avg_height
            print(f"  宽度变异系数: {width_cv:.3f}")
            print(f"  高度变异系数: {height_cv:.3f}")
            
            if width_cv < 0.1 and height_cv < 0.1:
                print(f"  框大小稳定性评价: 优秀")
            elif width_cv < 0.2 and height_cv < 0.2:
                print(f"  框大小稳定性评价: 良好")
            else:
                print(f"  框大小稳定性评价: 一般")
    print()
    
    print("【连续性评估】")
    # 检查跟踪连续性
    if box_positions:
        max_gap = 0
        current_gap = 0
        gap_start = None
        
        for i in range(1, len(box_positions)):
            if box_positions[i][0] - box_positions[i-1][0] > 1:
                # 发现间隙
                gap_size = box_positions[i][0] - box_positions[i-1][0] - 1
                if gap_size > max_gap:
                    max_gap = gap_size
        
        if max_gap == 0:
            print(f"  连续性评价: 完全连续（无间隙）")
        elif max_gap <= 2:
            print(f"  连续性评价: 很好（最大间隙{max_gap}帧）")
        else:
            print(f"  连续性评价: 一般（最大间隙{max_gap}帧）")
    print()
    
    print("【综合评分】")
    # 计算综合评分
    score = 0
    
    # 覆盖率 (40分)
    score += coverage_rate * 0.4
    
    # 稳定性 (30分)
    if trajectory_distances:
        stability_score = 30
        jumps = sum(1 for d in trajectory_distances if d > 50)
        stability_score -= min(jumps * 5, 15)  # 每次跳动扣5分，最多扣15分
        stability_score -= min(std_distance, 10)  # 标准差也会影响
        score += max(stability_score, 10)
    
    # 连续性 (20分)
    if max_gap == 0:
        score += 20
    elif max_gap <= 2:
        score += 15
    else:
        score += max(20 - max_gap * 2, 5)
    
    # 精度 (10分)
    if box_positions:
        precision_score = 10
        if std_width / avg_width > 0.2:
            precision_score -= 3
        if std_height / avg_height > 0.2:
            precision_score -= 3
        score += precision_score
    
    print(f"  综合评分: {score:.1f}/100")
    if score >= 85:
        print(f"  评级: A（优秀）")
    elif score >= 75:
        print(f"  评级: B（良好）")
    elif score >= 65:
        print(f"  评级: C（及格）")
    else:
        print(f"  评级: D（不及格）")
    
    print("=" * 60)

if __name__ == "__main__":
    evaluate_tracking()
