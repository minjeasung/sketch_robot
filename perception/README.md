# Perception — 작업면 인식

ZED 카메라 + RANSAC 평면 추출 작업물.

## 파일

- `ransac_multiplane.py` — Yi et al. (2026) 기반 RANSAC plane fitting prototype
  - Synthetic point cloud 입력 (T/cross/H/complex)
  - Original vs Improved RANSAC 비교
  - Plane optimization + weld seam extraction
  - 630줄, standalone Python (ROS 패키지 아님)
  - 실행: `python3 ransac_multiplane.py` (open3d, numpy 필요)

## 다음 단계 (Session 5+)

- ZED 실 점군 입력으로 교체
- ROS2 노드로 변환 (`/zed/depth/points` 구독)
- 평면 파라미터를 robot frame 으로 변환 후 PoseStamped 발행
