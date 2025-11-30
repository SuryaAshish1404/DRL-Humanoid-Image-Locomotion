# DRL-Humanoid-Image-Locomotion

## Troubleshooting Log

### Pose-estimation quirks (2025-11-30)
- **Symptom:** YOLO crops passed directly to MediaPipe occasionally clipped elbows/knees on tight bounding boxes, degrading θ_init accuracy.
- **Action:** Documented need to pad the crop slightly before skeleton extraction. (Pending implementation.)

### Humanoid simulation jitter (2025-11-30)
- **Symptom:** Holding θ_init in `--live` preview caused visible oscillations because position-control targets were reapplied instantly under full gravity.
- **Fixes to apply:**
  1. Smooth commanded joint targets with a low-pass filter in `step`.
  2. Validate θ_init → URDF joint-name mapping so unspecified joints remain at stable defaults.
  3. Extend `_settle_pose` time and momentarily soften gravity during the settling phase to avoid initial shocks.

### Rapid squatting under gravity (2025-11-30)
- **Symptom:** Even with zero action the humanoid immediately squatted because leg servos lacked torque/damping to hold θ_init.
- **Fix:** Increase per-joint motor forces for hips/knees and add explicit PD gains in `setJointMotorControl2` so pose targets remain upright.

These notes serve as a running record—append future issues/fixes here for quick reference.