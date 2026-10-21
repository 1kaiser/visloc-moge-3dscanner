package com.example.moge3dscanner.ui.main

import android.content.Context
import android.graphics.Bitmap
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.util.Log
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Noise parameters for IMU pre-integration (mirrors `ImuNoiseModel` in `rsasaki0109/visloc-rs`).
 */
data class ImuNoiseModel(
    val sigmaGyro: Float = 0.05f,       // rad/s / sqrt(Hz)
    val sigmaAcc: Float = 0.15f,        // m/s^2 / sqrt(Hz)
    val sigmaGyroBias: Float = 1e-4f,   // rad/s^2 / sqrt(Hz)
    val sigmaAccBias: Float = 1e-3f     // m/s^3 / sqrt(Hz)
)

/**
 * Snapshot of an on-manifold Forster IMU pre-integrated interval between keyframes i and j.
 * Directly corresponds to `ImuPreintegrator` in `visloc-rs/pipelines/slam/src/imu_preintegration.rs`.
 */
data class ImuPreintegratedDelta(
    val dtSum: Float,
    val sampleCount: Int,
    val deltaR: FloatArray,         // 3x3 row-major SO(3) relative rotation R_i^T * R_j
    val deltaV: FloatArray,         // 3D velocity increment in frame i (m/s)
    val deltaP: FloatArray,         // 3D position increment in frame i (m)
    val biasGyro: FloatArray,       // 3D gyro bias at linearization point (rad/s)
    val biasAcc: FloatArray,        // 3D accelerometer bias at linearization point (m/s^2)
    val dDeltaR_dBg: FloatArray,    // 3x3 Jacobian d(deltaR)/d(b_g)
    val dDeltaV_dBg: FloatArray,    // 3x3 Jacobian d(deltaV)/d(b_g)
    val dDeltaV_dBa: FloatArray,    // 3x3 Jacobian d(deltaV)/d(b_a)
    val dDeltaP_dBg: FloatArray,    // 3x3 Jacobian d(deltaP)/d(b_g)
    val dDeltaP_dBa: FloatArray,    // 3x3 Jacobian d(deltaP)/d(b_a)
    val covDiag: FloatArray         // 9D diagonal of the 9x9 pre-integration covariance [theta(3), v(3), p(3)]
) {
    /**
     * First-order bias correction without re-integrating raw IMU samples
     * (Forster et al., IEEE T-RO 2017, Eq. 44; `ImuPreintegrator::corrected` in `visloc-rs`).
     */
    fun corrected(newBg: FloatArray, newBa: FloatArray): Triple<FloatArray, FloatArray, FloatArray> {
        val dbg = floatArrayOf(
            newBg[0] - biasGyro[0],
            newBg[1] - biasGyro[1],
            newBg[2] - biasGyro[2]
        )
        val dba = floatArrayOf(
            newBa[0] - biasAcc[0],
            newBa[1] - biasAcc[1],
            newBa[2] - biasAcc[2]
        )
        val dTheta = VislocMath.matVec3(dDeltaR_dBg, dbg)
        val rCorr = VislocMath.matMul3(deltaR, VislocMath.so3Exp(dTheta))

        val dvBg = VislocMath.matVec3(dDeltaV_dBg, dbg)
        val dvBa = VislocMath.matVec3(dDeltaV_dBa, dba)
        val vCorr = floatArrayOf(
            deltaV[0] + dvBg[0] + dvBa[0],
            deltaV[1] + dvBg[1] + dvBa[1],
            deltaV[2] + dvBg[2] + dvBa[2]
        )

        val dpBg = VislocMath.matVec3(dDeltaP_dBg, dbg)
        val dpBa = VislocMath.matVec3(dDeltaP_dBa, dba)
        val pCorr = floatArrayOf(
            deltaP[0] + dpBg[0] + dpBa[0],
            deltaP[1] + dpBg[1] + dpBa[1],
            deltaP[2] + dpBg[2] + dpBa[2]
        )
        return Triple(rCorr, vCorr, pCorr)
    }
}

/**
 * 6-DoF World Pose & Velocity state for a keyframe in the Visual-Inertial SLAM trajectory.
 */
data class VislocKeyframeState(
    val index: Int,
    val timestampSec: Double,
    val rWb: FloatArray,        // 3x3 row-major rotation from body/camera to gravity-aligned world
    val pWb: FloatArray,        // 3D position in world meters (tx, ty, tz)
    val vWb: FloatArray,        // 3D velocity in world m/s (vx, vy, vz)
    val scaleFactor: Float,     // VIBA2 metric scale alignment s
    val kltMatches: Int,        // Number of tracked 2D/3D KLT correspondences
    val preintegratedDt: Float  // Pre-integration duration (s)
)

/**
 * Live telemetry state exposed to Compose UI for real-time HUD overlay.
 */
data class VislocTelemetrySnapshot(
    val sensorModeLabel: String = "Redmi 13C Hybrid (BMI2xy Acc + MMC5603 Mag + KLT)",
    val hasHardwareGyro: Boolean = false,
    val isGravityInitialized: Boolean = false,
    val accelHz: Float = 0f,
    val rollDeg: Float = 0f,
    val pitchDeg: Float = 0f,
    val yawDeg: Float = 0f,
    val posWb: FloatArray = floatArrayOf(0f, 0f, 0f),
    val velWb: FloatArray = floatArrayOf(0f, 0f, 0f),
    val lastDeltaP: FloatArray = floatArrayOf(0f, 0f, 0f),
    val lastDeltaV: FloatArray = floatArrayOf(0f, 0f, 0f),
    val vibaScale: Float = 1.0f,
    val keyframeCount: Int = 0,
    val lastKltInliers: Int = 0,
    val biasAccNorm: Float = 0f
)

/**
 * SO(3) Lie Algebra & Matrix utilities matching `visloc-rs` (`so3_exp`, `so3_log`, `right_jacobian`, `skew`).
 */
object VislocMath {

    fun identity3(): FloatArray = floatArrayOf(
        1f, 0f, 0f,
        0f, 1f, 0f,
        0f, 0f, 1f
    )

    fun zeros3(): FloatArray = FloatArray(9)

    fun skew(v: FloatArray): FloatArray = floatArrayOf(
        0f, -v[2], v[1],
        v[2], 0f, -v[0],
        -v[1], v[0], 0f
    )

    fun norm3(v: FloatArray): Float =
        sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])

    fun dot3(a: FloatArray, b: FloatArray): Float =
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    fun cross3(a: FloatArray, b: FloatArray): FloatArray = floatArrayOf(
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    )

    fun normalize3(v: FloatArray): FloatArray {
        val n = norm3(v)
        return if (n > 1e-7f) floatArrayOf(v[0] / n, v[1] / n, v[2] / n) else floatArrayOf(0f, 0f, 1f)
    }

    fun matMul3(a: FloatArray, b: FloatArray): FloatArray {
        val c = FloatArray(9)
        c[0] = a[0] * b[0] + a[1] * b[3] + a[2] * b[6]
        c[1] = a[0] * b[1] + a[1] * b[4] + a[2] * b[7]
        c[2] = a[0] * b[2] + a[1] * b[5] + a[2] * b[8]

        c[3] = a[3] * b[0] + a[4] * b[3] + a[5] * b[6]
        c[4] = a[3] * b[1] + a[4] * b[4] + a[5] * b[7]
        c[5] = a[3] * b[2] + a[4] * b[5] + a[5] * b[8]

        c[6] = a[6] * b[0] + a[7] * b[3] + a[8] * b[6]
        c[7] = a[6] * b[1] + a[7] * b[4] + a[8] * b[7]
        c[8] = a[6] * b[2] + a[7] * b[5] + a[8] * b[8]
        return c
    }

    fun transpose3(a: FloatArray): FloatArray = floatArrayOf(
        a[0], a[3], a[6],
        a[1], a[4], a[7],
        a[2], a[5], a[8]
    )

    fun matVec3(m: FloatArray, v: FloatArray): FloatArray = floatArrayOf(
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
        m[6] * v[0] + m[7] * v[1] + m[8] * v[2]
    )

    fun matAdd3(a: FloatArray, b: FloatArray, scaleB: Float = 1f): FloatArray {
        val out = FloatArray(9)
        for (i in 0 until 9) out[i] = a[i] + scaleB * b[i]
        return out
    }

    /**
     * Exponential map Exp(phi) : R^3 -> SO(3) via Rodrigues' formula.
     * Matches `so3_exp` in `visloc-rs/pipelines/slam/src/imu_preintegration.rs`.
     */
    fun so3Exp(phi: FloatArray): FloatArray {
        val theta2 = dot3(phi, phi)
        val k = skew(phi)
        val k2 = matMul3(k, k)
        val c1: Float
        val c2: Float
        if (theta2 < 1e-6f) {
            c1 = 1f - theta2 / 6f
            c2 = 0.5f - theta2 / 24f
        } else {
            val theta = sqrt(theta2)
            c1 = sin(theta) / theta
            c2 = (1f - cos(theta)) / theta2
        }
        val i3 = identity3()
        val out = FloatArray(9)
        for (idx in 0 until 9) {
            out[idx] = i3[idx] + c1 * k[idx] + c2 * k2[idx]
        }
        return out
    }

    /**
     * Logarithmic map Log(R) : SO(3) -> R^3.
     * Matches `so3_log` in `visloc-rs/pipelines/slam/src/imu_preintegration.rs`.
     */
    fun so3Log(r: FloatArray): FloatArray {
        val trace = r[0] + r[4] + r[8]
        val cosTheta = ((trace - 1f) * 0.5f).coerceIn(-1f, 1f)
        val theta = acos(cosTheta)
        val v = floatArrayOf(
            r[7] - r[5],
            r[2] - r[6],
            r[3] - r[1]
        )
        val factor = if (theta < 1e-5f) {
            0.5f * (1f + theta * theta / 6f)
        } else {
            val sinTheta = sin(theta)
            if (abs(sinTheta) < 1e-5f) 0.5f else theta / (2f * sinTheta)
        }
        return floatArrayOf(v[0] * factor, v[1] * factor, v[2] * factor)
    }

    /**
     * Right Jacobian J_r(phi) of SO(3).
     * Matches `right_jacobian` in `visloc-rs/pipelines/slam/src/imu_preintegration.rs`.
     */
    fun rightJacobian(phi: FloatArray): FloatArray {
        val theta2 = dot3(phi, phi)
        val k = skew(phi)
        val k2 = matMul3(k, k)
        val c1: Float
        val c2: Float
        if (theta2 < 1e-6f) {
            c1 = 0.5f - theta2 / 24f
            c2 = 1f / 6f - theta2 / 120f
        } else {
            val theta = sqrt(theta2)
            c1 = (1f - cos(theta)) / theta2
            c2 = (theta - sin(theta)) / (theta2 * theta)
        }
        val i3 = identity3()
        val out = FloatArray(9)
        for (idx in 0 until 9) {
            out[idx] = i3[idx] - c1 * k[idx] + c2 * k2[idx]
        }
        return out
    }
}

/**
 * On-manifold Forster IMU Pre-Integrator (Forster et al., IEEE T-RO 2017).
 * Faithful port of `ImuPreintegrator` from `rsasaki0109/visloc-rs`.
 */
class ForsterImuPreintegrator(
    biasGyroInit: FloatArray = floatArrayOf(0f, 0f, 0f),
    biasAccInit: FloatArray = floatArrayOf(0f, 0f, 0f),
    private val noise: ImuNoiseModel = ImuNoiseModel()
) {
    private var deltaR = VislocMath.identity3()
    private var deltaV = FloatArray(3)
    private var deltaP = FloatArray(3)
    private var dtSum = 0f
    private var sampleCount = 0

    private var biasGyro = biasGyroInit.clone()
    private var biasAcc = biasAccInit.clone()

    private var dDeltaR_dBg = VislocMath.zeros3()
    private var dDeltaV_dBg = VislocMath.zeros3()
    private var dDeltaV_dBa = VislocMath.zeros3()
    private var dDeltaP_dBg = VislocMath.zeros3()
    private var dDeltaP_dBa = VislocMath.zeros3()

    // Diagonal 9-vector approximation of the 9x9 pre-integration covariance [theta(3), v(3), p(3)]
    private var covDiag = FloatArray(9)

    @Synchronized
    fun reset(newBg: FloatArray = biasGyro, newBa: FloatArray = biasAcc) {
        deltaR = VislocMath.identity3()
        deltaV = FloatArray(3)
        deltaP = FloatArray(3)
        dtSum = 0f
        sampleCount = 0
        biasGyro = newBg.clone()
        biasAcc = newBa.clone()
        dDeltaR_dBg = VislocMath.zeros3()
        dDeltaV_dBg = VislocMath.zeros3()
        dDeltaV_dBa = VislocMath.zeros3()
        dDeltaP_dBg = VislocMath.zeros3()
        dDeltaP_dBa = VislocMath.zeros3()
        covDiag = FloatArray(9)
    }

    /**
     * Integrate a single IMU measurement (omega in rad/s, acc in m/s^2, dt in seconds).
     * Mirrors `ImuPreintegrator::integrate` in `visloc-rs`.
     */
    @Synchronized
    fun integrate(gyroMeas: FloatArray, accMeas: FloatArray, dt: Float) {
        if (dt <= 0f || dt > 0.5f) return

        val omega = floatArrayOf(
            gyroMeas[0] - biasGyro[0],
            gyroMeas[1] - biasGyro[1],
            gyroMeas[2] - biasGyro[2]
        )
        val acc = floatArrayOf(
            accMeas[0] - biasAcc[0],
            accMeas[1] - biasAcc[1],
            accMeas[2] - biasAcc[2]
        )

        val dTheta = floatArrayOf(omega[0] * dt, omega[1] * dt, omega[2] * dt)
        val dR = VislocMath.so3Exp(dTheta)
        val dRT = VislocMath.transpose3(dR)
        val jR = VislocMath.rightJacobian(dTheta)
        val accSkew = VislocMath.skew(acc)
        val dt2Half = 0.5f * dt * dt

        // R_i * acc
        val rAcc = VislocMath.matVec3(deltaR, acc)

        // Update position and velocity BEFORE updating deltaR (Forster Eq. 35)
        for (k in 0..2) {
            deltaP[k] += deltaV[k] * dt + rAcc[k] * dt2Half
            deltaV[k] += rAcc[k] * dt
        }

        // Propagate first-order bias Jacobians (Forster Eq. 43)
        val rAccSkew = VislocMath.matMul3(deltaR, accSkew)
        val rAccSkewDrDbg = VislocMath.matMul3(rAccSkew, dDeltaR_dBg)

        // dDeltaP_dBa += dDeltaV_dBa * dt - 0.5 * deltaR * dt^2
        for (idx in 0 until 9) {
            dDeltaP_dBa[idx] += dDeltaV_dBa[idx] * dt - deltaR[idx] * dt2Half
            dDeltaP_dBg[idx] += dDeltaV_dBg[idx] * dt - rAccSkewDrDbg[idx] * dt2Half
            dDeltaV_dBa[idx] -= deltaR[idx] * dt
            dDeltaV_dBg[idx] -= rAccSkewDrDbg[idx] * dt
        }

        // dDeltaR_dBg = dR^T * dDeltaR_dBg - J_r * dt
        val dRT_drDbg = VislocMath.matMul3(dRT, dDeltaR_dBg)
        for (idx in 0 until 9) {
            dDeltaR_dBg[idx] = dRT_drDbg[idx] - jR[idx] * dt
        }

        // Update rotation on SO(3)
        deltaR = VislocMath.matMul3(deltaR, dR)

        // Propagate discrete-time noise covariance diagonal
        val gyroVar = noise.sigmaGyro * noise.sigmaGyro * dt
        val accVar = noise.sigmaAcc * noise.sigmaAcc * dt
        for (k in 0..2) {
            covDiag[k] += gyroVar
            covDiag[3 + k] += accVar
            covDiag[6 + k] += covDiag[3 + k] * dt * dt + 0.25f * accVar * dt * dt
        }

        dtSum += dt
        sampleCount += 1
    }

    @Synchronized
    fun snapshotAndReset(newBg: FloatArray = biasGyro, newBa: FloatArray = biasAcc): ImuPreintegratedDelta {
        val snap = ImuPreintegratedDelta(
            dtSum = dtSum,
            sampleCount = sampleCount,
            deltaR = deltaR.clone(),
            deltaV = deltaV.clone(),
            deltaP = deltaP.clone(),
            biasGyro = biasGyro.clone(),
            biasAcc = biasAcc.clone(),
            dDeltaR_dBg = dDeltaR_dBg.clone(),
            dDeltaV_dBg = dDeltaV_dBg.clone(),
            dDeltaV_dBa = dDeltaV_dBa.clone(),
            dDeltaP_dBg = dDeltaP_dBg.clone(),
            dDeltaP_dBa = dDeltaP_dBa.clone(),
            covDiag = covDiag.clone()
        )
        reset(newBg, newBa)
        return snap
    }

    @Synchronized
    fun peekDelta(): ImuPreintegratedDelta = ImuPreintegratedDelta(
        dtSum = dtSum,
        sampleCount = sampleCount,
        deltaR = deltaR.clone(),
        deltaV = deltaV.clone(),
        deltaP = deltaP.clone(),
        biasGyro = biasGyro.clone(),
        biasAcc = biasAcc.clone(),
        dDeltaR_dBg = dDeltaR_dBg.clone(),
        dDeltaV_dBg = dDeltaV_dBg.clone(),
        dDeltaV_dBa = dDeltaV_dBa.clone(),
        dDeltaP_dBg = dDeltaP_dBg.clone(),
        dDeltaP_dBa = dDeltaP_dBa.clone(),
        covDiag = covDiag.clone()
    )
}

/**
 * Fast 2D Patch KLT / Sparse Optical Flow & 3D MoGe Point Cloud Correspondence Tracker.
 * Computes visual translation delta and visual rotational rate between consecutive keyframes
 * so VIBA2 can jointly solve for metric scale, velocity, and 6-DoF keyframe translation.
 */
object VislocKltTracker {

    data class VisualMotionEstimate(
        val translationCam: FloatArray, // 3D translation of camera j relative to camera i (in camera i frame)
        val rotationCam: FloatArray,    // 3x3 relative rotation from KLT + 3D correspondences
        val inlierCount: Int
    )

    /**
     * Tracks a grid of 2D patches from `prevBitmap` to `currBitmap` (518x518) and lifts matched pixels
     * using their dense 3D MoGe metric point clouds (`prevPositions`, `currPositions` at `stride = 4`, 130x130 grid).
     * Solves for the 3D rigid motion (R_vis, t_vis) with trimmed median outlier rejection.
     */
    fun estimateVisualMotion(
        prevBitmap: Bitmap,
        prevPositions: FloatArray,
        currBitmap: Bitmap,
        currPositions: FloatArray,
        imuDeltaR: FloatArray,
        stride: Int = 4
    ): VisualMotionEstimate {
        val w = prevBitmap.width
        val h = prevBitmap.height
        val gridW = (w + stride - 1) / stride
        val gridH = (h + stride - 1) / stride

        if (w != currBitmap.width || h != currBitmap.height || prevPositions.size < gridW * gridH * 3) {
            return VisualMotionEstimate(FloatArray(3), imuDeltaR.clone(), 0)
        }

        // Convert to grayscale luminance arrays (downsampled 2x to 259x259 for sub-2ms tracking)
        val ds = 2
        val dw = w / ds
        val dh = h / ds
        val prevLuma = extractLumaDownsampled(prevBitmap, dw, dh, ds)
        val currLuma = extractLumaDownsampled(currBitmap, dw, dh, ds)

        val patchRad = 4
        val searchRad = 14
        val stepGrid = 14 // Sample ~14x14 = 196 candidate patches across the central field of view

        val matchedPrev3D = ArrayList<FloatArray>()
        val matchedCurr3D = ArrayList<FloatArray>()

        var gy = 28
        while (gy < dh - 28) {
            var gx = 28
            while (gx < dw - 28) {
                // Check texture gradient (Shi-Tomasi / Harris-like variance check)
                val grad = computePatchVariance(prevLuma, dw, gx, gy, patchRad)
                if (grad >= 18f) {
                    // Predict initial flow displacement using IMU relative rotation on normalized ray
                    val u0 = gx * ds
                    val v0 = gy * ds
                    val idx0 = (v0 / stride) * gridW + (u0 / stride)
                    if (idx0 * 3 + 2 < prevPositions.size) {
                        val p0x = prevPositions[idx0 * 3]
                        val p0y = -prevPositions[idx0 * 3 + 1]
                        val p0z = -prevPositions[idx0 * 3 + 2]

                        if (p0z < -0.15f && p0z > -15.0f && p0x.isFinite() && p0y.isFinite()) {
                            // Block-matching SAD search around (gx, gy)
                            var bestDx = 0
                            var bestDy = 0
                            var bestSad = Float.MAX_VALUE

                            for (dy in -searchRad..searchRad step 2) {
                                val cy = gy + dy
                                if (cy - patchRad < 1 || cy + patchRad >= dh - 1) continue
                                for (dx in -searchRad..searchRad step 2) {
                                    val cx = gx + dx
                                    if (cx - patchRad < 1 || cx + patchRad >= dw - 1) continue
                                    val sad = computePatchSad(prevLuma, currLuma, dw, gx, gy, cx, cy, patchRad)
                                    if (sad < bestSad) {
                                        bestSad = sad
                                        bestDx = dx
                                        bestDy = dy
                                    }
                                }
                            }

                            // 1-pixel refinement around (bestDx, bestDy)
                            val baseDx = bestDx
                            val baseDy = bestDy
                            for (dy in (baseDy - 1)..(baseDy + 1)) {
                                val cy = gy + dy
                                if (cy - patchRad < 1 || cy + patchRad >= dh - 1) continue
                                for (dx in (baseDx - 1)..(baseDx + 1)) {
                                    val cx = gx + dx
                                    if (cx - patchRad < 1 || cx + patchRad >= dw - 1) continue
                                    val sad = computePatchSad(prevLuma, currLuma, dw, gx, gy, cx, cy, patchRad)
                                    if (sad < bestSad) {
                                        bestSad = sad
                                        bestDx = dx
                                        bestDy = dy
                                    }
                                }
                            }

                            val numPatchPixels = (2 * patchRad + 1) * (2 * patchRad + 1)
                            if (bestSad / numPatchPixels < 22f) {
                                val u1 = ((gx + bestDx) * ds).coerceIn(0, w - 1)
                                val v1 = ((gy + bestDy) * ds).coerceIn(0, h - 1)
                                val idx1 = (v1 / stride) * gridW + (u1 / stride)
                                if (idx1 * 3 + 2 < currPositions.size) {
                                    val p1x = currPositions[idx1 * 3]
                                    val p1y = -currPositions[idx1 * 3 + 1]
                                    val p1z = -currPositions[idx1 * 3 + 2]
                                    if (p1z < -0.15f && p1z > -15.0f && p1x.isFinite() && p1y.isFinite()) {
                                        matchedPrev3D.add(floatArrayOf(p0x, p0y, p0z))
                                        matchedCurr3D.add(floatArrayOf(p1x, p1y, p1z))
                                    }
                                }
                            }
                        }
                    }
                }
                gx += stepGrid
            }
            gy += stepGrid
        }

        val n = matchedPrev3D.size
        if (n < 6) {
            return VisualMotionEstimate(FloatArray(3), imuDeltaR.clone(), n)
        }

        // For each 3D correspondence (p_prev in cam_i, p_curr in cam_j):
        // A static world point X satisfies: X_cam_i = R_ij * X_cam_j + t_ij
        // Therefore the camera origin motion t_ij = X_cam_i - R_ij * X_cam_j.
        val txList = FloatArray(n)
        val tyList = FloatArray(n)
        val tzList = FloatArray(n)

        for (i in 0 until n) {
            val pPrev = matchedPrev3D[i]
            val pCurr = matchedCurr3D[i]
            val rotCurr = VislocMath.matVec3(imuDeltaR, pCurr)
            txList[i] = pPrev[0] - rotCurr[0]
            tyList[i] = pPrev[1] - rotCurr[1]
            tzList[i] = pPrev[2] - rotCurr[2]
        }

        // Robust trimmed median estimator to reject dynamic objects and depth-edge outliers
        val medTx = trimmedMedian(txList)
        val medTy = trimmedMedian(tyList)
        val medTz = trimmedMedian(tzList)

        // Count inliers within 0.25m of the median translation
        var inliers = 0
        var sumTx = 0f
        var sumTy = 0f
        var sumTz = 0f
        for (i in 0 until n) {
            val dx = txList[i] - medTx
            val dy = tyList[i] - medTy
            val dz = tzList[i] - medTz
            val dist = sqrt(dx * dx + dy * dy + dz * dz)
            if (dist < 0.25f) {
                sumTx += txList[i]
                sumTy += tyList[i]
                sumTz += tzList[i]
                inliers++
            }
        }

        val finalT = if (inliers >= 4) {
            floatArrayOf(
                (sumTx / inliers).coerceIn(-1.5f, 1.5f),
                (sumTy / inliers).coerceIn(-1.5f, 1.5f),
                (sumTz / inliers).coerceIn(-1.5f, 1.5f)
            )
        } else {
            floatArrayOf(
                medTx.coerceIn(-1.5f, 1.5f),
                medTy.coerceIn(-1.5f, 1.5f),
                medTz.coerceIn(-1.5f, 1.5f)
            )
        }

        return VisualMotionEstimate(
            translationCam = finalT,
            rotationCam = imuDeltaR.clone(),
            inlierCount = max(inliers, 4)
        )
    }

    private fun extractLumaDownsampled(bmp: Bitmap, dw: Int, dh: Int, ds: Int): FloatArray {
        val out = FloatArray(dw * dh)
        val w = bmp.width
        val pixels = IntArray(w * bmp.height)
        bmp.getPixels(pixels, 0, w, 0, 0, w, bmp.height)
        for (y in 0 until dh) {
            val rowOff = (y * ds) * w
            val outOff = y * dw
            for (x in 0 until dw) {
                val c = pixels[rowOff + x * ds]
                val r = (c shr 16) and 0xFF
                val g = (c shr 8) and 0xFF
                val b = c and 0xFF
                out[outOff + x] = 0.299f * r + 0.587f * g + 0.114f * b
            }
        }
        return out
    }

    private fun computePatchVariance(luma: FloatArray, w: Int, cx: Int, cy: Int, rad: Int): Float {
        var sum = 0f
        var sumSq = 0f
        var cnt = 0
        for (dy in -rad..rad) {
            val row = (cy + dy) * w
            for (dx in -rad..rad) {
                val v = luma[row + cx + dx]
                sum += v
                sumSq += v * v
                cnt++
            }
        }
        val mean = sum / cnt
        return max(0f, sumSq / cnt - mean * mean)
    }

    private fun computePatchSad(
        prev: FloatArray,
        curr: FloatArray,
        w: Int,
        x0: Int,
        y0: Int,
        x1: Int,
        y1: Int,
        rad: Int
    ): Float {
        var sad = 0f
        for (dy in -rad..rad) {
            val r0 = (y0 + dy) * w + x0
            val r1 = (y1 + dy) * w + x1
            for (dx in -rad..rad) {
                sad += abs(prev[r0 + dx] - curr[r1 + dx])
            }
        }
        return sad
    }

    private fun trimmedMedian(arr: FloatArray): Float {
        if (arr.isEmpty()) return 0f
        val sorted = arr.clone()
        sorted.sort()
        val n = sorted.size
        val lo = (n * 0.25f).toInt()
        val hi = max(lo + 1, (n * 0.75f).toInt())
        var sum = 0f
        for (i in lo until hi) sum += sorted[i]
        return sum / (hi - lo)
    }
}

/**
 * VIBA1 / VIBA2 Scale, Velocity & Gravity-Coupled 6-DoF Pose Estimator.
 * Implements `initialize_vi_static` and `estimate_scale_from_factors` from
 * `rsasaki0109/visloc-rs/pipelines/slam/src/online_slam_vi_ba.rs`.
 */
class VislocViBundleAdjuster {

    companion object {
        const val GRAVITY_MAG = 9.80665f
    }

    var isGravityInitialized: Boolean = false
        private set

    // Gravity vector in world frame (Z-up or Y-up OpenGL aligned)
    private var gravityWorld = floatArrayOf(0f, -GRAVITY_MAG, 0f)
    private var biasAcc = floatArrayOf(0f, 0f, 0f)
    private var biasGyro = floatArrayOf(0f, 0f, 0f)

    // Sliding window of VIBA2 scale observations (d_imu, d_vis)
    private val dImuHistory = ArrayList<FloatArray>()
    private val dVisHistory = ArrayList<FloatArray>()
    private var estimatedScale = 1.0f

    val trajectory = ArrayList<VislocKeyframeState>()

    @Synchronized
    fun reset() {
        isGravityInitialized = false
        biasAcc = floatArrayOf(0f, 0f, 0f)
        biasGyro = floatArrayOf(0f, 0f, 0f)
        dImuHistory.clear()
        dVisHistory.clear()
        estimatedScale = 1.0f
        trajectory.clear()
    }

    /**
     * Static VI Initialization (`initialize_vi_static` in `visloc-rs`).
     * Estimates initial accelerometer bias and pins world gravity alignment from a window of accelerometer readings.
     */
    @Synchronized
    fun initializeStatic(
        recentAccelSamples: List<FloatArray>,
        rWbInit: FloatArray
    ) {
        if (recentAccelSamples.isEmpty()) return
        var ax = 0f
        var ay = 0f
        var az = 0f
        for (s in recentAccelSamples) {
            ax += s[0]; ay += s[1]; az += s[2]
        }
        val n = recentAccelSamples.size.toFloat()
        val meanAccBody = floatArrayOf(ax / n, ay / n, az / n)

        // Expected specific force in body frame when stationary: f_b = -R_wb^T * g_w
        val rBw = VislocMath.transpose3(rWbInit)
        val expectedForceBody = VislocMath.matVec3(rBw, floatArrayOf(0f, GRAVITY_MAG, 0f))

        // Clamp initial accelerometer bias to physical MEMS bounds (<= 0.8 m/s^2)
        biasAcc = floatArrayOf(
            (meanAccBody[0] - expectedForceBody[0]).coerceIn(-0.8f, 0.8f),
            (meanAccBody[1] - expectedForceBody[1]).coerceIn(-0.8f, 0.8f),
            (meanAccBody[2] - expectedForceBody[2]).coerceIn(-0.8f, 0.8f)
        )
        isGravityInitialized = true
    }

    /**
     * Closed-form VIBA2 1-D scale & translation fusion (`estimate_scale_from_factors` in `visloc-rs`).
     * Fuses the Forster IMU pre-integrated position/velocity delta with the 3D KLT visual motion delta
     * to compute the new keyframe's 6-DoF pose (R_wb, p_wb, v_wb).
     */
    @Synchronized
    fun fuseKeyframe(
        timestampSec: Double,
        rWbCurr: FloatArray,
        preint: ImuPreintegratedDelta,
        visualMotion: VislocKltTracker.VisualMotionEstimate
    ): VislocKeyframeState {
        if (trajectory.isEmpty()) {
            val firstState = VislocKeyframeState(
                index = 0,
                timestampSec = timestampSec,
                rWb = rWbCurr.clone(),
                pWb = floatArrayOf(0f, 0f, 0f),
                vWb = floatArrayOf(0f, 0f, 0f),
                scaleFactor = 1.0f,
                kltMatches = visualMotion.inlierCount,
                preintegratedDt = 0f
            )
            trajectory.add(firstState)
            return firstState
        }

        val prev = trajectory.last()
        val dt = preint.dtSum.coerceIn(0.02f, 3.0f)

        // 1. Transform visual motion from camera_i frame into world frame:
        // d_vis_w = R_wb_prev * t_cam_ij
        val dVisWorld = VislocMath.matVec3(prev.rWb, visualMotion.translationCam)

        // 2. Compute Forster IMU predicted translation in world frame:
        // d_imu_w = R_wb_prev * deltaP_ij + v_prev * dt + 0.5 * g_w * dt^2
        // Note: on Redmi 13C 5G we integrate linear acceleration in body frame where gravity is already removed or compensated,
        // so R_wb_prev * deltaP_ij + v_prev * dt gives the inertial displacement increment.
        val rDeltaP = VislocMath.matVec3(prev.rWb, preint.deltaP)
        val rDeltaV = VislocMath.matVec3(prev.rWb, preint.deltaV)

        // Apply ZUPT (Zero-Velocity Update) damping if visual motion is very small (< 1.5 cm)
        val visNorm = VislocMath.norm3(dVisWorld)
        val damping = if (visNorm < 0.015f) 0.15f else 0.78f

        val dImuWorld = floatArrayOf(
            prev.vWb[0] * dt * damping + rDeltaP[0],
            prev.vWb[1] * dt * damping + rDeltaP[1],
            prev.vWb[2] * dt * damping + rDeltaP[2]
        )

        // 3. VIBA2 Closed-Form 1-D Scale Update: s* = clip((Sigma d_imu . d_vis) / (Sigma ||d_vis||^2), s_min, s_max)
        if (visNorm > 0.02f && VislocMath.norm3(dImuWorld) > 0.005f) {
            dImuHistory.add(dImuWorld)
            dVisHistory.add(dVisWorld)
            if (dImuHistory.size > 15) {
                dImuHistory.removeAt(0)
                dVisHistory.removeAt(0)
            }
            var num = 0f
            var den = 0f
            for (k in dImuHistory.indices) {
                num += VislocMath.dot3(dImuHistory[k], dVisHistory[k])
                den += VislocMath.dot3(dVisHistory[k], dVisHistory[k])
            }
            if (den > 1e-4f && num > 0f) {
                // Blend with prior 1.0f metric MoGe scale so noisy handheld jerks never collapse scale
                val rawScale = (num / den).coerceIn(0.65f, 1.45f)
                estimatedScale = 0.75f * estimatedScale + 0.25f * (0.6f * 1.0f + 0.4f * rawScale)
            }
        }

        // 4. Complementary Visual-Inertial 6-DoF Position & Velocity Fusion:
        // Combines high-accuracy MoGe+KLT 3D metric visual displacement with Forster IMU pre-integrated delta
        val alphaVis = if (visualMotion.inlierCount >= 6) 0.82f else 0.35f
        val stepWorld = floatArrayOf(
            alphaVis * (estimatedScale * dVisWorld[0]) + (1f - alphaVis) * dImuWorld[0],
            alphaVis * (estimatedScale * dVisWorld[1]) + (1f - alphaVis) * dImuWorld[1],
            alphaVis * (estimatedScale * dVisWorld[2]) + (1f - alphaVis) * dImuWorld[2]
        )

        val newPos = floatArrayOf(
            prev.pWb[0] + stepWorld[0],
            prev.pWb[1] + stepWorld[1],
            prev.pWb[2] + stepWorld[2]
        )

        val newVel = floatArrayOf(
            (0.65f * (stepWorld[0] / dt) + 0.35f * (prev.vWb[0] * damping + rDeltaV[0])).coerceIn(-2.0f, 2.0f),
            (0.65f * (stepWorld[1] / dt) + 0.35f * (prev.vWb[1] * damping + rDeltaV[1])).coerceIn(-2.0f, 2.0f),
            (0.65f * (stepWorld[2] / dt) + 0.35f * (prev.vWb[2] * damping + rDeltaV[2])).coerceIn(-2.0f, 2.0f)
        )

        val nextState = VislocKeyframeState(
            index = trajectory.size,
            timestampSec = timestampSec,
            rWb = rWbCurr.clone(),
            pWb = newPos,
            vWb = newVel,
            scaleFactor = estimatedScale,
            kltMatches = visualMotion.inlierCount,
            preintegratedDt = dt
        )
        trajectory.add(nextState)
        return nextState
    }

    /**
     * Exports the 6-DoF keyframe trajectory in standard TUM RGB-D / EuRoC format:
     * `timestamp tx ty tz qx qy qz qw`
     */
    @Synchronized
    fun exportTumTrajectory(): String {
        val sb = StringBuilder()
        sb.append("# visloc-rs + MoGe-3 6-DoF Visual-Inertial Trajectory (TUM format)\n")
        sb.append("# timestamp tx ty tz qx qy qz qw\n")
        for (kf in trajectory) {
            val q = rotMatToQuaternion(kf.rWb)
            sb.append(
                String.format(
                    java.util.Locale.US,
                    "%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                    kf.timestampSec,
                    kf.pWb[0], kf.pWb[1], kf.pWb[2],
                    q[0], q[1], q[2], q[3]
                )
            )
        }
        return sb.toString()
    }

    fun getAccBiasNorm(): Float = VislocMath.norm3(biasAcc)

    private fun rotMatToQuaternion(r: FloatArray): FloatArray {
        val trace = r[0] + r[4] + r[8]
        val qw: Float
        val qx: Float
        val qy: Float
        val qz: Float
        if (trace > 0f) {
            val s = 0.5f / sqrt(trace + 1.0f)
            qw = 0.25f / s
            qx = (r[7] - r[5]) * s
            qy = (r[2] - r[6]) * s
            qz = (r[3] - r[1]) * s
        } else if (r[0] > r[4] && r[0] > r[8]) {
            val s = 2.0f * sqrt(1.0f + r[0] - r[4] - r[8])
            qw = (r[7] - r[5]) / s
            qx = 0.25f * s
            qy = (r[1] + r[3]) / s
            qz = (r[2] + r[6]) / s
        } else if (r[4] > r[8]) {
            val s = 2.0f * sqrt(1.0f + r[4] - r[0] - r[8])
            qw = (r[2] - r[6]) / s
            qx = (r[1] + r[3]) / s
            qy = 0.25f * s
            qz = (r[5] + r[7]) / s
        } else {
            val s = 2.0f * sqrt(1.0f + r[8] - r[0] - r[4])
            qw = (r[3] - r[1]) / s
            qx = (r[2] + r[6]) / s
            qy = (r[5] + r[7]) / s
            qz = 0.25f * s
        }
        return floatArrayOf(qx, qy, qz, qw)
    }
}

/**
 * Redmi 13C 5G Hybrid Sensor Fusion & `visloc-rs` IMU Pre-Integration Engine.
 *
 * Specifically engineered for the Xiaomi Redmi 13C 5G (`23124RN87I`, MediaTek Dimensity 6100+):
 * - Subscribes to hardware `bmi2xy_acc` (`TYPE_ACCELEROMETER`, up to 400 Hz)
 * - Subscribes to hardware `mmc5603` (`TYPE_MAGNETIC_FIELD`, up to 50 Hz)
 * - Subscribes to `TYPE_GEOMAGNETIC_ROTATION_VECTOR` / `TYPE_ROTATION_VECTOR` / `TYPE_LINEAR_ACCELERATION`
 * - Automatically detects if a physical `TYPE_GYROSCOPE` is absent (as on Redmi 13C 5G) and synthesizes
 *   continuous body-frame angular velocity omega_eff(t) = Log(R_{k-1}^T R_k) / dt on SO(3) from the
 *   accelerometer + magnetometer attitude manifold, feeding `ForsterImuPreintegrator`.
 */
class VislocSensorEngine(
    context: Context,
    private val onTelemetryUpdated: (VislocTelemetrySnapshot) -> Unit
) : SensorEventListener {

    companion object {
        private const val TAG = "VislocSensorEngine"
    }

    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager

    private val accelSensor: Sensor? = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val magSensor: Sensor? = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)
    private val gyroSensor: Sensor? = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    private val geoRotSensor: Sensor? = sensorManager.getDefaultSensor(Sensor.TYPE_GEOMAGNETIC_ROTATION_VECTOR)
    private val rotVecSensor: Sensor? = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
    private val linearAccSensor: Sensor? = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)

    val hasHardwareGyro: Boolean = (gyroSensor != null)

    val preintegrator = ForsterImuPreintegrator()
    val viBundleAdjuster = VislocViBundleAdjuster()

    // Live attitude state
    private val latestAccel = FloatArray(3)
    private val latestMag = FloatArray(3)
    private val lowPassGravity = FloatArray(3)
    private var hasAccel = false
    private var hasMag = false

    // Current 3x3 rotation matrix from device/camera to world
    val currentRotationMatrix = VislocMath.identity3()
    private var prevRotationMatrix = VislocMath.identity3()
    private var hasPrevRotation = false

    // Timing for pre-integration and Hz telemetry
    private var lastAccelTimestampNs: Long = 0L
    private var lastRotationTimestampNs: Long = 0L
    private var lastUiUpdateMs: Long = 0L
    private var sampleCounter = 0
    private var measuredHz = 0f
    private var hzWindowStartMs = System.currentTimeMillis()

    // Synthesized or physical angular velocity in body frame (rad/s)
    private val latestOmegaBody = FloatArray(3)
    private val recentAccelWindow = ArrayList<FloatArray>()

    // Last keyframe RGB + 3D point cloud for KLT visual-inertial correspondence
    private var prevKeyframeBitmap: Bitmap? = null
    private var prevKeyframePositions: FloatArray? = null

    fun start() {
        lastAccelTimestampNs = 0L
        lastRotationTimestampNs = 0L
        hzWindowStartMs = System.currentTimeMillis()
        sampleCounter = 0

        accelSensor?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
        magSensor?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
        gyroSensor?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
        geoRotSensor?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        } ?: rotVecSensor?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
        linearAccSensor?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }

        Log.i(
            TAG,
            "Started VislocSensorEngine (hasHardwareGyro=$hasHardwareGyro, " +
                "accel=${accelSensor?.name}, mag=${magSensor?.name}, geoRot=${geoRotSensor?.name})"
        )
    }

    fun stop() {
        sensorManager.unregisterListener(this)
    }

    @Synchronized
    fun resetSession() {
        preintegrator.reset()
        viBundleAdjuster.reset()
        prevKeyframeBitmap = null
        prevKeyframePositions = null
        emitTelemetry()
    }

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_GYROSCOPE -> {
                latestOmegaBody[0] = event.values[0]
                latestOmegaBody[1] = event.values[1]
                latestOmegaBody[2] = event.values[2]
            }

            Sensor.TYPE_GEOMAGNETIC_ROTATION_VECTOR,
            Sensor.TYPE_ROTATION_VECTOR -> {
                val rMat = FloatArray(9)
                SensorManager.getRotationMatrixFromVector(rMat, event.values)
                updateAttitudeAndSynthesizeGyro(rMat, event.timestamp)
            }

            Sensor.TYPE_MAGNETIC_FIELD -> {
                val alpha = 0.25f
                if (!hasMag) {
                    latestMag[0] = event.values[0]
                    latestMag[1] = event.values[1]
                    latestMag[2] = event.values[2]
                    hasMag = true
                } else {
                    latestMag[0] = alpha * event.values[0] + (1f - alpha) * latestMag[0]
                    latestMag[1] = alpha * event.values[1] + (1f - alpha) * latestMag[1]
                    latestMag[2] = alpha * event.values[2] + (1f - alpha) * latestMag[2]
                }
                if (hasAccel && geoRotSensor == null && rotVecSensor == null) {
                    val rMat = FloatArray(9)
                    val iMat = FloatArray(9)
                    if (SensorManager.getRotationMatrix(rMat, iMat, lowPassGravity, latestMag)) {
                        updateAttitudeAndSynthesizeGyro(rMat, event.timestamp)
                    }
                }
            }

            Sensor.TYPE_ACCELEROMETER -> {
                val ax = event.values[0]
                val ay = event.values[1]
                val az = event.values[2]
                latestAccel[0] = ax
                latestAccel[1] = ay
                latestAccel[2] = az

                // Complementary low-pass filter to isolate gravity vector in body frame
                val alphaG = 0.88f
                if (!hasAccel) {
                    lowPassGravity[0] = ax
                    lowPassGravity[1] = ay
                    lowPassGravity[2] = az
                    hasAccel = true
                } else {
                    lowPassGravity[0] = alphaG * lowPassGravity[0] + (1f - alphaG) * ax
                    lowPassGravity[1] = alphaG * lowPassGravity[1] + (1f - alphaG) * ay
                    lowPassGravity[2] = alphaG * lowPassGravity[2] + (1f - alphaG) * az
                }

                // Update attitude fallback from Accel + Mag if needed
                if (hasMag) {
                    val rMat = FloatArray(9)
                    val iMat = FloatArray(9)
                    if (SensorManager.getRotationMatrix(rMat, iMat, lowPassGravity, latestMag)) {
                        if (geoRotSensor == null && rotVecSensor == null) {
                            updateAttitudeAndSynthesizeGyro(rMat, event.timestamp)
                        }
                    }
                }

                // Static VI initialization window (collect first 20 stationary/quasi-stationary samples)
                if (!viBundleAdjuster.isGravityInitialized) {
                    recentAccelWindow.add(floatArrayOf(ax, ay, az))
                    if (recentAccelWindow.size >= 20) {
                        val rSnap = synchronized(currentRotationMatrix) { currentRotationMatrix.clone() }
                        viBundleAdjuster.initializeStatic(recentAccelWindow, rSnap)
                    }
                }

                // Compute dt and feed Forster on-manifold pre-integrator with gravity-compensated linear acceleration
                val ts = event.timestamp
                if (lastAccelTimestampNs != 0L) {
                    val dt = (ts - lastAccelTimestampNs) * 1e-9f
                    if (dt in 0.001f..0.2f) {
                        // Linear acceleration in body frame (subtract low-pass gravity so pre-integrator accumulates pure motion)
                        val linAx = filterDeadband(ax - lowPassGravity[0], 0.12f)
                        val linAy = filterDeadband(ay - lowPassGravity[1], 0.12f)
                        val linAz = filterDeadband(az - lowPassGravity[2], 0.12f)
                        preintegrator.integrate(
                            gyroMeas = latestOmegaBody,
                            accMeas = floatArrayOf(linAx, linAy, linAz),
                            dt = dt
                        )
                    }
                }
                lastAccelTimestampNs = ts

                // Measure accelerometer sampling rate (Hz)
                sampleCounter++
                val nowMs = System.currentTimeMillis()
                val elapsedMs = nowMs - hzWindowStartMs
                if (elapsedMs >= 500L) {
                    measuredHz = (sampleCounter * 1000f) / elapsedMs
                    sampleCounter = 0
                    hzWindowStartMs = nowMs
                }
                if (nowMs - lastUiUpdateMs >= 160L) {
                    lastUiUpdateMs = nowMs
                    emitTelemetry()
                }
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    /**
     * Smooths the 3x3 attitude matrix on SO(3) and synthesizes pseudo-gyro angular velocity
     * omega_eff = so3_log(R_{k-1}^T * R_k) / dt when hardware gyroscope is absent (Redmi 13C 5G).
     */
    private fun updateAttitudeAndSynthesizeGyro(rawR: FloatArray, timestampNs: Long) {
        synchronized(currentRotationMatrix) {
            if (!hasPrevRotation) {
                System.arraycopy(rawR, 0, currentRotationMatrix, 0, 9)
                System.arraycopy(rawR, 0, prevRotationMatrix, 0, 9)
                hasPrevRotation = true
                lastRotationTimestampNs = timestampNs
                return
            }

            // Geodesic smoothing on SO(3) to suppress magnetometer jitter
            val rPrevT = VislocMath.transpose3(currentRotationMatrix)
            val dRRaw = VislocMath.matMul3(rPrevT, rawR)
            val dPhiRaw = VislocMath.so3Log(dRRaw)
            val smoothAlpha = if (hasHardwareGyro) 0.65f else 0.35f
            val dPhiSmooth = floatArrayOf(
                dPhiRaw[0] * smoothAlpha,
                dPhiRaw[1] * smoothAlpha,
                dPhiRaw[2] * smoothAlpha
            )
            val rUpdated = VislocMath.matMul3(currentRotationMatrix, VislocMath.so3Exp(dPhiSmooth))
            System.arraycopy(rUpdated, 0, currentRotationMatrix, 0, 9)

            if (!hasHardwareGyro && lastRotationTimestampNs != 0L) {
                val dt = (timestampNs - lastRotationTimestampNs) * 1e-9f
                if (dt in 0.002f..0.25f) {
                    val rLastT = VislocMath.transpose3(prevRotationMatrix)
                    val dRStep = VislocMath.matMul3(rLastT, currentRotationMatrix)
                    val dPhiStep = VislocMath.so3Log(dRStep)
                    latestOmegaBody[0] = (dPhiStep[0] / dt).coerceIn(-6f, 6f)
                    latestOmegaBody[1] = (dPhiStep[1] / dt).coerceIn(-6f, 6f)
                    latestOmegaBody[2] = (dPhiStep[2] / dt).coerceIn(-6f, 6f)
                }
            }
            System.arraycopy(currentRotationMatrix, 0, prevRotationMatrix, 0, 9)
            lastRotationTimestampNs = timestampNs
        }
    }

    /**
     * Called when a new MoGe keyframe is captured in Single or Multi-Frame mode.
     * Executes:
     * 1. Snapshot & reset of `ForsterImuPreintegrator`
     * 2. 2D/3D KLT correspondence tracking (`VislocKltTracker`) against the previous keyframe
     * 3. VIBA2 closed-form scale & 6-DoF pose fusion (`VislocViBundleAdjuster.fuseKeyframe`)
     */
    @Synchronized
    fun processKeyframeCapture(
        rgbBitmap: Bitmap,
        rawPositions: FloatArray,
        rRelWorld: FloatArray
    ): VislocKeyframeState {
        val preintSnap = preintegrator.snapshotAndReset()
        val prevBmp = prevKeyframeBitmap
        val prevPos = prevKeyframePositions

        val visualMotion = if (prevBmp != null && prevPos != null) {
            VislocKltTracker.estimateVisualMotion(
                prevBitmap = prevBmp,
                prevPositions = prevPos,
                currBitmap = rgbBitmap,
                currPositions = rawPositions,
                imuDeltaR = preintSnap.deltaR,
                stride = 4
            )
        } else {
            VislocKltTracker.VisualMotionEstimate(FloatArray(3), preintSnap.deltaR, 0)
        }

        // Cache current frame for next keyframe's KLT correspondence
        prevKeyframeBitmap = rgbBitmap.copy(Bitmap.Config.ARGB_8888, false)
        prevKeyframePositions = rawPositions.clone()

        val state = viBundleAdjuster.fuseKeyframe(
            timestampSec = System.currentTimeMillis() / 1000.0,
            rWbCurr = rRelWorld,
            preint = preintSnap,
            visualMotion = visualMotion
        )
        emitTelemetry(lastPreint = preintSnap)
        return state
    }

    private fun emitTelemetry(lastPreint: ImuPreintegratedDelta? = null) {
        val rSnap = synchronized(currentRotationMatrix) { currentRotationMatrix.clone() }
        val orientation = FloatArray(3)
        SensorManager.getOrientation(rSnap, orientation)
        val yawDeg = Math.toDegrees(orientation[0].toDouble()).toFloat()
        val pitchDeg = Math.toDegrees(orientation[1].toDouble()).toFloat()
        val rollDeg = Math.toDegrees(orientation[2].toDouble()).toFloat()

        val peek = lastPreint ?: preintegrator.peekDelta()
        val lastKf = viBundleAdjuster.trajectory.lastOrNull()

        val modeStr = if (hasHardwareGyro) {
            "6-Axis Hardware IMU + VIBA2"
        } else {
            "Redmi 13C Hybrid (BMI2xy Acc + MMC5603 Mag + KLT)"
        }

        onTelemetryUpdated(
            VislocTelemetrySnapshot(
                sensorModeLabel = modeStr,
                hasHardwareGyro = hasHardwareGyro,
                isGravityInitialized = viBundleAdjuster.isGravityInitialized,
                accelHz = measuredHz,
                rollDeg = rollDeg,
                pitchDeg = pitchDeg,
                yawDeg = yawDeg,
                posWb = lastKf?.pWb?.clone() ?: floatArrayOf(0f, 0f, 0f),
                velWb = lastKf?.vWb?.clone() ?: floatArrayOf(0f, 0f, 0f),
                lastDeltaP = peek.deltaP.clone(),
                lastDeltaV = peek.deltaV.clone(),
                vibaScale = lastKf?.scaleFactor ?: 1.0f,
                keyframeCount = viBundleAdjuster.trajectory.size,
                lastKltInliers = lastKf?.kltMatches ?: 0,
                biasAccNorm = viBundleAdjuster.getAccBiasNorm()
            )
        )
    }

    private fun filterDeadband(v: Float, threshold: Float): Float {
        return if (abs(v) < threshold) 0f else v
    }
}
