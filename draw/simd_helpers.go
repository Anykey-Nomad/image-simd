//go:build goexperiment.simd && amd64

package draw

import "simd/archsimd"

// accumulateHorizontalRGBA performs SIMD-accelerated horizontal weighted accumulation
// of RGBA pixels from src into accumulators (pr, pg, pb, pa).
// Uses Float64x4 to process all 4 channels simultaneously.
//
// Optimizations applied:
//   - BCE hints eliminate bounds checks inside the hot loop
//   - Weight array (wBuf) is reused across iterations to avoid re-allocation
//   - weightVec is updated in-place instead of creating a new [4]float64 each time
func accumulateHorizontalRGBA(src []uint8, contribs []contrib, srcStride, y, startIdx int32) [4]float64 {
	var sum archsimd.Float64x4
	var buf [4]float64
	var wBuf [4]float64

	for i := startIdx; i < int32(len(contribs)); i++ {
		c := &contribs[i]
		pi := y*srcStride + c.coord*4

		// BCE hint: prove to the compiler that all 4 channel bytes are within bounds.
		_ = src[pi : pi+4]

		// 1. Скалярно делаем только приведение типов (это дешевле умножений)
		buf[0] = float64(src[pi])
		buf[1] = float64(src[pi+1])
		buf[2] = float64(src[pi+2])
		buf[3] = float64(src[pi+3])

		pixelVec := archsimd.LoadFloat64x4(buf[:])

		// 2. Вычисляем итоговый вес для этого пикселя один раз
		// 257.0 — это float64 эквивалент 0x101 (ускоряет конвертацию int -> float)
		w := c.weight * 257.0

		// 3. Бродкаст веса — переиспользуем wBuf, обновляем in-place
		wBuf[0] = w
		wBuf[1] = w
		wBuf[2] = w
		wBuf[3] = w

		// 4. Векторное умножение + сложение (SIMD берет на себя всю тяжелую математику)
		weightVec := archsimd.LoadFloat64x4(wBuf[:])
		sum = sum.Add(pixelVec.Mul(weightVec))
	}

	var out [4]float64
	sum.Store(out[:])
	return out
}

// accumulateHorizontalNRGBA performs SIMD-accelerated horizontal weighted accumulation
// of NRGBA pixels (with alpha correction) into accumulators.
//
// Optimizations applied:
//   - BCE hints eliminate bounds checks inside the hot loop
//   - Weight array (wBuf) is reused across iterations
//   - Alpha correction done via weight vector adjustment instead of reloading pixel
//     with alpha=1.0 (eliminates one LoadFloat64x4 per iteration)
func accumulateHorizontalNRGBA(src []uint8, contribs []contrib, srcStride, y, startIdx int32) [4]float64 {
	var sum archsimd.Float64x4
	var buf [4]float64
	var wBuf [4]float64

	for i := startIdx; i < int32(len(contribs)); i++ {
		c := &contribs[i]
		pi := y*srcStride + c.coord*4

		// BCE hint: prove to the compiler that all 4 channel bytes are within bounds.
		_ = src[pi : pi+4]

		// Загружаем каналы — ОДИН раз, без повторной загрузки
		buf[0] = float64(src[pi])
		buf[1] = float64(src[pi+1])
		buf[2] = float64(src[pi+2])
		buf[3] = float64(src[pi+3])

		pixelVec := archsimd.LoadFloat64x4(buf[:])

		// Считаем скалярные множители
		pau := buf[3] * 257.0
		alphaVal := buf[3] // исходное значение альфы в float64

		// Формула для RGB: pixel * (pau / 65535.0) * weight * 257.0
		// Формула для A: 1.0 * pau * weight  (альфа не домножается на саму себя)
		//
		// Вместо того чтобы менять pixelVec[3] = 1.0 и перезагружать его,
		// мы корректируем вес для альфа-канала:
		//   wBuf[3] = alphaMultiplier / alphaVal (если alphaVal != 0)
		// Тогда pixelVec[3] * wBuf[3] = alphaVal * (alphaMultiplier / alphaVal) = alphaMultiplier
		rgbMultiplier := (pau / 65535.0) * c.weight * 257.0
		alphaMultiplier := pau * c.weight

		wBuf[0] = rgbMultiplier
		wBuf[1] = rgbMultiplier
		wBuf[2] = rgbMultiplier

		// Корректируем вес альфа-канала, чтобы компенсировать исходное значение в пикселе
		if alphaVal != 0 {
			wBuf[3] = alphaMultiplier / alphaVal
		} else {
			wBuf[3] = 0
		}

		weightVec := archsimd.LoadFloat64x4(wBuf[:])
		sum = sum.Add(pixelVec.Mul(weightVec))
	}

	var out [4]float64
	sum.Store(out[:])
	return out
}

// accumulateVertical performs SIMD-accelerated vertical weighted accumulation.
//
// Optimizations applied:
//   - Loop unrolling: processes 2 contributions per iteration
//     to reduce branch overhead and improve FMA pipelining
//   - Weight arrays reused across iterations
func accumulateVertical(tmp [][4]float64, contribs []contrib, dw, dx, sStart, sEnd int32) [4]float64 {
	var sum archsimd.Float64x4
	var wBuf [4]float64

	// Loop unrolling: process 2 contributions per iteration
	i := sStart

	// Main unrolled loop: 2 contributions per iteration
	for ; i+1 < sEnd; i += 2 {
		// --- Contribution 0 ---
		c0 := &contribs[i]
		pixelVec0 := archsimd.LoadFloat64x4(tmp[c0.coord*int32(dw)+dx][:])
		wBuf[0] = c0.weight
		wBuf[1] = c0.weight
		wBuf[2] = c0.weight
		wBuf[3] = c0.weight
		weightVec0 := archsimd.LoadFloat64x4(wBuf[:])

		// --- Contribution 1 ---
		c1 := &contribs[i+1]
		pixelVec1 := archsimd.LoadFloat64x4(tmp[c1.coord*int32(dw)+dx][:])
		wBuf[0] = c1.weight
		wBuf[1] = c1.weight
		wBuf[2] = c1.weight
		wBuf[3] = c1.weight
		weightVec1 := archsimd.LoadFloat64x4(wBuf[:])

		// Векторные операции: обе пары умножаются и складываются
		// Process both multiplications before additions for better pipelining
		mul0 := pixelVec0.Mul(weightVec0)
		mul1 := pixelVec1.Mul(weightVec1)
		sum = sum.Add(mul0).Add(mul1)
	}

	// Scalar tail: handle remaining contribution (0 or 1)
	if i < sEnd {
		c := &contribs[i]
		pixelVec := archsimd.LoadFloat64x4(tmp[c.coord*int32(dw)+dx][:])
		wBuf[0] = c.weight
		wBuf[1] = c.weight
		wBuf[2] = c.weight
		wBuf[3] = c.weight
		weightVec := archsimd.LoadFloat64x4(wBuf[:])
		sum = sum.Add(pixelVec.Mul(weightVec))
	}

	var out [4]float64
	sum.Store(out[:])
	return out
}
