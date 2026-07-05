//go:build !goexperiment.simd || !amd64

package draw

// accumulateHorizontalRGBA is the scalar fallback for horizontal RGBA accumulation.
//
// Optimizations applied:
//   - BCE hint eliminates bounds checks inside the hot loop
//   - Weight precomputation: w257 = 0x101 * c.weight computed once per contrib
func accumulateHorizontalRGBA(src []uint8, contribs []contrib, srcStride, y, tIdx int32) [4]float64 {
	var pr, pg, pb, pa float64
	for i := tIdx; i < int32(len(contribs)); i++ {
		c := &contribs[i]
		pi := y*srcStride + c.coord*4
		// BCE hint
		_ = src[pi+3]
		// Precompute combined weight to avoid repeated 0x101 multiplication
		w257 := 0x101 * c.weight
		pr += float64(src[pi]) * w257
		pg += float64(src[pi+1]) * w257
		pb += float64(src[pi+2]) * w257
		pa += float64(src[pi+3]) * w257
	}
	return [4]float64{pr, pg, pb, pa}
}

// accumulateHorizontalNRGBA is the scalar fallback for horizontal NRGBA accumulation.
//
// Optimizations applied:
//   - BCE hint eliminates bounds checks inside the hot loop
//   - Precomputed factors to reduce redundant multiplications per pixel
func accumulateHorizontalNRGBA(src []uint8, contribs []contrib, srcStride, y, tIdx int32) [4]float64 {
	var pr, pg, pb, pa float64
	for i := tIdx; i < int32(len(contribs)); i++ {
		c := &contribs[i]
		pi := y*srcStride + c.coord*4
		// BCE hint
		_ = src[pi+3]
		pau := float64(src[pi+3]) * 0x101
		// Precompute rgbFactor once instead of computing it per-channel
		rgbFactor := 0x101 * pau / 0xffff * c.weight
		pr += float64(src[pi]) * rgbFactor
		pg += float64(src[pi+1]) * rgbFactor
		pb += float64(src[pi+2]) * rgbFactor
		pa += pau * c.weight
	}
	return [4]float64{pr, pg, pb, pa}
}

// accumulateVertical is the scalar fallback for vertical accumulation.
//
// Optimizations applied:
//   - Loop unrolling: processes 2 contributions per iteration
//     to reduce branch overhead
//   - Precomputed weight to avoid repeated field access
func accumulateVertical(tmp [][4]float64, contribs []contrib, dw, dx, sStart, sEnd int32) [4]float64 {
	var pr, pg, pb, pa float64

	// Main unrolled loop: 2 contributions per iteration
	i := sStart
	for ; i+1 < sEnd; i += 2 {
		c0 := &contribs[i]
		p0 := &tmp[c0.coord*int32(dw)+dx]
		w0 := c0.weight
		c1 := &contribs[i+1]
		p1 := &tmp[c1.coord*int32(dw)+dx]
		w1 := c1.weight

		pr += p0[0]*w0 + p1[0]*w1
		pg += p0[1]*w0 + p1[1]*w1
		pb += p0[2]*w0 + p1[2]*w1
		pa += p0[3]*w0 + p1[3]*w1
	}

	// Scalar tail: handle remaining contribution (0 or 1)
	if i < sEnd {
		c := &contribs[i]
		p := &tmp[c.coord*int32(dw)+dx]
		w := c.weight
		pr += p[0] * w
		pg += p[1] * w
		pb += p[2] * w
		pa += p[3] * w
	}

	return [4]float64{pr, pg, pb, pa}
}
