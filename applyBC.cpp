#include "BoundaryCompensation.h"

#include "CommonLib/Slice.h"
#include "CommonLib/InterPrediction.h"

#include <algorithm>
#include <cmath>

namespace BoundaryComp
{

// ============================================================================
// 12-tap interpolation filters
// Same coefficients as the Python Simple12TapInterp
// ============================================================================

static const int16_t s_interp12[16][12] =
{
  {  0,  0,  0,  0,  0, 256,   0,   0,  0,  0,  0,  0 },
  { -1,  2, -3,  6, -14, 254,  16,  -7,  4, -2,  1,  0 },
  { -1,  3, -7, 12, -26, 249,  35, -15,  8, -4,  2,  0 },
  { -2,  5, -9, 17, -36, 241,  54, -22, 12, -6,  3, -1 },
  { -2,  5,-11, 21, -43, 230,  75, -29, 15, -8,  4, -1 },
  { -2,  6,-13, 24, -48, 216,  97, -36, 19,-10,  4, -1 },
  { -2,  7,-14, 25, -51, 200, 119, -42, 22,-12,  5, -1 },
  { -2,  7,-14, 26, -51, 181, 140, -46, 24,-13,  6, -2 },
  { -2,  6,-13, 25, -50, 162, 162, -50, 25,-13,  6, -2 },
  { -2,  6,-13, 24, -46, 140, 181, -51, 26,-14,  7, -2 },
  { -1,  5,-12, 22, -42, 119, 200, -51, 25,-14,  7, -2 },
  { -1,  4,-10, 19, -36,  97, 216, -48, 24,-13,  6, -2 },
  { -1,  4, -8, 15, -29,  75, 230, -43, 21,-11,  5, -2 },
  { -1,  3, -6, 12, -22,  54, 241, -36, 17, -9,  5, -2 },
  {  0,  2, -4,  8, -15,  35, 249, -26, 12, -7,  3, -1 },
  {  0,  1, -2,  4,  -7,  16, 254, -14,  6, -3,  2, -1 }
};

// ============================================================================
// Basic helpers
// ============================================================================

static inline Pel getClampedPelAt(const CPelBuf& buf, int x, int y)
{
  x = std::max(0, std::min(x, buf.width  - 1));
  y = std::max(0, std::min(y, buf.height - 1));
  return buf.buf[y * buf.stride + x];
}

static inline void splitMv16Safe(int mv, int& intPart, int& fracPart)
{
  intPart  = int(std::floor(double(mv) / 16.0));
  fracPart = mv - (intPart << 4);
  CHECK(fracPart < 0 || fracPart > 15, "fracPart out of range");
}

static inline int clipIntToBitDepthRange(int v, const CPelBuf& refBuf)
{
  // This helper is not used directly, but kept for completeness if needed.
  // Since refBuf itself does not carry bit depth, clipping is generally handled later.
  return v;
}

// ============================================================================
// 12-tap horizontal/vertical filtering
// Python equivalent:
//   frac_x != 0 -> conv2d with [1,1,1,12]
//   frac_y != 0 -> conv2d with [1,1,12,1]
// Border handling: replicate
// ============================================================================

static inline int filter12HorRaw256(const CPelBuf& ref, int xInt, int yInt, int fracX)
{
  CHECK(fracX < 0 || fracX > 15, "fracX out of range");

  const int16_t* f = s_interp12[fracX];
  int sum = 0;

  for (int k = 0; k < 12; ++k)
  {
    const int sx = xInt - 5 + k;
    sum += int(f[k]) * int(getClampedPelAt(ref, sx, yInt));
  }

  return sum; // scaled by 256
}

static inline int filter12VerRaw(
    const CPelBuf& ref,
    int xInt, int yInt,
    int fracY)
{
  CHECK(fracY < 0 || fracY > 15, "fracY out of range");

  const int16_t* f = s_interp12[fracY];
  int sum = 0;

  for (int k = 0; k < 12; ++k)
  {
    const int sy = yInt - 5 + k;
    sum += int(f[k]) * int(getClampedPelAt(ref, xInt, sy));
  }

  return sum; // scaled by 256
}

static inline int filter12VerFromTmpRaw256(
    const int tmpCol[12],
    int fracY)
{
  CHECK(fracY < 0 || fracY > 15, "fracY out of range");

  const int16_t* f = s_interp12[fracY];
  int sum = 0;

  for (int k = 0; k < 12; ++k)
  {
    sum += int(f[k]) * tmpCol[k];
  }

  return sum; // input already scaled by 256 => output scaled by 256*256
}

// ============================================================================
// Sample one point from reference using 12-tap interpolation
// MV unit: 1/16 pel
//
// Equivalent to Python extract_block() per output sample
// ============================================================================

static inline float sampleRef12TapFloat(
    const CPelBuf& refBuf,
    int x, int y,
    int mvX, int mvY)
{
  int intX, fracX, intY, fracY;
  splitMv16Safe(mvX, intX, fracX);
  splitMv16Safe(mvY, intY, fracY);

  const int refX = x + intX;
  const int refY = y + intY;

  if (fracX == 0 && fracY == 0)
  {
    return float(getClampedPelAt(refBuf, refX, refY));
  }
  else if (fracX != 0 && fracY == 0)
  {
    const int sum = filter12HorRaw256(refBuf, refX, refY, fracX);
    const int val = (sum + 128) >> 8;
    return float(val);
  }
  else if (fracX == 0 && fracY != 0)
  {
    const int sum = filter12VerRaw(refBuf, refX, refY, fracY);
    const int val = (sum + 128) >> 8;
    return float(val);
  }
  else
  {
    int tmpCol[12];

    for (int k = 0; k < 12; ++k)
    {
      const int sy = refY - 5 + k;
      tmpCol[k] = filter12HorRaw256(refBuf, refX, sy, fracX); // scaled by 256
    }

    const int sum = filter12VerFromTmpRaw256(tmpCol, fracY);   // scaled by 256*256
    const int val = (sum + 32768) >> 16;
    return float(val);
  }
}

// ============================================================================
// Fill one rectangular stripe in ext tensor from ref picture
//
// dstTensor shape: [1, h+4, w+4]
// Fill region:
//   dstTensor[0, dstY0 : dstY0+dstH, dstX0 : dstX0+dstW]
//
// Reference sample position for local pixel (xx,yy):
//   global = (baseX + xx, baseY + yy)
//   then motion-compensated with (mvX,mvY)
// ============================================================================

static void extractBlock12TapStripe(
    const CPelBuf& refBuf,
    int dstW,
    int dstH,
    int dstX0,
    int dstY0,
    int baseX,
    int baseY,
    int mvX,
    int mvY,
    SimpleNN::Tensor& dstTensor)
{
  CHECK(dstTensor.C != 1, "dstTensor must have C=1");

  for (int yy = 0; yy < dstH; ++yy)
  {
    for (int xx = 0; xx < dstW; ++xx)
    {
      const float v = sampleRef12TapFloat(
          refBuf,
          baseX + xx,
          baseY + yy,
          mvX,
          mvY);

      dstTensor.at(0, dstY0 + yy, dstX0 + xx) = v;
    }
  }
}

// ============================================================================
// Pred ext tensor
//
// shape: [1, h+4, w+4]
// center [4:,4:] is srcBuf predictor
// rest initialized to 0
// ============================================================================

SimpleNN::Tensor makePredExtTensor(const CPelBuf& srcBuf)
{
  const int h = srcBuf.height;
  const int w = srcBuf.width;

  SimpleNN::Tensor predExt(1, h + 4, w + 4);
  std::fill(predExt.data.begin(), predExt.data.end(), 0.0f);

  for (int y = 0; y < h; ++y)
  {
    const Pel* row = srcBuf.buf + y * srcBuf.stride;
    for (int x = 0; x < w; ++x)
    {
      predExt.at(0, y + 4, x + 4) = float(row[x]);
    }
  }

  return predExt;
}

// ============================================================================
// Retrieve one CU-level translational MV/reference
//
// IMPORTANT:
// - This is intentionally written for the common "single translational inter CU"
//   case.
// - If your content includes affine / sub-PU / DMVR / BDOF / special modes,
//   you need extra handling.
// - For bi-pred CU, this implementation prioritizes L0 if valid, otherwise L1.
// ============================================================================

static void getCuRefInfoSingleMV(
    const CodingUnit& cu,
    const CPelBuf*& refBufOut,
    int& mvXOut,
    int& mvYOut)
{
  CHECK(!CU::isInter(cu), "getCuRefInfoSingleMV expects inter CU");
  CHECK(cu.firstPU == nullptr, "CU has no PU");

  const PredictionUnit& pu = *cu.firstPU;

  RefPicList refList = REF_PIC_LIST_0;
  if (pu.refIdx[REF_PIC_LIST_0] < 0)
  {
    CHECK(pu.refIdx[REF_PIC_LIST_1] < 0, "No valid reference index in either list");
    refList = REF_PIC_LIST_1;
  }

  const int refIdx = pu.refIdx[refList];
  CHECK(refIdx < 0, "Invalid refIdx");

  Picture* refPic = cu.cs->slice->getRefPic(refList, refIdx);
  CHECK(refPic == nullptr, "refPic is null");

  static CPelBuf s_dummy;
  s_dummy = refPic->getRecoBuf(COMPONENT_Y);
  refBufOut = &s_dummy;

  const Mv mv = pu.mv[refList];
  mvXOut = mv.getHor();
  mvYOut = mv.getVer();
}

// ============================================================================
// Ref ext tensor
//
// shape: [1, h+4, w+4]
//
// Fill only limited regions to save computation:
//   1) left stripe   : [4:4+h, 0:8]   <- width 8, height h
//   2) top stripe    : [0:8,   4:4+w] <- width w, height 8
//   3) center region : [4:4+h, 4:4+w] <- width w, height h
//
// If you later decide that center is unnecessary, remove block (3).
// ============================================================================

SimpleNN::Tensor makeRefExtTensorLimited(const CodingUnit& cu)
{
  const CompArea& blk = cu.Y();
  const int x = blk.x;
  const int y = blk.y;
  const int w = blk.width;
  const int h = blk.height;

  SimpleNN::Tensor refExt(1, h + 4, w + 4);
  std::fill(refExt.data.begin(), refExt.data.end(), 0.0f);

  const CPelBuf* refBufPtr = nullptr;
  int mvX = 0;
  int mvY = 0;
  getCuRefInfoSingleMV(cu, refBufPtr, mvX, mvY);
  CHECK(refBufPtr == nullptr, "refBufPtr is null");

  const CPelBuf& refBuf = *refBufPtr;

  // 1) Left stripe: ext[4:4+h, 0:8]
  // global base position = (x - 4, y)
  extractBlock12TapStripe(
      refBuf,
      8, h,
      0, 4,
      x - 4, y,
      mvX, mvY,
      refExt);

  // 2) Top stripe: ext[0:8, 4:4+w]
  // global base position = (x, y - 4)
  extractBlock12TapStripe(
      refBuf,
      w, 8,
      4, 0,
      x, y - 4,
      mvX, mvY,
      refExt);

  // 3) Center CU area: ext[4:4+h, 4:4+w]
  extractBlock12TapStripe(
      refBuf,
      w, h,
      4, 4,
      x, y,
      mvX, mvY,
      refExt);

  return refExt;
}

// ============================================================================
// Output tensor -> PelBuf
// ============================================================================

void storeTensorToPelBufClipped(
    const SimpleNN::Tensor& tensor,
    PelBuf& dstBuf,
    const ClpRng& clpRng)
{
  CHECK(tensor.C != 1, "tensor C must be 1");
  CHECK(tensor.H != dstBuf.height || tensor.W != dstBuf.width,
        "tensor / dstBuf size mismatch");

  for (int y = 0; y < dstBuf.height; ++y)
  {
    Pel* dstRow = dstBuf.buf + y * dstBuf.stride;
    for (int x = 0; x < dstBuf.width; ++x)
    {
      const float v = tensor.at(0, y, x);
      const int iv = int(std::lround(v));
      dstRow[x] = Pel(Clip3(clpRng.min(), clpRng.max(), iv));
    }
  }
}

// ============================================================================
// Main API
// ============================================================================

void applyBoundaryCompensate(
    const BoundaryModel& boundaryModel,
    const CodingUnit& cu,
    const CPelBuf& srcBuf,
    PelBuf& dstBuf,
    const ClpRng& clpRng)
{
  const CompArea& blk = cu.Y();
  const int w = blk.width;
  const int h = blk.height;

  CHECK(srcBuf.width != w || srcBuf.height != h, "srcBuf size mismatch");
  CHECK(dstBuf.width != w || dstBuf.height != h, "dstBuf size mismatch");

  // 1) pred extension tensor
  const SimpleNN::Tensor predExt = makePredExtTensor(srcBuf);

  // 2) ref extension tensor (12-tap MC, limited stripes + center)
  const SimpleNN::Tensor refExt = makeRefExtTensorLimited(cu);

  // 3) run boundary model
  SimpleNN::Tensor outTensor;
  boundaryModel.forward(predExt, refExt, outTensor);

  CHECK(outTensor.C != 1 || outTensor.H != h || outTensor.W != w,
        "BoundaryModel output shape mismatch");

  // 4) store to dstBuf
  storeTensorToPelBufClipped(outTensor, dstBuf, clpRng);
}

} // namespace BoundaryComp






















#include "BoundaryModel.h"

#include <cmath>

// -----------------------------------------------------------------------------
// helpers
// -----------------------------------------------------------------------------

static SimpleNN::Tensor transposeHW(const SimpleNN::Tensor& in)
{
  SimpleNN::Tensor out(in.C, in.W, in.H);

  for (int c = 0; c < in.C; ++c)
  {
    for (int y = 0; y < in.H; ++y)
    {
      for (int x = 0; x < in.W; ++x)
      {
        out.at(c, x, y) = in.at(c, y, x);
      }
    }
  }
  return out;
}

// x_left = cat([pred[:, :, 4:, 0:8], ref[:, :, 4:, 0:8]], dim=1)
// pred/ref each: [1, h+4, w+4]
// output: [2, h, 8]
static SimpleNN::Tensor makeLeftBranchInput(
    const SimpleNN::Tensor& predExt,
    const SimpleNN::Tensor& refExt)
{
  CHECK(predExt.C != 1 || refExt.C != 1, "predExt/refExt must have C=1");
  CHECK(predExt.H != refExt.H || predExt.W != refExt.W, "predExt/refExt size mismatch");
  CHECK(predExt.H < 8 || predExt.W < 8, "predExt/refExt too small");

  const int h = predExt.H - 4;
  const int w = predExt.W - 4;

  (void)w;

  CHECK(h % 4 != 0, "Left branch length must be multiple of 4");

  SimpleNN::Tensor out(2, h, 8);

  for (int y = 0; y < h; ++y)
  {
    for (int x = 0; x < 8; ++x)
    {
      out.at(0, y, x) = predExt.at(0, y + 4, x);
      out.at(1, y, x) = refExt.at(0, y + 4, x);
    }
  }

  return out;
}

// x_top = cat([pred[:, :, 0:8, 4:], ref[:, :, 0:8, 4:]], dim=1)
// before transpose: [2, 8, w]
// after transpose : [2, w, 8]
static SimpleNN::Tensor makeTopBranchInputTransposed(
    const SimpleNN::Tensor& predExt,
    const SimpleNN::Tensor& refExt)
{
  CHECK(predExt.C != 1 || refExt.C != 1, "predExt/refExt must have C=1");
  CHECK(predExt.H != refExt.H || predExt.W != refExt.W, "predExt/refExt size mismatch");
  CHECK(predExt.H < 8 || predExt.W < 8, "predExt/refExt too small");

  const int h = predExt.H - 4;
  const int w = predExt.W - 4;

  (void)h;

  CHECK(w % 4 != 0, "Top branch length must be multiple of 4");

  SimpleNN::Tensor tmp(2, 8, w);

  for (int y = 0; y < 8; ++y)
  {
    for (int x = 0; x < w; ++x)
    {
      tmp.at(0, y, x) = predExt.at(0, y, x + 4);
      tmp.at(1, y, x) = refExt.at(0, y, x + 4);
    }
  }

  return transposeHW(tmp); // [2, w, 8]
}

// pred[:, :, 4:, 4:8] += tanh(gate) * val
static void applyLeftBranchOutput(
    SimpleNN::Tensor& predExt,
    const SimpleNN::Tensor& outLeft)
{
  CHECK(predExt.C != 1, "predExt must have C=1");
  CHECK(outLeft.C != 2, "outLeft must have C=2");
  CHECK(outLeft.H != predExt.H - 4 || outLeft.W != 4, "outLeft shape mismatch");

  for (int y = 0; y < outLeft.H; ++y)
  {
    for (int x = 0; x < 4; ++x)
    {
      const float val  = outLeft.at(0, y, x);
      const float gate = std::tanh(outLeft.at(1, y, x));
      predExt.at(0, y + 4, x + 4) += gate * val;
    }
  }
}

// pred[:, :, 4:8, 4:] += tanh(gate) * val
static void applyTopBranchOutput(
    SimpleNN::Tensor& predExt,
    const SimpleNN::Tensor& outTop)
{
  CHECK(predExt.C != 1, "predExt must have C=1");
  CHECK(outTop.C != 2, "outTop must have C=2");
  CHECK(outTop.H != 4 || outTop.W != predExt.W - 4, "outTop shape mismatch");

  for (int y = 0; y < 4; ++y)
  {
    for (int x = 0; x < outTop.W; ++x)
    {
      const float val  = outTop.at(0, y, x);
      const float gate = std::tanh(outTop.at(1, y, x));
      predExt.at(0, y + 4, x + 4) += gate * val;
    }
  }
}

// return pred[:, :, 4:, 4:]
static SimpleNN::Tensor cropCenterOutput(const SimpleNN::Tensor& predExt)
{
  CHECK(predExt.C != 1, "predExt must have C=1");
  CHECK(predExt.H < 5 || predExt.W < 5, "predExt too small");

  const int h = predExt.H - 4;
  const int w = predExt.W - 4;

  SimpleNN::Tensor out(1, h, w);

  for (int y = 0; y < h; ++y)
  {
    for (int x = 0; x < w; ++x)
    {
      out.at(0, y, x) = predExt.at(0, y + 4, x + 4);
    }
  }

  return out;
}

// -----------------------------------------------------------------------------
// BoundaryModel
// -----------------------------------------------------------------------------

BoundaryModel::BoundaryModel(const BoundaryPredictionModel& predModel)
  : m_predModel(predModel)
{
}

void BoundaryModel::forward(
    const SimpleNN::Tensor& predExt,
    const SimpleNN::Tensor& refExt,
    SimpleNN::Tensor& outTensor) const
{
  CHECK(predExt.C != 1 || refExt.C != 1, "predExt/refExt must have C=1");
  CHECK(predExt.H != refExt.H || predExt.W != refExt.W, "predExt/refExt size mismatch");
  CHECK(predExt.H < 8 || predExt.W < 8, "predExt/refExt too small");

  const int h = predExt.H - 4;
  const int w = predExt.W - 4;

  CHECK(h % 4 != 0, "h must be multiple of 4");
  CHECK(w % 4 != 0, "w must be multiple of 4");

  // local mutable copy, because PyTorch forward updates pred in-place
  SimpleNN::Tensor predWork = predExt;

  // 1) left branch
  {
    SimpleNN::Tensor xLeft   = makeLeftBranchInput(predWork, refExt); // [2,h,8]
    SimpleNN::Tensor outLeft;                                         // [2,h,4]
    m_predModel.forward(xLeft, outLeft);
    applyLeftBranchOutput(predWork, outLeft);
  }

  // 2) top branch
  {
    SimpleNN::Tensor xTopT   = makeTopBranchInputTransposed(predWork, refExt); // [2,w,8]
    SimpleNN::Tensor outTopT;                                                  // [2,w,4]
    m_predModel.forward(xTopT, outTopT);

    SimpleNN::Tensor outTop = transposeHW(outTopT); // [2,4,w]
    applyTopBranchOutput(predWork, outTop);
  }

  // 3) crop center
  outTensor = cropCenterOutput(predWork); // [1,h,w]
}















static SimpleNN::Tensor makePredExtTensor(
    const CodingUnit& cu,
    const CPelBuf& srcBuf)
{
  const CompArea& blk = cu.Y();

  const int x = blk.x;
  const int y = blk.y;
  const int w = blk.width;
  const int h = blk.height;

  CHECK(srcBuf.width != w || srcBuf.height != h, "srcBuf size mismatch");

  const CPelBuf recoBuf = cu.cs->picture->getRecoBuf(COMPONENT_Y);
  const int picW = cu.cs->picture->Y().width;
  const int picH = cu.cs->picture->Y().height;

  // output: [1, h+4, w+4]
  SimpleNN::Tensor predExt(1, h + 4, w + 4);

  // 1) fill whole tensor with recon context
  // local (yy, xx) corresponds to global (y - 4 + yy, x - 4 + xx)
  for (int yy = 0; yy < h + 4; ++yy)
  {
    for (int xx = 0; xx < w + 4; ++xx)
    {
      const int globalY = y - 4 + yy;
      const int globalX = x - 4 + xx;

      float v = 0.0f;

      if (globalX >= 0 && globalX < picW && globalY >= 0 && globalY < picH)
      {
        v = static_cast<float>(recoBuf.buf[globalY * recoBuf.stride + globalX]);
      }

      predExt.at(0, yy, xx) = v;
    }
  }

  // 2) overwrite current CU region with predictor
  // predExt[4:, 4:] = srcBuf
  for (int yy = 0; yy < h; ++yy)
  {
    const Pel* srcRow = srcBuf.buf + yy * srcBuf.stride;
    for (int xx = 0; xx < w; ++xx)
    {
      predExt.at(0, yy + 4, xx + 4) = static_cast<float>(srcRow[xx]);
    }
  }

  return predExt;
}












