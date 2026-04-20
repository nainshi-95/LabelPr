#pragma once

#include "CommonDef.h"
#include "Buffer.h"
#include "CommonLib/Unit.h"
#include "CommonLib/Picture.h"
#include "SimpleNN.h"
#include "BoundaryModel.h"

#include <vector>
#include <cstdint>

namespace BoundaryComp
{

// Main entry
void applyBoundaryCompensate(
    const BoundaryModel& boundaryModel,
    const CodingUnit& cu,
    const CPelBuf& srcBuf,
    PelBuf& dstBuf,
    const ClpRng& clpRng);

// -----------------------------------------------------------------------------
// Helper APIs (exposed only if you want to unit-test them separately)
// -----------------------------------------------------------------------------

SimpleNN::Tensor makePredExtTensor(const CPelBuf& srcBuf);

SimpleNN::Tensor makeRefExtTensorLimited(const CodingUnit& cu);

void storeTensorToPelBufClipped(
    const SimpleNN::Tensor& tensor,
    PelBuf& dstBuf,
    const ClpRng& clpRng);















} // namespace BoundaryComp
