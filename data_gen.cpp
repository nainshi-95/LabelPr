#if DATA_GEN
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#endif







#if DATA_GEN
namespace
{
namespace fs = std::filesystem;

static bool dataGenIsEligibleInterCu(const CodingUnit &cu)
{
  return cu.predMode == MODE_INTER && cu.Y().valid() && cu.mergeFlag && cu.mergeType == MergeType::DEFAULT_N &&
    !cu.affine && !cu.geoFlag && !cu.ciipFlag && !cu.gpmIntraFlag && !CU::isIBC(cu) && !cu.mmvdMergeFlag &&
    !cu.mmvdSkip && (cu.interDir == 1 || cu.interDir == 2) && cu.lx() >= 2 && cu.ly() >= 2;
}

static bool dataGenPelToU16(const Pel pel, uint16_t &dst)
{
  if (pel < 0 || pel > std::numeric_limits<uint16_t>::max())
  {
    return false;
  }

  dst = static_cast<uint16_t>(pel);
  return true;
}

static bool dataGenAppendBinaryU16(const fs::path &filePath, const std::vector<uint16_t> &samples)
{
  FILE *fp = std::fopen(filePath.string().c_str(), "ab");
  if (fp == nullptr)
  {
    return false;
  }

  std::vector<uint8_t> bytes(samples.size() * 2);
  for (size_t idx = 0; idx < samples.size(); idx++)
  {
    const uint16_t value   = samples[idx];
    bytes[idx * 2 + 0]     = static_cast<uint8_t>(value & 0xff);
    bytes[idx * 2 + 1]     = static_cast<uint8_t>((value >> 8) & 0xff);
  }

  const bool ok = std::fwrite(bytes.data(), 1, bytes.size(), fp) == bytes.size();
  std::fclose(fp);
  return ok;
}

static bool dataGenAppendMetadata(const fs::path &filePath, const CodingUnit &cu, const RefPicList refList)
{
  FILE *fp = std::fopen(filePath.string().c_str(), "a");
  if (fp == nullptr)
  {
    return false;
  }

  const int result = std::fprintf(fp, "%d %d %d %d %d %d %d %d %d %d\n", cu.slice->m_poc, cu.lx(), cu.ly(),
                                  cu.lwidth(), cu.lheight(), cu.interDir, static_cast<int>(refList), cu.refIdx[refList],
                                  cu.mv[refList].getHor(), cu.mv[refList].getVer());
  std::fclose(fp);
  return result > 0;
}

static bool dataGenBuildPredictorPatch(const CodingUnit &cu, std::vector<uint16_t> &patch)
{
  const int patchW = cu.lwidth() + 2;
  const int patchH = cu.lheight() + 2;
  const int x0     = cu.lx();
  const int y0     = cu.ly();

  const CPelBuf predY = cu.cs->getPredBuf(cu).Y();
  const CPelBuf recoY = cu.cs->picture->getRecoBuf(COMP_Y);

  patch.resize(size_t(patchW) * size_t(patchH));

  for (int y = 0; y < 2; y++)
  {
    for (int x = 0; x < 2; x++)
    {
      if (!dataGenPelToU16(recoY.at(x0 - 2 + x, y0 - 2 + y), patch[size_t(y) * patchW + x]))
      {
        return false;
      }
    }
    for (int x = 0; x < cu.lwidth(); x++)
    {
      if (!dataGenPelToU16(recoY.at(x0 + x, y0 - 2 + y), patch[size_t(y) * patchW + 2 + x]))
      {
        return false;
      }
    }
  }

  for (int y = 0; y < cu.lheight(); y++)
  {
    for (int x = 0; x < 2; x++)
    {
      if (!dataGenPelToU16(recoY.at(x0 - 2 + x, y0 + y), patch[size_t(y + 2) * patchW + x]))
      {
        return false;
      }
    }
    for (int x = 0; x < cu.lwidth(); x++)
    {
      if (!dataGenPelToU16(predY.at(x, y), patch[size_t(y + 2) * patchW + 2 + x]))
      {
        return false;
      }
    }
  }

  return true;
}
}   // namespace
#endif



















#if DATA_GEN
void DecCu::setFileName(std::string binFileName, std::string dataRoot)
{
  m_binFileName = binFileName;
  m_dataRoot = dataRoot;
}
#endif








#if DATA_GEN
  if (dataGenIsEligibleInterCu(cu))
  {
    const RefPicList refList = cu.interDir == 1 ? RPL0 : RPL1;
    std::vector<uint16_t> predictorPatch;
    std::vector<uint16_t> referencePatch;
    const int             patchW = cu.lwidth() + 2;
    const int             patchH = cu.lheight() + 2;

    if (dataGenBuildPredictorPatch(cu, predictorPatch))
    {
      std::vector<Pel> referencePatchPel(size_t(patchW) * size_t(patchH), 0);
      PelBuf           referencePatchBuf(referencePatchPel.data(), patchW, patchW, patchH);
      const Area       referenceArea(Position(cu.lx() - 2, cu.ly() - 2), Size(patchW, patchH));

      if (m_pcInterPred->dataGenGetLumaUniMcPatch(cu, referenceArea, referencePatchBuf))
      {
        referencePatch.resize(referencePatchPel.size());
        bool referenceOk = true;
        for (size_t idx = 0; idx < referencePatchPel.size(); idx++)
        {
          referenceOk &= dataGenPelToU16(referencePatchPel[idx], referencePatch[idx]);
        }

        if (referenceOk)
        {
          const fs::path seqRoot =
            fs::path(m_dataRoot) / (fs::path(m_binFileName).stem().empty() ? "unknown_bitstream"
                                                                          : fs::path(m_binFileName).stem().string()) /
            "inter_merge_uni_default";
          const fs::path predictorDir = seqRoot / "predictor";
          const fs::path referenceDir = seqRoot / "reference";
          const fs::path metadataDir  = seqRoot / "metadata";

          // File format:
          // - predictor/<w>x<h>.bin: appended samples, each sample is a (h+2) x (w+2) luma patch in row-major order.
          // - reference/<w>x<h>.bin: appended samples aligned 1:1 with predictor, same uint16 little-endian layout.
          // - metadata/<w>x<h>.txt: one line per sample:
          //   poc x y w h interDir refList refIdx mvx mvy
          if (fs::create_directories(predictorDir) || fs::exists(predictorDir))
          {
            if (fs::create_directories(referenceDir) || fs::exists(referenceDir))
            {
              if (fs::create_directories(metadataDir) || fs::exists(metadataDir))
              {
                const std::string baseName =
                  std::to_string(cu.lwidth()) + "x" + std::to_string(cu.lheight());
                const fs::path predictorPath = predictorDir / (baseName + ".bin");
                const fs::path referencePath = referenceDir / (baseName + ".bin");
                const fs::path metadataPath  = metadataDir / (baseName + ".txt");

                if (dataGenAppendBinaryU16(predictorPath, predictorPatch) &&
                    dataGenAppendBinaryU16(referencePath, referencePatch))
                {
                  (void)dataGenAppendMetadata(metadataPath, cu, refList);
                }
              }
            }
          }
        }
      }
    }
  }
#endif










#if DATA_GEN
  bool dataGenGetLumaUniMcPatch(const CodingUnit &cu, const Area &patchArea, PelBuf dstBuf);
#endif





#if DATA_GEN
#include <memory>
#endif






#if DATA_GEN
bool InterPrediction::dataGenGetLumaUniMcPatch(const CodingUnit &cu, const Area &patchArea, PelBuf dstBuf)
{
  if (patchArea.size() != Size(dstBuf.width, dstBuf.height) || (cu.interDir != 1 && cu.interDir != 2))
  {
    return false;
  }

  const RefPicList refList = cu.interDir == 1 ? RPL0 : RPL1;
  const int        refIdx  = cu.refIdx[refList];
  if (refIdx < 0)
  {
    return false;
  }

  const Picture *refPic = cu.slice->getRefPic(refList, refIdx);
  if (refPic == nullptr || refPic->isRefScaled(cu.cs->pps) || refPic->isWrapAroundEnabled(cu.cs->pps))
  {
    return false;
  }

  std::unique_ptr<bool[]> mcMaskStore(new bool[patchArea.width * patchArea.height]());
  if (isMvOOB(cu.mv[refList], patchArea.pos(), patchArea.size(), cu.slice->m_sps, cu.cs->pps, mcMaskStore.get(),
              nullptr, true))
  {
    return false;
  }

  CodingUnit patchCu(cu.chromaFormat, patchArea);
  patchCu = static_cast<const InterPredictionData &>(cu);
  patchCu.cs        = cu.cs;
  patchCu.slice     = cu.slice;
  patchCu.chType    = ChannelType::LUMA;
  patchCu.predMode  = cu.predMode;
  patchCu.mergeType = MergeType::DEFAULT_N;
  patchCu.obmcFlag  = false;
  patchCu.licFlag   = false;
  patchCu.ciipFlag  = false;
  patchCu.geoFlag   = false;
  patchCu.affine    = false;

  PelUnitBuf dstUnit(cu.chromaFormat, dstBuf);
  xPredInterBlk(COMP_Y, patchCu, refPic, patchCu.mv[refList], dstUnit, false, cu.slice->clpRng(COMP_Y), false, false,
                refList);
  return true;
}
#endif











