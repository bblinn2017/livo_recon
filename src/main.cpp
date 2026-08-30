#include "livo_recon/livo_recon_node.h"
#include "livo_recon/utils/algo/omp_utils.h"

int main(int argc, char** argv)
{
  // Must be the very first statement -- see pinOmpThreadsForDeterminism()'s
  // doc comment. Fixes a confirmed run-to-run non-determinism in VIO's
  // parallel HtH/Htz accumulation.
  livo_recon::pinOmpThreadsForDeterminism();

  ros::init(argc, argv, "livo_recon");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  livo_recon::LivoReconNode node(nh, pnh);
  // See LivoReconNode::offline_bag_path_'s doc comment (loadParameters())
  // -- reproducible-eval alternative to run(), opt-in via
  // common/offline_bag_path. Empty (the default) means run() unchanged.
  if (!node.offline_bag_path_.empty())
    node.runOffline(node.offline_bag_path_);
  else
    node.run();

  return 0;
}