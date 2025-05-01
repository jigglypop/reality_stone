#include "poincare_forward.h"

using namespace std;
using namespace torch;

namespace hyper_butterfly {
namespace geometry {
namespace poincare {

// forward.h에 인라인으로 구현되어 있음

} // namespace poincare
} // namespace geometry
} // namespace hyper_butterfly

// Python 바인딩을 위한 함수들
torch::Tensor log_map_cpu_export(torch::Tensor x, float c) {
    return hyper_butterfly::geometry::poincare::log_map_cpu(x, c);
}
torch::Tensor exp_map_cpu_export(torch::Tensor v, float c) {
    return hyper_butterfly::geometry::poincare::exp_map_cpu(v, c);
}