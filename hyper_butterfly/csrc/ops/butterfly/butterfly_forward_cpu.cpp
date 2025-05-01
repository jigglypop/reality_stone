#include "butterfly_forward.h"

using namespace std;
using namespace torch;

namespace hyper_butterfly {
namespace ops {
namespace butterfly {

// 헤더에 인라인으로 이미 구현되어 있음

} // namespace butterfly
} // namespace transforms
} // namespace core

// Python 바인딩을 위한 함수
torch::Tensor butterfly_forward_cpu_export(
    torch::Tensor input,
    torch::Tensor params,
    int layer_idx,
    int batch_size,
    int dim) {
    return hyper_butterfly::ops::butterfly::butterfly_forward_cpu(
        input, params, layer_idx, batch_size, dim);
}