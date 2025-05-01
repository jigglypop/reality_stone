#include "forward_poincare.h"

use namespace std;
use namespace torch;

namespace hyper_butterfly
{
    namespace geometry
    {
        namespace poincare
        {

            // forward.h에 인라인으로 구현되어 있음

        } // namespace poincare
    } // namespace geometry
} // namespace hyper_butterfly

// Python 바인딩을 위한 함수들
Tensor log_map_cpu_export(Tensor x, float c)
{
    return hyper_butterfly::geometry::poincare::log_map_cpu(x, c);
}
Tensor exp_map_cpu_export(Tensor v, float c)
{
    return hyper_butterfly::geometry::poincare::exp_map_cpu(v, c);
}