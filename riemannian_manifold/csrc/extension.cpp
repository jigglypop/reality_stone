#include <torch/extension.h>

// CPU 구현
torch::Tensor add_tensors(torch::Tensor a, torch::Tensor b)
{
    return a + b;
}

// 포인카레 볼 지수 사상 구현
torch::Tensor poincare_exp_map(torch::Tensor p, torch::Tensor v, double c)
{
    auto p_squared_norm = torch::sum(p * p, -1, true);
    auto lambda_p = 2.0 / (1.0 - c * p_squared_norm);

    auto v_norm = torch::norm(v, 2, -1, true);
    // 0으로 나누기 방지
    auto eps = 1e-8;
    v_norm = torch::clamp(v_norm, eps, INFINITY);

    auto scale = torch::tanh(sqrt(c) * lambda_p * v_norm / 2.0) / (sqrt(c) * v_norm);
    auto second_term = v * scale;

    auto numerator = (1.0 - c * p_squared_norm) * second_term;
    auto p_dot_second = torch::sum(p * second_term, -1, true);
    auto second_squared = torch::sum(second_term * second_term, -1, true);
    auto denominator = 1.0 - 2.0 * c * p_dot_second + pow(c, 2) * p_squared_norm * second_squared;

    return p + numerator / denominator;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("add_tensors", &add_tensors, "Add two tensors");
    m.def("poincare_exp_map", &poincare_exp_map, "Poincare ball exponential map");
}