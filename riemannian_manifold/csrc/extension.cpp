#include <torch/extension.h>
#include <cmath>

// CPU 구현
torch::Tensor add_tensors(torch::Tensor a, torch::Tensor b)
{
    return a + b;
}

// 포인카레 볼 지수 사상 구현
torch::Tensor poincare_exp_map(torch::Tensor x, torch::Tensor v, double c)
{
    auto x_norm_squared = torch::sum(x * x, -1, true);
    auto lambda_x = 2.0 / (1.0 - c * x_norm_squared);
    
    auto v_norm = torch::norm(v, 2, -1, true);
    v_norm = torch::clamp(v_norm, 1e-8, INFINITY);
    
    auto c_tensor = torch::tensor(c, v.options());
    auto sqrt_c = torch::sqrt(c_tensor);
    
    auto second_term = torch::tanh(sqrt_c * lambda_x * v_norm / 2.0) / (sqrt_c * v_norm) * v;
    
    auto numerator = (1.0 - c * x_norm_squared) * second_term;
    auto denominator = 1.0 - 2.0 * c * torch::sum(x * second_term, -1, true) + c * c * x_norm_squared * torch::sum(second_term * second_term, -1, true);
    
    return x + numerator / denominator;
}

torch::Tensor poincare_log_map(torch::Tensor x, torch::Tensor y, double c) {
    auto x_norm_squared = torch::sum(x * x, -1, true);
    auto lambda_x = 2.0 / (1.0 - c * x_norm_squared);
    
    auto diff = y - x;
    auto diff_norm_squared = torch::sum(diff * diff, -1, true);
    auto y_norm_squared = torch::sum(y * y, -1, true);
    
    auto c_tensor = torch::tensor(c, x.options());
    auto sqrt_c = torch::sqrt(c_tensor);
    
    auto transport_vector = (-x * y_norm_squared + y * (1.0 + c * x_norm_squared) - 2 * c * torch::sum(x * y, -1, true) * x) / (1.0 - c * x_norm_squared);
    auto transport_norm = torch::norm(transport_vector, 2, -1, true);
    
    auto numerator = 2 * sqrt_c * torch::atanh(sqrt_c * transport_norm);
    auto denominator = sqrt_c * lambda_x * transport_norm;
    
    return numerator / denominator * transport_vector;
}

torch::Tensor poincare_distance(torch::Tensor x, torch::Tensor y, double c) {
    auto norm_x = torch::sum(x * x, -1, true);
    auto norm_y = torch::sum(y * y, -1, true);
    auto xy_inner = torch::sum(x * y, -1, true);
    
    auto c_tensor = torch::tensor(c, x.options());
    auto sqrt_c = torch::sqrt(c_tensor);
    
    auto numerator = 2 * sqrt_c * torch::norm(x - y, 2, -1, true);
    auto denominator = torch::sqrt((1 - c * norm_x) * (1 - c * norm_y)) + sqrt_c * xy_inner;
    
    return 2 * torch::atanh(numerator / denominator) / sqrt_c;
}

torch::Tensor butterfly_factor(torch::Tensor input, torch::Tensor params, int layer) {
    int n = input.size(0);
    int block_size = 1 << layer;
    int num_blocks = n / block_size;
    
    auto result = input.clone();
    
    // 인덱스 접근이 params 텐서 크기를 넘지 않도록 제한
    int param_idx = 0;
    int total_params = params.size(0);
    
    for (int b = 0; b < num_blocks; b++) {
        for (int i = 0; i < block_size; i += 2) {
            if (b * block_size + i + 1 >= n) break; // 배열 범위 체크
            if (param_idx + 1 >= total_params) break; // 파라미터 배열 범위 체크
            
            int idx = b * block_size + i;
            double a = params[param_idx].item<double>();
            double b_val = params[param_idx + 1].item<double>();
            param_idx += 2;
            
            auto temp1 = a * input[idx] + b_val * input[idx + 1];
            auto temp2 = -b_val * input[idx] + a * input[idx + 1];
            
            result[idx] = temp1;
            result[idx + 1] = temp2;
        }
    }
    
    return result;
}

torch::Tensor hyper_butterfly_forward(torch::Tensor x, torch::Tensor params, double c, int L) {
    auto zeros = torch::zeros_like(x);
    auto u = poincare_log_map(zeros, x, c);
    
    // 각 레이어당 필요한 파라미터 개수 계산
    int n = x.size(0);
    for (int l = 0; l < L; l++) {
        int block_size = 1 << l;
        int num_blocks = n / block_size;
        int params_needed = num_blocks * 2;
        
        // 실제 사용 가능한 파라미터 개수 체크
        int actual_params = std::min(static_cast<int>(params.size(0)), params_needed);
        if (actual_params < 2) break; // 최소 한 블록 이상 필요
        
        auto layer_params = params.slice(0, 0, actual_params);
        u = butterfly_factor(u, layer_params, l);
        
        // 다음 레이어로 파라미터 슬라이싱 준비
        if (actual_params < params.size(0)) {
            params = params.slice(0, actual_params);
        } else {
            break; // 남은 파라미터 없음
        }
    }
    
    return poincare_exp_map(zeros, u, c);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("add_tensors", &add_tensors, "Add two tensors");
    m.def("poincare_exp_map", &poincare_exp_map, "Poincare ball exponential map");
    m.def("poincare_log_map", &poincare_log_map, "Poincare ball logarithmic map");
    m.def("poincare_distance", &poincare_distance, "Poincare ball distance");
    m.def("butterfly_factor", &butterfly_factor, "Butterfly factor transform");
    m.def("hyper_butterfly_forward", &hyper_butterfly_forward, "Hyper-Butterfly forward pass");
}