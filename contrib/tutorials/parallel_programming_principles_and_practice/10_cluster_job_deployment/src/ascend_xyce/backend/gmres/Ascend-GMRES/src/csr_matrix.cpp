#include "csr_matrix.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ascend_gmres {

std::vector<MatrixSpec> default_matrix_specs() {
    return course_sparse::default_benchmark_specs();
}

CSRMatrix generate_uniform_matrix(const MatrixSpec& spec, std::uint32_t seed) {
    return make_gmres_ready_matrix(course_sparse::generate_uniform_matrix(spec, seed));
}

CSRMatrix generate_long_tail_matrix(const MatrixSpec& spec, std::uint32_t seed) {
    return make_gmres_ready_matrix(course_sparse::generate_long_tail_matrix(spec, seed));
}

CSRMatrix generate_block_matrix(const MatrixSpec& spec, std::int32_t block_size,
                                std::uint32_t seed) {
    return make_gmres_ready_matrix(
        course_sparse::generate_block_matrix(spec, block_size, seed));
}

CSRMatrix make_gmres_ready_matrix(const CSRMatrix& input) {
    return course_sparse::make_diagonally_dominant(input);
}

std::vector<float> generate_solution_vector(std::int32_t size, std::uint32_t seed) {
    return course_sparse::generate_signed_vector(size, seed);
}

double relative_error(const std::vector<float>& reference,
                      const std::vector<float>& candidate) {
    double numerator = 0.0;
    double denominator = 0.0;
    if (reference.size() != candidate.size()) return std::numeric_limits<double>::infinity();
    const std::size_t size = reference.size();
    for (std::size_t index = 0; index < size; ++index) {
        if (!std::isfinite(reference[index]) || !std::isfinite(candidate[index])) {
            return std::numeric_limits<double>::infinity();
        }
        const double difference = static_cast<double>(reference[index]) -
                                  static_cast<double>(candidate[index]);
        numerator += difference * difference;
        denominator += static_cast<double>(reference[index]) *
                       static_cast<double>(reference[index]);
    }
    return denominator == 0.0 ? std::sqrt(numerator)
                              : std::sqrt(numerator / denominator);
}

}  // namespace ascend_gmres
