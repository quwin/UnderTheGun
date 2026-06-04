#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cfr_gpu.hpp"

namespace {

    constexpr std::size_t kBinariesByIdSize = 52;
    constexpr std::size_t kSuitBitByIdSize = 52;
    constexpr std::size_t kFlushSize = 8192;
    constexpr std::size_t kNoFlush7Size = 49205;
    constexpr std::size_t kSuitsSize = 4609;
    constexpr std::size_t kDpSize = 700;

template <typename T>
std::vector<T> load_integer_table(
    const std::string& path,
    std::size_t expected_count
) {
    std::ifstream input(path);

    if (!input) {
        throw std::runtime_error("Failed to open evaluator table: " + path);
    }

    std::vector<T> values;
    values.reserve(expected_count);

    std::string token;
    while (input >> token) {
        const unsigned long long parsed = std::stoull(token, nullptr, 0); // base 0 accepts decimal and 0x...

        values.push_back(static_cast<T>(parsed));
    }

    if (values.size() != expected_count) {
        throw std::runtime_error(
            "Evaluator table has wrong size: " + path +
            ", got " + std::to_string(values.size()) +
            ", expected " + std::to_string(expected_count)
        );
    }

    return values;
}

poker::HostHandEvaluatorTables load_hand_evaluator_tables(
    const std::string& data_dir
) {
        poker::HostHandEvaluatorTables tables;
        tables.binaries_by_id =
            load_integer_table<short>(
                data_dir + "/binaries_by_id_data.txt",
                kBinariesByIdSize
            );
        tables.suitbit_by_id =
            load_integer_table<short>(
                data_dir + "/suitbit_by_id_data.txt",
                kSuitBitByIdSize
            );
        tables.flush =
            load_integer_table<short>(
                data_dir + "/flush_data.txt",
                kFlushSize
            );
        tables.noflush7 =
            load_integer_table<short>(
                data_dir + "/noflush7_data.txt",
                kNoFlush7Size
            );
        tables.suits =
            load_integer_table<unsigned char>(
                data_dir + "/suits_data.txt",
                kSuitsSize
            );
        tables.dp =
            load_integer_table<int>(
                data_dir + "/dp_data.txt",
                kDpSize
            );
        return tables;
    }

} // namespace