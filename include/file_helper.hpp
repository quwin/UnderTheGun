#include <filesystem>
#include <stdexcept>
#include <string>

#ifdef _WIN32
    #include <windows.h>
#endif

std::filesystem::path executable_dir() {
#ifdef _WIN32
    char buffer[MAX_PATH];

    const DWORD length = GetModuleFileNameA(
        nullptr,
        buffer,
        static_cast<DWORD>(sizeof(buffer))
    );

    if (length == 0 || length == sizeof(buffer)) {
        throw std::runtime_error("Failed to resolve executable path.");
    }

    return std::filesystem::path(buffer).parent_path();
#else
    return std::filesystem::current_path();
#endif
}

std::filesystem::path evaluator_table_dir_next_to_exe() {
    return executable_dir() / "evaluator_tables";
}