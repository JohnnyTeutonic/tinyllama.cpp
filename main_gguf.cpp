#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>

#include "gguf_parser.h" // Include the parser header

// --- Main Function (Simplified) ---
int main(int argc, char **argv) {
    std::cout << "GGUF Loader - Starting...\n";

    // TODO: Add file path argument parsing (e.g., from argv)
    std::string filename = "data/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"; // Hardcoded for now

    try {
        // Load metadata and tensor info using the parser function
        GGUFData gguf_data = load_gguf_meta(filename);

        // Print a summary (optional)
        std::cout << "\n--- GGUF Summary ---\n";
        std::cout << "Version: " << gguf_data.header.version << "\n";
        std::cout << "Metadata Pairs: " << gguf_data.header.metadata_kv_count << "\n";
        std::cout << "Tensors: " << gguf_data.header.tensor_count << "\n";
        // Example: Print a specific metadata value if it exists
        if (gguf_data.metadata.count("general.architecture")) {
            try {
                 std::cout << "Architecture: " << std::get<std::string>(gguf_data.metadata["general.architecture"]) << "\n";
            } catch (const std::bad_variant_access& e) {
                 std::cerr << "Warning: Could not access general.architecture as string.\n";
            }
        }
         std::cout << "--------------------\n";

    } catch (const std::exception& e) {
        std::cerr << "\n*** Error loading GGUF file ***\n" << e.what() << "\n";
        return 1;
    }

    std::cout << "\nGGUF Loader - Finished successfully.\n";
    return 0;
} 