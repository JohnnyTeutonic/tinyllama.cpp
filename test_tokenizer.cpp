#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <sstream> // For stringstream logging
#include <regex>     // <<< ADDED for Strategy 5
#include <boost/regex.hpp> // <<< ADDED for Boost.Regex (Strategy 7)
#include <algorithm> // <<< ADDED for std::transform (lowercase)

#include "tokenizer.h"
#include "logger.h"

// Modified to test and log multiple strategies
void run_single_test(const Tokenizer& tokenizer, const std::string& test_text, int test_num) {
    Logger::info("--- Test Case " + std::to_string(test_num) + " ---");
    Logger::info("Input Text: '" + test_text + "'");

    // Helper lambda to log results for a strategy
    auto log_results = [&](const std::string& strategy_name, const std::vector<std::string>& tokens) {
        Logger::info(strategy_name);
        std::stringstream ss_tokens;
        ss_tokens << "  Tokens (str): [";
        for (size_t i = 0; i < tokens.size(); ++i) {
            ss_tokens << "'" << tokens[i] << "'" << (i + 1 < tokens.size() ? ", " : "");
        }
        ss_tokens << "]";
        Logger::info(ss_tokens.str());

        try {
            std::vector<int> ids = tokenizer.tokens_to_ids(tokens);
            std::stringstream ss_ids;
            ss_ids << "  Tokens (IDs): [";
            for (size_t i = 0; i < ids.size(); ++i) {
                ss_ids << ids[i] << (i + 1 < ids.size() ? ", " : "");
            }
            ss_ids << "]";
            Logger::info(ss_ids.str());

            std::vector<std::string> decoded_tokens = tokenizer.ids_to_tokens(ids);
            std::stringstream ss_decoded;
            ss_decoded << "  Decoded Tokens: [";
            for (size_t i = 0; i < decoded_tokens.size(); ++i) {
                ss_decoded << "'" << decoded_tokens[i] << "'" << (i + 1 < decoded_tokens.size() ? ", " : "");
            }
            ss_decoded << "]";
            Logger::info(ss_decoded.str());
        } catch (const std::exception& e) {
            Logger::error("  Exception during ID conversion/decoding: " + std::string(e.what()));
        }
    };

    // --- Strategy 1: Full BPE (Current Implementation) ---
    try {
        std::vector<std::string> tokens = tokenizer.bpe_tokenize(test_text); // Call BPE directly
        log_results("Strategy 1: Full BPE (Current)", tokens);
    } catch (const std::exception& e) {
        Logger::error("  Exception during BPE test: " + std::string(e.what()));
    }

    // --- Strategy 2: Simple Regex Split Only (Tokenizer Class Method) ---
    try {
        std::vector<std::string> tokens = tokenizer.regex_tokenize(test_text); // Call Tokenizer::regex_tokenize
        log_results("Strategy 2: Simple Regex Split Only (Tokenizer Method)", tokens);
    } catch (const std::exception& e) {
        Logger::error("  Exception during Simple Regex test: " + std::string(e.what()));
    }

    // --- Strategy 4: Lowercase Input + Full BPE ---
    try {
        std::string lower_text = test_text;
        std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(),
                       [](unsigned char c){ return std::tolower(c); });
        std::vector<std::string> tokens = tokenizer.bpe_tokenize(lower_text);
        log_results("Strategy 4: Lowercase + Full BPE", tokens);
    } catch (const std::exception& e) {
        Logger::error("  Exception during Lowercase+BPE test: " + std::string(e.what()));
    }
    
    // --- Strategy 5: Llama.cpp Regex PreTok + Full BPE ---
    try {
        std::vector<std::string> final_tokens;
        // Regex from llama.cpp (approximated)
        // Handles: 's, 't, 're, 've, 'm, 'll, 'd, optional_space+letters, optional_space+digits, optional_space+other_non_space, space(s)_not_followed_by_non_space, other_space(s)
        std::regex llama_regex(R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)");
        
        std::string text_to_search = test_text;
        auto words_begin = std::sregex_iterator(text_to_search.begin(), text_to_search.end(), llama_regex);
        auto words_end = std::sregex_iterator();

        for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
            std::smatch match = *i;
            std::string unit = match.str(0);
            if (!unit.empty()) {
                // Tokenize each unit found by the regex using the standard BPE
                std::vector<std::string> unit_tokens = tokenizer.bpe_tokenize(unit);
                final_tokens.insert(final_tokens.end(), unit_tokens.begin(), unit_tokens.end());
            }
        }
         // Note: This simplified regex loop might miss trailing characters if they don't match.
         // A more robust implementation might check match.suffix().str() like in tokenizer.cpp

        log_results("Strategy 5: Llama Regex PreTok + Full BPE", final_tokens);

    } catch (const std::exception& e) {
        Logger::error("  Exception during Llama Regex PreTok+BPE test: " + std::string(e.what()));
    }

    // --- Strategy 6: Lowercase Preprocessing + Full BPE (Simulating SP Normalization Effect) ---
    try {
        std::string lower_text = test_text;
        std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(),
                       [](unsigned char c){ return std::tolower(c); });
        // Use the same BPE function as Strategy 1, but on lowercase input
        std::vector<std::string> tokens = tokenizer.bpe_tokenize(lower_text); 
        log_results("Strategy 6: Lowercase Preprocessing + BPE", tokens);
    } catch (const std::exception& e) {
        Logger::error("  Exception during Lowercase Preprocessing + BPE test: " + std::string(e.what()));
    }

    // --- Strategy 7: Llama.cpp Regex PreTok + Our BPE ---
    // Moved definition outside try block for catch scope
    // MODIFIED: Using ASCII POSIX classes [:alpha:], [:digit:] instead of Unicode \p{L}, \p{N} for compatibility testing
    const std::string LlamaRegexStr = R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S))";
    try {
        std::vector<std::string> final_tokens;
        // Use the specific regex identified for Llama 2 / GPT-2 style BPE
        // Using Boost.Regex instead of std::regex
        boost::regex llama_regex(LlamaRegexStr);

        std::string text_to_search = test_text;
        // Using Boost iterators and match objects
        boost::sregex_iterator words_begin(text_to_search.begin(), text_to_search.end(), llama_regex);
        boost::sregex_iterator words_end;
        long last_pos = 0;

        for (boost::sregex_iterator i = words_begin; i != words_end; ++i) {
            boost::smatch match = *i;
            long current_pos = match.position();
            std::string unit = match.str(0);

            // Check for unmatched text between the previous match and this one
            if (current_pos > last_pos) {
                std::string unmatched_unit = text_to_search.substr(last_pos, current_pos - last_pos);
                Logger::debug("[STRAT 7 DEBUG] Unmatched pre-regex segment: '" + unmatched_unit + "' - Applying BPE.");
                std::vector<std::string> unit_tokens = tokenizer.bpe_tokenize(unmatched_unit);
                final_tokens.insert(final_tokens.end(), unit_tokens.begin(), unit_tokens.end());
            }

            // Process the matched unit
            if (!unit.empty()) {
                 Logger::debug("[STRAT 7 DEBUG] Matched regex segment: '" + unit + "' - Applying BPE.");
                 std::vector<std::string> unit_tokens = tokenizer.bpe_tokenize(unit);
                 final_tokens.insert(final_tokens.end(), unit_tokens.begin(), unit_tokens.end());
            }
            last_pos = current_pos + match.length();
        }

        // Handle any trailing text that didn't match the regex at all
        if (last_pos < text_to_search.length()) {
            std::string trailing_unit = text_to_search.substr(last_pos);
            Logger::debug("[STRAT 7 DEBUG] Trailing unmatched segment: '" + trailing_unit + "' - Applying BPE.");
            std::vector<std::string> unit_tokens = tokenizer.bpe_tokenize(trailing_unit);
            final_tokens.insert(final_tokens.end(), unit_tokens.begin(), unit_tokens.end());
        }

        log_results("Strategy 7: Llama.cpp Regex PreTok + Our BPE", final_tokens);

    } catch (const boost::regex_error& e) { // Catch Boost regex specific errors
         Logger::error("  Boost.Regex error during Llama Regex PreTok+BPE test: " + std::string(e.what()) + " - Pattern: " + LlamaRegexStr);
    } catch (const std::exception& e) {
        Logger::error("  Exception during Llama Regex PreTok+BPE test: " + std::string(e.what()));
    }

    Logger::info("--------------------\n");
}

int main() {
    // Initialize logger (adjust path and level as needed)
    // Logger::init("tokenizer_test.log", Logger::Level::DEBUG); 
    Logger::info("===== Starting Tokenizer Test Suite =====");

    std::vector<std::string> test_queries = {
        // 1. Simple sentence
        "This is a test.",
        // 2. Punctuation/numbers
        "Score: 95%, Rank #1!",
        // 3. Mixed case (Names)
        "USA President Joe Biden",
        // 4. Known problem word (geography)
        "Studying geography is fun.",
        // 5. Known problem word (please)
        "Please tokenize correctly.",
        // 6. Leading/trailing/internal spaces
        "  extra spaces  here ",
        // 7. Word expected with leading space prefix
        " France capital.", 
        // 8. Special characters / Newline
        "€§± ~ \n newline test",
        // 9. Instruction manual test
        "The Instruction manual is an instruction manual for performing multiple tasks."
    };

    try {
        // Load tokenizer (using the same path as main_gguf.cpp)
        // Ensure this path is correct relative to where the test executable will run
        std::string tokenizer_path = "data/tiny-llama-pure/tokenizer.json"; 
        Logger::info("Loading tokenizer from: " + tokenizer_path);
        Tokenizer tokenizer(tokenizer_path, tokenizer_path); 
        Logger::info("Tokenizer loaded successfully.");

        // Run tests
        for (int i = 0; i < test_queries.size(); ++i) {
            run_single_test(tokenizer, test_queries[i], i + 1);
        }

    } catch (const std::exception& e) {
        Logger::error("Failed to initialize tokenizer or run tests: " + std::string(e.what()));
        return 1;
    }

    Logger::info("===== Tokenizer Test Suite Finished =====");
    return 0;
} 