#include "tokenizer.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <regex>
#include <iostream>
#include <iomanip>

BPETokenizer::BPETokenizer() : loaded_(false) {
    initByteEncoder();
}

BPETokenizer::~BPETokenizer() {
}

void BPETokenizer::initByteEncoder() {
    // GPT-2 style byte encoder: maps bytes to Unicode characters
    // This avoids control characters that BPE doesn't handle well
    
    // First, add printable ASCII characters directly
    std::vector<uint8_t> bs;
    for (int i = static_cast<int>('!'); i <= static_cast<int>('~'); ++i) {
        bs.push_back(static_cast<uint8_t>(i));
    }
    for (int i = 0xA1; i <= 0xAC; ++i) {
        bs.push_back(static_cast<uint8_t>(i));
    }
    for (int i = 0xAE; i <= 0xFF; ++i) {
        bs.push_back(static_cast<uint8_t>(i));
    }
    
    std::vector<uint32_t> cs_unicode;
    for (size_t i = 0; i < bs.size(); ++i) {
        cs_unicode.push_back(static_cast<uint32_t>(bs[i]));
    }
    
    // Add remaining bytes (0-255 not in bs) with Unicode mapping
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), static_cast<uint8_t>(b)) == bs.end()) {
            bs.push_back(static_cast<uint8_t>(b));
            cs_unicode.push_back(256 + n);
            n++;
        }
    }
    
    // Convert Unicode code points to UTF-8 strings
    // Create bidirectional mappings
    for (size_t i = 0; i < bs.size(); ++i) {
        uint32_t code_point = cs_unicode[i];
        std::string utf8_char;
        
        // Convert Unicode code point to UTF-8
        if (code_point <= 0x7F) {
            // Single byte
            utf8_char = std::string(1, static_cast<char>(code_point));
        } else if (code_point <= 0x7FF) {
            // Two bytes
            utf8_char += static_cast<char>(0xC0 | (code_point >> 6));
            utf8_char += static_cast<char>(0x80 | (code_point & 0x3F));
        } else if (code_point <= 0xFFFF) {
            // Three bytes
            utf8_char += static_cast<char>(0xE0 | (code_point >> 12));
            utf8_char += static_cast<char>(0x80 | ((code_point >> 6) & 0x3F));
            utf8_char += static_cast<char>(0x80 | (code_point & 0x3F));
        } else {
            // Four bytes
            utf8_char += static_cast<char>(0xF0 | (code_point >> 18));
            utf8_char += static_cast<char>(0x80 | ((code_point >> 12) & 0x3F));
            utf8_char += static_cast<char>(0x80 | ((code_point >> 6) & 0x3F));
            utf8_char += static_cast<char>(0x80 | (code_point & 0x3F));
        }
        
        byte_encoder_[bs[i]] = utf8_char;
        byte_decoder_[utf8_char] = bs[i];
    }
}

bool BPETokenizer::parseVocabJSON(const std::string& json_content) {
    // Simple JSON parser for vocab.json
    // Format: {"token1": id1, "token2": id2, ...}
    // This is a simplified parser that handles the basic structure
    
    vocab_.clear();
    
    // Find opening brace
    size_t start = json_content.find('{');
    if (start == std::string::npos) {
        std::cerr << "Error: vocab.json missing opening brace" << std::endl;
        return false;
    }
    
    size_t pos = start + 1;
    bool in_string = false;
    bool in_escape = false;
    std::string current_key;
    std::string current_value;
    bool reading_key = true;
    
    while (pos < json_content.length()) {
        char c = json_content[pos];
        
        if (in_escape) {
            // Handle escaped characters
            if (c == '\\' || c == '"' || c == 'n' || c == 't' || c == 'r') {
                if (reading_key) {
                    current_key += (c == 'n' ? '\n' : (c == 't' ? '\t' : (c == 'r' ? '\r' : c)));
                } else {
                    current_value += c;
                }
            } else {
                // Keep the escape character for other cases
                if (reading_key) {
                    current_key += '\\';
                    current_key += c;
                } else {
                    current_value += '\\';
                    current_value += c;
                }
            }
            in_escape = false;
            pos++;
            continue;
        }
        
        if (c == '\\') {
            in_escape = true;
            pos++;
            continue;
        }
        
        if (c == '"') {
            in_string = !in_string;
            if (!in_string && reading_key) {
                // Finished reading key, expect colon next
                reading_key = false;
            }
            pos++;
            continue;
        }
        
        // Skip whitespace outside strings
        if (!in_string && (c == ' ' || c == '\n' || c == '\t' || c == '\r')) {
            pos++;
            continue;
        }
        
        if (!in_string) {
            if (c == '}') {
                // End of JSON - save last entry
                if (!current_key.empty() && !current_value.empty()) {
                    try {
                        int32_t id = std::stoi(current_value);
                        vocab_[current_key] = id;
                    } catch (...) {
                        std::cerr << "Warning: Failed to parse token ID: " << current_value << std::endl;
                    }
                }
                break;
            } else if (c == ':') {
                // Separator between key and value
                reading_key = false;
                pos++;
                continue;
            } else if (c == ',') {
                // Save current entry and start new one
                if (!current_key.empty() && !current_value.empty()) {
                    try {
                        int32_t id = std::stoi(current_value);
                        vocab_[current_key] = id;
                    } catch (...) {
                        std::cerr << "Warning: Failed to parse token ID: " << current_value << std::endl;
                    }
                }
                current_key.clear();
                current_value.clear();
                reading_key = true;
                pos++;
                continue;
            }
        }
        
        if (in_string) {
            if (reading_key) {
                current_key += c;
            } else {
                current_value += c;
            }
        } else {
            // Reading value (number)
            if (!reading_key && (std::isdigit(c) || c == '-')) {
                current_value += c;
            }
        }
        
        pos++;
    }
    
    if (vocab_.empty()) {
        std::cerr << "Error: Failed to parse vocab.json or vocabulary is empty" << std::endl;
        return false;
    }
    
    return true;
}

bool BPETokenizer::parseMergesFile(const std::string& merges_content) {
    bpe_ranks_.clear();
    
    std::istringstream stream(merges_content);
    std::string line;
    int rank = 0;
    
    // Skip first line if it's a version header (e.g., "#version: 0.2")
    bool first_line = true;
    
    while (std::getline(stream, line)) {
        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        
        // Skip empty lines
        if (line.empty()) {
            continue;
        }
        
        // Skip version header line
        if (first_line && (line[0] == '#' || line.find("version") != std::string::npos)) {
            first_line = false;
            continue;
        }
        first_line = false;
        
        // Split by whitespace (should be exactly two tokens)
        std::istringstream line_stream(line);
        std::string token1, token2;
        line_stream >> token1 >> token2;
        
        if (token1.empty() || token2.empty()) {
            continue; // Skip malformed lines
        }
        
        // Store merge with rank (lower rank = higher priority)
        bpe_ranks_[std::make_pair(token1, token2)] = rank;
        rank++;
    }
    
    if (bpe_ranks_.empty()) {
        std::cerr << "Error: No BPE merges found in merges.txt" << std::endl;
        return false;
    }
    
    return true;
}

bool BPETokenizer::loadFromFiles(const std::string& vocab_path, const std::string& merges_path) {
    // Load vocab.json
    std::ifstream vocab_file(vocab_path);
    if (!vocab_file.is_open()) {
        std::cerr << "Error: Failed to open vocab file: " << vocab_path << std::endl;
        return false;
    }
    
    std::string vocab_content((std::istreambuf_iterator<char>(vocab_file)),
                             std::istreambuf_iterator<char>());
    vocab_file.close();
    
    if (!parseVocabJSON(vocab_content)) {
        std::cerr << "Error: Failed to parse vocab.json" << std::endl;
        return false;
    }
    
    // Build reverse vocabulary map for decoding
    id_to_token_.clear();
    for (const auto& pair : vocab_) {
        id_to_token_[pair.second] = pair.first;
    }
    
    // Load merges.txt
    std::ifstream merges_file(merges_path);
    if (!merges_file.is_open()) {
        std::cerr << "Error: Failed to open merges file: " << merges_path << std::endl;
        return false;
    }
    
    std::string merges_content((std::istreambuf_iterator<char>(merges_file)),
                              std::istreambuf_iterator<char>());
    merges_file.close();
    
    if (!parseMergesFile(merges_content)) {
        std::cerr << "Error: Failed to parse merges.txt" << std::endl;
        return false;
    }
    
    loaded_ = true;
    return true;
}

std::vector<std::string> BPETokenizer::preTokenize(const std::string& text) {
    // GPT-2 style pre-tokenization
    // Pattern: 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
    // Key: The " ?" means optional space that becomes part of the token if present
    // This matches GPT-2's behavior where " world" (space+word) is one token
    
    std::vector<std::string> tokens;
    
    if (text.empty()) {
        return tokens;
    }
    
    size_t i = 0;
    while (i < text.length()) {
        // Skip any whitespace we're not capturing
        while (i < text.length() && std::isspace(text[i]) && text[i] != ' ') {
            i++; // Skip tabs, newlines, etc. that aren't part of the pattern
        }
        
        if (i >= text.length()) break;
        
        std::string token;
        
        // Check for optional leading space (single space only)
        bool has_space = (i < text.length() && text[i] == ' ');
        if (has_space) {
            token = " ";
            i++;
        }
        
        if (i >= text.length()) {
            // Just a trailing space
            if (!token.empty()) {
                tokens.push_back(token);
            }
            break;
        }
        
        // Now match the pattern based on what follows
        char c = text[i];
        
        // Check for contractions first (before letters)
        if (c == '\'' && i + 1 < text.length()) {
            std::string contraction = "'";
            i++;
            if (i < text.length()) {
                char next = text[i];
                contraction += next;
                i++;
                
                // Check for longer contractions: 're, 've, 'll
                if (i < text.length() && (next == 'r' || next == 'v' || next == 'l')) {
                    contraction += text[i];
                    i++;
                    if (i < text.length() && text[i] == 'l' && next == 'l') {
                        contraction += text[i];
                        i++;
                    }
                }
            }
            tokens.push_back(token + contraction);
            continue;
        }
        
        // Match letters: ?\p{L}+ (optional space + one or more letters)
        if (std::isalpha(c)) {
            while (i < text.length() && std::isalpha(text[i])) {
                token += text[i];
                i++;
            }
            if (!token.empty()) {
                tokens.push_back(token);
            }
            continue;
        }
        
        // Match digits: ?\p{N}+ (optional space + one or more digits)
        if (std::isdigit(c)) {
            while (i < text.length() && std::isdigit(text[i])) {
                token += text[i];
                i++;
            }
            if (!token.empty()) {
                tokens.push_back(token);
            }
            continue;
        }
        
        // Match punctuation/non-alphanumeric: ?[^\s\p{L}\p{N}]+
        // (optional space + one or more non-space, non-letter, non-digit)
        if (!std::isspace(c)) {
            while (i < text.length() && !std::isspace(text[i]) && 
                   !std::isalnum(text[i])) {
                token += text[i];
                i++;
            }
            if (!token.empty()) {
                tokens.push_back(token);
            }
            continue;
        }
        
        // Match whitespace sequences: \s+(?!\S)|\s+
        // (whitespace not followed by non-whitespace, or any whitespace)
        if (std::isspace(c)) {
            std::string whitespace;
            while (i < text.length() && std::isspace(text[i])) {
                whitespace += text[i];
                i++;
            }
            // Only add if it's significant (not part of a word prefix)
            // For GPT-2, we typically don't add standalone whitespace sequences
            // unless they're trailing
            if (i >= text.length() && !whitespace.empty()) {
                // Trailing whitespace - add as separate token
                tokens.push_back(whitespace);
            }
            continue;
        }
        
        // Fallback: consume one character
        token += c;
        i++;
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }
    
    return tokens;
}

std::vector<std::string> BPETokenizer::byteEncode(const std::string& text) {
    // Convert text to bytes, then map bytes to Unicode characters
    std::vector<std::string> encoded;
    
    for (unsigned char byte : text) {
        auto it = byte_encoder_.find(byte);
        if (it != byte_encoder_.end()) {
            encoded.push_back(it->second);
        } else {
            // Fallback: use the byte as-is (shouldn't happen)
            encoded.push_back(std::string(1, static_cast<char>(byte)));
        }
    }
    
    return encoded;
}

std::vector<std::pair<std::string, std::string>> BPETokenizer::getPairs(const std::vector<std::string>& tokens) {
    std::vector<std::pair<std::string, std::string>> pairs;
    
    if (tokens.size() < 2) {
        return pairs;
    }
    
    for (size_t i = 0; i < tokens.size() - 1; ++i) {
        pairs.push_back(std::make_pair(tokens[i], tokens[i + 1]));
    }
    
    return pairs;
}

std::vector<std::string> BPETokenizer::applyBPE(const std::vector<std::string>& tokens) {
    if (tokens.empty()) {
        return tokens;
    }
    
    std::vector<std::string> word = tokens;
    
    // Apply BPE merges iteratively
    while (true) {
        auto pairs = getPairs(word);
        
        if (pairs.empty()) {
            break;
        }
        
        // Find the pair with the lowest rank (highest priority)
        int best_rank = -1;
        std::pair<std::string, std::string> best_pair;
        bool found = false;
        
        for (const auto& pair : pairs) {
            auto it = bpe_ranks_.find(pair);
            if (it != bpe_ranks_.end()) {
                int rank = it->second;
                if (best_rank == -1 || rank < best_rank) {
                    best_rank = rank;
                    best_pair = pair;
                    found = true;
                }
            }
        }
        
        if (!found) {
            break; // No more merges possible
        }
        
        // Apply the merge
        std::vector<std::string> new_word;
        size_t i = 0;
        
        while (i < word.size()) {
            if (i < word.size() - 1 && 
                word[i] == best_pair.first && 
                word[i + 1] == best_pair.second) {
                new_word.push_back(best_pair.first + best_pair.second);
                i += 2;
            } else {
                new_word.push_back(word[i]);
                i++;
            }
        }
        
        word = new_word;
    }
    
    return word;
}

std::vector<int32_t> BPETokenizer::mapToIDs(const std::vector<std::string>& tokens) {
    std::vector<int32_t> ids;
    
    for (const auto& token : tokens) {
        auto it = vocab_.find(token);
        if (it != vocab_.end()) {
            ids.push_back(it->second);
        } else {
            // Unknown token - this should rarely happen if tokenization is correct
            // Try to find UNK token or use a default
            // For GPT-2 style, we might use the last token or a special token
            auto unk_it = vocab_.find("<|endoftext|>");
            if (unk_it != vocab_.end()) {
                ids.push_back(unk_it->second);
            } else if (!vocab_.empty()) {
                // Fallback: use first token ID
                ids.push_back(vocab_.begin()->second);
            }
        }
    }
    
    return ids;
}

std::vector<int32_t> BPETokenizer::tokenize(const std::string& text) {
    if (!loaded_) {
        std::cerr << "Error: Tokenizer not loaded. Call loadFromFiles() first." << std::endl;
        return std::vector<int32_t>();
    }
    
    // Step 1: Pre-tokenize (split into words/punctuation)
    std::vector<std::string> pre_tokens = preTokenize(text);
    
    // Step 2: Apply BPE to each pre-token
    std::vector<std::string> bpe_tokens;
    for (const auto& pre_token : pre_tokens) {
        // Byte encode the pre-token
        std::vector<std::string> byte_tokens = byteEncode(pre_token);
        
        // Apply BPE
        std::vector<std::string> merged = applyBPE(byte_tokens);
        
        // Add to result
        bpe_tokens.insert(bpe_tokens.end(), merged.begin(), merged.end());
    }
    
    // Step 3: Map tokens to IDs
    return mapToIDs(bpe_tokens);
}

std::string BPETokenizer::decode(const std::vector<int32_t>& token_ids) {
    if (!loaded_) {
        std::cerr << "Error: Tokenizer not loaded. Call loadFromFiles() first." << std::endl;
        return "";
    }
    
    // Step 1: Map token IDs back to token strings
    std::vector<std::string> tokens;
    for (int32_t id : token_ids) {
        auto it = id_to_token_.find(id);
        if (it != id_to_token_.end()) {
            tokens.push_back(it->second);
        } else {
            // Unknown token ID - skip or use placeholder
            std::cerr << "Warning: Unknown token ID " << id << " in decode" << std::endl;
        }
    }
    
    // Step 2: Concatenate all token strings
    std::string encoded_text;
    for (const auto& token : tokens) {
        encoded_text += token;
    }
    
    // Step 3: Decode bytes back to UTF-8 text
    // The tokens are in the byte-encoded format, so we need to decode them
    // Each character in encoded_text represents a byte via the byte encoder mapping
    std::string decoded_text;
    size_t i = 0;
    while (i < encoded_text.length()) {
        // Try to match multi-byte UTF-8 sequences first (longest match)
        bool matched = false;
        
        // Try 4-byte, 3-byte, 2-byte, then 1-byte UTF-8 sequences
        for (int len = 4; len >= 1 && !matched; --len) {
            if (i + len <= encoded_text.length()) {
                std::string utf8_seq = encoded_text.substr(i, len);
                auto it = byte_decoder_.find(utf8_seq);
                if (it != byte_decoder_.end()) {
                    decoded_text += static_cast<char>(it->second);
                    i += len;
                    matched = true;
                }
            }
        }
        
        if (!matched) {
            // Fallback: treat as literal character
            decoded_text += encoded_text[i];
            i++;
        }
    }
    
    return decoded_text;
}

