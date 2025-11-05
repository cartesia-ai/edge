#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <map>

/**
 * Lightweight BPE (Byte Pair Encoding) tokenizer for GPT-2 style tokenization
 * Compatible with Hugging Face GPT-2 tokenizers (vocab.json + merges.txt)
 */
class BPETokenizer {
public:
    BPETokenizer();
    ~BPETokenizer();
    
    /**
     * Load tokenizer from files
     * @param vocab_path Path to vocab.json file
     * @param merges_path Path to merges.txt file
     * @return true if loaded successfully, false otherwise
     */
    bool loadFromFiles(const std::string& vocab_path, const std::string& merges_path);
    
    /**
     * Tokenize text into token IDs
     * @param text Input text to tokenize
     * @return Vector of token IDs
     */
    std::vector<int32_t> tokenize(const std::string& text);
    
    /**
     * Check if tokenizer is loaded
     * @return true if loaded, false otherwise
     */
    bool isLoaded() const { return loaded_; }
    
    /**
     * Get vocabulary size
     * @return Number of tokens in vocabulary
     */
    size_t getVocabSize() const { return vocab_.size(); }

private:
    // Vocabulary: token string -> token ID
    std::unordered_map<std::string, int32_t> vocab_;
    
    // BPE merge rules: (token1, token2) -> priority (lower = higher priority)
    std::map<std::pair<std::string, std::string>, int> bpe_ranks_;
    
    // Byte encoder: maps bytes to Unicode characters (stored as strings for UTF-8)
    std::unordered_map<uint8_t, std::string> byte_encoder_;
    std::unordered_map<std::string, uint8_t> byte_decoder_;
    
    bool loaded_;
    
    // Helper functions
    void initByteEncoder();
    std::vector<std::string> preTokenize(const std::string& text);
    std::vector<std::string> byteEncode(const std::string& text);
    std::vector<std::string> applyBPE(const std::vector<std::string>& tokens);
    std::vector<std::pair<std::string, std::string>> getPairs(const std::vector<std::string>& tokens);
    std::vector<int32_t> mapToIDs(const std::vector<std::string>& tokens);
    
    // Simple JSON parser for vocab.json (just key-value pairs)
    bool parseVocabJSON(const std::string& json_content);
    
    // Parse merges.txt
    bool parseMergesFile(const std::string& merges_content);
};

#endif // TOKENIZER_H

