#pragma once

#include <string>
#include <vector>
#include <optional>
#include <memory>

namespace brain19 {

// OllamaConfig: Configuration for Ollama connection
struct OllamaConfig {
    std::string host = "http://localhost:11434";
    std::string model = "llama3.2:3b";  // Default: fast & small
    float temperature = 0.7;
    int num_predict = 512;  // max tokens
    bool stream = false;    // streaming disabled for now
};

// OllamaMessage: Chat message format
struct OllamaMessage {
    std::string role;     // "system", "user", or "assistant"
    std::string content;
};

// OllamaResponse: Response from Ollama
struct OllamaResponse {
    std::string content;
    bool success;
    std::string error_message;
    int tokens_generated;
    double total_duration_ms;
};

// OllamaClient: HTTP client for Ollama API
//
// Communicates with local Ollama instance via REST API
// Supports chat completion with context
class OllamaClient {
public:
    OllamaClient();
    ~OllamaClient();
    
    // Initialize with config
    bool initialize(const OllamaConfig& config);
    
    // Check if Ollama is available
    bool is_available();
    
    // Get list of available models
    std::vector<std::string> list_models();
    
    // Chat completion (single turn)
    OllamaResponse generate(
        const std::string& prompt,
        const std::vector<OllamaMessage>& context = {}
    );
    
    // Chat completion (multi-turn)
    OllamaResponse chat(
        const std::vector<OllamaMessage>& messages
    );
    
    // Get current config
    const OllamaConfig& get_config() const { return config_; }
    
    // Set model
    void set_model(const std::string& model) { config_.model = model; }
    
private:
    // HTTP request helper
    std::string http_post(
        const std::string& endpoint,
        const std::string& json_body
    );
    
    // Parse JSON response
    OllamaResponse parse_response(const std::string& json);
    
    OllamaConfig config_;
    bool initialized_;
};

} // namespace brain19
