#include "ollama_client.hpp"
#include <curl/curl.h>
#include <sstream>
#include <iostream>
#include <cstring>
#include <mutex>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace brain19 {

// Helper: Write callback for curl
static size_t write_callback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

// Thread-safe curl global init via call_once
static std::once_flag curl_init_flag;

OllamaClient::OllamaClient()
    : initialized_(false)
{
    std::call_once(curl_init_flag, []() {
        curl_global_init(CURL_GLOBAL_DEFAULT);
    });
}

OllamaClient::~OllamaClient() {
    // Note: curl_global_cleanup() intentionally omitted.
    // With multiple OllamaClient instances, cleanup by one would
    // invalidate curl for others. Cleanup happens at process exit.
}

bool OllamaClient::initialize(const OllamaConfig& config) {
    config_ = config;
    
    // Test connection
    if (!is_available()) {
        std::cerr << "Ollama not available at " << config_.host << "\n";
        std::cerr << "Make sure Ollama is running: ollama serve\n";
        return false;
    }
    
    initialized_ = true;
    return true;
}

bool OllamaClient::is_available() const {
    try {
        std::string response = http_post("/api/tags", "{}");
        return !response.empty();
    } catch (...) {
        return false;
    }
}

std::vector<std::string> OllamaClient::list_models() const {
    std::vector<std::string> models;
    
    try {
        std::string response = http_post("/api/tags", "{}");
        
        auto j = json::parse(response);
        if (j.contains("models")) {
            for (const auto& model : j["models"]) {
                if (model.contains("name")) {
                    models.push_back(model["name"]);
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error listing models: " << e.what() << "\n";
    }
    
    return models;
}

OllamaResponse OllamaClient::generate(
    const std::string& prompt,
    const std::vector<OllamaMessage>& context
) const {
    // Build messages array
    std::vector<OllamaMessage> messages = context;
    messages.push_back({"user", prompt});
    
    return chat(messages);
}

OllamaResponse OllamaClient::chat(
    const std::vector<OllamaMessage>& messages
) const {
    OllamaResponse response;
    
    if (!initialized_) {
        response.success = false;
        response.error_message = "Client not initialized";
        return response;
    }
    
    try {
        // Build JSON request
        json req;
        req["model"] = config_.model;
        req["stream"] = config_.stream;
        req["options"]["temperature"] = config_.temperature;
        req["options"]["num_predict"] = config_.num_predict;
        
        // Add messages
        json msgs = json::array();
        for (const auto& msg : messages) {
            msgs.push_back({
                {"role", msg.role},
                {"content", msg.content}
            });
        }
        req["messages"] = msgs;
        
        // Send request
        std::string json_str = req.dump();
        std::string resp_str = http_post("/api/chat", json_str);
        
        // Parse response
        response = parse_response(resp_str);
        
    } catch (const std::exception& e) {
        response.success = false;
        response.error_message = std::string("Exception: ") + e.what();
    }
    
    return response;
}

std::string OllamaClient::http_post(
    const std::string& endpoint,
    const std::string& json_body
) const {
    CURL* curl = curl_easy_init();
    std::string response;
    
    if (!curl) {
        throw std::runtime_error("Failed to init curl");
    }
    
    std::string url = config_.host + endpoint;
    
    // Setup request
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_body.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 300L);  // 5 min timeout
    
    // Set headers
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    
    // Perform request
    CURLcode res = curl_easy_perform(curl);
    
    // Cleanup
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    
    if (res != CURLE_OK) {
        throw std::runtime_error(
            std::string("curl_easy_perform failed: ") + curl_easy_strerror(res)
        );
    }
    
    return response;
}

OllamaResponse OllamaClient::parse_response(const std::string& json_str) const {
    OllamaResponse response;
    
    try {
        auto j = json::parse(json_str);
        
        if (j.contains("message") && j["message"].contains("content")) {
            response.content = j["message"]["content"];
            response.success = true;
        } else if (j.contains("error")) {
            response.success = false;
            response.error_message = j["error"];
        } else {
            response.success = false;
            response.error_message = "Invalid response format";
        }
        
        // Optional metrics
        if (j.contains("total_duration")) {
            response.total_duration_ms = j["total_duration"].get<double>() / 1e6;
        }
        
        if (j.contains("eval_count")) {
            response.tokens_generated = j["eval_count"];
        }
        
    } catch (const std::exception& e) {
        response.success = false;
        response.error_message = std::string("Parse error: ") + e.what();
    }
    
    return response;
}

} // namespace brain19
