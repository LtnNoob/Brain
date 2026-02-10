#pragma once

#include <string>
#include <stdexcept>

namespace brain19 {

// Simple HTTP client for GET requests using libcurl
class HttpClient {
public:
    struct HttpResponse {
        long status_code;
        std::string body;
        bool success;
        std::string error;
    };

    // Perform HTTP GET request
    static HttpResponse http_get(const std::string& url);

    // URL-encode a string
    static std::string url_encode(const std::string& str);
};

} // namespace brain19
