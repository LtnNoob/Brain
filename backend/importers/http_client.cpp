#include "http_client.hpp"
#include <curl/curl.h>
#include <mutex>

namespace brain19 {

static size_t write_callback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

static std::once_flag curl_init_flag;

HttpClient::HttpResponse HttpClient::http_get(const std::string& url) {
    std::call_once(curl_init_flag, []() {
        curl_global_init(CURL_GLOBAL_DEFAULT);
    });

    HttpResponse response{0, "", false, ""};

    CURL* curl = curl_easy_init();
    if (!curl) {
        response.error = "Failed to init curl";
        return response;
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response.body);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "User-Agent: Brain19/1.0 (cognitive-architecture)");
    headers = curl_slist_append(headers, "Accept: application/json");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    CURLcode res = curl_easy_perform(curl);

    if (res != CURLE_OK) {
        response.error = std::string("curl error: ") + curl_easy_strerror(res);
    } else {
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response.status_code);
        response.success = (response.status_code >= 200 && response.status_code < 300);
        if (!response.success) {
            response.error = "HTTP " + std::to_string(response.status_code);
        }
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    return response;
}

std::string HttpClient::url_encode(const std::string& str) {
    CURL* curl = curl_easy_init();
    if (!curl) return str;

    char* encoded = curl_easy_escape(curl, str.c_str(), static_cast<int>(str.length()));
    std::string result(encoded ? encoded : str);
    if (encoded) curl_free(encoded);
    curl_easy_cleanup(curl);
    return result;
}

} // namespace brain19
