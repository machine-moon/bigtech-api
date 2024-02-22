#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <curl/curl.h>

// Define the API endpoint and your API key
#define API_ENDPOINT "https://api.openai.com/v1/engines/davinci-codex/completions"
#define API_KEY "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"

// Callback function to handle the HTTP response
static size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp) {
    size_t total_size = size * nmemb;
    printf("%.*s", total_size, (char *)contents);
    return total_size;
}

int main() {
    // Initialize cURL
    CURL *curl;
    CURLcode res;
    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();

    if (curl) {
        // Set the HTTP request headers
        struct curl_slist *headers = NULL;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        headers = curl_slist_append(headers, "Authorization: Bearer " API_KEY);

        // Set the request data
        const char *data = "{\"prompt\": \"Translate the following English text to French: 'Hello, world!'\"}";

        // Set cURL options
        curl_easy_setopt(curl, CURLOPT_URL, API_ENDPOINT);
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, data);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);

        // Perform the HTTP request
        res = curl_easy_perform(curl);

        if (res != CURLE_OK) {
            fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
        }

        // Cleanup
        curl_easy_cleanup(curl);
        curl_slist_free_all(headers);
    }

    // Cleanup cURL global state
    curl_global_cleanup();

    return 0;
}
