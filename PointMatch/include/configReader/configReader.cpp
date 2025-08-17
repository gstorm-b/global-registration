#include "configReader.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cctype>

// ============ Public API ============

ConfigReader::ConfigReader() = default;

ConfigReader::ConfigReader(const std::string& path) {
    std::string err;
    if (!loadFromFile(path, &err)) {
        throw std::runtime_error("ConfigReader: cannot load '" + path + "': " + err);
    }
}

bool ConfigReader::loadFromFile(const std::string& path, std::string* err) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        setErr(err, "failed to open file");
        return false;
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return loadFromString(oss.str(), err);
}

bool ConfigReader::loadFromString(const std::string& json_text, std::string* err) {
    try {
        root_ = json::parse(json_text);
        return true;
    } catch (const std::exception& e) {
        setErr(err, e.what());
        return false;
    }
}

bool ConfigReader::has(const std::string& path) const {
    const json* node = resolve(path);
    return node != nullptr && !node->is_null();
}

// ============ Private Helpers ============

void ConfigReader::setErr(std::string* err, const std::string& msg) {
    if (err) *err = msg;
}

std::vector<ConfigReader::Token> ConfigReader::tokenize(const std::string& path) {
    std::vector<Token> tokens;
    Token cur;

    auto flush_key = [&]() {
        if (!cur.key.empty() || cur.index != -1) {
            tokens.push_back(cur);
            cur = Token{};
        }
    };

    for (size_t i = 0; i < path.size(); ) {
        if (path[i] == '.') {
            flush_key();
            ++i;
            continue;
        }
        if (path[i] == '[') {
            ++i;
            int sign = 1;
            if (i < path.size() && path[i] == '-') { sign = -1; ++i; }
            int val = 0;
            bool hasDigit = false;
            while (i < path.size() && std::isdigit(static_cast<unsigned char>(path[i]))) {
                hasDigit = true;
                val = val * 10 + (path[i] - '0');
                ++i;
            }
            if (i >= path.size() || path[i] != ']' || !hasDigit) {
                throw std::runtime_error("ConfigReader: invalid array syntax in path: " + path);
            }
            cur.index = sign * val;
            ++i; // skip ']'
            flush_key();
            continue;
        }
        size_t j = i;
        while (j < path.size() && path[j] != '.' && path[j] != '[') ++j;
        cur.key = path.substr(i, j - i);
        i = j;
    }
    flush_key();
    return tokens;
}

const ConfigReader::json* ConfigReader::resolve(const std::string& path) const {
    if (path.empty()) return &root_;
    const json* node = &root_;
    std::vector<Token> tokens;
    try {
        tokens = tokenize(path);
    } catch (...) {
        return nullptr;
    }

    for (const auto& t : tokens) {
        if (!t.key.empty()) {
            if (!node->is_object()) return nullptr;
            auto it = node->find(t.key);
            if (it == node->end()) return nullptr;
            node = &(*it);
        }
        if (t.index != -1) {
            if (!node->is_array()) return nullptr;
            int idx = t.index;
            if (idx < 0) idx = static_cast<int>(node->size()) + idx; // support negative index
            if (idx < 0 || static_cast<size_t>(idx) >= node->size()) return nullptr;
            node = &(*node)[static_cast<size_t>(idx)];
        }
    }
    return node;
}
