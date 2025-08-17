#ifndef CONFIG_READER_H
#define CONFIG_READER_H

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cctype>
#include <utility>

#include <vtkVersion.h>
#include <vtkPLYReader.h>
#include <vtkOBJReader.h>
#include <vtkTriangle.h>
#include <vtkTriangleFilter.h>
#include <vtkPolyDataMapper.h>

#include <vtknlohmannjson/include/vtknlohmann/json.hpp>

/**
 * @brief ConfigReader class for loading and accessing JSON configuration files.
 * 
 * This class uses nlohmann::json (bundled in VTK) to parse JSON files and 
 * provides convenient access methods with support for dot and array paths.
 * Example path: "camera.intrinsic[0][0]".
 */
class ConfigReader {
public:
    using json = nlohmann::json;

    ConfigReader();
    explicit ConfigReader(const std::string& path);

    /**
     * @brief Load configuration from a JSON file.
     * @param path Path to JSON file.
     * @param err Optional pointer to error message.
     * @return true if load succeeds, false otherwise.
     */
    bool loadFromFile(const std::string& path, std::string* err = nullptr);

    /**
     * @brief Load configuration from a raw JSON string.
     * @param json_text JSON string.
     * @param err Optional pointer to error message.
     * @return true if load succeeds, false otherwise.
     */
    bool loadFromString(const std::string& json_text, std::string* err = nullptr);

    /**
     * @brief Check if a path exists in the JSON structure.
     * @param path Dot/array style path.
     * @return true if exists and not null.
     */
    bool has(const std::string& path) const;

    /**
     * @brief Get value at path or return default if not found/mismatch.
     * @tparam T Desired type.
     * @param path Dot/array style path.
     * @param default_value Value returned if not found or invalid type.
     */
    template <typename T>
    T get(const std::string& path, const T& default_value) const {
        const json* node = resolve(path);
        if (!node || node->is_null()) return default_value;
        try {
            return node->get<T>();
        } catch (...) {
            return default_value;
        }
    }

    /**
     * @brief Get value at path, throw exception if not found or wrong type.
     * @tparam T Desired type.
     * @param path Dot/array style path.
     * @throws std::runtime_error if missing or type mismatch.
     */
    template <typename T>
    T require(const std::string& path) const {
        const json* node = resolve(path);
        if (!node || node->is_null()) {
            throw std::runtime_error("ConfigReader: missing required key: " + path);
        }
        try {
            return node->get<T>();
        } catch (const std::exception& e) {
            throw std::runtime_error("ConfigReader: type mismatch at '" + path + "': " + std::string(e.what()));
        }
    }

    /// Access the raw JSON object.
    const json& raw() const { return root_; }

    /// Dump entire JSON as a formatted string.
    std::string dump(int indent = 2) const { return root_.dump(indent); }

private:
    struct Token {
        std::string key;   ///< object field name
        int index = -1;    ///< array index (-1 if not array)
    };

    static void setErr(std::string* err, const std::string& msg);
    static std::vector<Token> tokenize(const std::string& path);
    const json* resolve(const std::string& path) const;

    json root_;
};


#endif
