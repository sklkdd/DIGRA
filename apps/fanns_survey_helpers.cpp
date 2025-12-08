#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <thread>
#include <atomic>
#include <chrono>
#include <unistd.h>
#include <utility>
#include <omp.h>

#include "global_thread_counter.h"


std::vector<std::vector<float>> read_fvecs(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Unable to open file for reading.\n";
        return {};
    }
    std::vector<std::vector<float>> dataset;
    while (file) {
        int d;
        if (!file.read(reinterpret_cast<char*>(&d), sizeof(int))) break;  // Read dimension
        std::vector<float> vec(d);
        if (!file.read(reinterpret_cast<char*>(vec.data()), d * sizeof(float))) break;  // Read vector data
        dataset.push_back(std::move(vec));
    }
    file.close();
    return dataset;
}

std::vector<std::vector<int>> read_ivecs(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Unable to open file for reading.\n";
        return {};
    }
    std::vector<std::vector<int>> dataset;
    while (file) {
        int d;
        if (!file.read(reinterpret_cast<char*>(&d), sizeof(int))) break;  // Read dimension
        std::vector<int> vec(d);
        if (!file.read(reinterpret_cast<char*>(vec.data()), d * sizeof(int))) break;  // Read vector data
        dataset.push_back(std::move(vec));
    }
    file.close();
    return dataset;
}

std::vector<int> read_one_int_per_line(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Error opening file: " + filename);
    }
    std::vector<int> result;
    std::string line;
    int line_number = 0;
    bool first_line = true;
    while (std::getline(file, line)) {
        ++line_number;
        // Skip empty lines
        if (line.empty()) {
            continue;
        }
        // Check if first line is a header (non-numeric)
        if (first_line) {
            first_line = false;
            std::stringstream test_ss(line);
            int test_value;
            if (!(test_ss >> test_value)) {
                // First line is a header, skip it
                continue;
            }
            // First line is numeric, process it normally
        }
        std::stringstream ss(line);
        int value;
        if (!(ss >> value)) {
            throw std::runtime_error("Non-integer or empty line at line " + std::to_string(line_number));
        }
        std::string extra;
        if (ss >> extra) {
            throw std::runtime_error("More than one value on line " + std::to_string(line_number));
        }
        result.push_back(value);
    }
    return result;
}

std::vector<std::vector<int>> read_multiple_ints_per_line(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Error opening file: " + filename);
    }
    std::vector<std::vector<int>> data;
    std::string line;
    int line_number = 0;
    while (std::getline(file, line)) {
        ++line_number;
        std::vector<int> row;
        std::stringstream ss(line);
        std::string token;
        while (std::getline(ss, token, ',')) {
            try {
                if (!token.empty()) {
                    row.push_back(std::stoi(token));
                }
            } catch (...) {
                throw std::runtime_error("Invalid integer on line " + std::to_string(line_number));
            }
        }
        data.push_back(std::move(row));
    }
    return data;
}

std::vector<std::pair<int, int>> read_two_ints_per_line(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Error opening file: " + filename);
    }
    std::vector<std::pair<int, int>> result;
    std::string line;
    int line_number = 0;
    bool first_line = true;
    while (std::getline(file, line)) {
        ++line_number;
        // Skip empty lines
        if (line.empty()) {
            continue;
        }
        // Check if first line is a header by trying to parse it
        if (first_line) {
            first_line = false;
            std::stringstream test_ss(line);
            std::string test_first, test_second;
            if (std::getline(test_ss, test_first, '-') && std::getline(test_ss, test_second)) {
                try {
                    std::stoi(test_first);
                    std::stoi(test_second);
                    // Successfully parsed as integers, not a header - process normally
                } catch (...) {
                    // Failed to parse as integers, it's a header - skip it
                    continue;
                }
            }
        }
        std::stringstream ss(line);
        std::string first, second;
        if (!std::getline(ss, first, '-') || !std::getline(ss, second)) {
            throw std::runtime_error("Invalid format at line " + std::to_string(line_number));
        }
        try {
            int a = std::stoi(first);
            int b = std::stoi(second);
            result.emplace_back(a, b);
        } catch (...) {
            throw std::runtime_error("Invalid integer value at line " + std::to_string(line_number));
        }
    }
    return result;
}

void peak_memory_footprint()
{

    unsigned iPid = (unsigned)getpid();

    std::cout << "PID: " << iPid << std::endl;

    std::string status_file = "/proc/" + std::to_string(iPid) + "/status";
    std::ifstream info(status_file);
    if (!info.is_open())
    {
        std::cout << "memory information open error!" << std::endl;
    }
    std::string tmp;
    while (getline(info, tmp))
    {
        if (tmp.find("Name:") != std::string::npos || tmp.find("VmPeak:") != std::string::npos || tmp.find("VmHWM:") != std::string::npos)
            std::cout << tmp << std::endl;
    }
    info.close();
}

void monitor_thread_count(std::atomic<bool>& done) {
    while (!done.load()) {
        int current = std::thread::hardware_concurrency();
        #pragma omp parallel
        {
            #pragma omp single
            {
                current = omp_get_num_threads();
            }
        }
        int expected = peak_threads.load();
        while (current > expected && !peak_threads.compare_exchange_weak(expected, current)) {}
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}
