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
#include <cstdio>

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
    std::cout << "DEBUG: read_ivecs() called with: " << filename << std::endl;
    std::ifstream file(filename, std::ios::binary);
    std::cout << "DEBUG: ifstream created (binary mode)" << std::endl;
    if (!file) {
        std::cerr << "DEBUG: File failed to open!" << std::endl;
        std::cerr << "Error: Unable to open file for reading.\n";
        return {};
    }
    std::cout << "DEBUG: File opened successfully" << std::endl;
    std::vector<std::vector<int>> dataset;
    int count = 0;
    while (file) {
        int d;
        if (!file.read(reinterpret_cast<char*>(&d), sizeof(int))) break;  // Read dimension
        std::vector<int> vec(d);
        if (!file.read(reinterpret_cast<char*>(vec.data()), d * sizeof(int))) break;  // Read vector data
        dataset.push_back(std::move(vec));
        ++count;
        if (count % 1000 == 0) {
            std::cout << "DEBUG: Loaded " << count << " groundtruth entries..." << std::endl;
        }
    }
    file.close();
    std::cout << "DEBUG: Finished reading file. Total entries: " << count << std::endl;
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
    std::cout << "DEBUG: read_two_ints_per_line() called with: " << filename << std::endl;
    
    // TEST: Try C-style file reading first
    std::cout << "DEBUG: Testing C-style file access..." << std::endl;
    FILE* test_fp = fopen(filename.c_str(), "r");
    if (test_fp) {
        char test_buf[256];
        if (fgets(test_buf, sizeof(test_buf), test_fp)) {
            std::cout << "DEBUG: C-style read successful, first line: " << test_buf << std::endl;
        } else {
            std::cout << "DEBUG: C-style fgets() failed" << std::endl;
        }
        fclose(test_fp);
    } else {
        std::cout << "DEBUG: C-style fopen() failed" << std::endl;
    }
    
    std::cout << "DEBUG: Creating ifstream..." << std::endl;
    std::cout.flush();
    std::ifstream file(filename);
    std::cout << "DEBUG: ifstream created" << std::endl;
    std::cout.flush();
    if (!file.is_open()) {
        std::cout << "DEBUG: File failed to open!" << std::endl;
        throw std::runtime_error("Error opening file: " + filename);
    }
    std::cout << "DEBUG: File opened successfully" << std::endl;
    std::cout << "DEBUG: file.good() = " << file.good() << std::endl;
    std::cout << "DEBUG: file.eof() = " << file.eof() << std::endl;
    std::cout << "DEBUG: file.fail() = " << file.fail() << std::endl;
    std::cout.flush();
    
    std::vector<std::pair<int, int>> result;
    std::string line;
    int line_number = 0;
    bool first_line = true;
    std::cout << "DEBUG: Starting while loop to read lines..." << std::endl;
    std::cout.flush();
    
    std::cout << "DEBUG: About to call std::getline() for first time..." << std::endl;
    std::cout.flush();
    while (std::getline(file, line)) {
        std::cout << "DEBUG: Inside while loop, line " << line_number + 1 << std::endl;
        std::cout.flush();
        ++line_number;
        if (line_number == 1) {
            std::cout << "DEBUG: First line content: [" << line << "]" << std::endl;
        }
        if (line_number % 1000 == 0) {
            std::cout << "DEBUG: Processed " << line_number << " lines..." << std::endl;
        }
        // Skip empty lines
        if (line.empty()) {
            std::cout << "DEBUG: Skipping empty line " << line_number << std::endl;
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
    std::cout << "DEBUG: Finished reading file. Total lines: " << line_number << ", result size: " << result.size() << std::endl;
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
