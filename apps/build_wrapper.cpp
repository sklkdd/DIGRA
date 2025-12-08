// build_wrapper.cpp - DIGRA index construction wrapper for FANNS benchmarking
// This wrapper builds a DIGRA RangeHNSW index for range-filtered ANN queries

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <atomic>
#include <omp.h>

#include "../TreeHNSW.hpp"
#include "../utils.hpp"
#include "fanns_survey_helpers.cpp"

using namespace std;
using namespace std::chrono;

int main(int argc, char** argv) {
    if (argc != 7) {
        cerr << "Usage: " << argv[0] << " <data.fvecs> <attributes.data> "
             << "<dim> <M> <ef_construction> <threads>\n";
        cerr << "\n";
        cerr << "Arguments:\n";
        cerr << "  data.fvecs         - Database vectors in .fvecs format\n";
        cerr << "  attributes.data    - Attribute file in 'key value' format\n";
        cerr << "  dim                - Vector dimension\n";
        cerr << "  M                  - HNSW degree parameter (max links per layer)\n";
        cerr << "  ef_construction    - Construction ef parameter\n";
        cerr << "  threads            - Number of threads for index construction\n";
        cerr << "\n";
        cerr << "Note: Index path is not used (DIGRA doesn't support serialization)\n";
        return 1;
    }

    string data_fvecs = argv[1];
    string attr_data = argv[2];
    int dim = stoi(argv[3]);
    int M = stoi(argv[4]);
    int ef_construction = stoi(argv[5]);
    int threads = stoi(argv[6]);

    // Set number of threads for construction
    omp_set_num_threads(threads);
    
    cout << "=== DIGRA Index Construction ===" << endl;
    cout << "Data: " << data_fvecs << endl;
    cout << "Attributes: " << attr_data << endl;
    cout << "Dimension: " << dim << endl;
    cout << "Parameters: M=" << M << ", ef_construction=" << ef_construction << endl;
    cout << "Threads: " << threads << endl;

    // ========== DATA LOADING (NOT TIMED) ==========
    cout << "\nLoading data..." << endl;
    
    // Load vectors using DIGRA's load_data function
    float* data = nullptr;
    int baseNum = 0;
    
    // Count number of vectors in fvecs file
    ifstream count_file(data_fvecs, ios::binary);
    if (!count_file) {
        cerr << "Error: Cannot open data file: " << data_fvecs << endl;
        return 1;
    }
    while (count_file) {
        int d;
        if (!count_file.read(reinterpret_cast<char*>(&d), sizeof(int))) break;
        if (d != dim) {
            cerr << "Error: Dimension mismatch in fvecs file" << endl;
            return 1;
        }
        count_file.seekg(d * sizeof(float), ios::cur);
        baseNum++;
    }
    count_file.close();
    
    cout << "Found " << baseNum << " vectors" << endl;
    
    // Load vectors
    load_data(data_fvecs.c_str(), data, baseNum, dim);
    cout << "Loaded " << baseNum << " vectors of dimension " << dim << endl;

    // Load attributes from .data file
    int* keys = new int[baseNum];
    int* values = new int[baseNum];
    
    ifstream attr_file(attr_data);
    if (!attr_file.is_open()) {
        cerr << "Error: Cannot open attribute file: " << attr_data << endl;
        delete[] data;
        return 1;
    }
    
    int count = 0;
    while (attr_file >> keys[count] >> values[count]) {
        count++;
        if (count > baseNum) {
            cerr << "Error: More attributes than vectors" << endl;
            delete[] data;
            delete[] keys;
            delete[] values;
            return 1;
        }
    }
    attr_file.close();
    
    if (count != baseNum) {
        cerr << "Error: Mismatch between data size (" << baseNum 
             << ") and attribute size (" << count << ")" << endl;
        delete[] data;
        delete[] keys;
        delete[] values;
        return 1;
    }
    cout << "Loaded " << count << " attribute values" << endl;

    // ========== INDEX CONSTRUCTION (TIMED) ==========
    cout << "\n--- Starting index construction (TIMED) ---" << endl;
    
    // Start thread monitoring
    atomic<bool> done_monitoring(false);
    thread monitor_thread(monitor_thread_count, ref(done_monitoring));
    
    auto start_time = high_resolution_clock::now();

    // Construct DIGRA RangeHNSW index
    // Constructor signature: RangeHNSW(dim, baseNum, maxBaseNum, data, key, value, m, ef_construction)
    RangeHNSW rangeHnsw(dim, baseNum, baseNum, data, keys, values, M, ef_construction);
    
    auto end_time = high_resolution_clock::now();
    
    // Stop thread monitoring
    done_monitoring = true;
    monitor_thread.join();
    
    cout << "--- Index construction complete ---\n" << endl;

    // ========== TIMING OUTPUT ==========
    double build_time_sec = duration_cast<duration<double>>(end_time - start_time).count();
    
    cout << "BUILD_TIME_SECONDS: " << build_time_sec << endl;
    cout << "PEAK_THREADS: " << peak_threads.load() << endl;
    
    // Memory footprint
    peak_memory_footprint();

    // Cleanup
    delete[] data;
    delete[] keys;
    delete[] values;

    cout << "\nNote: DIGRA does not support index serialization." << endl;
    cout << "Index must be rebuilt for query phase." << endl;

    return 0;
}
