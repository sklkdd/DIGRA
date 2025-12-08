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
    cout << "\n========================================" << endl;
    cout << "DEBUG: Starting data loading phase" << endl;
    cout << "========================================" << endl;
    
    // Load vectors using DIGRA's load_data function
    // Note: load_data takes num/dim by VALUE (not reference), so we need to calculate count ourselves
    float* data = nullptr;
    
    // First, manually read the file to get the actual vector count
    cout << "DEBUG: Opening data file: " << data_fvecs << endl;
    ifstream data_file(data_fvecs, ios::binary);
    if (!data_file.is_open()) {
        cerr << "ERROR: Cannot open data file: " << data_fvecs << endl;
        return 1;
    }
    cout << "DEBUG: Data file opened successfully" << endl;
    
    // Read dimension from first vector
    int file_dim;
    data_file.read((char*)&file_dim, 4);
    cout << "DEBUG: Read dimension from file: " << file_dim << endl;
    cout << "DEBUG: Expected dimension (from args): " << dim << endl;
    
    if (file_dim != dim) {
        cerr << "ERROR: Dimension mismatch. Expected " << dim << ", got " << file_dim << endl;
        data_file.close();
        return 1;
    }
    cout << "DEBUG: Dimension validation PASSED" << endl;
    
    // Calculate number of vectors from file size
    data_file.seekg(0, ios::end);
    size_t fsize = data_file.tellg();
    cout << "DEBUG: File size in bytes: " << fsize << endl;
    cout << "DEBUG: Calculation: fsize=" << fsize << " / (file_dim + 1)=" << (file_dim + 1) << " / 4" << endl;
    int baseNum = (unsigned)(fsize / (file_dim + 1) / 4);
    cout << "DEBUG: Calculated baseNum: " << baseNum << endl;
    data_file.close();
    
    // Now call load_data to actually allocate and read the data
    cout << "DEBUG: Calling load_data() to allocate and read vectors..." << endl;
    cout << "DEBUG: data pointer before load_data: " << (void*)data << endl;
    
    int dummy_num = 0;
    int dummy_dim = 0;
    load_data(data_fvecs.c_str(), data, dummy_num, dummy_dim);
    
    cout << "DEBUG: load_data() completed" << endl;
    cout << "DEBUG: data pointer after load_data: " << (void*)data << endl;
    
    if (data == nullptr) {
        cerr << "ERROR: load_data() returned null pointer!" << endl;
        return 1;
    }
    cout << "DEBUG: Data pointer is valid (not null)" << endl;
    cout << "DEBUG: Successfully loaded " << baseNum << " vectors of dimension " << file_dim << endl;

    // Load attributes from .data file
    cout << "\n========================================" << endl;
    cout << "DEBUG: Starting attribute loading phase" << endl;
    cout << "========================================" << endl;
    cout << "DEBUG: Allocating arrays for keys and values (size=" << baseNum << ")" << endl;
    
    int* keys = new int[baseNum];
    int* values = new int[baseNum];
    cout << "DEBUG: Allocated keys at: " << (void*)keys << endl;
    cout << "DEBUG: Allocated values at: " << (void*)values << endl;
    
    cout << "DEBUG: Opening attribute file: " << attr_data << endl;
    ifstream attr_file(attr_data);
    if (!attr_file.is_open()) {
        cerr << "ERROR: Cannot open attribute file: " << attr_data << endl;
        delete[] data;
        return 1;
    }
    cout << "DEBUG: Attribute file opened successfully" << endl;
    
    int count = 0;
    cout << "DEBUG: Reading attribute key-value pairs..." << endl;
    while (attr_file >> keys[count] >> values[count]) {
        if (count < 3 || count >= baseNum - 3) {
            cout << "DEBUG: Attribute[" << count << "]: key=" << keys[count] << ", value=" << values[count] << endl;
        } else if (count == 3) {
            cout << "DEBUG: ... (showing first 3 and last 3 only) ..." << endl;
        }
        count++;
        if (count > baseNum) {
            cerr << "ERROR: More attributes than vectors (read " << count << ", expected " << baseNum << ")" << endl;
            delete[] data;
            delete[] keys;
            delete[] values;
            return 1;
        }
    }
    attr_file.close();
    cout << "DEBUG: Read " << count << " attribute pairs" << endl;
    
    if (count != baseNum) {
        cerr << "ERROR: Mismatch between data size (" << baseNum 
             << ") and attribute size (" << count << ")" << endl;
        delete[] data;
        delete[] keys;
        delete[] values;
        return 1;
    }
    cout << "DEBUG: Attribute count validation PASSED" << endl;

    // ========== INDEX CONSTRUCTION (TIMED) ==========
    cout << "\n========================================" << endl;
    cout << "DEBUG: Starting index construction phase" << endl;
    cout << "========================================" << endl;
    cout << "DEBUG: Constructor parameters:" << endl;
    cout << "DEBUG:   dim = " << dim << endl;
    cout << "DEBUG:   baseNum = " << baseNum << endl;
    cout << "DEBUG:   maxBaseNum = " << baseNum << endl;
    cout << "DEBUG:   data = " << (void*)data << endl;
    cout << "DEBUG:   keys = " << (void*)keys << endl;
    cout << "DEBUG:   values = " << (void*)values << endl;
    cout << "DEBUG:   M = " << M << endl;
    cout << "DEBUG:   ef_construction = " << ef_construction << endl;
    
    // Start thread monitoring
    cout << "DEBUG: Starting thread monitor..." << endl;
    atomic<bool> done_monitoring(false);
    thread monitor_thread(monitor_thread_count, ref(done_monitoring));
    
    cout << "DEBUG: Recording start time..." << endl;
    auto start_time = high_resolution_clock::now();

    // Construct DIGRA RangeHNSW index
    cout << "DEBUG: About to call RangeHNSW constructor..." << endl;
    cout << "DEBUG: If crash occurs here, it's inside RangeHNSW constructor" << endl;
    cout << flush;  // Force flush to see this message
    
    RangeHNSW* rangeHnsw = nullptr;
    try {
        rangeHnsw = new RangeHNSW(dim, baseNum, baseNum, data, keys, values, M, ef_construction);
        cout << "DEBUG: RangeHNSW constructor returned successfully!" << endl;
        cout << "DEBUG: rangeHnsw pointer = " << (void*)rangeHnsw << endl;
    } catch (const exception& e) {
        cerr << "\nERROR: C++ exception during index construction: " << e.what() << endl;
        cerr << "ERROR: Exception type: " << typeid(e).name() << endl;
        done_monitoring = true;
        monitor_thread.join();
        delete[] data;
        delete[] keys;
        delete[] values;
        return 1;
    } catch (...) {
        cerr << "\nERROR: Unknown exception during index construction (not std::exception)" << endl;
        cerr << "ERROR: This is likely a segfault or memory corruption" << endl;
        done_monitoring = true;
        monitor_thread.join();
        delete[] data;
        delete[] keys;
        delete[] values;
        return 1;
    }
    
    cout << "DEBUG: Recording end time..." << endl;
    auto end_time = high_resolution_clock::now();
    
    // Stop thread monitoring
    cout << "DEBUG: Stopping thread monitor..." << endl;
    done_monitoring = true;
    monitor_thread.join();
    
    cout << "========================================" << endl;
    cout << "DEBUG: Index construction completed successfully!" << endl;
    cout << "========================================\n" << endl;

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
    if (rangeHnsw != nullptr) {
        delete rangeHnsw;
    }

    cout << "\nNote: DIGRA does not support index serialization." << endl;
    cout << "Index must be rebuilt for query phase." << endl;

    return 0;
}
