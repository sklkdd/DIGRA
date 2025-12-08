// search_wrapper.cpp - DIGRA query execution wrapper for FANNS benchmarking
// This wrapper performs range-filtered ANN queries using DIGRA RangeHNSW index

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <queue>
#include <set>
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
    if (argc != 19) {
        cerr << "Usage: " << argv[0] << " --data_path <data.fvecs> "
             << "--query_path <query.fvecs> --query_ranges_file <ranges.csv> "
             << "--groundtruth_file <gt.ivecs> --attributes_file <attrs.data> "
             << "--dim <dim> --ef_search <ef> --k <k> --M <M>\n";
        cerr << "\n";
        cerr << "Arguments:\n";
        cerr << "  --data_path          - Database vectors in .fvecs format\n";
        cerr << "  --query_path         - Query vectors in .fvecs format\n";
        cerr << "  --query_ranges_file  - Query ranges (low-high per line)\n";
        cerr << "  --groundtruth_file   - Groundtruth in .ivecs format\n";
        cerr << "  --attributes_file    - Attributes in 'key value' format\n";
        cerr << "  --dim                - Vector dimension\n";
        cerr << "  --ef_search          - Search ef parameter\n";
        cerr << "  --k                  - Number of neighbors to return\n";
        cerr << "  --M                  - HNSW degree (used for rebuild)\n";
        return 1;
    }

    // Parse command-line arguments
    string data_path, query_path, query_ranges_file, groundtruth_file, attributes_file;
    int dim = -1, ef_search = -1, k = -1, M = -1;

    for (int i = 1; i < argc; i += 2) {
        string arg = argv[i];
        if (arg == "--data_path") data_path = argv[i + 1];
        else if (arg == "--query_path") query_path = argv[i + 1];
        else if (arg == "--query_ranges_file") query_ranges_file = argv[i + 1];
        else if (arg == "--groundtruth_file") groundtruth_file = argv[i + 1];
        else if (arg == "--attributes_file") attributes_file = argv[i + 1];
        else if (arg == "--dim") dim = stoi(argv[i + 1]);
        else if (arg == "--ef_search") ef_search = stoi(argv[i + 1]);
        else if (arg == "--k") k = stoi(argv[i + 1]);
        else if (arg == "--M") M = stoi(argv[i + 1]);
    }

    // Validate inputs
    if (data_path.empty() || query_path.empty() || query_ranges_file.empty() || 
        groundtruth_file.empty() || attributes_file.empty()) {
        cerr << "Error: Missing required file arguments\n";
        return 1;
    }
    if (dim <= 0 || ef_search <= 0 || k <= 0 || M <= 0) {
        cerr << "Error: Invalid numeric parameters\n";
        cerr << "dim=" << dim << ", ef_search=" << ef_search << ", k=" << k << ", M=" << M << endl;
        return 1;
    }

    // Force single-threaded query execution
    omp_set_num_threads(1);

    cout << "=== DIGRA Query Execution ===" << endl;
    cout << "Data: " << data_path << endl;
    cout << "Query: " << query_path << endl;
    cout << "Query ranges: " << query_ranges_file << endl;
    cout << "Groundtruth: " << groundtruth_file << endl;
    cout << "Attributes: " << attributes_file << endl;
    cout << "Parameters: dim=" << dim << ", k=" << k << ", M=" << M << ", ef_search=" << ef_search << endl;

    // ========== DATA LOADING (NOT TIMED) ==========
    cout << "\n========================================" << endl;
    cout << "DEBUG: Starting data loading phase" << endl;
    cout << "========================================" << endl;
    
    // Load database vectors using DIGRA's load_data function
    // Note: load_data takes num/dim by VALUE (not reference), so we calculate counts ourselves
    float* data = nullptr;
    
    // Calculate database vector count from file
    cout << "DEBUG: Opening database file: " << data_path << endl;
    ifstream data_file(data_path, ios::binary);
    if (!data_file.is_open()) {
        cerr << "ERROR: Cannot open data file: " << data_path << endl;
        return 1;
    }
    cout << "DEBUG: Database file opened successfully" << endl;
    
    int file_dim;
    data_file.read((char*)&file_dim, 4);
    cout << "DEBUG: Read dimension from database file: " << file_dim << endl;
    cout << "DEBUG: Expected dimension (from args): " << dim << endl;
    
    if (file_dim != dim) {
        cerr << "ERROR: Dimension mismatch in data. Expected " << dim << ", got " << file_dim << endl;
        data_file.close();
        return 1;
    }
    cout << "DEBUG: Database dimension validation PASSED" << endl;
    
    data_file.seekg(0, ios::end);
    size_t data_fsize = data_file.tellg();
    cout << "DEBUG: Database file size: " << data_fsize << " bytes" << endl;
    int baseNum = (unsigned)(data_fsize / (file_dim + 1) / 4);
    cout << "DEBUG: Calculated baseNum: " << baseNum << endl;
    data_file.close();
    
    // Load data using DIGRA's function
    cout << "DEBUG: Calling load_data() for database vectors..." << endl;
    int dummy_num = 0, dummy_dim = 0;
    load_data(data_path.c_str(), data, dummy_num, dummy_dim);
    cout << "DEBUG: load_data() completed for database" << endl;
    cout << "DEBUG: data pointer: " << (void*)data << endl;
    
    if (data == nullptr) {
        cerr << "ERROR: load_data() returned null pointer for database!" << endl;
        return 1;
    }

    // Load query vectors
    float* query = nullptr;
    
    // Calculate query vector count from file
    cout << "\nDEBUG: Opening query file: " << query_path << endl;
    ifstream query_file(query_path, ios::binary);
    if (!query_file.is_open()) {
        cerr << "ERROR: Cannot open query file: " << query_path << endl;
        delete[] data;
        return 1;
    }
    cout << "DEBUG: Query file opened successfully" << endl;
    
    query_file.read((char*)&file_dim, 4);
    cout << "DEBUG: Read dimension from query file: " << file_dim << endl;
    
    if (file_dim != dim) {
        cerr << "ERROR: Dimension mismatch in queries. Expected " << dim << ", got " << file_dim << endl;
        query_file.close();
        delete[] data;
        return 1;
    }
    cout << "DEBUG: Query dimension validation PASSED" << endl;
    
    query_file.seekg(0, ios::end);
    size_t query_fsize = query_file.tellg();
    cout << "DEBUG: Query file size: " << query_fsize << " bytes" << endl;
    int queryNum = (unsigned)(query_fsize / (file_dim + 1) / 4);
    cout << "DEBUG: Calculated queryNum: " << queryNum << endl;
    query_file.close();
    
    // Load queries using DIGRA's function
    cout << "DEBUG: Calling load_data() for query vectors..." << endl;
    load_data(query_path.c_str(), query, dummy_num, dummy_dim);
    cout << "DEBUG: load_data() completed for queries" << endl;
    cout << "DEBUG: query pointer: " << (void*)query << endl;
    
    if (query == nullptr) {
        cerr << "ERROR: load_data() returned null pointer for queries!" << endl;
        delete[] data;
        return 1;
    }
    cout << "DEBUG: Successfully loaded " << baseNum << " database and " << queryNum << " query vectors" << endl;

    // Load attributes
    int* keys = new int[baseNum];
    int* values = new int[baseNum];
    
    ifstream attr_file(attributes_file);
    if (!attr_file.is_open()) {
        cerr << "Error: Cannot open attribute file: " << attributes_file << endl;
        delete[] data;
        delete[] query;
        return 1;
    }
    
    int count = 0;
    while (attr_file >> keys[count] >> values[count]) {
        count++;
        if (count > baseNum) break;
    }
    attr_file.close();
    
    if (count != baseNum) {
        cerr << "Error: Attribute count mismatch" << endl;
        delete[] data;
        delete[] query;
        delete[] keys;
        delete[] values;
        return 1;
    }
    cout << "Loaded " << count << " attributes" << endl;

    // Load query ranges
    cout << "\nDEBUG: Loading query ranges from: " << query_ranges_file << endl;
    vector<pair<int, int>> query_ranges;
    try {
        query_ranges = read_two_ints_per_line(query_ranges_file);
        cout << "DEBUG: Successfully read " << query_ranges.size() << " query ranges" << endl;
    } catch (const exception& e) {
        cerr << "ERROR: Failed to read query ranges: " << e.what() << endl;
        delete[] data;
        delete[] query;
        delete[] keys;
        delete[] values;
        return 1;
    }
    
    if (query_ranges.size() != (size_t)queryNum) {
        cerr << "ERROR: Number of query ranges (" << query_ranges.size() 
             << ") != number of queries (" << queryNum << ")\n";
        delete[] data;
        delete[] query;
        delete[] keys;
        delete[] values;
        return 1;
    }
    cout << "DEBUG: Query ranges count validation PASSED" << endl;

    // Load groundtruth
    cout << "\nDEBUG: Loading groundtruth from: " << groundtruth_file << endl;
    vector<vector<int>> groundtruth;
    try {
        groundtruth = read_ivecs(groundtruth_file);
        cout << "DEBUG: Successfully read " << groundtruth.size() << " groundtruth entries" << endl;
    } catch (const exception& e) {
        cerr << "ERROR: Failed to read groundtruth: " << e.what() << endl;
        delete[] data;
        delete[] query;
        delete[] keys;
        delete[] values;
        return 1;
    }
    
    if (groundtruth.size() != (size_t)queryNum) {
        cerr << "ERROR: Number of groundtruth entries (" << groundtruth.size() 
             << ") != number of queries (" << queryNum << ")\n";
        delete[] data;
        delete[] query;
        delete[] keys;
        delete[] values;
        return 1;
    }
    cout << "DEBUG: Groundtruth count validation PASSED" << endl;

    // ========== INDEX RECONSTRUCTION (NOT TIMED) ==========
    cout << "\n========================================" << endl;
    cout << "DEBUG: Starting index reconstruction phase" << endl;
    cout << "========================================" << endl;
    
    // Note: ef_construction is not provided, using a reasonable default
    int ef_construction = max(200, ef_search * 2);
    cout << "DEBUG: Using ef_construction = " << ef_construction << " (max of 200 and " << (ef_search * 2) << ")" << endl;
    
    cout << "DEBUG: Constructor parameters:" << endl;
    cout << "DEBUG:   dim = " << dim << endl;
    cout << "DEBUG:   baseNum = " << baseNum << endl;
    cout << "DEBUG:   maxBaseNum = " << baseNum << endl;
    cout << "DEBUG:   data = " << (void*)data << endl;
    cout << "DEBUG:   keys = " << (void*)keys << endl;
    cout << "DEBUG:   values = " << (void*)values << endl;
    cout << "DEBUG:   M = " << M << endl;
    cout << "DEBUG:   ef_construction = " << ef_construction << endl;
    
    RangeHNSW* rangeHnsw = nullptr;
    cout << "DEBUG: About to call RangeHNSW constructor..." << endl;
    cout << flush;  // Force flush
    
    try {
        rangeHnsw = new RangeHNSW(dim, baseNum, baseNum, data, keys, values, M, ef_construction);
        cout << "DEBUG: RangeHNSW constructor completed successfully!" << endl;
        cout << "DEBUG: rangeHnsw pointer = " << (void*)rangeHnsw << endl;
    } catch (const exception& e) {
        cerr << "\nERROR: C++ exception during index reconstruction: " << e.what() << endl;
        delete[] data;
        delete[] query;
        delete[] keys;
        delete[] values;
        return 1;
    } catch (...) {
        cerr << "\nERROR: Unknown exception during index reconstruction" << endl;
        delete[] data;
        delete[] query;
        delete[] keys;
        delete[] values;
        return 1;
    }
    
    cout << "DEBUG: Index reconstruction completed successfully" << endl;

    // ========== QUERY EXECUTION (TIMED, excludes recall computation) ==========
    cout << "\n--- Starting query execution (TIMED) ---" << endl;
    
    // Start thread monitoring
    atomic<bool> done_monitoring(false);
    thread monitor_thread(monitor_thread_count, ref(done_monitoring));

    // Store results for later recall calculation (NOT TIMED)
    vector<vector<int>> query_results(queryNum);

    auto start_time = high_resolution_clock::now();

    // Execute queries
    try {
        for (int i = 0; i < queryNum; i++) {
            int rangeL = query_ranges[i].first;
            int rangeR = query_ranges[i].second;
            
            // Perform range-filtered query
            // queryRange signature: queryRange(vecData, rangeL, rangeR, k, ef_s)
            auto result = rangeHnsw->queryRange(query + i * dim, rangeL, rangeR, k, ef_search);
            
            // Extract IDs from priority queue
            query_results[i].reserve(k);
            while (!result.empty()) {
                query_results[i].push_back(result.top().second);
                result.pop();
            }
            
            if ((i + 1) % 1000 == 0) {
                cout << "  Processed " << (i + 1) << " / " << queryNum << " queries" << endl;
            }
        }
    } catch (const exception& e) {
        cerr << "Error during query execution: " << e.what() << endl;
        done_monitoring = true;
        monitor_thread.join();
        delete[] data;
        delete[] query;
        delete[] keys;
        delete[] values;
        if (rangeHnsw != nullptr) delete rangeHnsw;
        return 1;
    } catch (...) {
        cerr << "Unknown error during query execution" << endl;
        done_monitoring = true;
        monitor_thread.join();
        delete[] data;
        delete[] query;
        delete[] keys;
        delete[] values;
        if (rangeHnsw != nullptr) delete rangeHnsw;
        return 1;
    }

    auto end_time = high_resolution_clock::now();
    
    // Stop thread monitoring
    done_monitoring = true;
    monitor_thread.join();
    
    cout << "--- Query execution complete ---\n" << endl;

    // ========== TIMING OUTPUT ==========
    double query_time_sec = duration_cast<duration<double>>(end_time - start_time).count();
    double qps = queryNum / query_time_sec;

    // ========== RECALL CALCULATION (NOT TIMED) ==========
    int total_true_positives = 0;
    for (int i = 0; i < queryNum; i++) {
        // Convert query results to set for faster lookup
        set<int> result_set(query_results[i].begin(), query_results[i].end());
        
        // Count true positives (compare with first k groundtruth results)
        int gt_size = min(k, (int)groundtruth[i].size());
        for (int j = 0; j < gt_size; j++) {
            if (result_set.count(groundtruth[i][j])) {
                total_true_positives++;
            }
        }
    }

    float recall = static_cast<float>(total_true_positives) / (queryNum * k);

    // ========== OUTPUT RESULTS ==========
    cout << "QUERY_TIME_SECONDS: " << query_time_sec << endl;
    cout << "QPS: " << qps << endl;
    cout << "RECALL: " << recall << endl;
    cout << "PEAK_THREADS: " << peak_threads.load() << endl;
    
    // Memory footprint
    peak_memory_footprint();

    // Cleanup
    delete[] data;
    delete[] query;
    delete[] keys;
    delete[] values;
    if (rangeHnsw != nullptr) {
        delete rangeHnsw;
    }

    return 0;
}
