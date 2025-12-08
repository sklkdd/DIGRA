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
    cout << "\nLoading data..." << endl;
    
    // Load database vectors
    float* data = nullptr;
    int baseNum = 0;
    
    ifstream count_file(data_path, ios::binary);
    if (!count_file) {
        cerr << "Error: Cannot open data file: " << data_path << endl;
        return 1;
    }
    while (count_file) {
        int d;
        if (!count_file.read(reinterpret_cast<char*>(&d), sizeof(int))) break;
        if (d != dim) {
            cerr << "Error: Dimension mismatch in data fvecs file" << endl;
            return 1;
        }
        count_file.seekg(d * sizeof(float), ios::cur);
        baseNum++;
    }
    count_file.close();
    
    load_data(data_path.c_str(), data, baseNum, dim);
    cout << "Loaded " << baseNum << " database vectors" << endl;

    // Load query vectors
    float* query = nullptr;
    int queryNum = 0;
    
    ifstream qcount_file(query_path, ios::binary);
    if (!qcount_file) {
        cerr << "Error: Cannot open query file: " << query_path << endl;
        delete[] data;
        return 1;
    }
    while (qcount_file) {
        int d;
        if (!qcount_file.read(reinterpret_cast<char*>(&d), sizeof(int))) break;
        if (d != dim) {
            cerr << "Error: Dimension mismatch in query fvecs file" << endl;
            delete[] data;
            return 1;
        }
        qcount_file.seekg(d * sizeof(float), ios::cur);
        queryNum++;
    }
    qcount_file.close();
    
    load_data(query_path.c_str(), query, queryNum, dim);
    cout << "Loaded " << queryNum << " query vectors" << endl;

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
    vector<pair<int, int>> query_ranges = read_two_ints_per_line(query_ranges_file);
    if (query_ranges.size() != queryNum) {
        cerr << "Error: Number of query ranges (" << query_ranges.size() 
             << ") != number of queries (" << queryNum << ")\n";
        delete[] data;
        delete[] query;
        delete[] keys;
        delete[] values;
        return 1;
    }
    cout << "Loaded " << query_ranges.size() << " query ranges" << endl;

    // Load groundtruth
    vector<vector<int>> groundtruth = read_ivecs(groundtruth_file);
    if (groundtruth.size() != queryNum) {
        cerr << "Error: Number of groundtruth entries (" << groundtruth.size() 
             << ") != number of queries (" << queryNum << ")\n";
        delete[] data;
        delete[] query;
        delete[] keys;
        delete[] values;
        return 1;
    }
    cout << "Loaded groundtruth" << endl;

    // ========== INDEX RECONSTRUCTION (NOT TIMED) ==========
    cout << "\nRebuilding index (NOT TIMED)..." << endl;
    // Note: ef_construction is not provided, using a reasonable default
    int ef_construction = max(200, ef_search * 2);
    RangeHNSW rangeHnsw(dim, baseNum, baseNum, data, keys, values, M, ef_construction);
    cout << "Index rebuilt with M=" << M << ", ef_construction=" << ef_construction << endl;

    // ========== QUERY EXECUTION (TIMED, excludes recall computation) ==========
    cout << "\n--- Starting query execution (TIMED) ---" << endl;
    
    // Start thread monitoring
    atomic<bool> done_monitoring(false);
    thread monitor_thread(monitor_thread_count, ref(done_monitoring));

    // Store results for later recall calculation (NOT TIMED)
    vector<vector<int>> query_results(queryNum);

    auto start_time = high_resolution_clock::now();

    // Execute queries
    for (int i = 0; i < queryNum; i++) {
        int rangeL = query_ranges[i].first;
        int rangeR = query_ranges[i].second;
        
        // Perform range-filtered query
        // queryRange signature: queryRange(vecData, rangeL, rangeR, k, ef_s)
        auto result = rangeHnsw.queryRange(query + i * dim, rangeL, rangeR, k, ef_search);
        
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

    return 0;
}
