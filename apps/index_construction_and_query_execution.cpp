// index_construction_and_query_execution.cpp - DIGRA combined wrapper for FANNS benchmarking
// This wrapper builds a DIGRA RangeHNSW index once and queries it multiple times
// with different ef_search values

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <queue>
#include <set>
#include <chrono>
#include <thread>
#include <atomic>
#include <algorithm>
#include <omp.h>

#include "../TreeHNSW.hpp"
#include "../utils.hpp"
#include "fanns_survey_helpers.cpp"

using namespace std;
using namespace std::chrono;

// Parse comma-separated list of integers (e.g., "4,8,16,32,64")
vector<int> parse_int_list(const string& input) {
    string cleaned = input;
    // Remove brackets if present
    cleaned.erase(remove_if(cleaned.begin(), cleaned.end(),
                  [](char c) { return c == '[' || c == ']'; }),
                  cleaned.end());

    vector<int> result;
    stringstream ss(cleaned);
    string token;

    while (getline(ss, token, ',')) {
        result.push_back(stoi(token));
    }

    return result;
}

int main(int argc, char** argv) {
    if (argc != 12) {
        cerr << "Usage: " << argv[0] << " <data.fvecs> <attributes.data> "
             << "<query.fvecs> <query_ranges.csv> <groundtruth.ivecs> "
             << "<dim> <M> <ef_construction> <ef_search_list> <k> <threads>\n";
        cerr << "\n";
        cerr << "Arguments:\n";
        cerr << "  data.fvecs          - Database vectors in .fvecs format\n";
        cerr << "  attributes.data     - Attribute file in 'key value' format\n";
        cerr << "  query.fvecs         - Query vectors in .fvecs format\n";
        cerr << "  query_ranges.csv    - Query ranges (low-high per line)\n";
        cerr << "  groundtruth.ivecs   - Groundtruth in .ivecs format\n";
        cerr << "  dim                 - Vector dimension\n";
        cerr << "  M                   - HNSW degree parameter (max links per layer)\n";
        cerr << "  ef_construction     - Construction ef parameter\n";
        cerr << "  ef_search_list      - Comma-separated list of ef_search values (e.g., 4,8,16,32,64)\n";
        cerr << "  k                   - Number of neighbors to return\n";
        cerr << "  threads             - Number of threads for index construction\n";
        return 1;
    }

    // Parse arguments
    string data_fvecs = argv[1];
    string attr_data = argv[2];
    string query_fvecs = argv[3];
    string query_ranges_file = argv[4];
    string groundtruth_file = argv[5];
    int dim = stoi(argv[6]);
    int M = stoi(argv[7]);
    int ef_construction = stoi(argv[8]);
    vector<int> ef_search_list = parse_int_list(argv[9]);
    int k = stoi(argv[10]);
    int threads = stoi(argv[11]);

    // Set number of threads for construction
    omp_set_num_threads(threads);
    
    cout << "=== DIGRA Index Construction and Query Execution ===" << endl;
    cout << "Data: " << data_fvecs << endl;
    cout << "Attributes: " << attr_data << endl;
    cout << "Queries: " << query_fvecs << endl;
    cout << "Query ranges: " << query_ranges_file << endl;
    cout << "Groundtruth: " << groundtruth_file << endl;
    cout << "Parameters: dim=" << dim << ", M=" << M << ", ef_construction=" << ef_construction << ", k=" << k << endl;
    cout << "ef_search values: ";
    for (int ef : ef_search_list) cout << ef << " ";
    cout << endl;
    cout << "Threads: " << threads << endl;

    // ========== DATA LOADING (NOT TIMED) ==========
    cout << "\nLoading data..." << endl;
    
    // Load database vectors
    float* data = nullptr;
    
    // Calculate database vector count from file
    ifstream data_file(data_fvecs, ios::binary);
    if (!data_file.is_open()) {
        cerr << "ERROR: Cannot open data file: " << data_fvecs << endl;
        return 1;
    }
    
    int file_dim;
    data_file.read((char*)&file_dim, 4);
    
    if (file_dim != dim) {
        cerr << "ERROR: Dimension mismatch. Expected " << dim << ", got " << file_dim << endl;
        data_file.close();
        return 1;
    }
    
    data_file.seekg(0, ios::end);
    size_t data_fsize = data_file.tellg();
    int baseNum = (unsigned)(data_fsize / (file_dim + 1) / 4);
    data_file.close();
    
    // Load data using DIGRA's function
    int dummy_num = 0, dummy_dim = 0;
    load_data(data_fvecs.c_str(), data, dummy_num, dummy_dim);
    
    if (data == nullptr) {
        cerr << "ERROR: load_data() returned null pointer for database!" << endl;
        return 1;
    }
    cout << "Loaded " << baseNum << " database vectors of dimension " << dim << endl;

    // Load query vectors
    float* query = nullptr;
    
    ifstream query_file(query_fvecs, ios::binary);
    if (!query_file.is_open()) {
        cerr << "ERROR: Cannot open query file: " << query_fvecs << endl;
        delete[] data;
        return 1;
    }
    
    query_file.read((char*)&file_dim, 4);
    
    if (file_dim != dim) {
        cerr << "ERROR: Dimension mismatch in queries. Expected " << dim << ", got " << file_dim << endl;
        query_file.close();
        delete[] data;
        return 1;
    }
    
    query_file.seekg(0, ios::end);
    size_t query_fsize = query_file.tellg();
    int queryNum = (unsigned)(query_fsize / (file_dim + 1) / 4);
    query_file.close();
    
    load_data(query_fvecs.c_str(), query, dummy_num, dummy_dim);
    
    if (query == nullptr) {
        cerr << "ERROR: load_data() returned null pointer for queries!" << endl;
        delete[] data;
        return 1;
    }
    cout << "Loaded " << queryNum << " query vectors" << endl;

    // Load attributes
    int* keys = new int[baseNum];
    int* values = new int[baseNum];
    
    ifstream attr_file(attr_data);
    if (!attr_file.is_open()) {
        cerr << "ERROR: Cannot open attribute file: " << attr_data << endl;
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
        cerr << "ERROR: Mismatch between data size (" << baseNum 
             << ") and attribute size (" << count << ")" << endl;
        delete[] data;
        delete[] query;
        delete[] keys;
        delete[] values;
        return 1;
    }
    cout << "Loaded " << count << " attribute pairs" << endl;

    // Load query ranges
    vector<pair<int, int>> query_ranges;
    try {
        query_ranges = read_two_ints_per_line(query_ranges_file);
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
    cout << "Loaded " << query_ranges.size() << " query ranges" << endl;

    // Load groundtruth
    vector<vector<int>> groundtruth;
    try {
        groundtruth = read_ivecs(groundtruth_file);
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
    
    // Truncate groundtruth to k
    for (auto& gt : groundtruth) {
        if (gt.size() > (size_t)k) {
            gt.resize(k);
        }
    }
    cout << "Loaded " << groundtruth.size() << " groundtruth entries" << endl;

    // ========== INDEX CONSTRUCTION (TIMED) ==========
    cout << "\n--- Starting index construction (TIMED) ---" << endl;
    
    // Start thread monitoring for index construction
    atomic<bool> done_build(false);
    peak_threads = 1;  // Reset
    thread monitor_build(monitor_thread_count, ref(done_build));
    
    auto start_build = high_resolution_clock::now();

    // Construct DIGRA RangeHNSW index
    RangeHNSW* rangeHnsw = nullptr;
    try {
        rangeHnsw = new RangeHNSW(dim, baseNum, baseNum, data, keys, values, M, ef_construction);
    } catch (const exception& e) {
        cerr << "ERROR: Exception during index construction: " << e.what() << endl;
        done_build = true;
        monitor_build.join();
        delete[] data;
        delete[] query;
        delete[] keys;
        delete[] values;
        return 1;
    }
    
    auto end_build = high_resolution_clock::now();
    
    // Stop thread monitoring for build phase
    done_build = true;
    monitor_build.join();
    int build_threads = peak_threads.load();
    
    double build_time_sec = duration_cast<duration<double>>(end_build - start_build).count();
    cout << "--- Index construction complete ---" << endl;

    // ========== QUERY EXECUTION (TIMED per ef_search value) ==========
    cout << "\n--- Starting query execution ---" << endl;
    
    // Force single-threaded query execution (standard for benchmarking)
    omp_set_num_threads(1);
    
    // Start thread monitoring for query phase
    atomic<bool> done_query(false);
    peak_threads = 1;  // Reset
    thread monitor_query(monitor_thread_count, ref(done_query));

    // Store results for each ef_search value
    vector<double> recall_list;
    vector<double> qps_list;

    for (int ef_search : ef_search_list) {
        vector<vector<int>> query_results(queryNum);
        
        auto start_query = high_resolution_clock::now();

        // Execute queries
        for (int i = 0; i < queryNum; i++) {
            int rangeL = query_ranges[i].first;
            int rangeR = query_ranges[i].second;
            
            // Perform range-filtered query
            auto result = rangeHnsw->queryRange(query + i * dim, rangeL, rangeR, k, ef_search);
            
            // Extract IDs from priority queue
            query_results[i].reserve(k);
            while (!result.empty()) {
                query_results[i].push_back(result.top().second);
                result.pop();
            }
        }

        auto end_query = high_resolution_clock::now();
        
        // Compute timing
        double query_time_sec = duration_cast<duration<double>>(end_query - start_query).count();
        double qps = queryNum / query_time_sec;

        // Compute recall
        int total_true_positives = 0;
        for (int i = 0; i < queryNum; i++) {
            set<int> result_set(query_results[i].begin(), query_results[i].end());
            
            int gt_size = min(k, (int)groundtruth[i].size());
            for (int j = 0; j < gt_size; j++) {
                if (result_set.count(groundtruth[i][j])) {
                    total_true_positives++;
                }
            }
        }
        
        double recall = (double)total_true_positives / (queryNum * k);
        
        recall_list.push_back(recall);
        qps_list.push_back(qps);
    }

    // Stop thread monitoring for query phase
    done_query = true;
    monitor_query.join();
    int query_threads = peak_threads.load();

    cout << "--- Query execution complete ---\n" << endl;

    // ========== OUTPUT RESULTS ==========
    // Output in same format as SeRF for compatibility with serf.py parsing
    peak_memory_footprint();
    printf("Maximum number of threads during index construction: %d\n", build_threads - 1);  // Subtract monitoring thread
    printf("Maximum number of threads during query execution: %d\n", query_threads - 1);
    printf("Index construction time: %.3f s\n", build_time_sec);
    
    for (size_t i = 0; i < ef_search_list.size(); i++) {
        printf("ef_search: %d QPS: %.3f Recall: %.5f\n", ef_search_list[i], qps_list[i], recall_list[i]);
    }

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
