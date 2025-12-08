// csv_to_data_converter.cpp - Convert FANNS .csv attributes to DIGRA .data format
// This converter reads a CSV file with a header line and converts it to DIGRA's
// simple "key value" text format where key is the 0-indexed position

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

using namespace std;

int main(int argc, char** argv) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <input.csv> <output.data>\n";
        cerr << "\n";
        cerr << "Converts FANNS .csv attribute file (with header) to DIGRA .data format\n";
        cerr << "Input CSV format: header line + one integer value per line\n";
        cerr << "Output .data format: 'key value' pairs (0-indexed keys)\n";
        return 1;
    }

    string input_csv = argv[1];
    string output_data = argv[2];

    // Open input file
    ifstream infile(input_csv);
    if (!infile.is_open()) {
        cerr << "Error: Cannot open input file: " << input_csv << endl;
        return 1;
    }

    // Read all attribute values (skip header)
    vector<int> values;
    string line;
    int line_number = 0;
    
    while (getline(infile, line)) {
        line_number++;
        
        // Skip header line
        if (line_number == 1) {
            cout << "Skipping header: " << line << endl;
            continue;
        }
        
        // Skip empty lines
        if (line.empty()) {
            continue;
        }
        
        // Parse integer value
        stringstream ss(line);
        int value;
        if (!(ss >> value)) {
            cerr << "Error: Invalid integer at line " << line_number << ": " << line << endl;
            return 1;
        }
        
        values.push_back(value);
    }
    infile.close();
    
    cout << "Read " << values.size() << " attribute values from " << input_csv << endl;

    // Write output file in .data format
    ofstream outfile(output_data);
    if (!outfile.is_open()) {
        cerr << "Error: Cannot open output file: " << output_data << endl;
        return 1;
    }

    for (size_t i = 0; i < values.size(); i++) {
        outfile << i << " " << values[i] << "\n";
    }
    outfile.close();
    
    cout << "Wrote " << values.size() << " key-value pairs to " << output_data << endl;
    cout << "Conversion complete!" << endl;

    return 0;
}
