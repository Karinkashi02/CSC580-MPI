#include <random>
#include <fstream>
#include <vector>
#include <iostream>
#include <algorithm>  // for std::min

void generateBinaryDataset(const std::string& filename, long long numElements) {
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 10000.0);
    
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;  // FIXED: << instead of B
        return;
    }
    
    // Write header
    file.write(reinterpret_cast<const char*>(&numElements), sizeof(numElements));
    
    // Generate and write in chunks to manage memory
    const long long CHUNK_SIZE = 1000000;  // 1M elements at a time
    std::vector<double> buffer(CHUNK_SIZE);
    
    for (long long processed = 0; processed < numElements; processed += CHUNK_SIZE) {
        long long currentChunk = std::min(CHUNK_SIZE, numElements - processed);
        for (long long i = 0; i < currentChunk; ++i) {
            buffer[i] = dist(rng);
        }
        file.write(reinterpret_cast<const char*>(buffer.data()), 
                   currentChunk * sizeof(double));
    }
    file.close();
    
    std::cout << "Generated " << numElements << " elements to " << filename << std::endl;
}

int main() {
    std::cout << "Generating datasets...\n";
    generateBinaryDataset("dataset_1M.bin", 1000000);
    generateBinaryDataset("dataset_10M.bin", 10000000);
    generateBinaryDataset("dataset_100M.bin", 100000000);
    std::cout << "All datasets generated successfully!\n";
    return 0;
}